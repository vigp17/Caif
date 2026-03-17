"""
grpo.py
Stage 5 — GRPO Fine-tuning (Group Relative Policy Optimization)

Uses the reward model as the training signal to further align the SFT model.
For each question, generates G responses, scores them with the RM,
computes relative advantages, and updates the policy.

Output: outputs/grpo/

Usage:
    python src/rl/grpo.py --smoke_test
    python src/rl/grpo.py
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import copy

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SFT_MODEL_PATH = "outputs/sft/final"
RM_PATH = "outputs/reward_model/best_rm.pt"
PAIRS_FILE = "data/processed/preference_pairs.jsonl"
OUTPUT_DIR = "outputs/grpo"

SYSTEM_PROMPT = (
    "You are a financial education assistant. You provide clear, accurate, "
    "and balanced information about finance and investing. You are not a "
    "licensed financial advisor and do not provide personalized investment advice."
)

G = 4           # group size — responses per question
BETA = 0.05     # KL penalty coefficient
LR = 1e-5
MAX_NEW_TOKENS = 200
MAX_INPUT_LENGTH = 512

# ── Reward Model Head ─────────────────────────────────────────────────────────

class RewardHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=False).cuda().float()

    def forward(self, hidden):
        return self.linear(hidden.float()).squeeze(-1)

# ── Load Models ───────────────────────────────────────────────────────────────

def load_policy_model(device: str):
    """Load SFT model as the trainable policy."""
    print("Loading policy model (SFT)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        attn_implementation="eager",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, SFT_MODEL_PATH)
    model.train()
    return model


def load_reference_model(device: str):
    """Load frozen SFT model as KL reference."""
    print("Loading reference model (frozen SFT)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        attn_implementation="eager",
        device_map="auto",
    )
    ref_model = PeftModel.from_pretrained(base, SFT_MODEL_PATH)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


def load_reward_head(hidden_size: int) -> RewardHead:
    """Load trained reward head."""
    print("Loading reward head...")
    head = RewardHead(hidden_size)
    head.linear.load_state_dict(torch.load(RM_PATH, map_location="cuda"))
    head.eval()
    for param in head.parameters():
        param.requires_grad = False
    return head

# ── Score Response ────────────────────────────────────────────────────────────

def score_response(
    model,
    reward_head: RewardHead,
    tokenizer,
    question: str,
    response: str,
    device: str,
) -> float:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=MAX_INPUT_LENGTH, padding="max_length"
    ).to(device)

    with torch.no_grad():
        outputs = model(
            **enc,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :].float()
        score = reward_head(last_hidden).item()
    return score

# ── Generate Response ─────────────────────────────────────────────────────────

def generate_response(model, tokenizer, question: str, device: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ── Compute Log Probs ─────────────────────────────────────────────────────────

def compute_log_prob(model, tokenizer, question: str, response: str, device: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH
    ).to(device)

    outputs = model(**enc)
    logits = outputs.logits[:, :-1, :]
    labels = enc["input_ids"][:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.mean()

# ── GRPO Training Loop ────────────────────────────────────────────────────────

def main(smoke_test: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load questions
    print("Loading questions...")
    with open(PAIRS_FILE) as f:
        pairs = [json.loads(line) for line in f]

    questions = list(set([p["question"] for p in pairs]))
    if smoke_test:
        questions = questions[:5]
    print(f"Questions: {len(questions)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # Load models
    policy = load_policy_model(device)
    ref_model = load_reference_model(device)
    hidden_size = policy.config.hidden_size
    reward_head = load_reward_head(hidden_size)

    # Optimizer — only LoRA params
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=LR,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_epochs = 1 if smoke_test else 2

    for epoch in range(num_epochs):
        total_loss = 0
        total_reward = 0

        for i, question in enumerate(questions):
            # Step 1 — Generate G responses
            responses = [
                generate_response(policy, tokenizer, question, device)
                for _ in range(G)
            ]

            # Step 2 — Score all responses with reward model
            scores = [
                score_response(policy, reward_head, tokenizer, question, r, device)
                for r in responses
            ]
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            # Step 3 — Compute group relative advantages
            mean_score = scores_tensor.mean()
            std_score = scores_tensor.std() + 1e-8
            advantages = (scores_tensor - mean_score) / std_score

            # Step 4 — Policy gradient with KL penalty
            loss = torch.tensor(0.0, requires_grad=True, device=device)

            for response, advantage in zip(responses, advantages):
                # Policy log prob
                log_prob = compute_log_prob(policy, tokenizer, question, response, device)

                # Reference log prob (for KL)
                with torch.no_grad():
                    ref_log_prob = compute_log_prob(ref_model, tokenizer, question, response, device)

                # KL penalty
                kl = log_prob - ref_log_prob

                # GRPO objective: maximize advantage, minimize KL drift
                step_loss = -(advantage.to(device) * log_prob - BETA * kl)
                loss = loss + step_loss / G

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            total_reward += scores_tensor.mean().item()

            if (i + 1) % 5 == 0:
                avg_loss = total_loss / (i + 1)
                avg_reward = total_reward / (i + 1)
                print(f"Epoch {epoch+1} [{i+1}/{len(questions)}] Loss: {avg_loss:.4f} | Avg Reward: {avg_reward:.4f}")

        print(f"Epoch {epoch+1} complete | Loss: {total_loss/len(questions):.4f} | Avg Reward: {total_reward/len(questions):.4f}")

    # Save final policy
    print(f"Saving policy to {OUTPUT_DIR}/final")
    policy.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)