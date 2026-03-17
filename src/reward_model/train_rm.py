"""
train_rm.py
Stage 4 — Reward Model Training

Trains a reward model on the preference pairs generated in Stage 3.
Architecture: SFT model backbone + linear head on last hidden state.
Loss: Bradley-Terry — good responses should score higher than bad ones.

Output: outputs/reward_model/

Usage:
    python src/reward_model/train_rm.py --smoke_test
    python src/reward_model/train_rm.py
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import wandb

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SFT_MODEL_PATH = "outputs/sft/final"
PAIRS_FILE = "data/processed/preference_pairs.jsonl"
OUTPUT_DIR = "outputs/reward_model"
WANDB_MODE = os.environ.get("WANDB_MODE", "disabled")

SYSTEM_PROMPT = (
    "You are a financial education assistant. You provide clear, accurate, "
    "and balanced information about finance and investing. You are not a "
    "licensed financial advisor and do not provide personalized investment advice."
)

# ── Reward Model ──────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """SFT backbone + scalar head."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use last token's hidden state as the reward signal
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        reward = self.reward_head(last_hidden).squeeze(-1)
        return reward

# ── Dataset ───────────────────────────────────────────────────────────────────

class PreferenceDataset(Dataset):
    def __init__(self, pairs: list, tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def format_text(self, question: str, response: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        good_text = self.format_text(pair["question"], pair["good_response"])
        bad_text = self.format_text(pair["question"], pair["bad_response"])

        good_enc = self.tokenizer(
            good_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        bad_enc = self.tokenizer(
            bad_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )

        return {
            "good_input_ids": good_enc["input_ids"].squeeze(0),
            "good_attention_mask": good_enc["attention_mask"].squeeze(0),
            "bad_input_ids": bad_enc["input_ids"].squeeze(0),
            "bad_attention_mask": bad_enc["attention_mask"].squeeze(0),
        }

# ── Training ──────────────────────────────────────────────────────────────────

def main(smoke_test: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load pairs
    print("Loading preference pairs...")
    with open(PAIRS_FILE) as f:
        pairs = [json.loads(line) for line in f]

    if smoke_test:
        pairs = pairs[:20]

    # 90/10 split
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # Load SFT backbone
    print("Loading SFT backbone...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
        device_map="auto" if device == "cuda" else None,
    )
    backbone = PeftModel.from_pretrained(base_model, SFT_MODEL_PATH)

    # Wrap with reward head
    model = RewardModel(backbone)
    if device != "cuda":
        model = model.to(device)

    # Only train the reward head + LoRA params
    for name, param in model.named_parameters():
        if "reward_head" in name or "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    # Datasets
    train_dataset = PreferenceDataset(train_pairs, tokenizer)
    val_dataset = PreferenceDataset(val_pairs, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    # Training loop
    num_epochs = 1 if smoke_test else 3
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_acc = 0

        for batch in train_loader:
            good_ids = batch["good_input_ids"].to(device)
            good_mask = batch["good_attention_mask"].to(device)
            bad_ids = batch["bad_input_ids"].to(device)
            bad_mask = batch["bad_attention_mask"].to(device)

            good_scores = model(good_ids, good_mask)
            bad_scores = model(bad_ids, bad_mask)

            # Bradley-Terry loss
            loss = -torch.log(torch.sigmoid(good_scores - bad_scores)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_acc += (good_scores > bad_scores).float().mean().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for batch in val_loader:
                good_ids = batch["good_input_ids"].to(device)
                good_mask = batch["good_attention_mask"].to(device)
                bad_ids = batch["bad_input_ids"].to(device)
                bad_mask = batch["bad_attention_mask"].to(device)

                good_scores = model(good_ids, good_mask)
                bad_scores = model(bad_ids, bad_mask)

                loss = -torch.log(torch.sigmoid(good_scores - bad_scores)).mean()
                val_loss += loss.item()
                val_acc += (good_scores > bad_scores).float().mean().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_rm.pt")
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

    print(f"\nDone. Best reward model saved to {OUTPUT_DIR}/best_rm.pt")
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)