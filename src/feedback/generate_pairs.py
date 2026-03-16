
"""
generate_pairs.py
Stage 3 — AI Feedback Loop (Constitutional AI core)

For each financial question:
1. Generate a response from the SFT model
2. Send it to Claude with a constitution principle
3. Get back a critique + revised response
4. Save (question, bad_response, good_response) as preference pairs

Output: data/processed/preference_pairs.jsonl

Usage:
    python src/feedback/generate_pairs.py --num_samples 500
"""

import argparse
import json
import os
import random
import torch
from anthropic import Anthropic
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

SFT_MODEL_PATH = "outputs/sft/final"
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
TRAIN_FILE = "data/processed/sft_train.jsonl"
OUTPUT_FILE = "data/processed/preference_pairs.jsonl"
CONSTITUTION_FILE = "constitution/financial_constitution.md"

SYSTEM_PROMPT = (
    "You are a financial education assistant. You provide clear, accurate, "
    "and balanced information about finance and investing. You are not a "
    "licensed financial advisor and do not provide personalized investment advice."
)

# ── Load Constitution ─────────────────────────────────────────────────────────

def load_principles(path: str) -> list[str]:
    """Parse constitution markdown into individual principles."""
    principles = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("## Principle"):
                if current:
                    principles.append("\n".join(current).strip())
                current = [line]
            elif current:
                current.append(line)
    if current:
        principles.append("\n".join(current).strip())
    print(f"Loaded {len(principles)} principles from constitution")
    return principles

# ── Load SFT Model ────────────────────────────────────────────────────────────

def load_sft_model(device: str):
    print(f"Loading base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.unk_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
        device_map="auto" if device == "cuda" else None,
    )

    print(f"Loading LoRA weights from: {SFT_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, SFT_MODEL_PATH)
    model.eval()

    if device == "mps":
        model = model.to("mps")

    return model, tokenizer

# ── Generate SFT Response ─────────────────────────────────────────────────────

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
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ── Claude Critique + Revision ────────────────────────────────────────────────

def get_critique_and_revision(
    client: Anthropic,
    question: str,
    response: str,
    principle: str,
) -> tuple[str, str]:
    prompt = f"""You are evaluating a financial AI assistant's response against a constitutional principle.

QUESTION:
{question}

RESPONSE TO EVALUATE:
{response}

CONSTITUTIONAL PRINCIPLE:
{principle}

Your task:
1. Write a brief CRITIQUE explaining whether and how the response violates this principle.
2. Write a REVISED RESPONSE that fully complies with the principle while still being helpful.

Format your answer exactly like this:
CRITIQUE: <your critique here>
REVISED: <your revised response here>"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    content = message.content[0].text
    critique, revised = "", ""

    for line in content.split("\n"):
        if line.startswith("CRITIQUE:"):
            critique = line[len("CRITIQUE:"):].strip()
        elif line.startswith("REVISED:"):
            revised = line[len("REVISED:"):].strip()

    return critique, revised

# ── Main ──────────────────────────────────────────────────────────────────────

def main(num_samples: int = 500):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load constitution
    principles = load_principles(CONSTITUTION_FILE)

    # Load questions from training data
    print("Loading questions...")
    with open(TRAIN_FILE) as f:
        all_examples = [json.loads(line) for line in f]

    # Extract just the user questions
    questions = [
        ex["messages"][1]["content"]
        for ex in all_examples
        if len(ex["messages"]) >= 2
    ]
    random.shuffle(questions)
    questions = questions[:num_samples]
    print(f"Sampled {len(questions)} questions")

    # Load SFT model
    model, tokenizer = load_sft_model(device)

    # Init Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try loading from .env
        with open(".env") as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
    client = Anthropic(api_key=api_key)

    # Generate preference pairs
    os.makedirs("data/processed", exist_ok=True)
    pairs = []
    errors = 0

    with open(OUTPUT_FILE, "w") as out_f:
        for i, question in enumerate(questions):
            try:
                # Pick a random principle for this example
                principle = random.choice(principles)

                # Generate SFT response (potentially bad)
                bad_response = generate_response(model, tokenizer, question, device)

                # Claude critique + revision
                critique, good_response = get_critique_and_revision(
                    client, question, bad_response, principle
                )

                if not good_response:
                    errors += 1
                    continue

                pair = {
                    "question": question,
                    "principle": principle,
                    "bad_response": bad_response,
                    "critique": critique,
                    "good_response": good_response,
                }
                out_f.write(json.dumps(pair) + "\n")
                pairs.append(pair)

                if (i + 1) % 10 == 0:
                    print(f"[{i+1}/{len(questions)}] Generated {len(pairs)} pairs ({errors} errors)")

            except Exception as e:
                print(f"Error on question {i}: {e}")
                errors += 1
                continue

    print(f"\nDone. {len(pairs)} preference pairs saved to {OUTPUT_FILE}")
    print(f"Errors: {errors}")

    # Preview one pair
    if pairs:
        print("\n--- Sample Pair ---")
        p = pairs[0]
        print(f"Q: {p['question'][:100]}...")
        print(f"Principle: {p['principle'][:80]}...")
        print(f"Critique: {p['critique'][:150]}...")
        print(f"Bad: {p['bad_response'][:100]}...")
        print(f"Good: {p['good_response'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of preference pairs to generate")
    args = parser.parse_args()
    main(num_samples=args.num_samples)
