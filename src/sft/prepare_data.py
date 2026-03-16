"""
prepare_data.py
Downloads finance-alpaca dataset from HuggingFace and formats it for SFT training.
Output: data/processed/sft_train.jsonl and sft_val.jsonl
"""

import json
import os
from datasets import load_dataset

PROCESSED_DIR = "data/processed"
TRAIN_FILE = os.path.join(PROCESSED_DIR, "sft_train.jsonl")
VAL_FILE = os.path.join(PROCESSED_DIR, "sft_val.jsonl")

SYSTEM_PROMPT = (
    "You are a financial education assistant. You provide clear, accurate, "
    "and balanced information about finance and investing. You are not a "
    "licensed financial advisor and do not provide personalized investment advice."
)


def format_example(instruction: str, output: str) -> dict:
    """Format a Q&A pair into chat format for Phi-3."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": output.strip()},
        ]
    }


def main():
    print("Loading finance-alpaca dataset...")
    dataset = load_dataset("gbharti/finance-alpaca", split="train")
    print(f"Total raw examples: {len(dataset)}")

    examples = []
    for row in dataset:
        instruction = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        output = row.get("output", "").strip()

        # Combine instruction + input if input exists
        question = f"{instruction}\n{inp}" if inp else instruction

        if question and output and len(output) > 50:
            examples.append(format_example(question, output))

    print(f"Filtered examples: {len(examples)}")

    # 90/10 train/val split
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open(TRAIN_FILE, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(VAL_FILE, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved to {TRAIN_FILE} and {VAL_FILE}")

    # Preview one example
    print("\n--- Sample Example ---")
    print(json.dumps(train_examples[0], indent=2))


if __name__ == "__main__":
    main()