"""
train.py
LoRA fine-tuning of Phi-3 Mini on financial Q&A data.
Auto-detects device: CUDA (RunPod) or MPS (M1 Mac).

Usage:
    python src/sft/train.py                  # full run
    python src/sft/train.py --smoke_test     # 100 examples, 1 epoch
"""

import argparse
import json
import os
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
TRAIN_FILE = "data/processed/sft_train.jsonl"
VAL_FILE = "data/processed/sft_val.jsonl"
OUTPUT_DIR = "outputs/sft"
MAX_SEQ_LENGTH = 1024

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                   # rank — higher = more capacity, more memory
    lora_alpha=32,          # scaling factor (usually 2x rank)
    lora_dropout=0.05,
    target_modules=[        # Phi-3 attention layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)

# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        print("Device: CUDA")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Device: MPS (M1 Mac)")
        return "mps"
    else:
        print("Device: CPU")
        return "cpu"

# ── Data ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_messages(example: dict, tokenizer) -> dict:
    """Apply Phi-3 chat template to messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def prepare_dataset(path: str, tokenizer, smoke_test: bool = False) -> Dataset:
    data = load_jsonl(path)
    if smoke_test:
        data = data[:100]
    dataset = Dataset.from_list(data)
    dataset = dataset.map(
        lambda ex: format_messages(ex, tokenizer),
        remove_columns=["messages"],
    )
    return dataset

# ── Training ──────────────────────────────────────────────────────────────────

def main(smoke_test: bool = False):
    device = get_device()

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
        device_map="auto" if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to("mps")

    # Apply LoRA
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Load data
    print("Preparing datasets...")
    train_dataset = prepare_dataset(TRAIN_FILE, tokenizer, smoke_test)
    val_dataset = prepare_dataset(VAL_FILE, tokenizer, smoke_test)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Training args
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1 if smoke_test else 3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        eval_strategy="steps",
        eval_steps=100 if smoke_test else 500,
        save_strategy="steps",
        save_steps=100 if smoke_test else 500,
        logging_steps=10,
        fp16=device == "cuda",
        report_to="wandb" if not smoke_test else "none",
        run_name="caif-sft-smoke" if smoke_test else "caif-sft",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on 100 examples for 1 epoch to verify setup")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)