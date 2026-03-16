# CAIF — Constitutional AI for Finance

A from-scratch implementation of Anthropic's Constitutional AI pipeline scoped to financial Q&A.

## Pipeline
1. SFT — Fine-tune Phi-3 Mini on financial Q&A data
2. Constitution — Written principles for financial advice
3. AI Feedback — Claude critiques + revises SFT outputs
4. Reward Model — Trained on AI-generated preference pairs
5. GRPO — RL fine-tuning against the reward model

## Stack
- Model: Phi-3 Mini (3.8B) + LoRA
- RL: GRPO
- Framework: HuggingFace transformers + trl + peft
- AI Feedback: Anthropic Python SDK
