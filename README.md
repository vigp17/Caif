# CAIF — Constitutional AI for Finance

A from-scratch implementation of Anthropic's [Constitutional AI](https://arxiv.org/abs/2212.08073) pipeline scoped to financial Q&A. Built to understand the core mechanics of CAI: supervised fine-tuning, AI-generated preference data, reward modeling, and reinforcement learning with GRPO.

---

## Motivation

Anthropic's Constitutional AI paper introduced a powerful idea: instead of relying on expensive human preference labels, you can use an LLM to critique and revise model outputs against a written set of principles — a "constitution." This project implements that pipeline end-to-end in a constrained domain (financial advice) where alignment failures are concrete and evaluable.

---

## Pipeline

```
Raw Data → [SFT] → [AI Feedback Loop] → [Reward Model] → [GRPO] → Aligned Model
                         ↑
                   Constitution
               (10 financial principles)
```

### Stage 1 — Supervised Fine-Tuning
- Base model: `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)
- Fine-tuned with LoRA (r=16, α=32) on 51,013 financial Q&A pairs from `gbharti/finance-alpaca`
- Trained for 3 epochs on RunPod A40 (~10 hours)
- Final train loss: 1.246 | Val loss: 0.773 | Token accuracy: 79.4%

### Stage 2 — Constitution
10 explicit principles governing financial advice behavior, including:
- No specific security recommendations
- Always disclose uncertainty
- Defer to licensed professionals
- No guaranteed returns
- Acknowledge regulatory differences

See [`constitution/financial_constitution.md`](constitution/financial_constitution.md)

### Stage 3 — AI Feedback Loop
- Generated 500 preference pairs using Claude as judge
- For each question: SFT model generates a response → Claude critiques it against a random constitution principle → Claude produces a revised, compliant response
- Output: (question, bad\_response, critique, good\_response) pairs
- 0 errors across 500 generations

### Stage 4 — Reward Model
- Architecture: SFT backbone (frozen) + linear scalar head
- Trained on preference pairs using Bradley-Terry loss
- Final validation accuracy: **57.7%** on 50 held-out pairs

### Stage 5 — GRPO Fine-Tuning
- Group size G=4, KL penalty β=0.05
- 2 epochs over 495 unique questions
- Average reward improved from -3.65 → -1.33 across training

---

## Results

| Model | Compliance (0–10) | Helpfulness (0–10) |
|-------|:-----------------:|:------------------:|
| Base Phi-3 | 9.80 | 3.40 |
| SFT | 10.00 | 4.00 |
| GRPO | 10.00 | 0.20 |

*Scored by Claude as judge on 10 held-out financial questions.*

---

## Key Finding: Reward Hacking

GRPO compliance reached ceiling but helpfulness collapsed to near-zero — a textbook case of **reward hacking**. The policy exploited weaknesses in the reward model (57.7% accuracy) rather than learning genuine constitutional alignment. This is consistent with Goodhart's Law: when a measure becomes a target, it ceases to be a good measure.

This finding motivates three improvements:

1. **More preference pairs** — 5,000+ instead of 500 to train a stronger reward model
2. **Higher KL penalty** — β=0.15 instead of 0.05 to prevent policy drift
3. **Financial-only data filtering** — the training dataset contained non-financial questions (haikus, travel recommendations) which degraded reward model quality

---

## Stack

| Component | Tool |
|-----------|------|
| Base model | Phi-3 Mini 4k Instruct |
| Fine-tuning | HuggingFace `transformers` + `peft` (LoRA) |
| RL training | `trl` (GRPO) |
| AI Feedback | Anthropic Python SDK (`claude-sonnet-4-20250514`) |
| Compute | RunPod A40 (48GB VRAM) |

---

## Repo Structure

```
caif/
├── constitution/
│   └── financial_constitution.md   # 10 alignment principles
├── src/
│   ├── sft/
│   │   ├── prepare_data.py         # Download + format FiQA data
│   │   └── train.py                # LoRA fine-tuning
│   ├── feedback/
│   │   └── generate_pairs.py       # Claude AI feedback loop
│   ├── reward_model/
│   │   └── train_rm.py             # Bradley-Terry reward model
│   └── rl/
│       └── grpo.py                 # GRPO fine-tuning
└── eval/
    └── evaluate.py                 # 3-model comparison
```

---

## References

- Bai et al. (2022). [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). Anthropic.
- Shao et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). (GRPO)
- Rafailov et al. (2023). [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).