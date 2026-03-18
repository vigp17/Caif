# CAIF — Constitutional AI for Finance

A from-scratch implementation of Anthropic's [Constitutional AI](https://arxiv.org/abs/2212.08073) pipeline scoped to financial Q&A. Built to understand the core mechanics of CAI: supervised fine-tuning, AI-generated preference data, reward modeling, and reinforcement learning with GRPO.

---

## Motivation

Anthropic's Constitutional AI paper introduced a powerful idea: instead of relying on expensive human preference labels, you can use an LLM to critique and revise model outputs against a written set of principles — a "constitution." This project implements that pipeline end-to-end in a constrained domain (financial advice) where alignment failures are concrete and evaluable: recommending specific stocks, implying guaranteed returns, or giving advice that requires a licensed professional.

---

## Pipeline

```
Raw Data → [SFT] → [AI Feedback Loop] → [Reward Model] → [GRPO] → Aligned Model
                          ↑
                    Constitution
                (10 financial principles)
```

### Stage 1 — Supervised Fine-Tuning
Fine-tuned `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters) with LoRA (r=16, α=32) on 51,013 financial Q&A pairs from `gbharti/finance-alpaca`. Trained for 3 epochs on RunPod A40.

- Train loss: 1.246 → Val loss: 0.773
- Token accuracy: 79.4%
- Training time: ~10 hours

### Stage 2 — Constitution
10 explicit principles governing financial advice, including no specific security recommendations, mandatory uncertainty disclosure, deferral to licensed professionals, and prohibition on guaranteed return claims.

Full constitution: [`constitution/financial_constitution.md`](constitution/financial_constitution.md)

### Stage 3 — AI Feedback Loop
For each question, the SFT model generates a response. Claude then critiques it against a randomly selected constitution principle and produces a revised, compliant version. This generates (question, bad\_response, critique, good\_response) preference pairs entirely through AI feedback — no human labelers needed.

- 500 preference pairs generated
- 0 errors across all generations
- Avg generation time: ~5 min per 10 pairs

### Stage 4 — Reward Model
SFT backbone (frozen) with a linear scalar head trained on preference pairs using Bradley-Terry loss. The reward head learns to score responses — higher = more constitutional.

- Validation accuracy: 57.7% on 50 held-out pairs
- 3 epochs, AdamW optimizer

### Stage 5 — GRPO Fine-Tuning
Group Relative Policy Optimization with group size G=4 and KL penalty β=0.05. For each question, 4 responses are generated, scored by the reward model, and relative advantages are computed. The policy gradient pushes up responses that scored above the group mean.

- 2 epochs over 495 unique questions
- Average reward improved: -3.65 → -1.33

---

## Results

| Model | Compliance (0–10) | Helpfulness (0–10) |
|-------|:-----------------:|:------------------:|
| Base Phi-3 | 9.82 | 5.32 |
| SFT | 9.80 | 4.68 |
| GRPO | 9.70 | 0.04 |

*Scored by Claude as judge on 50 held-out financial questions.*

---

## Key Finding: Reward Hacking

GRPO helpfulness collapsed to near-zero while compliance held steady — a textbook case of **reward hacking**. The policy exploited weaknesses in the reward model (57.7% accuracy trained on only 500 pairs) rather than learning genuine constitutional alignment. This is consistent with Goodhart's Law: when a measure becomes a target, it ceases to be a good measure.

The SFT model actually achieved the best balance — 9.80 compliance with 4.68 helpfulness — suggesting that domain fine-tuning alone provides substantial alignment benefit without the instability introduced by RL against a weak reward model.

**Three improvements that would fix this:**

1. **More preference pairs** — 5,000+ instead of 500 to train a more accurate reward model
2. **Higher KL penalty** — β=0.15 instead of 0.05 to prevent policy drift from the SFT baseline
3. **Financial-only data filtering** — the training dataset contained non-financial questions (haikus, travel recommendations) which degraded reward model signal quality

---

## Stack

| Component | Tool |
|-----------|------|
| Base model | Phi-3 Mini 4k Instruct (3.8B) |
| Fine-tuning | HuggingFace `transformers` + `peft` (LoRA) |
| RL training | `trl` (GRPO) |
| AI Feedback | Anthropic Python SDK (`claude-sonnet-4-20250514`) |
| Compute | RunPod A40 (48GB VRAM) |
| Evaluation | Claude as judge (compliance + helpfulness) |

---

## Repo Structure

```
caif/
├── constitution/
│   └── financial_constitution.md   # 10 alignment principles
├── src/
│   ├── sft/
│   │   ├── prepare_data.py         # Download + format training data
│   │   └── train.py                # LoRA fine-tuning
│   ├── feedback/
│   │   └── generate_pairs.py       # Claude AI feedback loop
│   ├── reward_model/
│   │   └── train_rm.py             # Bradley-Terry reward model
│   └── rl/
│       └── grpo.py                 # GRPO fine-tuning
└── eval/
    └── evaluate.py                 # 3-model comparison (Claude as judge)
```

---

## Reproducing

```bash
git clone https://github.com/vigp17/Caif.git
cd Caif
python -m venv venv && source venv/bin/activate
pip install torch transformers peft trl accelerate anthropic wandb bitsandbytes datasets

cp .env.example .env
# Add your ANTHROPIC_API_KEY, WANDB_API_KEY, HF_TOKEN

python src/sft/prepare_data.py
python src/sft/train.py
python src/feedback/generate_pairs.py --num_samples 500
python src/reward_model/train_rm.py
python src/rl/grpo.py
python eval/evaluate.py --num_questions 50
```

Full training requires a GPU with 24GB+ VRAM. Estimated cost on RunPod A40: ~$8-$12

---

## References

- Bai et al. (2022). [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). Anthropic.
- Shao et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300). (GRPO)
- Rafailov et al. (2023). [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).