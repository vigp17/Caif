# AGENTS.md

## Cursor Cloud specific instructions

### What this project is
CAIF (Constitutional AI for Finance) is a **GPU-dependent Python ML training pipeline**, not a web app or long-running service. There is nothing to `docker compose up` and no server/port to expose. The pipeline is a sequence of CLI scripts (see `README.md` "Reproducing" for the canonical stage order and commands).

### Environment
- Dependencies live in a virtualenv at `venv/` (gitignored). Always invoke Python as `venv/bin/python ...` (or `source venv/bin/activate` first). There is **no `requirements.txt`/`pyproject.toml`** — the dependency list is the single `pip install` line in `README.md`, mirrored by the startup update script.
- `venv/bin/python` is Python 3.12. Creating the venv requires the `python3.12-venv` system package (installed once during environment setup; persisted in the VM snapshot).
- There is **no test suite and no lint config**. The closest thing to a test is the `--smoke_test` flag on the training scripts.

### Secrets (set in the Secrets panel, not committed)
- `ANTHROPIC_API_KEY` — required only for Stage 3 (`src/feedback/generate_pairs.py`) and Stage 6 (`eval/evaluate.py`), which call Claude. Scripts read it from the env first, then fall back to parsing `.env`, so `cp .env.example .env` and/or exporting the var both work.
- `HF_TOKEN` — optional. The `gbharti/finance-alpaca` dataset and `microsoft/Phi-3-mini-4k-instruct` are public; downloads work unauthenticated (just rate-limited). Set it to avoid the "unauthenticated requests" warning / rate limits.
- `WANDB_API_KEY` — optional. Only used by `src/sft/train.py` on full (non-smoke) runs; smoke runs set `report_to="none"`.

### GPU / hardware caveat (important)
- The default Cursor Cloud VM is **CPU-only with ~15GB RAM and no CUDA GPU**. `torch.cuda.is_available()` is `False`.
- Stages that hardcode `.cuda()` / `device_map="auto"` / `"cuda"` — reward model (`src/reward_model/train_rm.py`), GRPO (`src/rl/grpo.py`), and eval (`eval/evaluate.py`) — **cannot run here**; they need a CUDA GPU (README targets 24GB+ VRAM).
- Even `src/sft/train.py --smoke_test`, which falls back to CPU, **OOM-kills**: Phi-3-mini in float32 needs ~15GB just for weights, which meets/exceeds total RAM. Do not expect SFT/RM/GRPO/eval to complete on the default VM.
- What **does** run CPU-only: `src/sft/prepare_data.py` (downloads + formats the finance dataset into `data/processed/sft_{train,val}.jsonl`). This is the pipeline entry point and the practical smoke check that the environment is healthy.
- To exercise the full pipeline, run on a machine with a 24GB+ VRAM CUDA GPU.

### Regenerated artifacts
`data/`, `outputs/`, and `wandb/` are gitignored; nothing is checked in. Every stage regenerates its outputs, and later stages depend on earlier stage outputs on disk (see `README.md` for the dependency chain).
