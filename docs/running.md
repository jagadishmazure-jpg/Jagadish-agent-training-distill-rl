# Running

## Offline (default, free)

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
distillrl demo      # distillation, DPO and GRPO tables (about 25 s)
distillrl curves    # writes results/reward_curve.csv and results/reward_curve.svg
distillrl bench     # trains the MLP, quantises it, times everything, writes results/bench.json
python scripts/render_docs.py          # refresh README and docs blocks
python scripts/render_docs.py --check  # what CI runs
```

Requirements: Python 3.11+, about 1 GB of disk for the CPU torch wheel, any 4-core laptop.

## Checks run in CI

ruff lint and format, pytest, printed test count, demo, docs drift check, hashed brand check, and
gitleaks over the full history. CodeQL runs in a separate workflow.

## Optional bigger runs (never in CI, still free)

* **Real small language model.** The same DPO recipe can be run with an open 0.5B-parameter
  instruction model and the TRL library on CPU (slow, hours). Not included, because it would add a
  multi-gigabyte download and no deterministic numbers.
* **Serving runners.** See [serving runners](serving-runners.md) for vLLM and llama.cpp.

There is no cloud deployment and no paid API anywhere in this repo.
