# Polymarket Alpha Research

Source code for an academic research pipeline that studies public Polymarket
transaction data, wallet-level feature engineering, trader-label construction,
model training, validation, and backtesting.

This repository intentionally publishes code only. Local datasets, model
artifacts, manuscript drafts, LaTeX files, generated figures, experiment
outputs, and private handoff/work files are excluded from Git.

## What Is Included

```text
src/                 Core Python package
scripts/             Reproduction and analysis utilities
tests/               Pytest regression suite
static/              Static dashboard assets
run.py               Dashboard entry point
run_backtest.py      Backtest entry point
requirements.txt     Python dependencies
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run.py
```

For pipeline checks:

```bash
bash scripts/reproduce.sh
```

For tests:

```bash
python -m pytest tests/ -v
```

## Data And Artifacts

The code expects local data and generated artifacts under paths such as
`data/`, `results/`, and `models/`. These directories are ignored because they
may contain large files, wallet-level data, manuscript work, or private research
outputs.

No API keys are required for public Polymarket data collection paths. Optional
CLOB credentials can be supplied through environment variables when needed by
advanced clients; never commit `.env` files or credentials.

## License

This project is for academic research and educational use. Polymarket data is
subject to Polymarket's own terms and public API policies.
