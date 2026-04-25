# Setup And Reproducibility

This repo is designed so a new machine can reproduce the working environment from code plus downloaded data. Large raw data, processed matrices, feature-store snapshots, and report artifacts are intentionally **not** committed to git.

## What Is Committed

Committed:

- source code
- config
- tests
- model artifacts under `artifacts/models/`
- research / iteration logs

Not committed:

- `data/raw/`
- `data/processed/`
- `artifacts/feature_store/`
- `artifacts/reports/`
- generated parquet reports

This is controlled by `.gitignore`.

## New Machine Bootstrap

From a clean clone:

```bash
# 1. Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
./.venv/bin/pip install -r requirements.txt

# 3. Download historical market and macro data
PYTHONPATH=. ./.venv/bin/python scripts/download_data.py

# 4. Build the feature store
PYTHONPATH=. ./.venv/bin/python scripts/build_features.py

# 5. Run tests
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/ -q
```

At that point the repo has enough local state to:

- run backtests
- run holdout evaluation
- build posture research datasets
- use the recommender with the committed model artifacts

## Core Workflows

### Full backtest

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_backtest.py
```

### Neutral baseline backtest

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_backtest.py --no-rl
```

### Build posture research dataset

```bash
PYTHONPATH=. ./.venv/bin/python scripts/build_posture_dataset.py \
  --end-date 2016-12-31 \
  --horizon-rebalances 2 \
  --utility-mode full_utility
```

## Production Serving Assumptions

Serving uses the committed model artifacts when available, but it still needs local market data and a local feature store.

Minimum prerequisites for serving on a new machine:

- `scripts/download_data.py`
- `scripts/build_features.py`
- committed files under `artifacts/models/`

If those are present, the recommender can:

- serve the agreed production baseline
- fall back safely when live RL is unavailable

## Reproducibility Notes

### Reports

Generated reports under `artifacts/reports/` are not committed by default. They are reproducible outputs, not canonical source files.

### Feature store

The feature store under `artifacts/feature_store/` is also not committed. It is a deterministic local build artifact and should be rebuilt on a new machine.

### Data download

The canonical path for reconstructing local data is:

- `scripts/download_data.py`

If data providers change or rate-limit, that should be treated as a data-ingestion issue, not solved by trying to commit all raw market data into git.

## When To Commit Artifacts

Commit:

- model artifacts that define the current served behavior
- lightweight research summaries that are needed as references

Do not commit by default:

- raw downloaded data
- processed matrices
- feature-store parquet partitions
- large report dumps

That keeps the repo portable while preserving the ability to rebuild locally.
