# RL-Based Indian Equity Portfolio System

An Indian equity portfolio system with:

- LightGBM sector scoring
- LightGBM stock ranking
- RL-driven sector tilts
- CVXPY constrained portfolio construction

The repo currently runs in a **split mode**:

- production: `tilt_only_rl` with posture fixed to `neutral`
- research: posture learning from realized forward outcomes

The concise source of truth for that split is:

- [docs/CURRENT_SYSTEM_STATE.md](docs/CURRENT_SYSTEM_STATE.md)

The new-machine bootstrap path is:

- [docs/SETUP_AND_REPRODUCIBILITY.md](docs/SETUP_AND_REPRODUCIBILITY.md)

---

## What it builds now

A walk-forward portfolio that:
- Rebalances every **4 weeks** using only information available at that date (no lookahead)
- Uses **LightGBM LambdaRank** to cross-sectionally rank stocks within each of 15 NSE sectors
- Uses RL to learn **sector tilts**
- Uses **CVXPY mean-variance optimization** with realistic constraints (turnover, concentration, liquidity)
- Enforces transaction costs (25 bps one-way) + slippage (10 bps one-way)
- Outputs full performance metrics, attribution, and current portfolio recommendation

Important current behavior:

- posture is frozen to `neutral` in the production RL path
- dynamic posture control is still a research problem, not a production claim

Backtest period: **2013-01-01 → 2026-04-17** | Initial capital: **INR 5,00,000**

---

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv .venv
source .venv/bin/activate
./.venv/bin/pip install -r requirements.txt

# 2. Download 10+ years of NSE data
PYTHONPATH=. ./.venv/bin/python scripts/download_data.py

# 3. Build the feature store
PYTHONPATH=. ./.venv/bin/python scripts/build_features.py

# 4. Run the current production backtest path
PYTHONPATH=. ./.venv/bin/python scripts/run_backtest.py

# 5. Compare without RL overlay
PYTHONPATH=. ./.venv/bin/python scripts/run_backtest.py --no-rl

# 6. Run tests
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/ -q
```

Raw data, processed data, feature-store snapshots, and generated reports are **not** committed to git. The exact bootstrap and rebuild path is documented in [docs/SETUP_AND_REPRODUCIBILITY.md](docs/SETUP_AND_REPRODUCIBILITY.md).

---

## Performance History

| Run | Description | CAGR | Sharpe | MaxDD |
|-----|-------------|------|--------|-------|
| Benchmark (Nifty 50 B&H) | Buy-and-hold | 9.68% | — | — |
| run_002 | True baseline (all bugs fixed) | 16.47% | 0.65 | -24.79% |
| run_004 | Fix: sector feature dedup (all 15 sectors) | 23.71% | 0.92 | -31.29% |
| run_016 | 42 features + infeasibility retry | 23.78% | 0.98 | -26.15% |
| run_017b | Trading-day calendar + benchmark ffill fix | 15.74% | 0.63 | -28.73% |
| run_018 | Accumulated RL buffer on corrected data | 22.96% | 0.83 | -30.75% |
| run_019 | Buffer pass 2 (converged) | 22.13% | 0.84 | -30.13% |
| **run_020** | **12w retrain + no triggers + pruned features** | **22.19%** | **0.96** | **-23.49%** |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                  │
│  yfinance → parquet → price/volume/macro matrices            │
│  Trading-day filter: drops NSE holidays (208 rows removed)   │
├──────────────────────────────────────────────────────────────┤
│  FEATURE LAYER                                               │
│  Macro (60+ cols) │ Sector (30 cols) │ Stock (40 cols)       │
│  All features lagged ≥1 day • Benchmark ffilled              │
├──────────────────────────────────────────────────────────────┤
│  SECTOR SCORER  (LightGBM regression)                        │
│  Predicts 4-week forward sector returns                      │
├──────────────────────────────────────────────────────────────┤
│  RL OVERLAY  (PPO — stable-baselines3)                       │
│  State: 82-dim [macro(12) + sector(60) + portfolio(10)]      │
│  Action: sector tilts ×15 + cash + aggressiveness            │
│  Retrained every 12 weeks (Ablation 2 optimal)               │
│  Event triggers: DISABLED (hurt performance on clean data)   │
├──────────────────────────────────────────────────────────────┤
│  STOCK RANKER  (LightGBM LambdaRank, per-sector)             │
│  Cross-sectional ranking within each of 15 sectors           │
│  Top-5 stocks per sector selected                            │
├──────────────────────────────────────────────────────────────┤
│  PORTFOLIO OPTIMIZER  (CVXPY mean-variance)                  │
│  Max stock: 8% │ Max sector: 35% │ Turnover: 40% │ Cash: 30% │
│  Ledoit-Wolf shrinkage covariance                            │
├──────────────────────────────────────────────────────────────┤
│  RISK ENGINE                                                 │
│  Drawdown monitor │ Vol regime │ Liquidity stress            │
└──────────────────────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full design details including all ablation results.

---

## Key Research Findings

### 1. Data quality beats model complexity
The biggest CAGR jump (+7%) came from fixing a sector feature deduplication bug, not from any model change. Every complexity addition without a data fix hurt performance.

### 2. RL retrain frequency matters — less is more
Tested 7 configurations (4w / 6w / 8w / 12w / 26w / triggers). Results on clean data:

| Freq | CAGR | Sharpe |
|------|------|--------|
| 4w | 20.78% | 0.82 |
| 6w | 21.12% | 0.83 |
| **12w** | **24.98%** | **1.01** |
| 4w + triggers | 15.04% | 0.65 |

More frequent retraining causes RL overfitting. Event triggers fired 300+ times per run and destroyed performance in down years (2018: +20% → −10%). **12w, no triggers is the current config.**

### 3. Trading-day calendar matters
NSE has ~15 holidays/year not in standard calendars. Holiday rows caused `beta_3m` and `rel_str_1m` to be near-zero (NaN→0 in rolling windows), making the RL exploit artificial signals. Filtering rows where <50% of tickers have data removed 208 rows and corrected both features to 99%+ coverage.

### 4. Buffer accumulation is real but bounded
A fresh RL pass yields ~15-16% CAGR. After 2-3 accumulated walk-forward passes: 22-23% CAGR. Further passes don't compound. The ceiling is the quality of features, not buffer size.

---

## Configuration

All parameters in `config/base.yaml`. Key values:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Rebalance frequency | 4 weeks | Fixed |
| RL retrain frequency | 12 weeks | Ablation 2 optimal |
| Event triggers | disabled | Hurts on clean data |
| Max stock weight | 8% | Per CVXPY constraint |
| Max sector weight | 35% | Per CVXPY constraint |
| Max turnover/rebalance | 40% | Controls trading costs |
| Cash range | 0%–30% | RL-managed |
| Transaction cost | 25 bps one-way | |
| Slippage | 10 bps one-way | |

---

## Universe

~100 NSE stocks across 15 sectors (`config/universe.yaml`):

IT · Banking · FinancialServices · FMCG · Automobiles · Pharma · Energy · Metals ·
Telecom · Cement · CapitalGoods · ConsumerDiscretionary · Healthcare · RealEstate · Chemicals

---

## Outputs

All saved to `artifacts/reports/`:

| File | Contents |
|------|----------|
| `metrics.json` | Full performance statistics |
| `current_portfolio.json` | Latest portfolio recommendation |
| `nav_series.parquet` | Daily NAV for all strategies |
| `rebalance_log.csv` | Per-rebalance decision log |
| `attribution.json` | Sector/stock/regime attribution |
| `nav_chart.png` | NAV vs benchmark |
| `drawdown_chart.png` | Drawdown periods |
| `year_returns.png` | Annual return bar chart |

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system design, RL state/action/reward, ablation results |
| [docs/CURRENT_SYSTEM_STATE.md](docs/CURRENT_SYSTEM_STATE.md) | Current production vs research split and the decisions behind it |
| [docs/DECISION_PROTOCOL.md](docs/DECISION_PROTOCOL.md) | Rules for diagnosis, planning, experimentation, and promotion decisions |
| [docs/SETUP_AND_REPRODUCIBILITY.md](docs/SETUP_AND_REPRODUCIBILITY.md) | New-machine bootstrap and what is / is not committed |
| [docs/feature_experiment_plan.md](docs/feature_experiment_plan.md) | Step-by-step stock feature improvement plan (runs 021–025) |
| [docs/feature_update_flow.md](docs/feature_update_flow.md) | How the feature store cache and schema invalidation works |
| [NEXT_STEPS.md](NEXT_STEPS.md) | Full run history, ablation tables, lessons learned |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data download | yfinance |
| Data processing | pandas, pyarrow |
| ML models | lightgbm |
| RL agent | stable-baselines3 (PPO) |
| Portfolio optimizer | cvxpy |
| Data contracts | pydantic v2 |
| CLI | click |
| Testing | pytest |

---

## Committed Artifacts

| Artifact | Status | Notes |
|----------|--------|-------|
| `artifacts/models/sector_scorer.pkl` | committed | current saved scorer |
| `artifacts/models/stock_ranker.pkl` | committed | current saved ranker |
| `artifacts/models/rl_agent/experience_buffer.pkl` | committed | RL buffer state |
| `artifacts/models/rl_agent/meta.pkl` | committed | RL metadata |
| `artifacts/models/rl_agent/ppo_model.zip` | committed | PPO weights |

Not committed by default:

- `data/raw/`
- `data/processed/`
- `artifacts/feature_store/`
- `artifacts/reports/`
- generated parquet reports

Use [docs/SETUP_AND_REPRODUCIBILITY.md](docs/SETUP_AND_REPRODUCIBILITY.md) to recreate them on a new machine.
