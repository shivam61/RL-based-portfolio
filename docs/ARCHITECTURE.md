# Architecture Document — RL-Based Indian Equity Portfolio System

## Overview

A hierarchical, walk-forward portfolio management system for Indian equities.
Combines supervised ML (sector scoring, stock ranking), constrained optimization,
and an RL overlay for sector allocation decisions.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│  yfinance → parquet storage → price/volume/macro matrices           │
│  Point-in-time safe • No lookahead • Survivorship-bias aware        │
├─────────────────────────────────────────────────────────────────────┤
│                       FEATURE LAYER                                 │
│  Macro Features │ Sector Features │ Stock Features │ Portfolio Feats │
│  All features lagged ≥1 day before model use                        │
├─────────────────────────────────────────────────────────────────────┤
│                       MODELING LAYER                                │
│  Sector Scorer (LightGBM regression)                                │
│  Stock Ranker  (LightGBM LambdaRank, per-sector)                    │
├─────────────────────────────────────────────────────────────────────┤
│                         RL OVERLAY                                  │
│  PPO agent (stable-baselines3)                                      │
│  Action: sector tilts × 15 + cash target + aggressiveness           │
│  State : macro (12) + sector (60) + portfolio (10)                  │
│  Reward: return − λ_dd×dd − λ_to×turnover − λ_conc×hhi             │
├─────────────────────────────────────────────────────────────────────┤
│                     PORTFOLIO OPTIMIZER                             │
│  CVXPY mean-variance with:                                          │
│  • Max stock weight 8% • Max sector weight 35%                      │
│  • Max turnover 40%/rebalance • Cash [0%, 30%]                      │
│  • Ledoit-Wolf shrinkage covariance                                 │
├─────────────────────────────────────────────────────────────────────┤
│                       RISK ENGINE                                   │
│  Drawdown monitor • Volatility regime • Concentration (HHI)         │
│  Liquidity stress • Macro stress • Hard limits → emergency rebalance│
├─────────────────────────────────────────────────────────────────────┤
│                    EVENT/NEWS ENGINE                                │
│  Pre-catalogued events 2013–2026                                    │
│  Exponential decay → sector impact scores                           │
├─────────────────────────────────────────────────────────────────────┤
│                   WALK-FORWARD ENGINE                               │
│  4-week rebalance loop • Rolling 3-year training window             │
│  Models retrained every 4 weeks • RL retrained every 12 weeks       │
├─────────────────────────────────────────────────────────────────────┤
│                  ATTRIBUTION ENGINE                                 │
│  Brinson-Hood-Beebower • Sector/stock/interaction effects           │
│  Regime attribution • Drawdown episode analysis • SHAP              │
├─────────────────────────────────────────────────────────────────────┤
│                    REPORTING LAYER                                  │
│  JSON metrics • Parquet data • CSV logs • PNG charts                │
│  Console summary • Current portfolio recommendation                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Walk-Forward Loop (per rebalance)

```
1. Build features  ──→  [macro, sector, stock, portfolio_state]
2. Score sectors   ──→  sector_scorer.predict(features)
3. RL decision     ──→  {sector_tilts, cash_target, aggressiveness}
                        (falls back to rule-based if RL not trained)
4. Stock ranking   ──→  stock_ranker.rank_stocks(feats, sector, top_k=5)
5. Optimize        ──→  cvxpy.maximize(α'w − λ_risk·w'Σw − ...)
6. Risk check      ──→  apply cash floor, exclude illiquid, cap small-cap
7. Execute trades  ──→  simulate with TC=25bps + slippage=10bps (one-way)
8. Daily NAV       ──→  interpolate from holdings × prices
9. Record & log    ──→  RebalanceRecord, daily_log
10. RL experience  ──→  append step to experience buffer
```

---

## Repository Structure

```
rl-portfolio/
├── config/
│   ├── base.yaml          ← main config (backtest, optimizer, RL params)
│   └── universe.yaml      ← 100 NSE stocks + sector metadata
├── data/
│   ├── raw/equity/        ← per-ticker parquet files from yfinance
│   ├── processed/         ← price_matrix.parquet, volume_matrix.parquet, macro.parquet
│   └── features/          ← macro_features.parquet, sector_features.parquet, stock_features.parquet
├── src/
│   ├── config.py          ← YAML config loader
│   ├── data/
│   │   ├── contracts.py   ← Pydantic schemas for all inter-module data
│   │   ├── ingestion.py   ← yfinance download + parquet caching
│   │   ├── universe.py    ← point-in-time universe management
│   │   └── macro.py       ← macro/global data + RBI rate schedule
│   ├── features/
│   │   ├── base.py        ← shared rolling/stat utilities
│   │   ├── macro_features.py
│   │   ├── sector_features.py
│   │   ├── stock_features.py
│   │   └── portfolio_features.py
│   ├── models/
│   │   ├── sector_scorer.py  ← LightGBM regression on sector returns
│   │   └── stock_ranker.py   ← LightGBM LambdaRank within sectors
│   ├── rl/
│   │   ├── environment.py ← Gymnasium SectorAllocationEnv
│   │   └── agent.py       ← PPO wrapper + rule-based fallback
│   ├── optimizer/
│   │   └── portfolio_optimizer.py  ← CVXPY constrained optimizer
│   ├── risk/
│   │   └── risk_engine.py ← drawdown/vol/liquidity/concentration checks
│   ├── events/
│   │   └── event_engine.py ← event catalog + decay-weighted sector impacts
│   ├── backtest/
│   │   ├── simulator.py   ← trade execution, NAV, metrics
│   │   ├── walk_forward.py ← main 4-week loop
│   │   └── baselines.py   ← Nifty B&H, equal-weight, sector-momentum
│   ├── attribution/
│   │   └── attribution.py ← Brinson + regime + drawdown + SHAP
│   └── reporting/
│       └── report.py      ← JSON/CSV/PNG/console outputs
├── scripts/
│   ├── download_data.py   ← CLI: download 10+ years of data
│   ├── run_backtest.py    ← CLI: run full walk-forward + report
│   └── run_rl.py          ← CLI: dedicated RL training pass
├── tests/
│   ├── test_data.py
│   └── test_backtest.py
├── artifacts/
│   ├── models/            ← saved sector scorer, stock ranker, RL agent
│   ├── reports/           ← metrics.json, nav_series.parquet, charts
│   └── logs/
├── requirements.txt
├── setup.py
└── docs/ARCHITECTURE.md
```

---

## Data Sources

| Data | Source | Coverage |
|------|--------|----------|
| Equity OHLCV | yfinance (.NS suffix) | 2012–present |
| Global proxies | yfinance (^NSEI, ^VIX, CL=F, GC=F, ^GSPC, etc.) | 2012–present |
| RBI repo rate | Hardcoded schedule (src/data/macro.py) | 2012–present |
| RBI meeting dates | Hardcoded list | 2013–2026 |
| Budget dates | Hardcoded list | 2013–2026 |
| Election dates | Hardcoded windows | 2014, 2019, 2024 |
| Market events | Hardcoded catalog (src/events/event_engine.py) | 2013–2026 |

---

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Initial capital | INR 5,00,000 |
| Backtest period | 2013-01-01 → 2026-04-17 |
| Rebalance frequency | Every 4 weeks (28 days) |
| Warm-up (min training) | 2 years |
| Transaction costs | 25 bps one-way |
| Slippage | 10 bps one-way |
| Max stock weight | 8% |
| Max sector weight | 35% |
| Max turnover/rebalance | 40% |
| Cash range | 0%–30% |

---

## RL Agent Design

- **Algorithm**: PPO (Proximal Policy Optimization) via stable-baselines3
- **State**: 82-dimensional vector [macro(12) + sector(60) + portfolio(10)]
- **Action**: 18-dimensional continuous [sector_tilts(15) + cash + agg + rebalance]
- **Reward**: `return − 2.0×dd − 0.5×turnover − 0.3×hhi − 0.2×liquidity`
- **Training**: offline from walk-forward experience buffer
- **Retraining**: every 12 rebalances (~48 weeks)
- **Fallback**: rule-based allocation if RL not trained (safe default)

---

## Lookahead Prevention

1. All features shifted by `stock_lag=1` / `macro_lag=1` day before use
2. Walk-forward: models trained only on `[start, rebalance_date)` data
3. Forward return labels computed strictly after current date
4. Universe filtered to stocks with ≥252 days of history before as_of date
5. Point-in-time joins throughout (no future index alignment)

---

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download 10+ years of data (takes ~30 min)
python scripts/download_data.py

# 3. Run full walk-forward backtest
python scripts/run_backtest.py

# 4. (Optional) Train RL agent standalone
python scripts/run_rl.py --timesteps 200000

# 5. Run without RL overlay for baseline comparison
python scripts/run_backtest.py --no-rl

# 6. Run tests
pytest tests/ -v
```
