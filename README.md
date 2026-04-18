# RL-Based Indian Equity Portfolio System

A production-grade hierarchical AI portfolio management system for Indian equities.

## What it does

Simulates a **10+ year walk-forward backtest** (2013→2026) of an Indian equity portfolio:

- Starts with **INR 5,00,000** initial capital
- Rebalances every **4 weeks** using only information available at that date
- Uses **supervised ML** (LightGBM) for sector scoring and stock ranking
- Uses a **PPO RL agent** for dynamic sector tilting, cash management, and risk scaling
- Uses **CVXPY** for constrained mean-variance portfolio optimization
- Enforces **realistic transaction costs** (25 bps) + slippage (10 bps)
- Outputs full performance metrics, attribution, and a current portfolio recommendation

## Answering the key questions

| Question | Answer source |
|----------|---------------|
| What CAGR? | `metrics["cagr"]` in artifacts/reports/metrics.json |
| Max drawdown? | `metrics["max_drawdown"]` |
| Which sectors created value? | `artifacts/reports/sector_attribution.png` |
| Did RL help? | Compare `run_backtest.py` vs `run_backtest.py --no-rl` |
| Current portfolio? | `artifacts/reports/current_portfolio.json` |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download data (30-60 min, downloads 100 NSE stocks + global proxies)
python scripts/download_data.py

# Run full backtest with RL
python scripts/run_backtest.py

# Run without RL overlay (rule-based baseline)
python scripts/run_backtest.py --no-rl

# Run unit + integration tests
pytest tests/ -v
```

## System Architecture

```
Data → Features → Sector Scorer → RL Agent → Stock Ranker → Optimizer → Risk Engine → Execution
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details.

## Outputs

All outputs saved to `artifacts/reports/`:
- `metrics.json` — full performance statistics
- `nav_series.parquet` — daily NAV for all strategies
- `rebalance_log.parquet` / `.csv` — per-rebalance decision log
- `attribution.json` — sector/stock/regime attribution
- `current_portfolio.json` — latest portfolio recommendation
- `nav_chart.png` — NAV vs benchmark
- `drawdown_chart.png` — drawdown periods
- `year_returns.png` — annual return bar chart
- `sector_attribution.png` — sector contribution chart
- `rolling_returns.png` — rolling 1Y Sharpe

## Configuration

All parameters in `config/base.yaml`:
- Backtest period, initial capital, rebalance frequency
- Transaction costs, slippage
- Optimizer constraints (max stock weight, max sector weight, max turnover)
- Risk engine thresholds (drawdown limits, vol regime)
- RL hyperparameters (PPO, reward weights)

## Universe

~100 NSE stocks across 15 sectors defined in `config/universe.yaml`:
IT, Banking, FinancialServices, FMCG, Automobiles, Pharma, Energy, Metals,
Telecom, Cement, CapitalGoods, ConsumerDiscretionary, Healthcare, RealEstate, Chemicals

## Tech Stack

| Component | Library |
|-----------|---------|
| Data | yfinance + pandas + pyarrow |
| ML models | lightgbm |
| RL agent | stable-baselines3 (PPO) |
| Optimizer | cvxpy |
| Data contracts | pydantic v2 |
| CLI | click |
| Testing | pytest |
