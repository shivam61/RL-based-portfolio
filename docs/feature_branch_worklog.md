# Feature Branch Worklog

This file tracks measured work completed on the current feature branch.
Each task should record:
- scope
- code changes
- validation run
- outcome
- learning

## 2026-04-20

### Task: Data / PIT correctness baseline
- Scope:
  - point-in-time universe membership in feature generation
  - normalized sector index construction
  - shorter gap fill semantics
  - feature-store logic invalidation
- Validation:
  - `./.venv/bin/pytest tests -q` -> `89 passed`
  - full corrected backtest -> `15.75% CAGR`, `0.70 Sharpe`, `-29.82% MaxDD`, `17.49% avg turnover`
- Learning:
  - PIT fixes materially improved trustworthiness and recovered performance versus the earlier post-accounting low point.
  - The strategy still underperforms the historic high-water-mark runs, so more issues remain in optimizer and universe construction.

### Task: Optimizer turnover correctness
- Scope:
  - align optimizer turnover accounting with one-way turnover
  - repair solver outputs that violate turnover budget instead of silently accepting them
  - add regression tests for liquidation-only turnover math and repair behavior
- Validation:
  - `./.venv/bin/pytest tests -q` -> `91 passed`
  - full corrected backtest -> `18.94% CAGR`, `0.94 Sharpe`, `-21.47% MaxDD`, `21.92% avg turnover`
- Learning:
  - Making turnover accounting consistent with execution materially improved the measured frontier.
  - The optimizer now behaves more defensibly, but turnover is still a policy tradeoff rather than a fully solved hard-control problem.

### Next Queue
- Historical universe reconstruction beyond the static roster:
  - expand point-in-time coverage beyond the current fixed config names
  - preserve date-valid listing, sector, and investability logic
  - measure impact against the current branch baseline before proceeding further

### Investigation: Historical universe expansion
- Outcome:
  - Local-only universe expansion was not selected for implementation in this cycle.
- Evidence:
  - `price_matrix.parquet` already matches the configured NSE roster except for one missing ticker data hole (`TATAMOTORS.NS`).
  - Current PIT logic already delays post-2013 listings via price-coverage plus 252-day history gating.
  - Official Nifty archive files appear to be the right long-term source for date-effective roster reconstruction, but they were not harvestable reliably enough from this environment for a measured overnight change.
- Learning:
  - A meaningful historical-universe upgrade likely needs an external monthly constituent source, not just more logic on top of the current local cache.
  - The next best measured branch task was `run_021` rather than forcing a low-impact metadata-only universe change.

### Task: run_021 — Sharpe / Calmar stock features
- Scope:
  - added `sharpe_1m`, `sharpe_3m`, `sharpe_12m`, and `calmar_3m` to the stock feature set
  - extended feature validation with schema, fill-rate, and construction-consistency checks
- Validation:
  - `./.venv/bin/pytest tests/test_feature_validation.py -q` -> `44 passed`
  - `./.venv/bin/pytest tests -q` -> `93 passed`
  - full backtest -> `10.00% CAGR`, `0.31 Sharpe`, `-22.67% MaxDD`, `27.27% avg turnover`
- Decision:
  - reject and revert
- Learning:
  - Explicit return-over-volatility features pushed the system toward cash-heavy, concentrated RL behavior and materially degraded the stock ranker path.
  - The branch baseline remains the optimizer-fixed state (`18.94% CAGR`, `0.94 Sharpe`) until the next stock-feature experiment is measured.

### Task: run_022 — Cross-sectional return ranks
- Scope:
  - added `ret_1m_rank`, `ret_3m_rank`, and `ret_12m_rank` as percentile ranks across the investable stock universe
  - extended feature validation with schema, bounds, fill-rate, and cross-sectional consistency checks
- Validation:
  - `./.venv/bin/pytest tests/test_feature_validation.py -q` -> `44 passed`
  - `./.venv/bin/pytest tests -q` -> `93 passed`
  - full backtest -> `12.20% CAGR`, `0.43 Sharpe`, `-33.01% MaxDD`, `31.47% avg turnover`
- Decision:
  - reject and revert
- Learning:
  - Regime-invariant rank features did not help the current stack; they increased turnover and deepened drawdown while reducing both CAGR and Sharpe.
  - The branch baseline remains the optimizer-fixed state (`18.94% CAGR`, `0.94 Sharpe`) before `run_023`.
