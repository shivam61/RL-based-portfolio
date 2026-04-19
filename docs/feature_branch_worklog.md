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
