# RL Control Iteration Log

This file is the running audit trail for RL-as-controller work.

Each iteration should record:

- scope
- control levers changed
- evaluation artifacts used
- full-window result vs `neutral_full_stack`
- holdout result vs `neutral_full_stack`
- named stress-window behavior
- keep / reject decision
- concrete learning

## Permanent Gates

- `neutral_full_stack`
  - source: `artifacts/reports/rl_full_neutral_comparison.json`
  - role: primary RL gate
- `current_rl`
  - source: `artifacts/reports/metrics.json`
  - role: incumbent policy to beat
- `optimizer_only`
  - source: `artifacts/reports/rl_full_backtest_comparison.json`
  - role: structural baseline, not the main RL gate

Named stress windows:

- `2018_q4`
- `2020_covid`
- `2022_rate_shock`
- `2024_late_drawdown`
- `2025_prolonged_drawdown`
- `2026_early_weakness`

## Iteration 0 — Freeze Baseline And Diagnose Current RL

- Date:
  - `2026-04-22`
- Scope:
  - freeze the current reference modes
  - review RL as a portfolio-control overlay using the rebalance log and neutral comparisons
- Evaluation artifacts:
  - `artifacts/reports/metrics.json`
  - `artifacts/reports/rl_holdout_comparison.json`
  - `artifacts/reports/rl_full_neutral_comparison.json`
  - `artifacts/reports/rl_full_backtest_comparison.json`
  - `artifacts/reports/rebalance_log.csv`
- Full-window result vs `neutral_full_stack`:
  - `current_rl`: CAGR `18.27%`, Sharpe `0.750`, MaxDD `-32.62%`, avg turnover `28.03%`
  - `neutral_full_stack`: CAGR `17.85%`, Sharpe `0.720`, MaxDD `-32.80%`, avg turnover `27.43%`
  - uplift:
    - CAGR `+0.42 pts`
    - Sharpe `+0.029`
    - MaxDD `+0.18 pts`
    - turnover `+0.60 pts` worse
- Holdout result vs `neutral_full_stack`:
  - `current_rl`: CAGR `32.68%`, Sharpe `1.509`
  - `neutral_full_stack`: CAGR `32.39%`, Sharpe `1.465`
  - uplift:
    - CAGR `+0.29 pts`
    - Sharpe `+0.044`
- Stress-window behavior review:
  - average stress-window cash since 2017:
    - trained RL `5.96%`
    - neutral `9.52%`
  - average stress-window aggressiveness:
    - trained RL `1.037`
    - neutral `1.000`
  - average stress-window turnover:
    - trained RL `30.52%`
    - neutral `29.32%`
  - key observations:
    - `2020_covid`: RL stayed near fully invested and traded heavily
    - `2022_rate_shock`: RL cut cash to `0%` while drawdown deepened
    - `2024_late_drawdown`: RL increased aggressiveness instead of de-risking
    - `2025_prolonged_drawdown`: RL oscillated between `0%` and `15%` cash with no coherent stress posture
- Decision:
  - keep as baseline, but do not treat the current policy as a solved controller
  - proceed to Stage 0 evaluation hardening and Stage 1 risk-budget redesign
- Learning:
  - the base stack is strong
  - RL is causally valid and slightly additive
  - RL does not yet move cash, risk, or turnover decisively enough in stress
  - the next improvement should be control-specific and incremental, not a broad expansion of authority

## Iteration Template

Copy this block for each future change:

### Iteration N — <short title>

- Date:
- Scope:
- Control levers changed:
- Config flags:
- Evaluation artifacts:
- Full-window result vs `neutral_full_stack`:
- Holdout result vs `neutral_full_stack`:
- Stress-window behavior:
  - `2018_q4`:
  - `2020_covid`:
  - `2022_rate_shock`:
  - `2024_late_drawdown`:
  - `2025_prolonged_drawdown`:
  - `2026_early_weakness`:
- Decision:
  - keep / reject
- Learning:
