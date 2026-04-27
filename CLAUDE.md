# RL-Based NSE Portfolio — Agent Context

> **First thing every session**: read `docs/session_handoff.md` for current experiment state.
> Then check if the active run listed below is still alive: `ps aux | grep run_backtest`

---

## What this is

Walk-forward ML+RL portfolio on NSE Indian equities (2013–2026). INR 5L capital, 4-week rebalance.
Three-layer architecture: (1) stock selection alpha, (2) sector tilts via RL, (3) posture (research only).
Production track: `tilt_only_rl` — posture frozen to `neutral`, learned sector tilts live.

---

## Current State (updated 2026-04-27)

### Active run
- **full_rl v5** — PID 35477, launched 07:42 UTC 2026-04-27
- Config: `adaptive_top_k: true`, `trend_gate_enabled: true`, `trend_gate_pctile: 0.80`
- ETA: ~58h (CVXPY inside RL loop is single-threaded — more cores don't help)
- **Do not relaunch or change selection config until v5 completes**
- Log: `artifacts/logs/full_run_v5.log`

### Last completed experiments
| Experiment | Result | Status |
|---|---|---|
| TASK-9: adaptive top-k | IC +0.032, CAGR 12.39% | Marginal pass, led to TASK-10/11 |
| TASK-10: single trend gate p70 | IC +0.026, CAGR 10.93% | **REJECTED** — too blunt |
| TASK-11: combined gate (trend + weak dispersion) | IC +0.045, CAGR 11.02% | **PROMOTED to full RL** |
| Track 3: bidirectional fallback fix | cash_gap 7.65→1.04 pts | **CLOSED** |

### What to watch when v5 completes
Compare vs run_020 baseline: CAGR 22.19%, Sharpe 0.96.
Key question: does PPO increase aggressiveness in low-IC / trend regimes (recovering the 1.4pp CAGR lost from TASK-11's reduced breadth)?
Check: `aggressiveness` level in 2016-2019 periods, `sector_tilt_std`.

---

## Stable baselines (never redefine)
- `neutral_full_stack`: CAGR 17.85%, Sharpe 0.720 (full window)
- `current_rl` run_020: CAGR 22.19%, Sharpe 0.96 (full window, best to date)
- `optimizer_only`: CAGR 9.48%, Sharpe 0.234

---

## Architecture (3-layer mental model)
```
Layer 1: stock selection (alpha)       — StockRanker, adaptive top-k, TASK-11 gate live
Layer 2: sector tilts (RL, live)       — PPO, force_neutral_posture=true (posture frozen)
Layer 3: posture (research only)       — PARKED: needs n≥80 samples + Δ modeling
```
Rule: Layer 3 research must never gate Layer 1/2 progress.

---

## Governance (always follow)
- Read `docs/DECISION_PROTOCOL.md` before any code change
- **One major change per measured run** — never stack multiple hypotheses
- 8-field decision template required before implementing
- Root cause before fix — never bypass safety checks as a shortcut
- Rejected experiments must be logged in `NEXT_STEPS.md` with root cause

---

## Key files
| File | Purpose |
|---|---|
| `config/base.yaml` | All parameters |
| `NEXT_STEPS.md` | Full experiment history, findings, open tracks |
| `docs/DECISION_PROTOCOL.md` | Governance rules |
| `docs/session_handoff.md` | Deep session context (read this) |
| `scripts/run_backtest.py` | Main entry: `--mode [selection_only\|full_rl]` |
| `artifacts/logs/full_run_v5.log` | Active run log |
| `src/models/stock_ranker.py` | TASK-9/11 implementation |

---

## Parked tracks (do not reopen without explicit instruction)
- **Track 2 — Posture regression**: needs n≥80 samples + Δ modeling (pairwise regression). Currently n≈20. Reopens automatically as full backtests accumulate samples.
- **Track 4 — Earnings data**: Screener.in historical depth insufficient pre-2019. Needs paid data source.
- **TASK-10 threshold iteration**: TASK-11 promoted; don't re-tune the p70 gate.

---

## VM note
This is likely a preemptible GCP instance. Full RL runs take ~58h and have been killed 3× before.
If v5 is dead when you arrive: check `ps aux | grep run_backtest`, check last log timestamp,
then relaunch: `nohup .venv/bin/python scripts/run_backtest.py --mode full_rl > artifacts/logs/full_run_v6.log 2>&1 &`
After clearing stale artifacts: `rm -f artifacts/models/rl_agent/ppo_model.zip artifacts/models/rl_agent/meta.pkl artifacts/models/rl_agent/experience_buffer.pkl`

---

## Context update rules (ALWAYS follow — not optional)

### When to update
- **After every experiment result** (selection_only or full_rl completes)
- **After every major decision** (promote, reject, relaunch)
- **Before ending a session** — even if nothing changed, run the script to update the timestamp

### How to update
```bash
bash scripts/save_context.sh
git add CLAUDE.md docs/session_handoff.md
git commit -m "chore: update session context [date]"
git push origin codebase-analysis
```

### What to update manually in `docs/session_handoff.md`
- Add new experiment results to the chain table
- Update "What to do when v5 completes" if the run finishes
- Add any new key design decisions to the decision table
- Mark parked tracks as reopened if conditions are met

### Rule: never leave a session without persisting context
If you completed any experiment, made any decision, or changed any config — run `save_context.sh`
and push. The next agent has no memory of this session. Context in these files is all it has.
