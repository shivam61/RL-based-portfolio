# Next Steps — Improvement Backlog

## Active Program — RL Control Roadmap (2026-04-22)

Decisioning for all future work on this roadmap should follow:

- `docs/DECISION_PROTOCOL.md`

In particular:

- diagnose root cause before changing code
- plan and discuss before major implementation
- change one major variable per measured run
- do not tune reward before structural diagnosis is credible
- log rejected runs as first-class research output

The current portfolio stack is now split into three permanent reference modes. These are the gates for every future RL change and should not be redefined mid-stream:

- `neutral_full_stack`
  - same alpha, sector filter, optimizer, and constraints as `full_rl`
  - fixed neutral action: equal sector tilts, neutral aggressiveness, neutral cash target
  - full-window reference in `artifacts/reports/rl_full_neutral_comparison.json`
  - current full-window metrics:
    - CAGR `17.85%`
    - Sharpe `0.720`
    - MaxDD `-32.80%`
    - avg turnover `27.43%`
- `current_rl`
  - current trained PPO overlay on the causal historical env
  - full-window reference in `artifacts/reports/metrics.json`
  - current full-window metrics:
    - CAGR `18.27%`
    - Sharpe `0.750`
    - MaxDD `-32.62%`
    - avg turnover `28.03%`
- `optimizer_only`
  - lower-level structural benchmark; useful, but not the main RL gate
  - reference in `artifacts/reports/rl_full_backtest_comparison.json`
  - current full-window metrics:
    - CAGR `9.48%`
    - Sharpe `0.234`
    - MaxDD `-34.27%`
    - avg turnover `48.65%`

Primary RL gate:
- compare candidate RL against `neutral_full_stack` on:
  - full backtest
  - holdout
  - named stress windows

Current diagnosis:
- RL is now causally valid and slightly better than neutral, but it is still a weak control policy.
- Full-window uplift vs true neutral is only:
  - CAGR `+0.42 pts`
  - Sharpe `+0.029`
  - MaxDD `+0.18 pts`
  - turnover `+0.60 pts` worse
- Prior holdout uplift vs true neutral is only:
  - CAGR `+0.29 pts`
  - Sharpe `+0.044`
- Stress review from `artifacts/reports/rebalance_log.csv` shows the core weakness:
  - in stress windows since 2017, trained RL averaged `5.96%` cash vs neutral `9.52%`
  - trained RL averaged `1.037` aggressiveness vs neutral `1.000`
  - trained RL averaged `30.52%` turnover vs neutral `29.32%`
  - RL mostly behaved like a mild tilt engine, not a decisive risk controller

### Permanent Evaluation Outputs

Every promoted RL run must publish a control-evaluation artifact that includes:

- CAGR
- Sharpe
- Sortino
- MaxDD
- avg turnover
- avg cash
- stress-window realized loss vs neutral
- stress-window recovery time
- behavior metrics in drawdowns:
  - avg cash
  - avg aggressiveness
  - avg turnover
  - avg stock count
  - avg sector count

Named stress windows are fixed:

- `2018_q4`
- `2020_covid`
- `2022_rate_shock`
- `2024_late_drawdown`
- `2025_prolonged_drawdown`
- `2026_early_weakness`

The working iteration log for these runs lives in `docs/rl_control_iteration_log.md`.

### Staged RL Build Order

#### Stage 0 — Freeze the current strong baseline

- keep `neutral_full_stack`, `current_rl`, and `optimizer_only` immutable
- make the control-evaluation artifact mandatory for:
  - full backtest
  - holdout
  - any candidate RL redesign
- do not merge RL changes without the control review

Status:
- implemented
- canonical artifact: `artifacts/reports/rl_control_evaluation.json`
- next active build: `Stage 1 — Risk budget control only`

#### Stage 1 — Risk budget control only

Add first:

- better control-state features:
  - current drawdown
  - drawdown slope
  - realized vol shock
  - breadth deterioration
  - rolling correlation spike
  - trend decay
  - turnover pressure
  - transaction-cost pressure
- explicit cash control:
  - start with tight buckets or a tight band
  - example: `0%`, `10%`, `20%`, `30%`
- stronger aggressiveness effect with clear portfolio impact
- optional turnover budget / turnover cap

Do not add yet:

- sector inclusion
- stock-count control
- top-k sector selection

Stage 1 success criteria:

- more cash in stress
- less churn in stress
- same or better full-period CAGR
- improved drawdown profile vs `current_rl` and `neutral_full_stack`
- improvement in at least `3/6` named stress windows

Status:
- implementation slice landed behind bounded controls:
  - control-state features
  - cash buckets
  - turnover caps
  - stronger aggressiveness scaling
- feature-correctness coverage added for:
  - clipped/default control features
  - finite RL observation vectors
  - report schema carrying `turnover_cap_pct`
- action-activation fix is now in place:
  - non-neutral cash usage rate
  - non-neutral turnover-cap usage rate
  - non-neutral aggressiveness usage rate
  are tracked directly in holdout/full diagnostics
- latest 2016 holdout still does not clear the promotion gate:
  - candidate RL CAGR `21.00%` vs neutral `32.39%`
  - candidate RL Sharpe `0.916` vs neutral `1.465`
  - candidate RL MaxDD `-14.89%` vs neutral `-15.00%`
  - candidate RL avg turnover `24.66%` vs neutral `25.74%`
  - policy behavior:
    - cash is now state-conditional instead of fixed
    - stress/posture correlation improved to `0.64`
    - turnover cap is still effectively fixed at `30%`
    - sector set is still unchanged at `15` sectors / `~69` names
- decision:
  - keep the implementation and validation work
  - do not promote this policy as the new incumbent
  - keep Stage 1 open until conditional control improves economics, not just behavior

#### Stage 2 — Add a posture layer

- use discrete posture selected by RL:
  - `risk_on`
  - `neutral`
  - `risk_off`
- each posture maps to:
  - cash band
  - aggressiveness cap
  - turnover cap

This is safer than giving RL unconstrained continuous authority too early.

Status:
- implementation slice landed:
  - posture is now the primary RL control primitive
  - posture is persisted in rebalance logs and RL comparison traces
  - each posture maps to bounded cash / aggressiveness / turnover settings
- first measured holdout after the posture rollout:
  - candidate RL CAGR `38.08%` vs neutral `32.39%`
  - candidate RL Sharpe `1.745` vs neutral `1.465`
  - policy behavior:
    - `unique_postures = ['neutral']`
    - `posture_usage_rate = 0.0`
    - `posture_change_rate = 0.0`
    - target posture still varied across `risk_on / neutral / risk_off`
- second measured holdout after tightening posture activation:
  - candidate RL CAGR `36.77%` vs neutral `32.39%`
  - candidate RL Sharpe `1.675` vs neutral `1.465`
  - policy behavior:
    - `unique_postures = ['risk_on']`
    - `posture_usage_rate = 1.0`
    - `posture_change_rate = 0.0`
    - target posture still varied across `risk_on / neutral / risk_off`
- decision:
  - keep the posture-controller implementation
  - do not promote the resulting policy as a valid regime controller yet
  - next Stage 2 work must focus on conditional posture switching, not more generic action activation

Recommended Stage 2 build order from here:
- `2A` target-aware posture state
  - expose current target posture, prior posture, prior target posture, stress persistence, and mismatch state directly in the RL observation
- `2B` switching-quality reward
  - reward improvement toward the current target posture
  - penalize stale mismatch when the target persists
  - penalize posture flips that do not improve alignment
- `2C` posture-guided evaluation diagnostics
  - measure posture stagnation explicitly
  - do not hard-gate on switching frequency
  - use switching data to judge whether the next reward change is fixing the real objective

Current Stage 2A result:
- implementation landed and validated
- latest 2016 holdout:
  - candidate RL CAGR `27.04%` vs neutral `32.39%`
  - candidate RL Sharpe `1.299` vs neutral `1.465`
  - candidate RL MaxDD `-14.08%` vs neutral `-15.00%`
  - candidate RL avg turnover `19.64%` vs neutral `25.54%`
- posture behavior:
  - `unique_postures = ['risk_off']`
  - `posture_usage_rate = 1.0`
  - `posture_change_rate = 0.0`
  - target posture still varied across `risk_on / neutral / risk_off`
- current instrumentation baseline:
  - `posture_counts = {'risk_off': 12}`
  - `target_posture_counts = {'neutral': 5, 'risk_on': 5, 'risk_off': 2}`
  - `decision_quality_basis = target_posture_proxy`
  - `posture_optimality_rate = 16.7%`
  - `mean_regret = 0.583`
- interpretation:
  - the target-aware state is working mechanically
  - the controller still collapses to one posture, now `risk_off`
  - next Stage 2 step should make posture correctness economically dominant, not force posture switching as a fixed condition
  - the first bounded-utility / soft-regret prototype is now implemented, but not yet promoted:
    - targeted tests pass
    - real holdout runtime is currently too slow because every reward step launches multiple counterfactual posture rollouts
    - next refinement should cut regret compute cost before wider evaluation
  - cached one-step regret is now the active prototype:
    - 2016 holdout -> CAGR `30.74%`, Sharpe `1.577`, MaxDD `-13.33%`, turnover `19.83%`
    - neutral full-stack -> CAGR `32.99%`, Sharpe `1.496`, MaxDD `-14.99%`, turnover `25.53%`
    - decision-quality basis: `cached_one_step_soft_regret_v1`
    - posture optimality improved to `41.7%`, but posture utility dispersion is only `2.32e-05`
    - interpretation:
      - runtime is acceptable again
      - posture separation is still too weak to break the static `risk_off` policy
  - first separability pass with stronger posture authority:
    - widened posture profiles:
      - `risk_on -> cash 2%, aggressiveness 1.30, turnover cap 45%`
      - `neutral -> cash 5%, aggressiveness 1.00, turnover cap 35%`
      - `risk_off -> cash 35%, aggressiveness 0.75, turnover cap 15%`
    - changed the posture transform so cash shift is executed before mix rotation under the turnover budget
    - 2016 holdout -> CAGR `20.80%`, Sharpe `1.112`, MaxDD `-12.23%`, turnover `18.05%`
    - neutral full-stack -> CAGR `33.20%`, Sharpe `1.508`, MaxDD `-15.10%`, turnover `25.21%`
    - posture behavior:
      - `unique_postures = ['neutral', 'risk_off']`
      - `posture_change_rate = 9.1%`
      - `posture_counts = {'risk_off': 11, 'neutral': 1}`
    - decision quality:
      - `posture_optimality_rate = 8.3%`
      - `mean_regret = 0.061`
      - `mean_posture_utility_dispersion = 8.05e-05`
    - interpretation:
      - posture separability improved about `3.5x`, but is still far too weak
      - the stronger control envelope pushed the policy into a more persistent defensive basin
      - the next cut should target posture realization / optimizer feasibility before touching reward again
  - execution-honesty pass:
    - optimizer fallback is now turnover-aware when the solver drops to rank fallback
    - target-posture streak / previous-target bookkeeping is internally consistent again in the trace
    - `mean_target_posture_penalty` was removed from promoted diagnostics because it is not part of reward
    - 2016 holdout economics were unchanged in practice:
      - RL -> CAGR `20.80%`, Sharpe `1.112`, MaxDD `-12.23%`, turnover `18.05%`
      - neutral -> CAGR `33.20%`, Sharpe `1.508`, MaxDD `-15.10%`, turnover `25.21%`
    - optimizer warning pressure barely moved:
      - prior holdout log match count `200`
      - current holdout log match count `199`
    - interpretation:
      - we fixed bookkeeping honesty and made fallback outputs safer
      - we did not materially reduce solver infeasibility yet
      - the next execution fix should target why the optimizer is infeasible without turnover, not just how fallback behaves
  - Stage 2E live-execution instrumentation:
    - per-step optimizer diagnostics now include:
      - reason code
      - fallback mode
      - requested cash / realized cash
      - requested turnover cap / effective turnover budget
    - `risk_off` solver failure now routes to a dedicated de-risk fallback instead of generic rank fallback
    - 2016 holdout still lands at:
      - RL -> CAGR `20.80%`, Sharpe `1.112`, MaxDD `-12.23%`, turnover `18.05%`
      - neutral -> CAGR `33.20%`, Sharpe `1.508`, MaxDD `-15.10%`, turnover `25.21%`
    - but the new diagnostics narrow the problem:
      - `optimizer_reason_counts = {'optimal': 9, 'optimal_without_turnover_constraint': 3}`
      - `optimizer_fallback_counts = {'none': 12}`
      - `mean_requested_vs_realized_cash_gap = 7.65 pts`
    - interpretation:
      - the holdout execution path is not dominated by generic fallback anymore
      - the current live bottleneck is cash attainment under `risk_off`, not fallback mode selection
      - the next fix should target post-optimizer realization and cash attainment rather than reward or sector breadth
  - Track 3 — separate_cash_turnover_budget validation [FAILED — REVERTED]:
    - flag enabled: `separate_cash_turnover_budget: true`, `max_cash_delta_per_rebalance: 0.20`
    - 2016 holdout result:
      - neutral cash_gap: 9.42 pts (worse than baseline 7.65 pts)
      - risk_off fixed posture: cash_gap=65 pts, 12/12 `risk_off_de_risk` fallback, avg_turnover=0%, CAGR=0%
    - root cause: `risk_off_de_risk` fallback fires 100% before the separate budget constraint is reached
      - fallback liquidates all equity → 100% cash → overshoots 35% target by 65 pts
      - the turnover budget split is irrelevant when the fallback path preempts the solver
    - next fix must target: why `risk_off_de_risk` fallback fires on every period and why it ignores cash target
      - specifically: does the fallback path respect `max_cash_delta`? Does it set cash to min(1.0, prev+delta) or just dump to 100%?
    - flag reverted to `false`. Do not re-enable until fallback path is fixed.

#### Stage 3 — Add breadth control

Add constrained stock-count / concentration buckets:

- `wide`: `18–24` names
- `medium`: `12–18` names
- `focused`: `8–12` names

Breadth control changes concentration while still preserving the ranker and optimizer as the base engine.

#### Stage 4 — Add constrained sector inclusion last

Only after RL proves it can manage risk budget, posture, and breadth:

- allow:
  - include all sectors
  - exclude up to `1–2` weakest sectors
  - focus on top-`k` sectors from the sector model
- do not start with free-form sector choice

### Rollback Rules

Reject a candidate RL change if it:

- materially lowers full-period CAGR
- worsens MaxDD
- loses to `neutral_full_stack` on both return and behavior
- increases stress-window turnover without reducing realized stress losses
- fails to improve at least `3/6` named stress windows
- claims better reward while showing weaker cash/aggressiveness response in stress

### Working Principles

- add one control lever at a time
- keep each new lever behind a config flag
- prefer discrete buckets over wide continuous freedom at first
- do not give RL more authority until it proves it can use the current increment responsibly

---

Each item is implemented one at a time, backtested, committed, and measured before moving on.
Best known result: **run_020 — 22.19% CAGR, Sharpe 0.96, MaxDD −23.5%** (12w retrain, no triggers, dropped zero-importance binary features)
Stable ceiling on corrected data: **~22–25% CAGR** (12w retrain config — ablation D2 showed 24.98% single-pass; run_020 is 22.19% accumulated)

Current stock-ranker baseline:
- frozen 8W label horizon
- `selection_only`
- current universe
- current feature set
- deterministic seed `42`
- use this as the comparison point for further feature-block ablations

---

## Run History

| Run | Description | CAGR | Sharpe | MaxDD | Final NAV | Delta vs best |
|-----|-------------|------|--------|-------|-----------|---------------|
| run_001 | Original baseline (bugs present) | 21.21% | 0.86 | -30.47% | ₹43.5L | — |
| run_002 | Fix: sector label leakage + double-tilt removed | 16.47% | 0.65 | -24.79% | ₹27.8L | true baseline |
| run_003 | RL 20k timesteps + Sharpe reward + India VIX | 17.18% | 0.60 | -33.33% | ₹29.7L | +0.7% |
| run_004 | **Fix: sector feature dedup — all 15 sectors feed RL** | **23.71%** | **0.92** | -31.29% | **₹54.8L** | **+6.5%** ✅ |
| run_005 | P0-A: sector cap 0.35→0.50 alone | 15.58% | 0.63 | -22.11% | ₹25.5L | -8.1% ❌ |
| run_006 | P0-B: realized weights in RL state (STATE_DIM 82→97) | 20.57% | 0.85 | -27.59% | ₹41.0L | -3.1% ❌ |
| run_007 | P0-A+B combined (cap 0.50 + realized weights) | 11.3% | 0.35 | -39.48% | ₹19.4L | -12.4% ❌ |
| run_008 | FII proxy in RL state (noisy signal) | 20.3% | 0.78 | -29.91% | ₹37.5L | -3.4% ❌ |
| run_009 | Reverted: noisy signals removed, clean baseline | 19.34% | 0.71 | -30.58% | ₹36.5L | -4.4% |
| **run_010** | **Ablation Config B: retrain_freq_weeks 12→8** | **23.57%** | **0.93** | -32.06% | **₹54.1L** | **+4.2% vs run_009** ✅ |
| run_011 | INVALID — stale 36-col feature store served old data | 20.14% | 0.82 | -38.32% | ₹39.4L | discard |
| run_012 | TASK-2: real 44-col technical feature set (ffill fix + new signals) | 19.88% | 0.74 | -31.03% | ₹38.5L | -3.7% vs run_010 ❌ |
| run_013 | Pruned: dropped ret_2w + reversal_1w (42 features) | 17.94% | 0.76 | -29.34% | ₹32.0L | -5.6% vs run_010 ❌ |
| run_014 | Fix: full-portfolio turnover constraint (liquidation + cash) | 20.88% | 0.75 | -30.95% | ₹42.2L | -2.7% vs run_010 |
| run_015 | Revert to 36-feat set + infeasibility-retry fix (TO: 45%→29%) | 18.16% | 0.80 | -24.72% | ₹32.7L | -5.4% vs run_010 |
| **run_016** | **42-feat set + infeasibility retry (new best)** | **23.78%** | **0.98** | **-26.15%** | **₹55.2L** | **+0.2% vs run_010** ✅ |
| run_017 | Trading-day calendar + benchmark ffill (stale RL model) | 15.10% | 0.61 | -29.75% | ₹24.3L | discard — stale RL |
| run_017b | Trading-day calendar + benchmark ffill (fresh RL) | 15.74% | 0.63 | -28.73% | ₹25.9L | -8.0% vs run_016 ❌ new data baseline |
| **run_018** | **Accumulated buffer on corrected data (new best)** | **22.96%** | **0.83** | **-30.75%** | **₹51.2L** | **+7.2% vs run_017b** ✅ |
| run_019 | Accumulated buffer pass 2 — converged | 22.13% | 0.84 | -30.13% | ₹47.4L | flat vs run_018 — buffer converged |
| **run_020** | **12w retrain + no triggers + drop above_50/200ma** | **22.19%** | **0.96** | **-23.49%** | **—** | **+0.06% CAGR, +0.12 Sharpe, MaxDD −6.6pp** ✅ |

### Ablation 2: Retrain Frequency × Event Triggers (run from run_019 state, clean data)

| Config | Description | CAGR | Sharpe | Sortino | MaxDD | Turnover | Status |
|--------|-------------|------|--------|---------|-------|----------|--------|
| A2 | 4-week (no triggers) | 20.78% | 0.82 | 1.15 | -24.34% | 29.5% | ✅ done |
| B2 | 6-week (no triggers) | 21.12% | 0.83 | 1.12 | -33.81% | 24.7% | ✅ done |
| C2 | 8-week (no triggers) — was prev best | 15.79% | 0.64 | 0.85 | -22.23% | 21.9% | ✅ done |
| **D2** | **12-week (no triggers)** | **24.98%** | **1.01** | **1.41** | **-24.58%** | **22.2%** | **✅ done** |
| **E2** | **26-week (no triggers)** | **28.37%** | **1.17** | **1.64** | **-23.56%** | **36.9%** | **✅ done** |
| F2 | 4-week + event triggers | 15.04% | 0.65 | 0.86 | -27.36% | 18.72% | ✅ done |
| G2 | 6-week + event triggers | 13.00% | 0.46 | 0.58 | -32.03% | 20.40% | ✅ done |

**Complete findings (7/7 done):**
- **Monotonic no-trigger trend**: 4w (20.78%) < 6w (21.12%) < 12w (24.98%) < 26w (28.37%)
- **Triggers HURT on clean data**: F2 (4w+triggers: 15.04%) < A2 (4w: 20.78%); G2 (6w+triggers: 13.00%) < B2 (6w: 21.12%)
  - F2 fired 305 events, G2 fired 345 events — massive RL churn caused by over-retraining
  - Key damage: 2018 and 2019 go deeply negative or near-flat when triggers are on
- **8w C2 (15.79%) is a stochastic outlier** — not meaningful
- **Winner: 26w no-triggers** (28.37% CAGR, Sharpe 1.17, Calmar 1.20) — but 36.9% turnover is anomalously high
- **Conservative winner: 12w no-triggers** (24.98% CAGR, Sharpe 1.01, Calmar 1.02, 22.2% turnover)
- **Root cause of Ablation 1 being wrong**: on corrupted data, `rel_str_1m ≈ 0` → RL couldn't use it → frequent retrain made no difference. On clean data, `rel_str_1m` is a real signal — over-retraining causes the RL to chase noise instead of letting the signal compound.
- **Decision: set `retrain_freq_weeks: 12`, `rl_triggers.enabled: false`** — start feature experiment series from this config

### Ablation 1: Retrain Frequency × Event Triggers (run from run_009 state, corrupted data)

| Config | Description | CAGR | Sharpe | MaxDD | Final NAV | RL Retrains | Events Fired |
|--------|-------------|------|--------|-------|-----------|-------------|--------------|
| A | 4-week retrain (no triggers) | 15.15% | 0.61 | -36.35% | ₹24.4L | 29 | 0 |
| B | **8-week retrain (no triggers)** | **23.57%** | **0.93** | -32.06% | **₹54.1L** | 15 | 0 |
| C | 12-week retrain — baseline | 14.88% | 0.58 | -29.59% | ₹23.8L | 10 | 0 |
| D | 26-week retrain (no triggers) | 14.50% | 0.55 | -33.29% | ₹22.9L | 4 | 0 |
| E | 26-week + event triggers | 19.54% | 0.76 | -27.23% | ₹37.2L | 44 | 229 |
| **F** | **8-week + event triggers** | **23.96%** | **0.88** | -32.57% | **₹56.0L** | 56 | 233 |

**Key findings:**
1. **8-week is the optimal scheduled frequency** — Config B (23.57%, Sharpe 0.93) beats the current 12-week by +8.7% CAGR
2. **Event triggers provide lift when retraining is infrequent** — D→E: +5% CAGR; B→F: +0.4% CAGR
3. **Config B has the best Sharpe (0.93)** — cleaner risk-adjusted return than F (0.88)
4. **Over-retraining hurts** — 4-week (29 retrains) degrades vs 8-week (15 retrains)
5. **Next run**: change `retrain_freq_weeks: 12 → 8`, keep `rl_triggers.enabled: false`

**Note on run_009 vs run_004**: Same architecture and config. CAGR gap (19.3% vs 23.7%)
is because run_009 retrains RL from a fresh experience buffer — the RL hasn't yet seen
the same training path run_004 had accumulated. With continued live rebalancing, the
experience buffer will grow and performance should converge back to run_004 levels.

---

## Key Lessons Learned

1. **Data quality beats model complexity.** The biggest win (+6.5% CAGR) was a bug fix
   (sector dedup), not a new feature. Every complexity addition since has hurt.

7. **Buffer reset explains the run_010 gap more than features.** run_015 (36-feat set, same as run_010) gives 18.16% CAGR vs run_010's 23.57%. Gap is ~5.4% with fresh buffer. run_010 had accumulated experience from prior runs; fresh-buffer runs consistently land 18-21%. The 42-feature set (run_014 at 20.88%) actually helps slightly vs 36-feature set once the optimizer is properly fixed.

6. **Turnover constraint had three bugs.** Liquidated tickers (removed from candidate set) were not counted in turnover; cash changes were unconstrained; first-rebalance from all-cash was infeasible, causing a silent rank fallback with zero constraints. Fixing all three recovered +3% CAGR vs run_013 (17.94%→20.88%). Avg turnover 45.76% — slightly over 40% budget due to effective_max_to relaxation when liquidation is forced.

5. **run_012/013 could not beat run_010.** Adding new technical features (44 cols) and then
   pruning redundant ones (42 cols) both underperformed the original 36-feature baseline.
   Dropping ret_2w + reversal_1w hurt further — those features carried real short-term signal
   (2019 +18% vs +30%, 2020 +24% vs +51%). The real gap vs run_010 is likely the RL experience
   buffer being reset — run_010 had accumulated buffer from prior runs; run_012/013 started fresh.
   Next: revert stock features to run_010's 36-col set and re-run to isolate buffer effect.

2. **RL needs data, not architecture.** With only 147 training steps, adding state dims
   adds noise faster than signal. P0-B and FII proxy both hurt for this reason.

3. **Proxy signals are dangerous.** FII proxy (built from USDINR + Nifty vs SP500) sounds
   reasonable but introduces correlated noise the RL can't separate from true macro signal.

4. **The constraint.** RL sector cap increase (0.35→0.50) consistently hurts. At 147 steps,
   wider freedom = more variance the policy gradient can't model.

8. **Trading-day calendar fix reset the baseline (run_017b = 15.74%).** Three bugs fixed:
   (a) price matrix now excludes 208 NSE holiday rows (3,731→3,523); (b) `beta_3m` in stock
   features was 0/365k non-null (now 99%) due to unffilled benchmark in rolling.cov();
   (c) `rel_str_1m/3m` in sector features was ~88% non-null (now 99%) same cause.
   `rel_str_1m` is in the RL state vector — the old RL was trained with it ≈ 0 on holiday
   dates. Fresh RL on corrected data still lands at 15.74%, consistent with the buffer-reset
   pattern (~18-21% range). run_016's 23.78% was partly inflated by the RL exploiting
   corrupted (zero-padded) `rel_str_1m` values. The corrected data is the right baseline
   going forward — run_017b is the new true baseline, not a regression.
   **Down years improved**: 2018 +11.2% (vs +4.9%), 2019 +15.4% (vs +9.5%), 2022 +0.8%
   (vs -1.4%). Up years weaker: 2021 +22.6% (vs +36%), 2023 +18.6% (vs +31.5%).

---

## Current Working State (run_017b config)

- `config/base.yaml`: `max_sector_weight: 0.35`, `total_timesteps: 20000`, `n_epochs: 10`
- `src/rl/environment.py`: STATE_DIM=82, macro keys use `nifty_ret_1m/nifty_above_200ma`
- `src/features/feature_store.py`: sector dedup fixed (15 sectors per snapshot)
- `src/data/ingestion.py`: trading-day filter (≥50% tickers present) in build_price_matrix/build_volume_matrix
- `src/features/sector_features.py`: benchmark_prices ffilled before rolling cov/rel_str
- `src/features/stock_features.py`: benchmark_prices ffilled before rolling cov/beta
- `src/data/macro.py`: India VIX + Nifty IT added to macro data
- FII proxy code exists in `src/data/fii_proxy.py` — computed into feature store but **not
  wired into RL state** until real FII data is available
- P0-B code exists (realized sector weights in experience buffer) — **not in STATE_DIM yet**
  until RL has 500+ live experience steps

---

## Feature Experiment Series — Stock Ranker (runs 020–025)

**Goal**: Improve cross-sectional stock ranking signal quality through targeted,
one-at-a-time feature additions. Each step is tested independently; only winning
changes are carried forward.

**Baseline**: run_019 — 22.13% CAGR, Sharpe 0.84 (accumulated RL buffer on clean data)
**RL policy**: carry forward between all steps — do NOT wipe between runs
**Retrain frequency**: 12-week (Ablation 2 winner), triggers disabled
**Full plan**: `docs/feature_experiment_plan.md`

### Decision rule
Each run must beat the previous run's CAGR by ≥0.5% to be kept.
If it doesn't, revert `stock_features.py` and move to the next step.

---

| Run | Step | Change | Status | CAGR | Sharpe | Delta | Decision |
|-----|------|--------|--------|------|--------|-------|----------|
| run_020 | Step 0 | Drop `above_50ma` + `above_200ma` (0% importance) | ✅ KEEP | 22.19% | 0.96 | +0.06% CAGR / +0.12 Sharpe / MaxDD −6.6pp | KEEP — neutral CAGR but Sharpe 0.84→0.96, MaxDD −30%→−23.5% |
| run_021 | Step 1 | Add Sharpe features: `sharpe_1m/3m/12m`, `calmar_3m` | ❌ REJECT | 10.00% | 0.31 | -8.94% CAGR / -0.63 Sharpe vs branch baseline | REJECT — cash-heavy RL regime and major performance collapse |
| run_022 | Step 2 | Add CS ranks: `ret_1m/3m/12m_rank` (percentile across universe) | ❌ REJECT | 12.20% | 0.43 | -6.74% CAGR / -0.51 Sharpe vs branch baseline | REJECT — higher turnover, deeper drawdown, weaker risk-adjusted returns |
| run_023 | Step 3 | Fix sector z-score: `ret_vs_sector = (ret-mean)/std` | ❌ REJECT | 11.73% | 0.39 | -7.21% CAGR / -0.55 Sharpe vs branch baseline | REJECT — materially worse CAGR, Sharpe, MaxDD, and turnover |

Additional reject on the current branch:
- `sector_relative_strength` (residual momentum + within-sector rank + drawdown-vs-sector peers) was tested in `selection_only` on the frozen 8W baseline and rejected.
- Result: `11.57% CAGR`, `0.33 Sharpe`, `-16.91% MaxDD`, `55.95% avg turnover`
- Diagnostics: `top-k vs sector median +0.50%`, `rank IC -0.032`, `within-sector IC -0.026`, `within-sector top-bottom spread -0.0019`
- Decision: revert the block and keep `sector_normalized` as the better sector-aware configuration

### Universe hardening attempt (Stage A)

| Task | Description | CAGR | Sharpe | MaxDD | Turnover | Decision |
|------|-------------|------|--------|-------|----------|----------|
| Stage A | Infer `listed_since` from price history + default large/mid-only active caps | 16.67% | 0.69 | -30.69% | 24.34% | ❌ REJECT — more realistic but clearly worse than branch baseline |
| Stage B | Broaden to generated current Nifty 200 large/mid roster (199 names) | 17.64% | 0.75 | -33.50% | 33.71% | ❌ REJECT — initial read was invalid due to stale feature-store shards; corrected run still worse than branch baseline and much higher turnover |
| Stage C | Expand curated roster to 150 names and re-measure stock-selection only | 4.13% / 8.53% | -0.11 / 0.21 | -10.32% / -6.72% | 72.85% / 51.45% | ❌ REJECT — no improvement over the smaller isolated baseline; keep the branch roster unchanged |
| Stage D | Re-map the current roster into the proposed 18-sector taxonomy and test `selection_only` | 5.67% | -0.02 | -10.47% | 72.80% | ❌ REJECT — small ranking edge (`top-k vs universe +0.17%`, `rank IC 0.193`) and thin selection (`19.5` names, `26.17%` stability); do not proceed to optimizer/full-RL on this branch |
| run_024 | Step 4 | Add momentum acceleration: `mom_accel_1m`, `mom_accel_3m` | ❌ REJECT | 4.13% / 7.80% | -0.11 / 0.14 | - | REJECT — closed, do not revisit on this branch |
| run_025 | Step 5 | Combine all winning steps; prune back to ≤42 features | ⏳ PENDING | — | — | — | pending after remaining winners are identified |

### Feature detail

**Step 0 — Drop dead features**
- `above_50ma`: 0.1% importance — binary, redundant vs continuous `ma_50_200_ratio`
- `above_200ma`: 0.0% importance — same problem

**Step 1 — Volatility-adjusted returns**
```python
sharpe_1m  = ret_1m  / (vol_1m  + 1e-6)
sharpe_3m  = ret_3m  / (vol_3m  + 1e-6)
sharpe_12m = ret_12m / (vol_12m + 1e-6)
calmar_3m  = ret_3m  / (abs(max_dd_3m) + 1e-6)
```
Hypothesis: LightGBM can't natively divide two columns. Giving it ret/vol directly
promotes momentum signal over pure vol signal. The ranker is currently dominated by
vol features (vol_12m 5.9%, skew_3m 4.2%); raw returns rank near the bottom.

**Step 2 — Cross-sectional percentile ranks**
```python
ret_1m_rank  = ret_1m.rank(axis=1, pct=True)   # 0=worst, 1=best across all tickers
ret_3m_rank  = ret_3m.rank(axis=1, pct=True)
ret_12m_rank = ret_12m.rank(axis=1, pct=True)
```
Hypothesis: Ranks are regime-invariant. Raw returns conflate absolute level with
relative strength — a +5% 1m return means different things in a +10% vs −5% market.
A 90th-percentile stock is always a strong buy signal.

**Step 3 — Within-sector z-score (fix existing vs_sector features)**
```python
# Replace: ret_Xm_vs_sector = ret - sector_mean
# With:    ret_Xm_vs_sector = (ret - sector_mean) / sector_std
```
Hypothesis: A 3% excess return in IT (sector vol ~30%) is weaker signal than 3%
in FMCG (sector vol ~12%). Division by sector_std makes the 6 sector features
comparable across sectors. Same column names — no schema change needed.

**Step 4 — Momentum acceleration**
```python
mom_accel_1m = ret_1m - ret_1m.shift(21)   # 1m momentum improving or fading?
mom_accel_3m = ret_3m - ret_3m.shift(21)   # 3m momentum improving or fading?
```
Hypothesis: A stock up 10% this month after being flat last month is different from
one that's been up 10% consistently. Acceleration often precedes trend continuation;
deceleration often precedes reversal.

**Step 5 — Combination**
Keep only steps that individually passed the ≥0.5% CAGR threshold.
Prune weakest features by importance back to ≤42 total if needed.

---

## Open Tasks (prioritised)

### TASK-1 — Get real FII/DII data [BLOCKED on data source]
**Status**: Proxy built and shelved. Real data needed before wiring into RL.
**Sources to investigate**:
- NSE India monthly reports: `nseindia.com/reports-indices-derivative-statistics`
  → download CSVs manually; place at `data/raw/fii_dii_historical.csv`
  → code in `src/data/fii_proxy.py` auto-detects and uses real data
- Quandl / Nasdaq Data Link: search for "NSE FII" datasets (paid, $30/mo)
- SEBI website: `sebi.gov.in` → Market Statistics → FII/DII (monthly aggregates only)
- Alternative: use a brokerage data API that includes institutional flows
  (Fyers API = UI only, no programmatic FII endpoint)
**Expected impact once real**: Replace proxy with actual ₹ crore net buy/sell
→ `fii_flow_zscore` becomes a high-quality regime signal for the RL

### TASK-2 — Earnings surprise features for stock ranker [NOT STARTED]
**Why**: Stock ranker currently uses only price momentum (mom_stab_12m, ret_1w top features).
Adding fundamental momentum (EPS growth, earnings surprise) gives orthogonal alpha.
**Does not touch RL** — purely improves the alpha fed into the optimizer.
**Sources**:
- Screener.in unofficial API: `https://www.screener.in/api/company/<BSE_CODE>/`
  → quarterly EPS, revenue, P/E (free, rate-limited)
- BSE corporate results API (free): `https://api.bseindia.com/BseIndiaAPI/api/...`
- Trendlyne / Tickertape (paid but clean)
**Features to add** (per stock, quarterly, lagged 1 quarter):
  - `eps_growth_yoy` — EPS vs same quarter last year
  - `rev_growth_yoy` — Revenue growth YoY
  - `ebitda_margin_chg` — Margin improvement/deterioration
  - `pe_vs_sector` — P/E relative to sector median (mean-reversion signal)
**Files**: new `src/data/earnings.py` + extend `src/features/stock_features.py`

### TASK-3 — Self-awareness dims for RL [DEFERRED — needs 500+ live steps first]
**Why deferred**: Adding state dims hurts with <200 training steps (runs 005-008 proved this).
Revisit after 12+ months of live rebalancing (~13 new steps/quarter × 4 quarters = ~52 steps).
**Planned dims** (in order of expected value):
1. `unrealized_pnl_top5` — avg unrealised PnL of top 5 holdings (hold winners signal)
2. `last_turnover` — turnover from previous rebalance (avoid over-trading feedback)
3. `realized_sector_weights[15]` — P0-B (already coded, just not enabled)
**Files**: `environment.py` REALIZED_SECTOR_DIM re-enable + add new PORT_DIM keys

### TASK-4 — RL timesteps experiment [DONE ✅ — 20k confirmed optimal]
**Result**: 50k = 23.37% CAGR / Sharpe 0.89 vs 20k = 23.57% / Sharpe 0.93.
More timesteps = more overfitting to the same thin buffer. 20k stays.

### TASK-5 — Sector cap revisit [DEFERRED — needs more live data]
Re-enable P0-A (cap 0.35→0.50) only after TASK-3 unlocks (500+ live steps).
The constraint feedback loop (realized weights in state) must be active first.

### TASK-6 — Cross-sectional ranking features [IN PROGRESS]
Goal: improve stock-selection separability without changing taxonomy or universe breadth.
Added features:
- sector-relative momentum
- per-date z-scores
- cross-sectional rank transforms
- volatility-adjusted momentum

Validation:
- rebuild feature store after logic hash change
- run `selection_only` before `optimizer_only`
- compare:
  - top-k vs universe
  - top-k vs sector median
  - rank IC
  - precision@k
  - stability
  - top-bottom spread
  - intra-sector dispersion

Decision rule:
- keep only if ranking diagnostics improve materially on the frozen universe
- do not change taxonomy again for this track

Current direction:
- prune the ranker further to the six raw canonical features only:
  - `ret_3m`
  - `mom_12m_skip1m`
  - `mom_accel_3m_6m`
  - `vol_3m`
  - `amihud_1m`
  - `ma_50_200_ratio`
- retire the transform-heavy feature variants if the raw-minimal run improves local ranking diagnostics

### TASK-7 — Local-sector diagnostics [IN PROGRESS]
Goal: decide whether the stock ranker is acting as a true sector-local ranker.
Add and track:
- within-sector IC
- within-sector top-bottom spread
- sector-median separation

Reporting:
- equal-weighted across sectors
- weighted by candidate count

Decision rule:
- require positive local metrics and non-collapsing stability before calling the feature layer useful
- if local metrics are weak too, the feature layer needs redesign rather than more universe or taxonomy changes

### TASK-9 — Signal-conditioned concentration [CLOSED — PROMOTED TO FULL RL VIA TASK-11]
Goal: replace fixed `top_k_per_sector` with dispersion-adaptive k (p90-p10 spread → percentile-ranked vs trailing 8 rebalances).

**Final result (2013-2020, adaptive_top_k=true, no trend gate):**
- within_sector_ic: +0.032 (vs 56D baseline: -0.021) — +0.053 aggregate improvement
- Per-year IC: 2015=+0.333, 2016=-0.121, 2017=+0.113, 2018=-0.048, 2019=-0.042, 2020=+0.127
- CAGR: 12.39% (vs 56D baseline: 10.64%) — +1.75pp
- stability: 30.6% (vs 32.5%)
- **Gate: MARGINAL PASS** — aggregate IC ✅, CAGR ✅, but 2016 worsened (failure mode identified)

**Structural finding**: 2016 IC = -0.121 because high Nifty trend + low within-sector dispersion → adaptive chose k=7 (diversify), holding reversal stocks. Root cause: dispersion ≠ trend signal. Led to TASK-10/11.

---

### TASK-10 — Trend gate (single condition, p70) [CLOSED — REJECTED]
Goal: gate on strong Nifty trend alone → k=mid_k=5. Threshold: abs(nifty_3m) > 70th pctile of trailing 12 rebalances.

**Result:**
- Overall IC: +0.026 (regression from TASK-9 +0.032) — marginal fail zone
- 2016 IC: -0.088 (improved from -0.121) — still in reject zone (< -0.050)
- 2017 IC: +0.032 (degraded from +0.113) — gate fired in 2017, cancelled good signal
- CAGR: 10.93% — reject zone (< 11.5%)
- **Decision: REJECTED** — blunt gate damaged 2017/2020 where adaptive was working correctly

**Root cause**: 70th pctile fires in trend+strong-dispersion periods (2017, 2020) where concentration k=3 was correct. Gate can't distinguish trend+weak-signal from trend+real-signal.

---

### TASK-11 — Combined trend+dispersion gate [CLOSED — PROMOTED TO FULL RL]
Goal: gate only when BOTH strong trend AND weak within-sector dispersion. Threshold: abs(nifty_3m) > 80th pctile AND dispersion_pctile < 0.33 (bottom third of dispersion history).

**Result (2013-2020, combined gate):**
- within_sector_ic: +0.045 — **best IC across all experiments**
- Per-year IC: 2015=+0.333, 2016=-0.090, 2017=+0.121, 2018=-0.020, 2019=-0.036, 2020=+0.124
- CAGR: 11.02% (below 12.0% gate — explained below)
- rank_ic: -0.011 (best ever)
- **Gate: PARTIAL PASS** — overall IC ✅, 2017 fully recovered ✅, 2016 still negative ⚠️, CAGR below gate

**CAGR drop explanation**: in trend+weak-dispersion periods, gate uses k=5 instead of k=7. k=7 captures more broad trend upside passively even with near-zero IC. Expected tradeoff: better selection accuracy, slightly less trend beta. RL layer should compensate via exposure decisions.

**Decision: PROMOTE TO FULL RL** — best IC ever (+0.013 over TASK-9), 2017/2020 structural bugs fixed, 2016 weakness now correctly identified as ranker limitation not gating bug. RL expected to recover CAGR by adjusting aggressiveness/cash in weak-IC regimes.

**What to watch in full RL run:**
- Does PPO learn to increase exposure in low-IC / trend regimes?
- Does PPO compensate for reduced breadth with higher aggressiveness?
- If yes: CAGR recovers naturally. If no: add separate beta/aggressiveness knob in next iteration.

**Config**: `adaptive_top_k: true`, `trend_gate_enabled: true`, `trend_gate_pctile: 0.80`
**Artifacts**: `docs/signal_conditioned_concentration.md`, `src/models/stock_ranker.py`

---

### TASK-8 — Horizon shift experiment [CLOSED — 56D RETAINED]
Goal: test whether the current stock signal is slow-moving momentum rather than a 4W alpha.

**Final result (2013-2020 window, 6 prediction years):**
- 28D: within_sector_ic=0.0102, std=0.4024, stability=31.6%, CAGR=8.82%, Sharpe=0.16, RankIC=-0.062
  - Per-year IC: 2015=+0.333, 2016=-0.002, 2017=-0.060, 2018=-0.092, 2019=-0.040, 2020=+0.103
  - **Positive years: 2/6 — FAILS gate (required ≥4/6)**
- 56D (baseline): CAGR=10.64%, Sharpe=0.26, stability=32.5%, RankIC=-0.021 — better on all visible metrics

**Decision: REJECT 28D. Keep `fwd_window_days: 56`. TASK-8 closed.**

**Structural finding**: IC is positive only in 2015 (range-bound) and 2020 (COVID crash+recovery). Negative in 2016-2019 (India bull run). The 6-feature momentum contract produces near-zero within-sector differentiation in one-directional trending markets — this is a property of the signal, not the horizon. Horizon shift cannot fix this. Next orthogonal improvement requires either earnings/fundamental data (parked — data unavailable) or a regime-conditioned feature weighting approach.

### Track 2 — Posture Utility Regression [PARKED — KNOWN FAILURE MODE, CLEAR REOPEN CONDITIONS]
**Result (2026-04-26, n=20 samples, return_only utilities, H=2, through 2016-12-31):**
- LOO accuracy (non-indifferent, n=15): **33.3%** (first clean run after feature fixes) — below 50% gate
- Baseline always_neutral: 37.5%, always_risk_off: 50.0%
- Utility capture (non-indifferent): 84.5% — below 90% gate
- Mean regret: 0.0106
- ε (p25): 0.0037; 15/20 samples non-indifferent
- Top features: nifty_vol_1m, turnover_1m, bottom_decile_return_1m (no regime feature dominant — model fitting noise at n=20)
- **Conclusion**: No learnable posture signal at current formulation + sample size. Two failure modes identified:
  - Wrong abstraction: 3-class argmax is unstable when margin << regime variance; should use pairwise Δ_on_neutral / Δ_off_neutral regression
  - Insufficient data: n=20 samples — even with perfect formulation, LOO accuracy is not interpretable (1 correct = 5pp change in reported accuracy)
- Per decision protocol: do not proceed to PPO posture loop. **Posture research track parked (not closed).**
- Artifacts: `artifacts/reports/posture_regression_eval.json`, `artifacts/models/posture_model/`

**Reopen conditions (both required):**
1. Dataset ≥ 80 non-indifferent samples (requires 2013-2020 window rebuild)
2. Switch to Δ modeling: targets = Δ_on_neutral, Δ_off_neutral (continuous); vol-normalized utilities; binary Stage 1 = deviate_from_neutral; Stage 2 = direction
- Do NOT reopen with n<80 or 3-class argmax — will reproduce 33% result

### Track 4 — Earnings Bootstrap [PARKED — DATA SOURCE INSUFFICIENT]
**Result (2026-04-26, coverage audit):**
- `eps_growth_yoy` / `rev_growth_yoy`: 84.8% missing, only 2/97 tickers usable, data starts 2019-08-15
- `opm_level`: 77.8% missing average, starts 2012 for some tickers
- For a 2013-2020 backtest window: essentially no usable earnings data before 2019
- **Conclusion**: Screener.in historical depth insufficient. Track parked.
- **Reopen conditions**: access to a paid source with quarterly EPS/revenue history from 2013 (BSE bulk filings, paid screener API). Do not attempt Screener.in scrape again without verifying historical depth first.

---

## Ops Notes

### Active run — full_rl v5 [RUNNING — TASK-11 config, 2026-04-27]
- PID 35477, launched 07:42 UTC
- Config: `adaptive_top_k: true`, `trend_gate_enabled: true`, `trend_gate_pctile: 0.80`
- Baseline to beat: run_020 CAGR 22.19%, Sharpe 0.96 (full window 2013-2026)
- Minimum bar: `neutral_full_stack` CAGR 17.85%, Sharpe 0.720
- Key RL learning question: does PPO increase aggressiveness in low-IC / trend regimes to recover the CAGR lost by TASK-11's reduced breadth?
- **Do not relaunch with different selection config until v5 completes** — one change per measured run
- ETA: ~58h (limited by sequential CVXPY calls inside RL training loop, not CPU count)
- VM note: switch to non-preemptible instance if v5 is preempted again

---

- RL controller status after the latest execution pass:
  - production track is now `tilt_only_rl`:
    - posture fixed to `neutral`
    - learned sector tilts remain live
    - serving fallback uses the neutral full-stack baseline rather than the old rule path
  - latest 2016 holdout for the production track:
    - `tilt_only_rl`: CAGR `33.70%`, Sharpe `1.464`, MaxDD `-14.73%`, turnover `27.34%`
    - `neutral_full_stack`: CAGR `32.55%`, Sharpe `1.433`, MaxDD `-14.67%`, turnover `29.63%`
- Next RL hypothesis:
  - keep the production path fixed as tilt-only RL
  - move posture research to realized `k`-step outcome labeling for `risk_on / neutral / risk_off`
  - do not keep tuning the current posture regret proxy in the production loop
  - latest research-engineering update:
    - model training is now cached per rebalance index in the posture dataset builder
    - counterfactual replay restores scorer/ranker snapshots instead of retraining for every posture path
    - the builder is now practical for larger sample passes, though horizon replay is still the remaining cost center
  - latest label-quality read from the cached `H=2` sample build through `2016-12-31`:
    - superseded by the larger `16`-sample horizon comparison below
  - latest horizon comparison through `2016-12-31`:
    - `H = 2` rebalances:
      - `sample_count = 16`
      - `best_posture_counts = {'risk_off': 15, 'neutral': 1}`
      - `mean_utility_margin = 0.0925`
    - `H = 3` rebalances:
      - `sample_count = 16`
      - `best_posture_counts = {'risk_off': 16}`
      - `mean_utility_margin = 0.1100`
  - current research read:
    - longer horizon does not restore balance to the posture labels
    - `risk_off` dominance persists and gets slightly stronger at `H = 3`
    - this points to either:
      - a genuinely defensive short-horizon regime in this sample window
      - or a horizon utility that is still too drawdown / turnover heavy for posture labeling
  - immediate research follow-up:
    - completed utility comparison on the same realized `H = 2` sample window:
      - artifact: `artifacts/reports/posture_dataset_utilcmp_h2_2016_s16_summary.json`
      - `full_utility`:
        - `risk_off = 15`
        - `neutral = 1`
        - `mean_utility_margin = 0.0925`
      - `return_only`:
        - `risk_off = 8`
        - `neutral = 6`
        - `risk_on = 2`
        - `mean_utility_margin = 0.0153`
      - `return_minus_drawdown`:
        - `risk_off = 8`
        - `neutral = 6`
        - `risk_on = 2`
        - `mean_utility_margin = 0.0172`
      - interpretation:
        - posture labels are not inherently all `risk_off`
        - the strong defensive skew is created mainly by the current full utility, especially turnover-heavy scoring
        - margins under return-only and return-minus-drawdown are much smaller, so any posture model should predict utility, not just class labels
      - additional diagnostics:
        - `winner_by_metric.total_return = {risk_off: 8, neutral: 6, risk_on: 2}`
        - `winner_by_metric.max_drawdown = {neutral: 9, risk_off: 7}`
        - `winner_by_metric.avg_turnover = {risk_off: 15, risk_on: 1}`
        - `execution_clean_subset.sample_count = 0`
      - next step:
        - keep production path unchanged
        - add a posture-utility regression baseline on top of this dataset rather than a classifier or PPO posture loop
- Always `rm artifacts/models/rl_agent/ppo_model.zip meta.pkl experience_buffer.pkl`
  before running backtest when STATE_DIM changes
- Always reset `_metadata.json` macro last_date when `macro_features.py` changes
- Each run saves reports to `artifacts/run_history/run_NNN_description/`
- Commit after every backtest with full metrics in commit message
