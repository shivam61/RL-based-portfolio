# Feature Branch Worklog

This file tracks measured work completed on the current feature branch.
Each task should record:
- scope
- code changes
- validation run
- outcome
- learning

## 2026-04-22

### Task: RL control baseline freeze and evaluation-plan lock
- Scope:
  - promote the new causal RL stack into a permanent control-evaluation workflow
  - freeze three reference modes for future RL work:
    - `neutral_full_stack`
    - `current_rl`
    - `optimizer_only`
  - add the staged RL-control roadmap to `NEXT_STEPS.md`
  - add a dedicated iteration log for future RL redesigns in `docs/rl_control_iteration_log.md`
- Validation:
  - current full-window references:
    - `current_rl` -> `18.27% CAGR`, `0.750 Sharpe`, `-32.62% MaxDD`, `28.03% avg turnover`
    - `neutral_full_stack` -> `17.85% CAGR`, `0.720 Sharpe`, `-32.80% MaxDD`, `27.43% avg turnover`
    - `optimizer_only` -> `9.48% CAGR`, `0.234 Sharpe`, `-34.27% MaxDD`, `48.65% avg turnover`
  - current holdout reference:
    - `current_rl` vs neutral -> `+0.29% CAGR`, `+0.044 Sharpe`
  - stress-window control review from `rebalance_log.csv`:
    - trained RL average stress cash since 2017 -> `5.96%`
    - neutral average stress cash -> `9.52%`
    - trained RL average stress aggressiveness -> `1.037`
    - neutral average stress aggressiveness -> `1.000`
    - trained RL average stress turnover -> `30.52%`
    - neutral average stress turnover -> `29.32%`
- Decision:
  - keep the current RL stack as the incumbent policy
  - do not widen RL authority yet
  - Stage 1 should focus only on risk-budget control:
    - better control-state features
    - explicit cash control
    - stronger aggressiveness effect
    - optional turnover cap / budget
- Learning:
  - the causal RL path is now real enough to measure, but still too weak as a controller
  - the main gating problem is behavior in drawdowns, not headline CAGR
  - future RL iterations need a permanent audit trail or they will drift back into reward-first evaluation

### Task: Stage 0 control-evaluation harness
- Scope:
  - add a canonical RL control-evaluation artifact and CLI
  - unify:
    - full-window RL vs neutral
    - full-window RL vs optimizer-only baseline
    - holdout RL vs neutral
    - named stress-window behavior
  - extend future rebalance logs with selected sector / stock counts for control analysis
- Validation:
  - `./.venv/bin/python -m py_compile src/rl/control_evaluation.py scripts/evaluate_rl_control.py src/reporting/report.py src/backtest/walk_forward.py src/data/contracts.py`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_rl_control_evaluation.py tests/test_reporting_artifacts.py -q` -> `3 passed`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/ -q` -> `113 passed, 1 skipped`
  - `./.venv/bin/python scripts/evaluate_rl_control.py` generated `artifacts/reports/rl_control_evaluation.json`
  - current canonical full-window control result:
    - `current_rl` vs `neutral_full_stack` -> `+0.42 pts` CAGR, `+0.029` Sharpe, turnover worse by `+0.60 pts`
  - current canonical drawdown-behavior summary at `drawdown <= -8%`:
    - RL average cash `5.30%`
    - neutral average cash `9.70%`
    - RL average aggressiveness `1.024`
    - neutral average aggressiveness `1.000`
    - RL average turnover `30.34%`
    - neutral average turnover `28.96%`
- Decision:
  - keep
  - Stage 0 complete
  - move to Stage 1:
    - control-state features
    - explicit cash control
    - stronger aggressiveness effect
    - optional turnover cap / budget
- Learning:
  - the new artifact makes the control problem measurable in one place
  - the current policy still fails the economic smell test in stress even though it is causally valid and slightly additive on returns
  - future stages can now be rejected quickly when behavior gets worse even if reward improves

### Task: Stage 1 risk-budget controls and validation hardening
- Scope:
  - add bounded RL control levers for:
    - cash buckets
    - turnover caps
    - stronger aggressiveness scaling
  - enrich the RL portfolio-state surface with control-specific features:
    - drawdown slope
    - volatility shock
    - breadth deterioration
    - recent turnover pressure
    - recent cost pressure
    - risk cash floor
    - emergency flag
  - harden validation so the new control inputs cannot introduce silent data issues
- Validation:
  - `./.venv/bin/python -m py_compile src/rl/environment.py src/rl/agent.py src/rl/historical_executor.py src/backtest/walk_forward.py src/features/portfolio_features.py src/optimizer/portfolio_optimizer.py src/rl/policy_utils.py tests/test_portfolio_features.py`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_portfolio_features.py tests/test_rl_environment_contract.py tests/test_data.py::TestOptimizer tests/test_reporting_artifacts.py -q` -> `19 passed`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/ -q` -> `118 passed, 1 skipped`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 64`
    - candidate RL -> `28.01% CAGR`, `1.236 Sharpe`, `-15.15% MaxDD`, `26.18% avg turnover`
    - neutral full-stack -> `32.39% CAGR`, `1.465 Sharpe`, `-15.00% MaxDD`, `25.54% avg turnover`
- Decision:
  - keep the implementation and validation additions
  - do not promote the resulting policy as the new RL incumbent
  - continue Stage 1 until the candidate improves control behavior without losing to neutral
- Learning:
  - the new control levers are functioning and bounded, but the first candidate policy is still economically weaker than neutral
  - the added tests materially reduce data-risk on this surface:
    - control features are clipped/defaulted safely
    - RL observations remain finite at the expanded state dimension
    - rebalance reports now preserve turnover-cap metadata for audit
  - the next Stage 1 work should focus on the economics of how the policy uses these levers, not on widening authority further

### Task: Stage 1 action activation and stress-aware control diagnostics
- Scope:
  - make cash / turnover bucket actions easier to activate from PPO outputs
  - add direct control-usage diagnostics to holdout and neutral comparison traces
  - add stress-aware reward alignment so constant defensive posture is penalized relative to state stress
- Validation:
  - `./.venv/bin/python -m py_compile src/rl/environment.py src/rl/holdout.py src/rl/full_comparison.py src/rl/control_evaluation.py src/rl/historical_executor.py`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_rl_environment_contract.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py -q` -> `14 passed`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/ -q` -> `121 passed, 1 skipped`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 64`
    - candidate RL -> `24.07% CAGR`, `1.094 Sharpe`, `-14.58% MaxDD`, `23.25% avg turnover`
    - neutral full-stack -> `32.39% CAGR`, `1.465 Sharpe`, `-15.00% MaxDD`, `25.54% avg turnover`
    - control diagnostics:
      - cash usage rate `1.0`
      - turnover-cap usage rate `1.0`
      - executed cash fixed at `15%`
      - executed turnover cap fixed at `30%`
- Decision:
  - keep the action-activation and diagnostics work
  - do not promote the resulting policy as the new RL incumbent
- Learning:
  - the control-lever activation problem is fixed
  - the next problem is now explicit:
    - policy uses the controls
    - but uses them statically rather than conditionally
  - that is a better failure mode to debug than silent neutral collapse, but it is still below the economic gate

### Task: Stage 1 regime-conditioned control guidance
- Scope:
  - add stress-conditioned target controls for:
    - cash
    - aggressiveness
    - turnover cap
  - blend executed RL controls toward those targets at runtime
  - add alignment diagnostics between stress and realized posture
- Validation:
  - `./.venv/bin/python -m py_compile src/rl/historical_executor.py src/rl/holdout.py src/rl/full_comparison.py src/rl/control_evaluation.py`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_rl_environment_contract.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py -q` -> `16 passed`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/ -q` -> `123 passed, 1 skipped`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 64`
    - candidate RL -> `21.00% CAGR`, `0.916 Sharpe`, `-14.89% MaxDD`, `24.66% avg turnover`
    - neutral full-stack -> `32.39% CAGR`, `1.465 Sharpe`, `-15.00% MaxDD`, `25.74% avg turnover`
    - behavior:
      - mean cash target `13.36%`
      - unique cash targets from `5%` to `21.3%`
      - `stress_posture_correlation = 0.642`
- Decision:
  - keep the regime-conditioned guidance implementation
  - do not promote the resulting policy as the RL incumbent
- Learning:
  - the controller is now meaningfully state-conditional on cash
  - this is the first iteration where posture varies with stress instead of staying flat
  - but returns degraded further, so the remaining problem is economic quality of control decisions, not missing control movement

### Task: Stage 2 discrete posture controller
- Scope:
  - replace continuous risk-budget control with a posture controller:
    - `risk_on`
    - `neutral`
    - `risk_off`
  - map posture to bounded cash / aggressiveness / turnover settings
  - persist posture in:
    - rebalance logs
    - holdout traces
    - full neutral comparison diagnostics
- Validation:
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_rl_environment_contract.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py tests/test_reporting_artifacts.py -q` -> `18 passed`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/ -q` -> `123 passed, 1 skipped`
  - `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 128`
    - candidate RL -> `38.08% CAGR`, `1.745 Sharpe`, `-14.70% MaxDD`, `24.62% avg turnover`
    - neutral full-stack -> `32.39% CAGR`, `1.465 Sharpe`, `-15.00% MaxDD`, `25.54% avg turnover`
    - posture diagnostics:
      - `unique_postures = ['neutral']`
      - `posture_usage_rate = 0.0`
      - `posture_change_rate = 0.0`
- Decision:
  - keep the implementation
  - do not promote the resulting policy as a working posture controller
- Learning:
  - the discrete posture execution path is now correct and observable
  - better holdout economics alone are not enough here, because the policy never left neutral posture

### Task: Stage 2 posture activation tightening
- Scope:
  - reduce neutral-band stickiness for posture decoding
  - increase reward pressure on target-posture mismatch
- Validation:
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_rl_environment_contract.py tests/test_rl_holdout.py -q` -> `15 passed`
  - `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 128`
    - candidate RL -> `36.77% CAGR`, `1.675 Sharpe`, `-15.03% MaxDD`, `24.83% avg turnover`
    - neutral full-stack -> `32.39% CAGR`, `1.465 Sharpe`, `-15.00% MaxDD`, `25.54% avg turnover`
    - posture diagnostics:
      - `unique_postures = ['risk_on']`
      - `posture_usage_rate = 1.0`
      - `posture_change_rate = 0.0`
      - target posture still varied across `risk_on / neutral / risk_off`
- Decision:
  - keep the activation adjustments
  - keep Stage 2 open
- Learning:
  - posture now activates, but still does not switch conditionally
  - the immediate next problem is regime discrimination / posture switching quality, not missing action activation

## 2026-04-23

### Task: Stage 2 cash-target realization tightening
- Scope:
  - tighten optimizer handling of explicit RL cash targets
  - preserve solver cash through no-trade-band cleanup instead of renormalizing it downward
  - keep reward/regret unchanged so the measured effect stays execution-only
- Validation:
  - `./.venv/bin/python -m py_compile src/optimizer/portfolio_optimizer.py tests/test_data.py`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_data.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py -q` -> `33 passed`
  - `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 128`
    - candidate RL -> `17.97% CAGR`, `0.961 Sharpe`, `-11.18% MaxDD`, `17.21% avg turnover`
    - neutral full-stack -> `31.41% CAGR`, `1.460 Sharpe`, `-14.42% MaxDD`, `24.73% avg turnover`
    - execution diagnostics:
      - mean requested-vs-realized cash gap improved from `7.65 pts` to `5.56 pts`
      - optimizer fallback count in the live holdout path remained `0`
      - realized `risk_off` cash now sits close to `35%` after the early turnover-limited windows
- Decision:
  - keep the execution tightening
  - do not promote the resulting policy as the new RL incumbent
- Learning:
  - the optimizer was indeed too soft about explicit cash targets
  - tightening that path made posture execution more faithful, but did not improve economics because the controller is still choosing `risk_off` too often
  - this narrows the next hypothesis: the remaining issue is decision quality, not cash-target realization drift

### Task: Stage 2 sector-preserved stock breadth gate
- Scope:
  - add posture-specific stock breadth gates before optimization
  - keep reward/regret unchanged
  - keep a light sector-presence guard so all sectors remain represented
- Validation:
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_data.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py -q` -> `35 passed`
  - `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 128`
    - candidate RL -> `12.96% CAGR`, `0.575 Sharpe`, `-10.76% MaxDD`, `10.92% avg turnover`
    - neutral full-stack -> `28.96% CAGR`, `1.228 Sharpe`, `-14.89% MaxDD`, `29.32% avg turnover`
    - diagnostics:
      - `unique_postures = ['risk_off']`
      - `mean_posture_utility_dispersion = 8.42e-05`
      - `optimizer_fallback_counts = {'risk_off_de_risk': 7, 'none': 5}`
      - `mean_selected_stock_count = 42.0`
      - `mean_selected_sector_count = 15.0`
- Decision:
  - reject
- Learning:
  - stock-breadth masking alone was too diluted because all sectors stayed present
  - the active stock count changed, but the active sector count did not
  - that created a thinner `risk_off` book, not a meaningfully different posture
  - the next structural experiment should change sector breadth by posture before changing stock breadth again

### Task: Stage 2 sector-first breadth gate
- Scope:
  - replace rejected global stock-breadth masking with posture-specific sector breadth
  - apply posture-specific stock `top_k` inside the selected sectors
  - keep reward/regret unchanged
- Validation:
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_data.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py tests/test_rl_environment_contract.py -q` -> `53 passed`
  - `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 128`
    - candidate RL -> `34.79% CAGR`, `1.583 Sharpe`, `-12.86% MaxDD`, `27.90% avg turnover`
    - neutral full-stack -> `32.55% CAGR`, `1.433 Sharpe`, `-14.67% MaxDD`, `29.63% avg turnover`
    - diagnostics:
      - `unique_postures = ['neutral']`
      - `mean_posture_utility_dispersion = 5.33e-05`
      - `optimizer_fallback_counts = {'none': 12}`
      - `mean_selected_stock_count = 51.75`
      - `mean_selected_sector_count = 11.0`
- Decision:
  - reject as a solved RL-controller iteration
- Learning:
  - sector-first breadth is much healthier structurally than the prior stock-breadth-only pass
  - it improved economics and restored clean execution
  - but the policy used only `neutral`, so it did not actually demonstrate posture separability or multi-posture control

### Task: Stage 2 target-aware switching state and reward
- Scope:
  - expose control-target features directly in RL state:
    - current target posture
    - previous posture
    - previous target posture
    - stress persistence
    - mismatch memory
  - add switching-quality reward terms:
    - bonus for moving closer to target posture
    - penalty for staying in a mismatched posture while the target persists
    - penalty for posture flips that do not improve alignment
- Validation:
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_portfolio_features.py tests/test_rl_environment_contract.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py -q` -> `18 passed`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/ -q` -> `123 passed, 1 skipped`
  - `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 128`
    - candidate RL -> `27.04% CAGR`, `1.299 Sharpe`, `-14.08% MaxDD`, `19.64% avg turnover`
    - neutral full-stack -> `32.39% CAGR`, `1.465 Sharpe`, `-15.00% MaxDD`, `25.54% avg turnover`
    - posture diagnostics:
      - `unique_postures = ['risk_off']`
      - `posture_usage_rate = 1.0`
      - `posture_change_rate = 0.0`
      - `mean_posture_progress_bonus = -0.0025`
      - `mean_posture_stale_penalty = 0.0213`
- Decision:
  - keep the state / reward instrumentation
  - do not promote the resulting policy as the new RL incumbent
- Learning:
  - the state now carries enough explicit posture-target information to diagnose switching quality directly
  - the remaining failure is not hidden anymore: policy still collapses to one posture, now `risk_off`
  - the next Stage 2 build should introduce explicit posture-usage gates or constrained supervision rather than only more reward shaping

### Task: Stage 2 decision-quality instrumentation and advisory-only posture diagnostics
- Scope:
  - add explicit holdout diagnostics for:
    - posture counts
    - posture by stress bucket
    - proxy decision quality
    - realized control settings by posture and by stress bucket
  - remove hard posture-switch thresholds from config and evaluation
  - keep posture stagnation as an advisory diagnostic only
- Validation:
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_rl_holdout.py tests/test_rl_control_evaluation.py tests/test_rl_environment_contract.py -q` -> `17 passed`
  - `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 128`
    - candidate RL -> `29.96% CAGR`, `1.464 Sharpe`, `-13.58% MaxDD`, `20.77% avg turnover`
    - neutral full-stack -> `32.99% CAGR`, `1.496 Sharpe`, `-14.99% MaxDD`, `25.53% avg turnover`
    - posture behavior:
      - `posture_counts = {'risk_off': 12}`
      - `target_posture_counts = {'neutral': 5, 'risk_on': 5, 'risk_off': 2}`
    - decision quality, current proxy:
      - `decision_quality_basis = target_posture_proxy`
      - `posture_optimality_rate = 16.7%`
      - `mean_regret = 0.583`
- Decision:
  - keep
  - do not use posture switching as a promotion gate
  - use these diagnostics as the baseline for the next reward redesign
- Learning:
  - the current candidate is still statically defensive, but now that failure is measurable in economic terms
  - the next reward change should target posture correctness directly, not enforce switching frequency

### Task: Stage 2 bounded utility and soft-regret reward prototype
- Scope:
  - replace the Stage 2 reward backbone with bounded regime-weighted utility
  - add soft regret over posture counterfactuals using:
    - static postures
    - one-switch posture baselines
  - switch holdout diagnostics from target-proxy decision quality to reward-native decision quality
- Validation:
  - `./.venv/bin/python -m py_compile src/rl/historical_executor.py src/rl/holdout.py src/rl/control_evaluation.py tests/test_rl_environment_contract.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py`
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_rl_environment_contract.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py -q` -> `19 passed`
  - real holdout runs with the new reward did not complete on a practical research timescale:
    - `scripts/evaluate_rl_holdout.py --timesteps 128`
    - `scripts/evaluate_rl_holdout.py --timesteps 8`
    both remained materially slower than the prior objective because every reward step now launches multiple counterfactual rollouts
- Decision:
  - keep the implementation as a prototype, not as the new default research loop
  - do not promote the reward yet
  - next step is to reduce counterfactual cost before trusting economics from this objective
- Learning:
  - the reward shape is now closer to the intended control problem:
    - bounded regime weights
    - no forced posture-switch gate
    - soft regret instead of hard max
  - the first operational bottleneck is compute, not correctness
  - the counterfactual term needs approximation, caching, or a smaller candidate set before full holdout evaluation is practical

### Task: Stage 2 cached one-step soft-regret optimization
- Scope:
  - replace full rollout regret with cached one-step approximate regret from:
    - current weights
    - feasible target weights
    - realized next-step asset returns
    - observed turnover/cost
  - keep the new diagnostics and reward structure intact
- Validation:
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_rl_environment_contract.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py -q` -> `19 passed`
  - `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 128`
    - candidate RL -> `30.74% CAGR`, `1.577 Sharpe`, `-13.33% MaxDD`, `19.83% avg turnover`
    - neutral full-stack -> `32.99% CAGR`, `1.496 Sharpe`, `-14.99% MaxDD`, `25.53% avg turnover`
    - decision-quality diagnostics:
      - `decision_quality_basis = cached_one_step_soft_regret_v1`
      - `posture_optimality_rate = 41.7%`
      - `mean_regret = 0.057`
      - `mean_posture_utility_dispersion = 2.32e-05`
- Decision:
  - keep
  - use this as the active Stage 2 reward baseline, not the slower rollout version
- Learning:
  - the research loop is usable again
  - the dominant remaining problem is not compute; it is weak posture separability
  - posture utility dispersion is extremely small, which explains why the policy still collapses to static `risk_off`

### Task: Stage 2 stronger posture authority and cash-first realization
- Scope:
  - widen posture envelopes materially without changing the reward surface
  - make the posture transform spend turnover on cash movement first, then on equity-mix rotation
  - keep cached one-step soft regret as the active objective
- Validation:
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_rl_environment_contract.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py -q` -> `20 passed`
  - `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 128`
    - candidate RL -> `20.80% CAGR`, `1.112 Sharpe`, `-12.23% MaxDD`, `18.05% avg turnover`
    - neutral full-stack -> `33.20% CAGR`, `1.508 Sharpe`, `-15.10% MaxDD`, `25.21% avg turnover`
    - decision-quality diagnostics:
      - `unique_postures = ['neutral', 'risk_off']`
      - `posture_change_rate = 9.1%`
      - `posture_optimality_rate = 8.3%`
      - `mean_regret = 0.061`
      - `mean_posture_utility_dispersion = 8.05e-05`
- Decision:
  - keep as a measured regression
  - do not promote
- Learning:
  - posture separability improved about `3.5x`, but remained far too weak
  - the wider control envelope mostly increased defensive persistence rather than useful regime switching
  - the next Stage 2 step should attack posture realization quality and optimizer feasibility, not add more reward complexity yet

### Task: Stage 2 execution honesty and target-context cleanup
- Scope:
  - make fallback outputs safer when the optimizer drops to rank mode
  - fix target-streak / previous-target consistency in posture context
  - stop promoting target-posture penalty as a top-line diagnostic since it is not part of reward
- Validation:
  - `MPLCONFIGDIR=/tmp/mpl ./.venv/bin/pytest tests/test_data.py tests/test_rl_environment_contract.py tests/test_rl_holdout.py tests/test_rl_control_evaluation.py -q`
  - `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. ./.venv/bin/python scripts/evaluate_rl_holdout.py --holdout-start 2016-01-01 --holdout-end 2016-12-31 --timesteps 128`
- Result:
  - holdout economics were effectively unchanged
  - target-streak trace is now internally consistent on repeated regimes
  - fallback warning pressure barely changed
- Decision:
  - keep as a correctness / honesty improvement
  - do not count it as a solved execution issue
- Learning:
  - the real problem is still solver infeasibility upstream of fallback
  - the next optimizer pass should target infeasibility directly, not fallback cosmetics

## 2026-04-21

### Task: sector_relative_strength block
- Scope:
  - add a stronger sector-relative stock feature block with residual momentum, sector percentile rank, and drawdown-vs-sector peers
  - test the block in `selection_only` on the frozen 8W baseline
- Validation:
  - `./.venv/bin/python -m py_compile src/features/stock_features.py tests/test_feature_validation.py`
  - `./.venv/bin/python -m pytest tests/test_feature_validation.py::TestCrossFeatureConsistency::test_interaction_blocks_emit_expected_columns -q` -> `1 passed`
  - `selection_only` on `2013-01-01 → 2016-12-31` with `absolute_momentum + risk + liquidity + trend + sector_relative_strength`
    - `11.57% CAGR`
    - `0.33 Sharpe`
    - `-16.91% MaxDD`
    - `55.95% avg turnover`
    - selection diagnostics:
      - `top-k vs universe +0.54%`
      - `top-k vs sector median +0.66%`
      - `precision@k 53.86%`
      - `rank IC -0.032`
      - `within-sector IC -0.026`
      - `within-sector top-bottom spread -0.0019`
- Decision:
  - reject and revert the block
- Learning:
  - stronger sector-relative residuals and sector percentile features did not improve the stock ranker's local ordering
  - the block preserved the familiar top-k vs sector-median lift, but it worsened the actual separability metrics we care about
  - `sector_normalized` remains the better sector-aware configuration on this frozen window

### Task: Freeze truth baseline and run stock-feature ablations
- Scope:
  - revert horizon blending and lock the stock-ranker truth baseline to a single 8W label horizon
  - add block-based stock feature selection so the ranker can be ablated without changing architecture
  - add deterministic seed wiring for reproducible selection-only runs
  - run the following frozen-window ablations on `2013-01-01 → 2016-12-31`:
    - baseline blocks: `absolute_momentum + risk + liquidity + trend`
    - `absolute_momentum` only
    - `sector_relative_momentum` only
    - `absolute_momentum + sector_relative_momentum`
    - `volatility_adjusted_momentum`
- Validation:
  - `./.venv/bin/python -m py_compile src/models/stock_ranker.py src/models/sector_scorer.py src/backtest/walk_forward.py scripts/run_backtest.py src/features/stock_features.py src/reporting/report.py scripts/export_stock_ranker_importance.py`
  - `selection_only` baseline -> `11.55% CAGR`, `0.32 Sharpe`, `-19.58% MaxDD`, `54.45% avg turnover`
  - `absolute_momentum` only -> `7.05% CAGR`, `0.06 Sharpe`, `-23.70% MaxDD`, `56.32% avg turnover`
  - `sector_relative_momentum` only -> `6.53% CAGR`, `0.03 Sharpe`, `-24.79% MaxDD`, `56.66% avg turnover`
  - `absolute_momentum + sector_relative_momentum` -> `8.49% CAGR`, `0.14 Sharpe`, `-22.07% MaxDD`, `55.87% avg turnover`
  - `volatility_adjusted_momentum` -> `7.62% CAGR`, `0.09 Sharpe`, `-22.52% MaxDD`, `59.44% avg turnover`
- Learning:
  - the 8W truth baseline is the strongest of the tested feature slices on this frozen window
  - absolute momentum by itself is not enough
  - sector-relative momentum by itself is not enough
  - combining absolute + sector-relative helps relative to either alone, but still underperforms the full baseline that retains risk/liquidity/trend context
  - volatility-adjusted momentum alone does not solve the ranking problem
  - the next useful step is to prune weaker blocks one at a time from the 8W truth baseline rather than reintroducing horizon or optimizer complexity

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

### Task: run_023 — Sector-relative z-score
- Scope:
  - replaced sector-relative raw spreads with within-sector z-scores for `ret_1w`, `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m`, and `vol_3m`
  - added a direct validation test that checks `ret_1m_vs_sector` against the expected within-sector z-score formula
- Validation:
  - `./.venv/bin/pytest tests/test_feature_validation.py -q` -> `43 passed`
  - `./.venv/bin/pytest tests -q` -> `92 passed`
  - full backtest -> `11.73% CAGR`, `0.39 Sharpe`, `-34.98% MaxDD`, `30.79% avg turnover`
- Decision:
  - reject and revert
- Learning:
  - Making sector-relative features variance-scaled did not improve comparability in practice; it destabilized the ranker path and materially worsened turnover and drawdown.
  - Three consecutive stock-feature experiments failed against the optimizer-fixed baseline, which is a signal to pause feature churn and move back to data/foundation work.

### Task: Stage A universe hardening (inferred listing dates + large/mid focus)
- Scope:
  - inferred `listed_since` from first valid trade date in local price history when config metadata is absent
  - added optional inferred delisting support for long disappearances
  - applied a default `active_caps: [large, mid]` universe filter
  - added targeted universe tests for inferred listing dates and default cap filtering
- Validation:
  - `./.venv/bin/pytest tests/test_data.py::TestUniverseManager -q` -> `8 passed`
  - `./.venv/bin/pytest tests -q` -> `93 passed`
  - full backtest -> `16.67% CAGR`, `0.69 Sharpe`, `-30.69% MaxDD`, `24.34% avg turnover`
- Decision:
  - reject and revert
- Learning:
  - Tightening realism on the current ~100-stock roster without expanding historical breadth reduced both return and risk-adjusted quality.
  - This suggests the next useful universe project is broader historical coverage, not a stricter version of the current narrow roster.

### Task: Stage B universe broadening (current Nifty 200 large/mid roster)
- Scope:
  - expanded the equity universe from the hand-curated 99-stock roster to a generated 199-stock broad large/mid roster based on current Nifty 200 constituents
  - added `scripts/build_universe_from_nifty200.py` to make the config generation reproducible
  - stored source constituent files under `data/reference/indices/`
  - added integrity tests for the broadened universe config
  - fixed `scripts/download_data.py` stale earnings import so the equity-only refresh path works again
  - fixed `FeatureStore` logic hashing so stock/sector shards rebuild when the universe roster changes
  - normalized source-symbol issues during generation:
    - dropped `DUMMY*` mirror artifacts
    - remapped `ZOMATO` to `ETERNAL`
    - excluded `TATAMOTORS` from this pass because the post-demerger ticker/history path was not stable in the Yahoo-backed downloader
- Validation:
  - `./.venv/bin/pytest tests -q` -> `94 passed`
  - short smoke backtest (`2013-01-01` → `2015-06-30`) -> `6.88% CAGR`, `0.12 Sharpe`, `-2.85% MaxDD`, `14.88% avg turnover`
  - initial full backtest result (`14.91% CAGR`, `0.61 Sharpe`, `-25.45% MaxDD`, `24.60% avg turnover`) was invalid because the feature store reused stale stock/sector shards from the old universe roster
  - corrected full backtest after feature-store rebuild -> `17.64% CAGR`, `0.75 Sharpe`, `-33.50% MaxDD`, `33.71% avg turnover`
- Decision:
  - reject as new baseline
- Learning:
  - The first measurement caught a real infrastructure bug: feature-store freshness was not keyed on universe composition, so universe changes could silently produce invalid backtest reads.
  - After correcting that bug and rebuilding the store, the broadened roster still underperformed the branch baseline (`18.94% CAGR`, `0.94 Sharpe`, `-21.47% MaxDD`, `21.92% turnover`) on return, Sharpe, drawdown, and turnover.
  - The broader universe also materially increases runtime because stock-ranker training and RL retrain checkpoints scale with the larger candidate set.

### Task: sector taxonomy experiment (18-sector target, selection_only)
- Scope:
  - remapped the current roster into the proposed 18-sector taxonomy using currently available local price history
  - added clean extra names that already exist in the local price matrix where possible, including:
    - `TATAELXSI.NS`
    - `AUBANK.NS`
    - `HDFCAMC.NS`
    - `LUPIN.NS`
    - `ZYDUSLIFE.NS`
    - `ADANIGREEN.NS`
    - `NTPC.NS`
    - `POWERGRID.NS`
    - `NHPC.NS`
    - `JINDALSTEL.NS`
    - `ETERNAL.NS`
    - `PAYTM.NS`
  - kept the run isolated in `/tmp/sector18_eval so it did not touch the main branch artifacts`
- Validation:
  - `selection_only` short backtest (`2013-01-01` → `2015-06-30`) -> `5.67% CAGR`, `-0.02 Sharpe`, `-10.47% MaxDD`, `72.80% avg turnover`
  - selection diagnostics:
    - Avg selected names `19.5`
    - Top-k avg forward return `0.92%`
    - Top-k vs universe `+0.17%`
    - Top-k vs sector median `+0.01%`
    - Precision@k `48.62%`
    - Rank IC `0.193`
    - Selection stability `26.17%`
- Decision:
  - reject and do not proceed to optimizer-only or full-RL on this taxonomy in the current branch state
- Learning:
  - The proposed finer sector split is reasonable as a target taxonomy, but on the current local history it did not materially improve stock selection versus the existing control setup.
  - The raw ranking edge stayed small (`top-k vs universe` `+0.17%`, `top-k vs sector median` `+0.01%`) and the selection set remained thin (`avg selected names` `19.5`, `stability` `26.17%`), even though intra-sector dispersion was measurable (`0.0592`).
  - That is consistent with relabeling clusters rather than creating a meaningfully easier ranking problem.
  - The current 15-sector model remains the practical baseline until we have broader historical coverage for the missing names.
  - The evaluation now includes explicit intra-sector dispersion in the selection diagnostics, and the added-stock feature path was validated against the local price/volume history for the names actually used in this isolated run.

### Task: existing-sector universe expansion (added supported names only)
- Scope:
  - added the locally supported names into the existing sector buckets without changing the sector taxonomy:
    - `TATAELXSI.NS`, `KPITTECH.NS`
    - `AUBANK.NS`
    - `HDFCAMC.NS`
    - `TATACONSUM.NS`
    - `SONACOMS.NS`, `BALKRISIND.NS`
    - `LUPIN.NS`, `ZYDUSLIFE.NS`
    - `ADANIGREEN.NS`, `ATGL.NS`
    - `JINDALSTEL.NS`
    - `PHOENIXLTD.NS`
    - `SRF.NS`
  - validated that the added names already have local price and volume history, so no fresh download was required for this set
  - ran the isolated `selection_only` smoke backtest against the expanded roster in `/tmp/universe_additions_eval`
- Validation:
  - isolated `selection_only` short backtest (`2013-01-01` → `2015-06-30`) -> `8.13% CAGR`, `0.12 Sharpe`, `-12.06% MaxDD`, `72.85% avg turnover`
  - selection diagnostics:
    - Avg selected names `19.17`
    - Top-k avg forward return `1.11%`
    - Top-k vs universe `+0.23%`
    - Top-k vs sector median `-0.02%`
    - Precision@k `50.87%`
    - Rank IC `0.083`
    - Intra-sector dispersion `0.0577`
    - Top-bottom spread `-0.0006`
    - Selection stability `22.85%`
- Decision:
  - keep the roster additions in the live universe config for now, but do not treat this as a strong stock-selection win yet
- Learning:
  - The broader roster is valid and the added names are correctly wired through the data path.
  - The ranking diagnostics are still mixed: top-k edge improved modestly, but rank IC, stability, and top-bottom spread remain weak.
  - This is a better universe shape than a sector-taxonomy rewrite, but it still needs more evidence before we call it a durable alpha improvement.

### Task: isolated 150-name universe retry
- Scope:
  - expanded the current curated roster to 150 names by adding extra liquid names available in the processed price matrix
  - kept the experiment isolated in temp configs so it did not collide with the background full-RL run
  - measured the stock-selection layer independently with the new `selection_only` / `optimizer_only` split
- Validation:
  - `selection_only` short backtest (`2013-01-01` → `2015-06-30`) -> `4.13% CAGR`, `-0.11 Sharpe`, `-10.32% MaxDD`, `72.85% avg turnover`
  - `optimizer_only` short backtest (`2013-01-01` → `2015-06-30`) -> `8.53% CAGR`, `0.21 Sharpe`, `-6.72% MaxDD`, `51.45% avg turnover`
  - selection diagnostics improved slightly in the optimizer pass, but not enough to beat the earlier isolated 99-name baseline
- Decision:
  - reject and do not keep the 150-name roster as a new baseline
- Learning:
  - The current roster can be expanded, but a naive bump from 99 to 150 names does not improve the isolated stock-selection or optimizer layers.
  - This suggests the next meaningful universe step is not just more names, but more historically correct breadth.

### Task: run_022 retry — selection-only / optimizer-only cross-sectional ranks
- Scope:
  - reintroduced `ret_1m_rank`, `ret_3m_rank`, and `ret_12m_rank` under the new `selection_only` / `optimizer_only` split
  - measured the candidate independently from the RL overlay in an isolated temp workspace
- Validation:
  - `selection_only` short backtest (`2013-01-01` → `2015-06-30`) -> `4.93% CAGR`, `-0.06 Sharpe`, `-9.57% MaxDD`, `73.58% avg turnover`
  - `optimizer_only` short backtest (`2013-01-01` → `2015-06-30`) -> `7.38% CAGR`, `0.13 Sharpe`, `-5.91% MaxDD`, `50.12% avg turnover`
- Learning:
  - The cross-sectional rank idea did improve some selection diagnostics, but it did not improve the portfolio outcome versus the current baseline in either isolated mode.
  - `optimizer_only` remained better than `selection_only`, which means the optimizer is still adding value, but this candidate is not a clear keeper.
  - This is enough evidence to treat `run_022` as a reject and move on to the next stock-selection candidate.

### Task: run_024 — momentum acceleration
- Scope:
  - added `mom_accel_1m` and `mom_accel_3m` to the stock feature set
  - kept the new `selection_only` / `optimizer_only` split for isolated measurement
- Validation:
  - `selection_only` short backtest (`2013-01-01` → `2015-06-30`) -> `4.13% CAGR`, `-0.11 Sharpe`, `-10.32% MaxDD`, `72.85% avg turnover`
  - `optimizer_only` short backtest (`2013-01-01` → `2015-06-30`) -> `7.80% CAGR`, `0.14 Sharpe`, `-6.47% MaxDD`, `50.81% avg turnover`
- Learning:
  - Momentum acceleration improved over its own selection-only pass, which is the right direction, but it still did not beat the earlier isolated optimizer-only baseline.
  - The optimizer is still contributing more than raw selection alone, but this candidate is also a reject for now.
  - This feature family is closed and should not be revisited on this branch.

### Task: cross-sectional ranking feature generation
- Scope:
  - added sector-relative momentum, per-date z-scores, rank transforms, and volatility-adjusted momentum to the stock feature builder
  - widened the feature store invalidation path through a new stock feature logic hash so the expanded feature set rebuilds cleanly
  - kept the universe fixed and reused the current expanded roster as the control set
- Validation:
  - `scripts/build_features.py` rebuilt the feature store with the new logic hash
  - focused regression tests passed:
    - `tests/test_data.py::TestFeatures::test_stock_features_cross_sectional_transforms`
    - `tests/test_data.py::TestFeatures::test_taxonomy_additions_have_price_and_feature_coverage`
  - stored stock feature parquet now includes:
    - `mom_3m_vol_adj`
    - `mom_3m_vol_adj_z`
    - `mom_3m_vol_adj_rank`
    - `ret_3m_z`
    - `ret_3m_rank`
    - `ret_12m_z`
    - `ret_12m_rank`
    - `mom_12m_skip1m_z`
    - `mom_12m_skip1m_rank`
- Learning:
  - This is the right layer to attack next because the issue is ranking signal quality, not taxonomy.
  - The expanded universe and the stock-selection diagnostics can now be tested on a richer, more discriminative feature set without changing sectors again.

### Task: local-sector signal diagnostics
- Scope:
  - froze the current expanded universe as the control set
  - added candidate-level score propagation into the walk-forward selection diagnostics
  - added within-sector IC and within-sector top-bottom spread, plus sector-median separation, aggregated equal-weight and by candidate count
- Validation:
  - `tests/test_selection_diagnostics.py` now checks the new local-sector metrics
  - `selection_diagnostics.json` can now report:
    - `within_sector_ic`
    - `within_sector_ic_weighted`
    - `within_sector_top_bottom_spread`
    - `within_sector_top_bottom_spread_weighted`
    - `within_sector_top_k_minus_sector_median`
    - `within_sector_top_k_minus_sector_median_weighted`
- Learning:
  - The ranking layer is now measurable both globally and locally inside each sector.
  - This is the right diagnostic split for deciding whether the model should be read as a flat cross-sectional ranker or a sector-local ranker inside a hierarchical selection pipeline.

### Task: stock ranker minimal raw v2
- Scope:
  - hard-prune the stock feature builder to the six raw canonical features only:
    - `ret_3m`
    - `mom_12m_skip1m`
    - `mom_accel_3m_6m`
    - `vol_3m`
    - `amihud_1m`
    - `ma_50_200_ratio`
  - retire all `_z`, `_rank`, `_resid`, `_vs_sector`, benchmark-relative, and duplicate-horizon transforms from the stock-ranker experiment series
  - update the schema tests to match the new contract
- Rationale:
  - the ranker is still fragmenting momentum signal across too many correlated transforms
  - the next test is whether a concentrated raw set can finally move within-sector IC and top-bottom spread

### Task: stock horizon shift [NEXT]
- Scope:
  - keep the raw-minimal stock contract fixed
  - vary only the stock label horizon at 4W / 8W / 12W
  - expose the horizon as `stock_model.fwd_window_days` plus a `run_backtest.py` CLI override
- Rationale:
  - the ranker has little to no 4W signal; the next hypothesis is that the signal is slower-moving momentum rather than a broken model wiring path

### Task: production split to tilt-only RL
- Scope:
  - froze the production RL path to neutral posture while preserving learned sector tilts
  - aligned serving fallback to the neutral full-stack baseline instead of the old rule path
  - retained fixed posture holdout baselines for posture research
- Validation:
  - focused suites passed:
    - `tests/test_api_recommender.py`
    - `tests/test_rl_environment_contract.py`
    - `tests/test_rl_holdout.py`
    - `tests/test_rl_control_evaluation.py`
    - `tests/test_data.py`
  - 2016 holdout:
    - `tilt_only_rl`: `33.70% CAGR`, `1.464 Sharpe`, `-14.73% MaxDD`, `27.34% turnover`
    - `neutral_full_stack`: `32.55% CAGR`, `1.433 Sharpe`, `-14.67% MaxDD`, `29.63% turnover`
- Learning:
  - the production edge survives with posture frozen, so the live RL value is still in sector tilts
  - posture should move to a separate realized-outcome research track instead of staying inside the current PPO reward loop

### Task: posture research dataset builder
- Scope:
  - added a realized forward-outcome builder for `risk_on / neutral / risk_off`
  - uses the neutral full-stack path as the reference state stream
  - saves both a parquet row dataset and a JSON summary
- Validation:
  - focused tests passed:
    - `tests/test_posture_dataset.py`
    - `tests/test_rl_holdout.py`
    - `tests/test_api_recommender.py`
  - sample run:
    - `scripts/build_posture_dataset.py --end-date 2016-12-31 --horizon-rebalances 2 --max-samples 4`
    - summary:
      - `sample_count = 4`
      - `best_posture_counts = {'risk_off': 4}`
      - `mean_utility_margin = 0.0751`
- Learning:
  - the research dataset path is now real and labelable from realized outcomes instead of proxy regret
  - the first implementation is compute-heavy because it retrains models redundantly across counterfactual paths
  - the next engineering improvement should be rebalance-date model caching before scaling the dataset

### Task: posture dataset model-state caching
- Scope:
  - added an incremental model-state timeline in `src/rl/posture_dataset.py`
  - caches trained scorer/ranker snapshots by rebalance index on a separate helper engine
  - restores cached model state into counterfactual executors so posture horizon replay runs with `allow_model_retraining=False`
- Validation:
  - focused suites passed:
    - `tests/test_posture_dataset.py`
    - `tests/test_rl_holdout.py`
    - `tests/test_api_recommender.py`
  - cache snapshot round-trip is now covered in `tests/test_posture_dataset.py`
  - larger sample run:
    - `scripts/build_posture_dataset.py --end-date 2016-12-31 --horizon-rebalances 2 --max-samples 8`
- Learning:
  - duplicate model retraining across posture paths is removed
  - larger sample builds are now feasible enough to inspect label balance before any posture model training
  - first larger sample still shows heavy `risk_off` dominance:
    - `best_posture_counts = {'risk_off': 7, 'neutral': 1}`
    - `mean_utility_margin = 0.0817`
  - next question is no longer builder correctness; it is label economics:
    - whether `risk_off` is genuinely best over short horizons
    - or whether the current horizon utility overweights drawdown / turnover relative to return

### Task: posture horizon comparison
- Scope:
  - added artifact prefix support to `scripts/build_posture_dataset.py`
  - ran side-by-side posture dataset builds for:
    - `H = 2` rebalances
    - `H = 3` rebalances
  - both through `2016-12-31` with `max_samples = 16`
- Artifacts:
  - `artifacts/reports/posture_dataset_h2_2016_s16_summary.json`
  - `artifacts/reports/posture_dataset_h3_2016_s16_summary.json`
- Result:
  - `H = 2`:
    - `best_posture_counts = {'risk_off': 15, 'neutral': 1}`
    - `mean_utility_margin = 0.0925`
  - `H = 3`:
    - `best_posture_counts = {'risk_off': 16}`
    - `mean_utility_margin = 0.1100`
- Learning:
  - extending the realized label horizon did not reduce defensive bias
  - `risk_off` dominance persisted and strengthened slightly at `H = 3`
  - the next posture-research question is not “longer horizon or shorter horizon”
  - it is “what utility definition should define the label?”
  - next experiment should compare label balance under:
    - return-only
    - return minus drawdown
    - current full utility
