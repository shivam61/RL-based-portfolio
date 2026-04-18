# Next Steps — Improvement Backlog

Each item should be implemented one at a time, backtested, committed, and measured before moving on.

## Run History

| Run | Description | CAGR | Sharpe | MaxDD | Final NAV |
|-----|-------------|------|--------|-------|-----------|
| run_001 | Original baseline (with bugs) | 21.21% | 0.86 | -30.47% | ₹43.5L |
| run_002 | Fix: sector label leakage + double-tilt | 16.47% | 0.65 | -24.79% | ₹27.8L |
| run_003 | Feat: RL 20k timesteps, Sharpe reward, India VIX, fix macro path | 17.18% | 0.60 | -33.33% | ₹29.7L |
| run_004 | **Fix: sector feature store dedup bug (all 15 sectors now feed RL)** | **23.71%** | **0.92** | -31.29% | **₹54.8L** |

## P0 — Make RL Constraint-Aware (PLANNED, not yet started)

### P0-A: Raise sector cap (1 line)
- File: `config/base.yaml`
- Change: `max_sector_weight: 0.45` → `0.50`
- Rationale: Let RL express strong sector conviction without hitting ceiling

### P0-B: Feed realized sector weights into RL state
- Problem: RL sets tilt=2.0 for IT, optimizer clips to 45%. RL never learns what the actual realized weight was.
- Fix: Add `realized_sector_weights` (15 floats, post-optimizer per-sector allocation) to experience buffer `outcome` dict and to RL observation state
- STATE_DIM: 82 → 97
- Files: `walk_forward.py` (step M + decide() call), `environment.py` (constants + _get_obs), `agent.py` (decide() + _build_obs)
- Important: Delete old ppo_model.zip before running (STATE_DIM mismatch will crash SB3)

## Phase 1 — FII/DII Institutional Flow Data

- Source: NSE India free daily FII/DII reports (jugaad-data or manual CSV backfill)
- New file: `src/data/fii_dii.py` (mirrors MacroDataManager pattern)
- Features to add to macro feature builder:
  - `fii_net_5d`, `fii_net_20d` — rolling net FII buy in ₹ crore
  - `fii_flow_zscore` — (fii_net_20d - mean_1y) / std_1y
  - `dii_net_5d`, `dii_absorb_ratio` — DII absorbing FII selling signal
- Add to RL state (feeds directly into macro_state dict)
- Expected impact: RL can identify FII-led bull/bear regimes; sector scorer gets cross-sectional FII flow signal

## Phase 2 — Earnings Surprise Features

- Source: Screener.in unofficial API or BSE quarterly results
- Must lag by 1 quarter (point-in-time discipline)
- Features per stock:
  - `eps_growth_yoy`, `rev_growth_yoy`
  - `eps_surprise_pct` (actual vs prior quarter as proxy if no consensus)
  - `pe_vs_sector` — P/E relative to sector median
- Add to `src/features/stock_features.py`
- Expected impact: Stock ranker gains fundamental momentum signal alongside pure price momentum

## Phase 3 — RL Architecture Upgrade (after phases 1-2 add more state signal)

- Current: PPO (continuous action), 20k timesteps, Sharpe reward
- Consider: SAC (Soft Actor-Critic) — better for continuous action spaces, more sample-efficient
- Reward shaping: add drawdown-conditioned scaling (penalize harder when portfolio is near max_dd_hard)
- State: reduce noise with PCA on the 60 sector-momentum dims (now real data, worth compressing)

## Notes
- Always delete `artifacts/models/rl_agent/ppo_model.zip` and `meta.pkl` when STATE_DIM changes
- Run `scripts/build_features.py` before `run_backtest.py` after any feature store schema change
- Each run should copy reports to `artifacts/run_history/run_NNN_description/`
- The sector feature store dedup fix (run_004) is the biggest single improvement found so far:
  RL was flying completely blind on sector state (60/82 state dims were structural zeros)
