# Next Steps — Improvement Backlog

Each item should be implemented one at a time, backtested, committed, and measured before moving on.

## Run History

| Run | Description | CAGR | Sharpe | MaxDD | Final NAV |
|-----|-------------|------|--------|-------|-----------|
| run_001 | Original baseline (with bugs) | 21.21% | 0.86 | -30.47% | ₹43.5L |
| run_002 | Fix: sector label leakage + double-tilt | 16.47% | 0.65 | -24.79% | ₹27.8L |
| run_003 | Feat: RL 20k timesteps, Sharpe reward, India VIX, fix macro path | 17.18% | 0.60 | -33.33% | ₹29.7L |
| run_004 | **Fix: sector feature store dedup bug (all 15 sectors now feed RL)** | **23.71%** | **0.92** | -31.29% | **₹54.8L** |
| run_005 | P0-A: sector cap 0.35→0.50 alone (RL overtrades without feedback) | 15.58% | 0.63 | -22.11% | ₹25.5L |
| run_006 | P0-B: realized sector weights in RL state (STATE_DIM 82→97) | 20.57% | 0.85 | -27.59% | ₹41.0L |
| run_007 | P0-A+B combined (cap 0.50 + feedback): too much freedom for 147 steps | ~11% | 0.35 | -39.48% | ₹19.4L |
| run_008 | Phase 1: FII proxy flow features (fii_flow_zscore, fii_sell_regime in RL) | ~20% | 0.78 | -29.91% | ₹37.5L |

## P0 — Make RL Constraint-Aware (DONE — mixed results)

**Lesson learned**: P0-A (wider cap) alone and combined with P0-B both hurt. Root cause:
with only ~147 experience steps, the RL doesn't have enough training data to learn when
to use a wider cap. The cap increase creates more outcome variance the RL can't model.
P0-B code is correct and stays in — it will help more as experience buffer grows over live use.
Cap reverted to 0.35.

### P0-A: Raise sector cap — REVERTED (hurt performance)
- Tried 0.35→0.50. CAGR dropped from 23.71% to 15.58% alone, 11% combined.
- Root cause: RL has only 147 training steps; wider cap = more variance = worse learning.
- Keep at 0.35 until RL has more live experience (100+ additional rebalances).

### P0-B: Feed realized sector weights into RL state — DONE ✅
- STATE_DIM: 82 → 97 (+15 realized sector weight dims at positions 82-96)
- Slightly hurt alone (20.57% vs 23.71%) but architecturally correct.
- Will compound positively with more training data and when cap is raised later.

## Phase 1 — FII/DII Flow (DONE — synthetic proxy implemented)

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

## Phase 1 — FII/DII Flow (DONE — synthetic proxy)

**Status**: Implemented in run_008. Slightly below run_004 (RL macro keys changed).
**Note on real data**: Fyers API = UI only (no programmatic access). NSE provides monthly CSVs
manually at nseindia.com/reports. If downloaded and placed at `data/raw/fii_dii_historical.csv`,
the code (`src/data/fii_proxy.py`) will automatically use real data instead of proxy.

**Features added** (`src/data/fii_proxy.py` → `src/features/macro_features.py`):
- `fii_proxy_flow` — daily composite (INR strength 50% + Nifty vs SP500 35% - VIX premium 15%)
- `fii_flow_zscore` — regime signal (z-score of 20d rolling flow vs 1y baseline)
- `fii_sell_regime` — binary flag when z-score < -1.5 (heavy FII selling)
- `fii_buy_regime`  — binary flag when z-score > 1.5  (heavy FII buying)
- `india_vs_global` — Nifty 5d minus SP500 5d (FII preference signal)
- `inr_strength_5d` — inverted USDINR 5d return

**RL state**: replaced `nifty_ret_1m`, `nifty_above_200ma` with `fii_flow_zscore`, `fii_sell_regime`

**TODO**: Download real NSE FII/DII CSVs for better signal quality. Place at `data/raw/fii_dii_historical.csv`.

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
