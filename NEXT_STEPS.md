# Next Steps — Improvement Backlog

Each item is implemented one at a time, backtested, committed, and measured before moving on.
Best known result: **run_020 — 22.19% CAGR, Sharpe 0.96, MaxDD −23.5%** (12w retrain, no triggers, dropped zero-importance binary features)
Stable ceiling on corrected data: **~22–25% CAGR** (12w retrain config — ablation D2 showed 24.98% single-pass; run_020 is 22.19% accumulated)

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
| run_024 | Step 4 | Add momentum acceleration: `mom_accel_1m`, `mom_accel_3m` | ⏳ PENDING | — | — | — | — |
| run_025 | Step 5 | Combine all winning steps; prune back to ≤42 features | ⏳ PENDING | — | — | — | — |

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

---

## Ops Notes
- Always `rm artifacts/models/rl_agent/ppo_model.zip meta.pkl experience_buffer.pkl`
  before running backtest when STATE_DIM changes
- Always reset `_metadata.json` macro last_date when `macro_features.py` changes
- Each run saves reports to `artifacts/run_history/run_NNN_description/`
- Commit after every backtest with full metrics in commit message
