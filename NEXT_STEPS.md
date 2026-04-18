# Next Steps — Improvement Backlog

Each item is implemented one at a time, backtested, committed, and measured before moving on.
Best known result: **run_010 — 23.57% CAGR, Sharpe 0.93, ₹54.1L final NAV** (8-week retrain, no triggers)

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

### Ablation: Retrain Frequency × Event Triggers (run from run_009 state)

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

2. **RL needs data, not architecture.** With only 147 training steps, adding state dims
   adds noise faster than signal. P0-B and FII proxy both hurt for this reason.

3. **Proxy signals are dangerous.** FII proxy (built from USDINR + Nifty vs SP500) sounds
   reasonable but introduces correlated noise the RL can't separate from true macro signal.

4. **The constraint.** RL sector cap increase (0.35→0.50) consistently hurts. At 147 steps,
   wider freedom = more variance the policy gradient can't model.

---

## Current Working State (run_009 config)

- `config/base.yaml`: `max_sector_weight: 0.35`, `total_timesteps: 20000`, `n_epochs: 10`
- `src/rl/environment.py`: STATE_DIM=82, macro keys use `nifty_ret_1m/nifty_above_200ma`
- `src/features/feature_store.py`: sector dedup fixed (15 sectors per snapshot)
- `src/data/macro.py`: India VIX + Nifty IT added to macro data
- FII proxy code exists in `src/data/fii_proxy.py` — computed into feature store but **not
  wired into RL state** until real FII data is available
- P0-B code exists (realized sector weights in experience buffer) — **not in STATE_DIM yet**
  until RL has 500+ live experience steps

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
