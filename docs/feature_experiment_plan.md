# Feature Experiment Plan — Stock Ranker Improvement Series

For the run-by-run operating checklist, see
[docs/stock_ranker_experiment_protocol.md](./stock_ranker_experiment_protocol.md).

Current active stock-ranker contract:
- `ret_3m`
- `mom_12m_skip1m`
- `mom_accel_3m_6m`
- `vol_3m`
- `amihud_1m`
- `ma_50_200_ratio`

All `_z`, `_rank`, `_resid`, `_vs_sector`, and benchmark-relative stock features
have been retired for the current experiment series.

## Goal

Improve the cross-sectional stock ranking signal quality through targeted,
one-at-a-time feature additions. The LightGBM LambdaRank ranker currently
relies too heavily on raw volatility features (vol_12m: 5.9%, skew_3m: 4.2%)
while return-based signals rank near the bottom despite being the theoretically
correct factor for cross-sectional ranking.

---

## Baseline & Rules

| Item | Value |
|------|-------|
| Baseline run | run_019 — 22.13% CAGR, Sharpe 0.84 (accumulated buffer, clean data) |
| Config | `retrain_freq_weeks: 12`, `rl_triggers.enabled: false` (Ablation 2 winner) |
| RL policy | Carry forward between all steps — do NOT wipe RL artifacts |
| Gate | Each run must beat previous run's CAGR by ≥0.5% to keep the change |
| Revert rule | If delta < 0.5%, revert `stock_features.py` and note as "rejected" |
| Feature store | Schema hash auto-detects column additions. For value-only changes (same column names), call `store.invalidate("stock")` manually before the run |

---

## Experiment Table

| Run | Step | Change | Status | CAGR | Sharpe | Delta | Decision |
|-----|------|--------|--------|------|--------|-------|----------|
| run_020 | Step 0 | Drop `above_50ma` + `above_200ma` | ✅ KEEP | 22.19% | 0.96 | +0.06% CAGR / Sharpe 0.84→0.96 / MaxDD −30%→−23.5% | KEEP — Sharpe and drawdown both improved significantly |
| run_021 | Step 1 | Add Sharpe features | ⏳ PENDING | — | — | — | — |
| run_022 | Step 2 | Add CS percentile ranks | ⏳ PENDING | — | — | — | — |
| run_023 | Step 3 | Fix sector z-score normalisation | ⏳ PENDING | — | — | — | — |
| run_024 | Step 4 | Add momentum acceleration | ⏳ PENDING | — | — | — | — |
| run_025 | Step 5 | Combine all winners; prune to ≤42 features | ⏳ PENDING | — | — | — | — |

---

## Step-by-Step Detail

---

### Step 0 — Drop dead binary features (run_020)

**File**: `src/features/stock_features.py`

**Change**: Remove `above_50ma` and `above_200ma` from `feat_dict`.

```python
# REMOVED — 0% SHAP importance, redundant vs continuous ma_50_200_ratio
# feat_dict["above_50ma"]  = (prices > prices.rolling(50).mean()).astype(float)
# feat_dict["above_200ma"] = (prices > prices.rolling(200).mean()).astype(float)
```

**Hypothesis**: Binary 0/1 features carry no marginal information beyond the
continuous `ma_50_200_ratio = (MA50/MA200) − 1` already in the feature set.
Removing them frees two slots for the additions in steps 1–4 without going
above the target feature count.

**Expected effect**: Neutral to slightly positive. Fewer irrelevant features
reduce LightGBM's chance of splitting on noise. Feature count: 42 → 40.

**Schema impact**: Column removal → hash mismatch → auto-rebuild triggered.

---

### Step 1 — Volatility-adjusted return features (run_021)

**File**: `src/features/stock_features.py`

**Add after the Returns / Momentum block**:

```python
# ── Volatility-adjusted returns (Sharpe-style) ─────────────────────────
eps = 1e-6
feat_dict["sharpe_1m"]  = feat_dict["ret_1m"]  / (feat_dict["vol_1m"]  + eps)
feat_dict["sharpe_3m"]  = feat_dict["ret_3m"]  / (feat_dict["vol_3m"]  + eps)
feat_dict["sharpe_12m"] = feat_dict["ret_12m"] / (feat_dict["vol_12m"] + eps)
feat_dict["calmar_3m"]  = feat_dict["ret_3m"]  / (feat_dict["max_dd_3m"].abs() + eps)
```

**Hypothesis**: LightGBM cannot natively divide two input columns — it can
only split on individual feature values. Providing `ret/vol` directly as a
column promotes momentum signal quality, since a +5% return with vol=10% is
categorically different from +5% with vol=40%. `calmar_3m` combines 3-month
return with 3-month drawdown, rewarding consistent rather than volatile gains.

**Expected effect**: +1–3% CAGR. The ranker should shift weight away from
raw `vol_12m` toward the risk-adjusted signals. Feature count: 40 → 44.

**Dependencies**: `ret_1m`, `ret_3m`, `ret_12m`, `vol_1m`, `vol_3m`,
`vol_12m`, `max_dd_3m` must all be computed before this block.

**Schema impact**: New columns → hash mismatch → auto-rebuild.

---

### Step 2 — Cross-sectional percentile ranks (run_022)

**File**: `src/features/stock_features.py`

**Add after the Returns / Momentum block** (before sector-relative section):

```python
# ── Cross-sectional percentile ranks ─────────────────────────────────
# Computed on the wide (date × ticker) matrix before stacking to long format
for col, src in [("ret_1m_rank", "ret_1m"), ("ret_3m_rank", "ret_3m"),
                 ("ret_12m_rank", "ret_12m")]:
    if src in feat_dict:
        feat_dict[col] = feat_dict[src].rank(axis=1, pct=True)
```

**Hypothesis**: Raw return values conflate absolute magnitude with relative
strength. A +3% 1-month return means very different things in a bull (+15%
average) vs bear (−5% average) market. A 90th-percentile cross-sectional rank
is regime-invariant — it always signals a strong buy candidate. Ranks also
compress outliers, which improves LightGBM split quality on non-Gaussian data.

**Expected effect**: +0.5–2% CAGR. Should especially help in high-dispersion
periods (2017 bull, 2020 recovery) where absolute magnitudes are misleading.
Feature count: 44 → 47.

**Schema impact**: New columns → auto-rebuild.

---

### Step 3 — Within-sector z-score normalisation (run_023)

**File**: `src/features/stock_features.py`

**In the sector-relative features loop** — change raw mean-subtraction to
division by sector standard deviation:

```python
# BEFORE (current):
feat_dict[f"{feat_name}_vs_sector"] = feat_dict[feat_name] - sec_mean_df

# AFTER (fix):
sec_std_df = pd.DataFrame({
    t: feat_dict[feat_name][[t2 for t2 in tickers
                              if sector_map.get(t2) == sector_map[t]]].std(axis=1)
    for t in tickers if t in sector_map
})
feat_dict[f"{feat_name}_vs_sector"] = (
    (feat_dict[feat_name] - sec_mean_df)
    / sec_std_df.replace(0, np.nan)
)
```

**Hypothesis**: A 3% excess return vs sector peers means very different things
in IT (sector annualised vol ~30%) vs FMCG (sector annualised vol ~12%). The
current `ret − sector_mean` signal is scale-dependent and incomparable across
sectors. Division by `sector_std` produces a proper z-score, making the
6 `_vs_sector` features (ret_1w/1m/3m/6m/12m/vol_3m) interpretable in the
same unit across all 15 sectors.

**Expected effect**: +0.5–1.5% CAGR. Biggest gains in diversified multi-sector
portfolios where IT vs FMCG vs Metals comparisons currently have mismatched
scales. **Schema impact: column names unchanged → hash match → must call
`store.invalidate("stock")` manually before this run.**

**Note**: Sector std is undefined for single-stock sectors (if any). Handle
with `.replace(0, np.nan)` to avoid inf values.

---

### Step 4 — Momentum acceleration (run_024)

**File**: `src/features/stock_features.py`

**Add after the Returns / Momentum block**:

```python
# ── Momentum acceleration ─────────────────────────────────────────────
# Did momentum improve or fade vs last month?
feat_dict["mom_accel_1m"] = feat_dict["ret_1m"] - feat_dict["ret_1m"].shift(21)
feat_dict["mom_accel_3m"] = feat_dict["ret_3m"] - feat_dict["ret_3m"].shift(21)
```

**Hypothesis**: A stock up 8% this month that was flat last month has
accelerating momentum — the underlying driver is strengthening. A stock up 8%
this month that was up 15% last month is decelerating — the driver may be
fading. Momentum acceleration is a leading indicator of trend continuation vs
reversal. Raw momentum features cannot distinguish these two cases.

**Expected effect**: +0.5–2% CAGR. Best in trending regimes (2017, 2021, 2024).
Potential risk: adds noise in range-bound markets (2022). Feature count +2.

**Schema impact**: New columns → auto-rebuild.

---

### Step 5 — Combine winners and prune to ≤42 features (run_025)

**File**: `src/features/stock_features.py`

**Process**:
1. Keep all step changes that individually beat the gate (CAGR delta ≥ +0.5%)
2. Run SHAP feature importance on the combined feature set
3. Drop features ranked below the 30th percentile in importance that have
   a conceptual near-duplicate already in the set
4. Target: ≤42 features (current count minus removals plus net additions)

**Why prune?**: LightGBM performance degrades slightly with irrelevant features
because tree splits can be wasted on noise columns. Historical evidence:
run_013 (42-feat pruned) had marginally better Sharpe than run_012 (44-feat
unpruned) in controlled tests. Pruning also reduces model training time and
overfitting risk.

**Candidate features to prune** (pending importance analysis):
- `mom_stab_3m` / `mom_stab_12m`: fraction of positive days — may be
  dominated by `sharpe` features once those are added
- `skew_3m` / `kurt_3m`: distributional moments captured indirectly by
  `vol_ratio_1m_3m` and `calmar_3m`
- `price_to_52w_low`: often mirrors `price_to_52w_high`

---

## Sector Z-Score Fix — Implementation Notes

For Step 3, the per-ticker std computation needs care:

```python
sectors = sorted(set(sector_map.values()))
for feat_name in ["ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_3m"]:
    if feat_name not in feat_dict:
        continue
    sector_means: dict[str, pd.Series] = {}
    sector_stds:  dict[str, pd.Series] = {}
    for sec in sectors:
        sec_tickers = [t for t in tickers
                       if sector_map.get(t) == sec
                       and t in feat_dict[feat_name].columns]
        if not sec_tickers:
            continue
        sector_means[sec] = feat_dict[feat_name][sec_tickers].mean(axis=1)
        sector_stds[sec]  = feat_dict[feat_name][sec_tickers].std(axis=1)

    sec_mean_df = pd.DataFrame({
        t: sector_means.get(sector_map[t], pd.Series(np.nan, index=prices.index))
        for t in tickers if t in sector_map
    })
    sec_std_df = pd.DataFrame({
        t: sector_stds.get(sector_map[t],  pd.Series(np.nan, index=prices.index))
        for t in tickers if t in sector_map
    })
    feat_dict[f"{feat_name}_vs_sector"] = (
        (feat_dict[feat_name] - sec_mean_df)
        / sec_std_df.replace(0, np.nan)
    )
```

---

## Decision Log

| Run | Decision | Reason |
|-----|----------|--------|
| run_020 | KEEP | CAGR flat (+0.06%), Sharpe 0.84→0.96, MaxDD −30%→−23.5%. New baseline: 22.19% CAGR, Sharpe 0.96. |
| run_021 | — | Pending |
| run_022 | — | Pending |
| run_023 | — | Pending |
| run_024 | — | Pending |
| run_025 | — | Pending |

---

## After run_025

If the combined feature set beats the cumulative baseline by ≥1% CAGR, the
next focus area is the **RL state vector**:

1. Add accumulated `sharpe_1m/3m` sector-level signals to the 60-dim sector
   state (currently only `mom_1m`, `mom_3m`, `rel_str_1m`, `breadth_3m`)
2. Add portfolio Sharpe rolling 3m to the 10-dim portfolio state
3. Re-test 12w vs 26w retrain frequency with the enriched state

Feature experiments are intentionally decoupled from RL state changes to avoid
confounding results.
