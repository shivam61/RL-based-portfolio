# Signal-Conditioned Concentration — Experiment Plan

**Date**: 2026-04-26
**Author**: Shivam Gupta
**Status**: READY TO IMPLEMENT

---

## 1. Root Cause Diagnosis

Every feature experiment in the current series (run_021–024) failed despite testing
well-motivated additions (volatility-adjusted returns, CS ranks, sector z-scores,
momentum acceleration). All four showed >6% CAGR collapse. The cause is not the
features themselves — it is a structural weakness in the selection layer that features
cannot fix.

**Evidence chain:**

| Signal | Value | Interpretation |
|---|---|---|
| within_sector_ic 2016-2019 | negative all 4 years | Ranker produces no usable ordering in bull regimes |
| within_sector_ic 2015, 2020 | positive | Signal works in choppy/crisis regimes only |
| RankIC (56D baseline) | -0.021 | Near zero — essentially random cross-sectional ordering |
| Universe expansion (Stages A–D) | all worse | Adding names in weak-signal regime amplifies noise |
| Horizon shift (28D vs 56D) | 28D positive only 2/6 years | Not a horizon problem — structural signal absence |

**Root cause (agreed diagnosis):**

In bull markets (India 2016–2019), momentum scores cluster — all stocks trend up at
similar rates. `std(rank_scores)` collapses. The top-k selection is near-random in this
regime. The model still picks top-5 per sector and concentrates the portfolio, but those
top-5 names carry zero additional alpha over the next-5. Concentrated noise amplification.

In choppy or crisis markets (2015, 2020), scores genuinely spread. Top-k selection
discriminates. The signal is real and the portfolio earns it.

**The fix is not more features.** Features are adequate for choppy regimes and irrelevant
in trend regimes because the generative process (all stocks co-moving) does not produce
separable cross-sectional signal regardless of how many features you add.

**The fix is adaptive selection:** when scores are clustered (weak signal), diversify;
when scores are spread (strong signal), concentrate.

---

## 2. Decision Template (8-field)

**1. Current diagnosis:**
Stock ranker IC collapses in trending bull markets because momentum scores cluster when
all stocks co-move. Fixed top-k selection in this regime is concentrated noise, not alpha.
The selection count should be conditioned on whether the ranking actually discriminates.

**2. Why this is next:**
- All 4 feature experiments failed (run_021–024), establishing that features cannot fix structural signal absence
- Universe experiments (Stages A–D) all failed — adding names in weak-signal regime worsens results
- Horizon shift (TASK-8) closed — 28D fails IC gate, 56D retained
- This is the minimal, targeted intervention that directly addresses the diagnosed root cause
- No new data, no new features, no RL changes — one config flag

**3. Alternatives considered:**

| Alternative | Why not now |
|---|---|
| More features | Already tested 4 additions — all rejected. Features don't create signal where the generative process lacks it |
| Earnings/fundamental data | PARKED — Screener.in coverage 84.8% missing before 2019, unusable for 2013-2020 window |
| Shorter label horizon | CLOSED (TASK-8) — 28D fails IC gate. Not a horizon problem |
| Regime-switching feature weighting | Complex, more parameters, no clear signal for the switching mechanism |
| Universe contraction | Tested (Stages A–D) — not the lever |
| Fixed bucket breadth (Stage 3 original plan) | Does not adapt to signal quality; would still concentrate in weak-signal regimes |

**4. Expected upside:**
- IC stability improvement in 2016–2019 (currently all negative) — the primary target
- Turnover reduction in weak-signal periods (fewer churn trades from noise picks)
- CAGR improvement in bull years specifically, or at minimum no degradation
- Portfolio quality improvement: in weak-signal periods, a wider but signal-consistent
  portfolio beats a narrow noise-dominated one

**5. Side effects / risks:**

| Risk | Mitigation |
|---|---|
| Over-diversification in mixed-signal periods | Three-tier thresholds (p33/p67) limit extremes; mid-tier keeps current behavior |
| Threshold overfitting to 2013-2020 window | Thresholds are percentile-based against trailing 8-rebalance history, not hardcoded |
| Score-spread measure is garbage-in | Use within-sector p90-p10 of rank scores, not global std; more robust to outliers |
| Interaction with sector model | top_k_per_sector mechanism preserved — adaptation is in k, not in sector selection |
| Fewer stocks in strong-signal periods increases concentration risk | Hard floor of k=3 per sector prevents single-stock dominance |

**6. Success criteria (all required for promotion):**
- within_sector_ic improves in ≥2 of 4 years (2016-2019) vs 56D baseline
- CAGR ≥ 10.14% (selection_only 56D baseline was 10.64%; allow ≤0.5% slack)
- Turnover ≤ 56D baseline in weak-signal years
- Stability (Jaccard) ≥ 56D baseline
- No degradation in 2015 or 2020 (years where current IC is already positive)

**7. Rejection criteria:**
- within_sector_ic worsens in 2016–2019 compared to 56D baseline → signal-conditioned k adds noise, not alpha
- CAGR drops > 1 pt below 56D baseline → net harm from adaptive selection
- 2015 or 2020 IC degrades (regimes where the signal currently works must not be hurt)
- Thresholds fire on < 20% or > 80% of rebalances (degenerate calibration)

**8. Baseline:**
`selection_only` 56D on 2013-2020, seed 42.
- CAGR 10.64%, Sharpe 0.26, stability 32.5%, RankIC -0.021
- Per-year IC: 2015=positive, 2016=negative, 2017=negative, 2018=negative, 2019=negative, 2020=positive

---

## 3. Design

### 3.1 Core Mechanism

Replace the fixed `top_k_per_sector` with a dispersion-adaptive k, computed
fresh at each rebalance from the current period's within-sector score distribution.

```
High dispersion (strong signal) → small k → concentrate in genuine alpha names
Low dispersion (weak signal)    → large k → diversify across sector, reduce noise bet
```

### 3.2 Dispersion Measure

**Chosen: within-sector p90–p10 spread of rank scores (per sector, then averaged)**

```python
def _score_dispersion(sector_scores: pd.Series) -> float:
    """p90-p10 spread within a sector's predicted rank scores."""
    if len(sector_scores) < 4:
        return 0.0
    return float(sector_scores.quantile(0.9) - sector_scores.quantile(0.1))
```

**Why p90–p10 and not std:**
- `std` is inflated by outlier scores; a single extreme prediction (common in LightGBM)
  makes std look "high signal" when most scores are still clustered
- p90–p10 measures the bulk spread, not the tails
- Both are dimensionless relative to the score scale, so percentile comparison is valid

**Why per-sector and not global:**
- `top_k_per_sector` is the selection mechanism — each sector independently selects
  its top-k. The dispersion signal should be computed at the same level of granularity.
- Global dispersion would mix sectors with genuinely different alpha characteristics
- In a bull run, IT might have high dispersion (tech bifurcation) while Banking is flat.
  A per-sector measure captures this; global averaging obscures it.

**Aggregation across sectors:**
- Use median sector dispersion (not mean) to avoid outlier sectors dominating the threshold

### 3.3 Threshold Calibration

**Chosen: percentile-based against trailing 8-rebalance history (data-driven, not hardcoded)**

```python
def _adaptive_top_k(dispersion: float, dispersion_history: list[float], cfg: dict) -> int:
    """Compute adaptive top_k from current dispersion vs trailing history."""
    if len(dispersion_history) < 4:
        # Insufficient history: fall back to default k
        return cfg["stock_model"]["top_k_per_sector"]

    pct = percentile_rank(dispersion, dispersion_history)

    if pct < 0.33:   # bottom third historically → weak signal
        return cfg["stock_model"].get("top_k_low_signal", 7)
    elif pct < 0.67: # middle third → normal signal
        return cfg["stock_model"].get("top_k_mid_signal", 5)
    else:            # top third → strong signal
        return cfg["stock_model"].get("top_k_high_signal", 3)
```

**Why trailing 8 rebalances (~8 months)?**
- Short enough to adapt to regime shifts
- Long enough to avoid reacting to single-period noise
- Calibration is relative (percentile rank), so the thresholds are self-adjusting as
  the market's overall dispersion level shifts

**Why not hardcoded absolute thresholds?**
- LightGBM rank scores don't have a fixed scale. Score spread at period 50 (more
  training data) differs from period 20. Hardcoded thresholds overfit to the
  training window's specific score distribution.
- Percentile-based thresholds adapt as score scale evolves.

### 3.4 k Values

| Tier | Threshold | k | Rationale |
|---|---|---|---|
| Low signal | dispersion_pct < 0.33 | 7 | Spread risk across 7 names; reduce noise concentration |
| Mid signal | 0.33 ≤ pct < 0.67 | 5 | Current config default — no behavior change in mid-signal |
| High signal | pct ≥ 0.67 | 3 | Concentrate in genuine alpha; minimize dilution |

**Why 7/5/3 and not 24/16/10?**
- The mechanism is `top_k_per_sector`, not global top-k
- With 15 sectors active: 7 × 15 = 105 names (too wide), 3 × 15 = 45 names (reasonable)
- But not all 15 sectors are selected — sector model filters to ~8–11 active sectors
- 7 × 9 sectors = 63 names at most in weak-signal periods (reasonable diversification)
- 3 × 9 sectors = 27 names at most in strong-signal periods (concentrated but not single-name)

Starting values are conservative. The experiment measures whether the directional
mechanism improves diagnostics before tuning the k values.

### 3.5 Config (off by default)

```yaml
stock_model:
  adaptive_top_k: false               # off by default; enable for this experiment
  top_k_low_signal: 7                 # k when dispersion in bottom third
  top_k_mid_signal: 5                 # k when dispersion in middle third (= current default)
  top_k_high_signal: 3                # k when dispersion in top third
  adaptive_top_k_history_window: 8    # trailing rebalances for percentile calibration
```

---

## 4. Pushbacks Resolved

### Pushback 1: `std(rank_scores)` is wrong measure
**Raised**: LightGBM outputs unbounded scores; std is inflated by outliers and not
comparable across sectors or training windows.
**Resolved**: Using `p90–p10` within-sector. Measures bulk spread, not tail-driven spread.
More stable across sectors of different sizes and training history lengths.

### Pushback 2: Hardcoded thresholds overfit to training window
**Raised**: Absolute thresholds fit the 2013–2020 dispersion distribution specifically.
Will fail out-of-sample as score scale evolves.
**Resolved**: Using percentile rank against trailing 8-rebalance history. Thresholds
are relative (bottom/mid/top third), not absolute. Self-calibrating as regime shifts.

### Pushback 3: Global top-k breaks per-sector mechanism
**Raised**: A global top-k of 24 in weak signal gives 1.6 stocks/sector across 15 sectors —
guts some sectors entirely.
**Resolved**: Keeping `top_k_per_sector` as the mechanism; adapting the per-sector k.
Low-signal tier: k=7 per sector (not 24 global). Sector structure preserved.

### Pushback 4: Score dispersion ≠ predictive validity
**Raised**: A garbage ranker with extreme outliers looks "high signal" from score spread,
but actual predictive validity could be near zero.
**Considered**: Using trailing realized IC as the signal strength measure instead.
**Decision**: Start with score dispersion for simplicity (no extra state required).
If the experiment shows regime mismatch (dispersion high but IC still low), switch to
trailing IC as the conditioning variable in iteration 2.

---

## 5. Implementation Plan

### Files to modify

| File | Change |
|---|---|
| `src/models/stock_ranker.py` | Add `_score_dispersion()`, `_adaptive_top_k()`, dispersion history tracking; modify `predict()` to call adaptive k when flag is set |
| `config/base.yaml` | Add 5 new keys under `stock_model` (all off by default) |

### Files NOT to touch
- `src/features/stock_features.py` — feature contract frozen
- `src/rl/` — RL layer untouched
- `src/optimizer/` — optimizer untouched
- `config/universe.yaml` — universe frozen

### Code sketch (stock_ranker.py)

```python
# In StockRanker.__init__:
self._dispersion_history: list[float] = []   # trailing per-rebalance dispersions

# New helper:
@staticmethod
def _score_dispersion(scores: pd.Series) -> float:
    if len(scores) < 4:
        return 0.0
    return float(scores.quantile(0.9) - scores.quantile(0.1))

# New helper:
def _adaptive_top_k(self, current_dispersion: float) -> int:
    history = self._dispersion_history
    window = int(self._model_cfg.get("adaptive_top_k_history_window", 8))
    if len(history) < 4:
        return int(self._model_cfg.get("top_k_per_sector", 5))
    recent = history[-window:]
    pct = sum(1 for h in recent if h < current_dispersion) / len(recent)
    if pct < 0.33:
        return int(self._model_cfg.get("top_k_low_signal", 7))
    elif pct < 0.67:
        return int(self._model_cfg.get("top_k_mid_signal", 5))
    else:
        return int(self._model_cfg.get("top_k_high_signal", 3))

# In predict() — after computing sector scores, before top-k selection:
if self._model_cfg.get("adaptive_top_k", False):
    sector_dispersions = [
        self._score_dispersion(sector_scores[sector])
        for sector in active_sectors
        if len(sector_scores.get(sector, [])) >= 4
    ]
    if sector_dispersions:
        median_dispersion = float(np.median(sector_dispersions))
        k = self._adaptive_top_k(median_dispersion)
        self._dispersion_history.append(median_dispersion)
    else:
        k = int(self._model_cfg.get("top_k_per_sector", 5))
else:
    k = int(self._model_cfg.get("top_k_per_sector", 5))
```

---

## 6. Experiment Execution

### Step 1 — Implement and sanity-test

```bash
# Verify adaptive k fires correctly with test cases
PYTHONPATH=. .venv/bin/python -c "
from src.models.stock_ranker import StockRanker
# instantiate with adaptive_top_k: true, feed mock scores, verify k changes
"
```

### Step 2 — Run selection_only on 2013-2020

```bash
PYTHONPATH=. .venv/bin/python scripts/run_backtest.py \
  --mode selection_only \
  --start 2013-01-01 \
  --end 2020-12-31
```

Enable flag in config before running:
```yaml
stock_model:
  adaptive_top_k: true
```

### Step 3 — Evaluate against gate

Primary check: per-year IC from `artifacts/reports/selection_diagnostics.json`
```
Required: within_sector_ic improves in ≥2 of {2016, 2017, 2018, 2019}
Required: CAGR ≥ 10.14% (56D baseline 10.64% - 0.5pp slack)
Required: 2015 and 2020 IC not degraded
Required: threshold activation rate 20%-80% (not degenerate)
```

### Step 4 — If selection gate passes: run full backtest

```bash
PYTHONPATH=. .venv/bin/python scripts/run_backtest.py \
  --mode full_rl \
  --start 2013-01-01 \
  --end 2026-04-17
```

Compare vs run_020 baseline (22.19% CAGR, 0.96 Sharpe).

### Step 5 — Log result in NEXT_STEPS.md

Whether pass or fail: record per-year IC, CAGR, dispersion activation rate, decision.

---

## 7. What NOT to Do in This Experiment

- Do NOT change stock features — feature contract is frozen at 6-feature raw-minimal
- Do NOT reopen posture research (Track 2) — reopen conditions not yet met (n<80)
- Do NOT expand universe — all 4 universe experiments failed
- Do NOT add earnings features — data source insufficient through 2019
- Do NOT run 84D horizon — TASK-8 closed, 56D retained
- Do NOT touch RL reward, PPO config, or policy — RL layer frozen
- Do NOT combine this change with any other change in the same measured run

---

## 8. If This Experiment Fails

If the gate is not met (IC does not improve in 2016-2019, or CAGR drops > 1 pt):

**Interpretation**: The IC collapse in bull markets is not fixable by concentration tuning.
The ranker simply cannot differentiate stocks when the generative process (co-moving market)
does not produce separable signal. No selection-layer fix will help in this regime.

**Next options (do not decide until after failure is confirmed):**
1. Accept the structural IC floor and focus on the RL layer improving returns via sector tilts
2. Regime-conditioned feature selection (different feature weights in trending vs choppy)
3. Wait for fundamental data access (BSE bulk filings, 2013+ quarterly EPS/revenue)

**Do not prematurely close these options before the experiment runs.**

---

## 9. Iteration 2 (conditional on iteration 1 passing gate)

If score-dispersion conditioning improves IC but not enough:

- Switch conditioning variable from score dispersion to **trailing realized IC**
  (last 3–4 rebalances' actual IC as the signal strength proxy)
- Trailing IC directly measures predictive validity, not just score spread
- Requires tracking realized IC at each rebalance — already available in the diagnostics path

Do not implement iteration 2 until iteration 1 result is known.
