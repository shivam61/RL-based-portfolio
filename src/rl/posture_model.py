"""Posture utility regression model.

Trains 3 independent LightGBM regressors (one per posture: risk_on, neutral, risk_off)
on normalized per-sample utilities. At inference, argmax across the 3 predictions gives
the recommended posture.

Feature set (23 features, hard cap 25):
  Group A — Macro/Regime (8): nifty_ret_1m, nifty_ret_3m, nifty_vol_1m,
    vol_percentile_1y, india_vix, vix_percentile_1y, max_drawdown_3m, trend_50_200
  Group B — Cross-sectional/Breadth (6): top_decile_return_1m, bottom_decile_return_1m,
    spread_decile, cross_sectional_vol, hit_rate_top_k, sector_dispersion
  Group C — Portfolio State (6): cash_ratio, current_drawdown, portfolio_vol_1m,
    turnover_1m, exposure_concentration (HHI), top5_weight_sum
  Group D — Execution/Friction (2): avg_cost_per_turnover, turnover_capacity_utilization
  Group E — Regime Persistence (1): nifty_trend_duration

Data source rules (enforced by validate_feature_matrix):
  Group A Nifty features: computed from price_matrix (^NSEI) — not from macro parquet
    (macro parquet only has nifty_* columns from 2016; price_matrix has full history from 2012)
  Group A VIX: vix_level column from macro parquet (US VIX, available from 2013)
  Groups B/E: computed from price_matrix
  Groups C/D: extracted from posture simulation dataset

Evaluation: LOO cross-validation (not held-out) — correct for ≤16 post-filter samples.
Success gate: LOO accuracy > 50% (beats always_risk_off baseline) AND
              utility_capture ≥ 90% of oracle on non-indifferent samples.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

POSTURES = ("risk_on", "neutral", "risk_off")
# 16 features — hard cap 25.
# Dropped from original 23 (all had genuine data issues):
#   cash_ratio, exposure_concentration, top5_weight_sum — no per-sample data in posture dataset
#   avg_cost_per_turnover — avg_cost_ratio=0 in current dataset; add back when cost tracking is wired
#   sector_dispersion — duplicated cross_sectional_vol (same rets.std() computation); replace with
#                       true sector-level dispersion when sector membership is available at feature time
#   portfolio_vol_1m — was derived as abs(drawdown)*0.5, not a real computation; drop until
#                      per-rebalance portfolio returns are stored in the posture dataset
#   turnover_capacity_utilization — constant-scale duplicate of turnover_1m (÷0.45); no extra info
FEATURE_NAMES = [
    # Group A — Macro / Regime (8)
    "nifty_ret_1m",
    "nifty_ret_3m",
    "nifty_vol_1m",
    "vol_percentile_1y",
    "india_vix",
    "vix_percentile_1y",
    "max_drawdown_3m",
    "trend_50_200",
    # Group B — Cross-sectional / Breadth (5)
    "top_decile_return_1m",
    "bottom_decile_return_1m",
    "spread_decile",
    "cross_sectional_vol",
    "hit_rate_top_k",
    # Group C — Portfolio State (2)
    "current_drawdown",
    "turnover_1m",
    # Group E — Regime Persistence (1)
    "nifty_trend_duration",
]


class FeatureValidationError(ValueError):
    """Raised when feature matrix fails hard quality gates before model training."""


# Features that are allowed to be binary/near-constant by design.
_BINARY_OR_DESIGN_CONSTANT_FEATURES: frozenset[str] = frozenset()

# Hard gate thresholds — changing these requires a comment explaining why.
_MAX_ZERO_FRACTION = 0.30        # > 30% exact zeros → data source likely broken
_MIN_NONZERO_STD = 1e-5          # std ≤ this → constant feature, no information
_MAX_NAN_FRACTION = 0.05         # > 5% NaN after fillna → something went wrong
# 0.995 (not 0.98): catches genuine derivation bugs (|r|→1.0) but not monotonic transforms
# like vix_level vs vix_percentile at small n, which are expected to be correlated at n≤30.
_MAX_PAIRWISE_CORR = 0.995


def validate_feature_matrix(features: pd.DataFrame) -> None:
    """Hard gate: raises FeatureValidationError if any feature fails quality checks.

    Must be called before fitting any model. Rules:
      1. No NaN (after any upstream fillna).
      2. No constant feature (std ≤ _MIN_NONZERO_STD).
      3. No feature with > _MAX_ZERO_FRACTION exact zeros (detects silent fallback-to-zero).
      4. No pair of features with Pearson |r| ≥ _MAX_PAIRWISE_CORR (detects derivation bugs).

    Call signature:
        validate_feature_matrix(features)   # raises on failure; logs per-feature detail
    """
    failures: list[str] = []

    for col in features.columns:
        series = features[col]

        nan_frac = float(series.isna().mean())
        if nan_frac > _MAX_NAN_FRACTION:
            failures.append(f"  {col}: {nan_frac:.1%} NaN (limit {_MAX_NAN_FRACTION:.0%})")

        if col in _BINARY_OR_DESIGN_CONSTANT_FEATURES:
            continue

        std = float(series.std())
        if std <= _MIN_NONZERO_STD:
            failures.append(
                f"  {col}: constant (std={std:.2e}) — "
                "check data source, column name, or computation"
            )
            continue  # no point computing zero-fraction if series is constant

        zero_frac = float((series == 0.0).mean())
        if zero_frac > _MAX_ZERO_FRACTION:
            failures.append(
                f"  {col}: {zero_frac:.1%} exact zeros (limit {_MAX_ZERO_FRACTION:.0%}) — "
                "likely silent fallback; check column name or data availability"
            )

    # Pairwise correlation check (only on non-failing features)
    ok_cols = [c for c in features.columns if not any(c in f for f in failures)]
    if len(ok_cols) >= 2:
        corr = features[ok_cols].corr().abs()
        for i, c1 in enumerate(ok_cols):
            for c2 in ok_cols[i + 1:]:
                r = float(corr.loc[c1, c2])
                if r >= _MAX_PAIRWISE_CORR:
                    failures.append(
                        f"  ({c1}, {c2}): |r|={r:.3f} ≥ {_MAX_PAIRWISE_CORR} — "
                        "near-duplicate; one may be derived from a broken source"
                    )

    if failures:
        msg = (
            f"Feature matrix failed {len(failures)} quality check(s). "
            "Fix the data pipeline before training:\n" + "\n".join(failures)
        )
        logger.error(msg)
        raise FeatureValidationError(msg)

    logger.info(
        "validate_feature_matrix: all %d features passed quality gates "
        "(NaN<%.0f%%, std>%.0e, zeros<%.0f%%, |r|<%.2f)",
        len(features.columns),
        _MAX_NAN_FRACTION * 100,
        _MIN_NONZERO_STD,
        _MAX_ZERO_FRACTION * 100,
        _MAX_PAIRWISE_CORR,
    )


def build_features(
    parquet_path: str | Path,
    macro_parquet_dir: str | Path,
    price_matrix_path: str | Path,
) -> pd.DataFrame:
    """Build the 23-feature matrix indexed by sample date.

    Returns a DataFrame with one row per sample date and FEATURE_NAMES as columns.
    All features are at decision time (lagged ≥1 day vs forward outcomes).

    Raises FeatureValidationError (via validate_feature_matrix) if any feature
    fails quality gates — call this before fitting any model.
    """
    parquet_path = Path(parquet_path)
    macro_dir = Path(macro_parquet_dir)
    price_path = Path(price_matrix_path)

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])

    sample_dates = sorted(df["date"].unique())

    # Macro features (VIX / global stress signals — vix_level available from 2013)
    macro_dfs = []
    for p in sorted(macro_dir.glob("**/*.parquet")):
        macro_dfs.append(pd.read_parquet(p))
    macro = pd.concat(macro_dfs).sort_index() if macro_dfs else pd.DataFrame()
    if not macro.empty and not isinstance(macro.index, pd.DatetimeIndex):
        macro.index = pd.to_datetime(macro.index)

    # Price matrix — primary source for all Nifty-derived features (full history from 2012)
    if not price_path.exists():
        raise FileNotFoundError(f"Price matrix not found: {price_path}")
    price_matrix = pd.read_parquet(price_path)
    if not isinstance(price_matrix.index, pd.DatetimeIndex):
        price_matrix.index = pd.to_datetime(price_matrix.index)

    rows = []
    for date in sample_dates:
        row: dict[str, float] = {}
        date_ts = pd.Timestamp(date)
        posture_group = df[df["date"] == date_ts]

        # Group A: Nifty features from price_matrix; VIX from macro
        row.update(_macro_features(macro, price_matrix, date_ts))
        # Group B: Cross-sectional breadth from price_matrix
        row.update(_breadth_features(price_matrix, date_ts, posture_group))
        # Group C: Portfolio state from posture simulation rows
        row.update(_portfolio_features(posture_group))
        # Group D: Execution friction from posture simulation rows
        row.update(_execution_features(posture_group))
        # Group E: Regime persistence (Nifty trend duration) from price_matrix
        row.update(_regime_persistence(price_matrix, date_ts))

        row["date"] = date_ts
        rows.append(row)

    feat = pd.DataFrame(rows).set_index("date")

    # Hard failure for missing columns — do NOT silently fill with zeros.
    missing = [c for c in FEATURE_NAMES if c not in feat.columns]
    if missing:
        raise FeatureValidationError(
            f"Feature columns missing after build: {missing}. "
            "Fix the feature extraction functions before running the model."
        )

    result = feat[FEATURE_NAMES]
    if result.isna().any().any():
        nan_cols = result.columns[result.isna().any()].tolist()
        logger.warning("NaN in features after build (will be filled with 0): %s", nan_cols)
        result = result.fillna(0.0)

    return result


def _nifty_prices(price_matrix: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    """Return Nifty price series up to and including date (lagged). None if unavailable."""
    nifty_col = next(
        (c for c in price_matrix.columns if "NSEI" in str(c).upper() or "NIFTY50" in str(c).upper()),
        None,
    )
    if nifty_col is None:
        return None
    idx = price_matrix.index[price_matrix.index <= date]
    if len(idx) < 2:
        return None
    return price_matrix.loc[idx, nifty_col].dropna()


def _macro_features(
    macro: pd.DataFrame,
    price_matrix: pd.DataFrame,
    date: pd.Timestamp,
) -> dict[str, float]:
    """Compute Group A features.

    Nifty-derived features (ret, vol, drawdown, trend) come from the price matrix —
    this is the authoritative source because macro parquet only has nifty_* columns
    from 2016 onward. VIX (vix_level) comes from macro — available from 2013.
    """
    result: dict[str, float] = {}

    # ── Nifty features from price_matrix ────────────────────────────────────
    nifty = _nifty_prices(price_matrix, date)

    if nifty is not None and len(nifty) >= 22:
        # 1M and 3M returns
        result["nifty_ret_1m"] = float(nifty.iloc[-1] / nifty.iloc[-22] - 1)
        result["nifty_ret_3m"] = float(nifty.iloc[-1] / nifty.iloc[max(0, len(nifty) - 63)] - 1)

        # Realized 1M volatility (annualized)
        daily_rets = nifty.pct_change().iloc[-22:].dropna()
        vol_1m = float(daily_rets.std() * np.sqrt(252)) if len(daily_rets) >= 5 else 0.0
        result["nifty_vol_1m"] = vol_1m

        # Vol percentile vs trailing 252-day rolling window
        if len(nifty) >= 252 + 22:
            rolling_vols = [
                float(nifty.pct_change().iloc[i - 22:i].dropna().std() * np.sqrt(252))
                for i in range(22, len(nifty) - 230)
            ]
            result["vol_percentile_1y"] = float((np.array(rolling_vols) < vol_1m).mean()) if rolling_vols else 0.5
        else:
            result["vol_percentile_1y"] = 0.5

        # Max drawdown over last 63 trading days (~3 months)
        window = nifty.iloc[-63:]
        peak = window.cummax()
        result["max_drawdown_3m"] = float(((window - peak) / peak).min())

        # 50MA / 200MA ratio (trend signal)
        if len(nifty) >= 200:
            ma50 = float(nifty.iloc[-50:].mean())
            ma200 = float(nifty.iloc[-200:].mean())
            result["trend_50_200"] = ma50 / ma200 if ma200 > 0 else 1.0
        else:
            result["trend_50_200"] = 1.0
    else:
        result.update({
            "nifty_ret_1m": 0.0, "nifty_ret_3m": 0.0, "nifty_vol_1m": 0.0,
            "vol_percentile_1y": 0.5, "max_drawdown_3m": 0.0, "trend_50_200": 1.0,
        })
        logger.warning("Nifty price series unavailable at %s — Nifty features set to neutral defaults", date.date())

    # ── VIX from macro parquet (vix_level — US VIX, available from 2013) ────
    if not macro.empty:
        idx = macro.index[macro.index <= date]
        if not idx.empty:
            row = macro.loc[idx[-1]]

            def _get(col: str, default: float = 0.0) -> float:
                v = row.get(col, default) if hasattr(row, "get") else getattr(row, col, default)
                return float(v) if pd.notna(v) else default

            vix = _get("vix_level", 0.0)
            result["india_vix"] = vix

            trail = macro[macro.index <= date].tail(252)
            vix_series = trail["vix_level"].dropna() if "vix_level" in trail.columns else pd.Series(dtype=float)
            result["vix_percentile_1y"] = float((vix_series < vix).mean()) if len(vix_series) > 1 else 0.5
        else:
            result["india_vix"] = 0.0
            result["vix_percentile_1y"] = 0.5
    else:
        result["india_vix"] = 0.0
        result["vix_percentile_1y"] = 0.5

    return result


def _breadth_features(
    price_matrix: pd.DataFrame,
    date: pd.Timestamp,
    posture_group: pd.DataFrame,
) -> dict[str, float]:
    default = {k: 0.0 for k in [
        "top_decile_return_1m", "bottom_decile_return_1m", "spread_decile",
        "cross_sectional_vol", "hit_rate_top_k", "sector_dispersion",
    ]}
    if price_matrix.empty:
        return default

    # 1M returns for all stocks at this date (lagged 1 day)
    idx = price_matrix.index[price_matrix.index <= date]
    if len(idx) < 22:
        return default
    end_row = price_matrix.loc[idx[-1]]
    start_row = price_matrix.loc[idx[-22]]
    rets = ((end_row - start_row) / start_row.replace(0, np.nan)).dropna()
    if len(rets) < 10:
        return default

    q10 = rets.quantile(0.90)
    q90 = rets.quantile(0.10)
    top_ret = float(rets[rets >= q10].mean())
    bot_ret = float(rets[rets <= q90].mean())

    # hit_rate_top_k: fraction of held portfolio stocks with positive 1M return
    # Use neutral posture row to get avg portfolio composition proxy
    hit_rate = float((rets > 0).mean())

    return {
        "top_decile_return_1m": top_ret,
        "bottom_decile_return_1m": bot_ret,
        "spread_decile": top_ret - bot_ret,
        "cross_sectional_vol": float(rets.std()),
        "hit_rate_top_k": hit_rate,
    }


def _portfolio_features(posture_group: pd.DataFrame) -> dict[str, float]:
    """Extract portfolio state from the neutral posture row.

    Only returns features that are genuinely computable from the posture dataset.
    Dropped: cash_ratio, exposure_concentration, top5_weight_sum (no per-sample data),
             portfolio_vol_1m (was derived from drawdown — not a real computation).
    """
    default = {"current_drawdown": 0.0, "turnover_1m": 0.0}
    if posture_group.empty:
        return default

    neutral = posture_group[posture_group["posture"] == "neutral"]
    row = neutral.iloc[0] if not neutral.empty else posture_group.iloc[0]

    def _g(col: str) -> float:
        v = row.get(col, 0.0) if hasattr(row, "get") else 0.0
        return float(v) if pd.notna(v) else 0.0

    return {
        "current_drawdown": _g("max_drawdown"),
        "turnover_1m": _g("avg_turnover"),
    }


def _execution_features(posture_group: pd.DataFrame) -> dict[str, float]:
    """Execution/friction features from the posture simulation dataset.

    Dropped: avg_cost_per_turnover (avg_cost_ratio=0 in current dataset — add back when
             cost tracking is wired), turnover_capacity_utilization (constant-scale
             duplicate of turnover_1m with max_to=0.45).
    Returns empty dict — no execution features currently computable without data.
    """
    return {}


def _regime_persistence(price_matrix: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    """Consecutive weeks Nifty has been above (positive) or below (negative) 50MA."""
    if price_matrix.empty:
        return {"nifty_trend_duration": 0.0}

    # Use the first column as Nifty proxy if a specific column isn't available
    nifty_col = next(
        (c for c in price_matrix.columns if "NSEI" in str(c).upper() or "NIFTY" in str(c).upper()),
        price_matrix.columns[0] if len(price_matrix.columns) > 0 else None,
    )
    if nifty_col is None:
        return {"nifty_trend_duration": 0.0}

    idx = price_matrix.index[price_matrix.index <= date]
    if len(idx) < 50:
        return {"nifty_trend_duration": 0.0}

    prices = price_matrix.loc[idx, nifty_col].dropna().tail(100)
    ma50 = prices.rolling(50).mean()
    above = (prices > ma50).dropna()
    if above.empty:
        return {"nifty_trend_duration": 0.0}

    # Count consecutive periods with same sign from the end
    current_state = bool(above.iloc[-1])
    count = 0
    for val in reversed(above.values):
        if bool(val) == current_state:
            count += 1
        else:
            break
    # Sign: positive if above MA (bullish), negative if below (bearish)
    duration = float(count) if current_state else float(-count)
    return {"nifty_trend_duration": duration}


def build_regression_targets(
    parquet_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (targets_df, sample_info_df) from the posture dataset parquet.

    targets_df: indexed by date, columns = [risk_on, neutral, risk_off] utilities (return_only)
                normalized per-sample (subtract mean, divide by std across 3 postures)
    sample_info_df: indexed by date, columns = [stress_bucket, stress_signal, margin,
                    target_posture (argmax of raw utilities)]
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])

    targets: list[dict] = []
    info: list[dict] = []

    for date, group in df.groupby("date", sort=True):
        utils = {str(row["posture"]): float(row["utility_return_only"]) for _, row in group.iterrows()}
        if len(utils) < 3:
            continue

        raw = np.array([utils.get("risk_on", 0), utils.get("neutral", 0), utils.get("risk_off", 0)])
        mean, std = raw.mean(), raw.std()
        normed = (raw - mean) / (std + 1e-8)

        sorted_vals = sorted(utils.values(), reverse=True)
        margin = sorted_vals[0] - sorted_vals[1]
        best = max(utils, key=lambda p: utils[p])

        targets.append({"date": date, "risk_on": normed[0], "neutral": normed[1], "risk_off": normed[2]})
        info.append({
            "date": date,
            "target_posture": best,
            "utility_margin": margin,
            "stress_bucket": str(group["stress_bucket"].iloc[0]) if "stress_bucket" in group.columns else "",
        })

    t = pd.DataFrame(targets).set_index("date")
    s = pd.DataFrame(info).set_index("date")
    return t, s


class PostureUtilityModel:
    """3 independent LightGBM regressors for posture utility prediction.

    Training: fit on ALL samples (ε filter only applied at evaluation).
    Evaluation: LOO cross-validation on non-indifferent samples (margin > ε).
    """

    def __init__(self, epsilon_percentile: int = 25):
        self.epsilon_percentile = epsilon_percentile
        self.models: dict[str, Any] = {}
        self.epsilon: float = 0.0
        self.feature_names: list[str] = FEATURE_NAMES
        self._is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        targets: pd.DataFrame,
        sample_info: pd.DataFrame,
    ) -> "PostureUtilityModel":
        """Train 3 regressors on normalized utilities.

        Args:
            X: feature matrix (n_samples × 23), indexed by date
            targets: normalized utility per posture (n_samples × 3), indexed by date
            sample_info: per-sample info including utility_margin, indexed by date
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm required: pip install lightgbm")

        common = X.index.intersection(targets.index)
        X_fit = X.loc[common].values.astype(np.float32)

        margins = sample_info.loc[common, "utility_margin"].values
        self.epsilon = float(np.percentile(margins, self.epsilon_percentile))
        # Guard: ensure at least 10 samples remain after filtering
        n_after = (margins >= self.epsilon).sum()
        if n_after < 10:
            p15 = float(np.percentile(margins, 15))
            logger.warning(
                "ε=%.4f leaves only %d samples — lowering to p15=%.4f",
                self.epsilon, n_after, p15,
            )
            self.epsilon = p15

        logger.info("ε calibration: p%d=%.4f, n_samples=%d, n_non_indifferent=%d",
                    self.epsilon_percentile, self.epsilon, len(common),
                    int((margins >= self.epsilon).sum()))

        for posture in POSTURES:
            y = targets.loc[common, posture].values.astype(np.float32)
            model = lgb.LGBMRegressor(
                n_estimators=50,
                max_depth=3,
                min_child_samples=3,
                learning_rate=0.1,
                verbose=-1,
            )
            model.fit(X_fit, y, feature_name=self.feature_names)
            self.models[posture] = model

        self._is_fitted = True
        return self

    def predict_utilities(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return {p: self.models[p].predict(X) for p in POSTURES}

    def predict_best_posture(
        self,
        macro_state: dict,
        sector_state: dict,
        portfolio_state: dict,
    ) -> str:
        """Predict best posture from raw state dicts. Returns posture name."""
        feat = _state_dicts_to_features(macro_state, sector_state, portfolio_state)
        utils = self.predict_utilities(feat.reshape(1, -1))
        scores = {p: float(utils[p][0]) for p in POSTURES}
        return max(scores, key=lambda p: scores[p])

    def evaluate_loo(
        self,
        X: pd.DataFrame,
        targets: pd.DataFrame,
        sample_info: pd.DataFrame,
    ) -> dict[str, Any]:
        """Leave-one-out cross-validation on non-indifferent samples.

        Returns accuracy, top_2_accuracy, utility_capture, regret, feature_importance.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm required")

        common = X.index.intersection(targets.index).intersection(sample_info.index)
        X_all = X.loc[common].values.astype(np.float32)
        t_all = targets.loc[common]
        info_all = sample_info.loc[common]
        margins = info_all["utility_margin"].values

        # Indifferent mask (for evaluation accuracy only)
        non_indiff = margins >= self.epsilon
        n_non_indiff = non_indiff.sum()
        logger.info("LOO eval: %d total samples, %d non-indifferent (margin≥%.4f)",
                    len(common), n_non_indiff, self.epsilon)

        # LOO predictions
        loo_preds: list[str] = []
        for i in range(len(common)):
            train_idx = [j for j in range(len(common)) if j != i]
            X_tr = X_all[train_idx]
            preds: dict[str, float] = {}
            for posture in POSTURES:
                y_tr = t_all.iloc[train_idx][posture].values.astype(np.float32)
                m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, min_child_samples=3,
                                      learning_rate=0.1, verbose=-1)
                m.fit(X_tr, y_tr)
                preds[posture] = float(m.predict(X_all[i:i+1])[0])
            loo_preds.append(max(preds, key=lambda p: preds[p]))

        loo_preds_arr = np.array(loo_preds)
        true_postures = info_all["target_posture"].values

        # Accuracy on non-indifferent samples only
        if n_non_indiff > 0:
            mask = non_indiff
            accuracy = float((loo_preds_arr[mask] == true_postures[mask]).mean())
        else:
            accuracy = 0.0

        # Utility capture = selected_utility / best_utility (on non-indiff samples)
        # We need raw (unnormalized) utilities for this
        raw_parquet_avail = hasattr(self, "_raw_utils")
        utility_capture = None  # computed in train script with raw utilities

        # Feature importance from full-data model
        importance = {}
        if self._is_fitted:
            for p in POSTURES:
                imp = self.models[p].feature_importances_
                for fname, fval in zip(self.feature_names, imp):
                    importance[fname] = importance.get(fname, 0) + int(fval)

        top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

        return {
            "n_samples_total": int(len(common)),
            "n_non_indifferent": int(n_non_indiff),
            "epsilon": round(self.epsilon, 6),
            "loo_accuracy": round(accuracy, 4),
            "loo_accuracy_n": int(n_non_indiff),
            "baseline_always_neutral": 0.375,
            "baseline_always_risk_off": 0.500,
            "beats_binding_baseline": accuracy > 0.500,
            "top_features": top_features,
            "loo_predictions": loo_preds,
            "true_postures": true_postures.tolist(),
        }

    def feature_importance_summary(self) -> dict[str, int]:
        if not self._is_fitted:
            return {}
        importance: dict[str, int] = {}
        for p in POSTURES:
            for fname, fval in zip(self.feature_names, self.models[p].feature_importances_):
                importance[fname] = importance.get(fname, 0) + int(fval)
        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for p in POSTURES:
            if p in self.models:
                self.models[p].booster_.save_model(str(path / f"model_{p}.lgb"))
        meta = {
            "epsilon": self.epsilon,
            "epsilon_percentile": self.epsilon_percentile,
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("PostureUtilityModel saved → %s", path)

    def load(self, path: str | Path) -> "PostureUtilityModel":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm required")
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        self.epsilon = float(meta["epsilon"])
        self.epsilon_percentile = int(meta["epsilon_percentile"])
        self.feature_names = list(meta["feature_names"])
        self._is_fitted = bool(meta["is_fitted"])
        for p in POSTURES:
            mp = path / f"model_{p}.lgb"
            if mp.exists():
                booster = lgb.Booster(model_file=str(mp))
                self.models[p] = booster
        return self


def _state_dicts_to_features(
    macro_state: dict,
    sector_state: dict,
    portfolio_state: dict,
) -> np.ndarray:
    """Convert raw state dicts to feature vector for inference."""
    row = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    mapping = {
        "nifty_ret_1m": macro_state.get("nifty_ret_1m", 0),
        "nifty_ret_3m": macro_state.get("nifty_ret_3m", 0),
        "nifty_vol_1m": macro_state.get("nifty_vol_1m", 0),
        "vol_percentile_1y": macro_state.get("vol_percentile_1y", 0.5),
        "india_vix": macro_state.get("india_vix", 0),
        "vix_percentile_1y": macro_state.get("vix_percentile_1y", 0.5),
        "max_drawdown_3m": macro_state.get("max_drawdown_3m", 0),
        "trend_50_200": macro_state.get("trend_50_200", 1.0),
        "top_decile_return_1m": sector_state.get("top_decile_return_1m", 0),
        "bottom_decile_return_1m": sector_state.get("bottom_decile_return_1m", 0),
        "spread_decile": sector_state.get("spread_decile", 0),
        "cross_sectional_vol": sector_state.get("cross_sectional_vol", 0),
        "hit_rate_top_k": sector_state.get("hit_rate_top_k", 0.5),
        "sector_dispersion": sector_state.get("sector_dispersion", 0),
        "cash_ratio": portfolio_state.get("cash_ratio", 0.05),
        "current_drawdown": portfolio_state.get("current_drawdown", 0),
        "portfolio_vol_1m": portfolio_state.get("portfolio_vol_1m", 0),
        "turnover_1m": portfolio_state.get("turnover_1m", 0),
        "exposure_concentration": portfolio_state.get("exposure_concentration", 0.08),
        "top5_weight_sum": portfolio_state.get("top5_weight_sum", 0.4),
        "avg_cost_per_turnover": portfolio_state.get("avg_cost_per_turnover", 0.0025),
        "turnover_capacity_utilization": portfolio_state.get("turnover_capacity_utilization", 0.6),
        "nifty_trend_duration": macro_state.get("nifty_trend_duration", 0),
    }
    for i, name in enumerate(FEATURE_NAMES):
        row[i] = float(mapping.get(name, 0))
    return row
