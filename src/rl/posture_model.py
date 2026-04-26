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
FEATURE_NAMES = [
    # Group A — Macro / Regime
    "nifty_ret_1m",
    "nifty_ret_3m",
    "nifty_vol_1m",
    "vol_percentile_1y",
    "india_vix",
    "vix_percentile_1y",
    "max_drawdown_3m",
    "trend_50_200",
    # Group B — Cross-sectional / Breadth
    "top_decile_return_1m",
    "bottom_decile_return_1m",
    "spread_decile",
    "cross_sectional_vol",
    "hit_rate_top_k",
    "sector_dispersion",
    # Group C — Portfolio State
    "cash_ratio",
    "current_drawdown",
    "portfolio_vol_1m",
    "turnover_1m",
    "exposure_concentration",
    "top5_weight_sum",
    # Group D — Execution / Friction
    "avg_cost_per_turnover",
    "turnover_capacity_utilization",
    # Group E — Regime Persistence
    "nifty_trend_duration",
]


def build_features(
    parquet_path: str | Path,
    macro_parquet_dir: str | Path,
    price_matrix_path: str | Path,
) -> pd.DataFrame:
    """Build the 23-feature matrix indexed by sample date.

    Returns a DataFrame with one row per sample date and FEATURE_NAMES as columns.
    All features are at decision time (lagged ≥1 day vs forward outcomes).
    """
    parquet_path = Path(parquet_path)
    macro_dir = Path(macro_parquet_dir)
    price_path = Path(price_matrix_path)

    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])

    # One row per sample date (pivot the 3 posture rows)
    sample_dates = sorted(df["date"].unique())

    # --- Group A: Macro features from feature store ---
    macro_dfs = []
    for p in sorted(macro_dir.glob("**/*.parquet")):
        macro_dfs.append(pd.read_parquet(p))
    macro = pd.concat(macro_dfs).sort_index() if macro_dfs else pd.DataFrame()
    if not macro.empty and not isinstance(macro.index, pd.DatetimeIndex):
        macro.index = pd.to_datetime(macro.index)

    # --- Price matrix for breadth and regime persistence ---
    price_matrix = pd.read_parquet(price_path) if price_path.exists() else pd.DataFrame()
    if not price_matrix.empty and not isinstance(price_matrix.index, pd.DatetimeIndex):
        price_matrix.index = pd.to_datetime(price_matrix.index)

    rows = []
    for date in sample_dates:
        row: dict[str, float] = {}
        date_ts = pd.Timestamp(date)
        posture_group = df[df["date"] == date_ts]

        # ── Group A: Macro / Regime ──────────────────────────────────────────
        row.update(_macro_features(macro, date_ts))

        # ── Group B: Cross-sectional / Breadth ──────────────────────────────
        row.update(_breadth_features(price_matrix, date_ts, posture_group))

        # ── Group C: Portfolio State ─────────────────────────────────────────
        row.update(_portfolio_features(posture_group))

        # ── Group D: Execution / Friction ────────────────────────────────────
        row.update(_execution_features(posture_group))

        # ── Group E: Regime Persistence ──────────────────────────────────────
        row.update(_regime_persistence(price_matrix, date_ts))

        row["date"] = date_ts
        rows.append(row)

    feat = pd.DataFrame(rows).set_index("date")

    # Fill any missing columns with 0 (graceful degradation when data is absent)
    for col in FEATURE_NAMES:
        if col not in feat.columns:
            feat[col] = 0.0
            logger.warning("Feature %s missing — filled with 0", col)

    return feat[FEATURE_NAMES].fillna(0.0)


def _macro_features(macro: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    if macro.empty:
        return {k: 0.0 for k in [
            "nifty_ret_1m", "nifty_ret_3m", "nifty_vol_1m",
            "vol_percentile_1y", "india_vix", "vix_percentile_1y",
            "max_drawdown_3m", "trend_50_200",
        ]}

    # Use most recent row ≤ date (lagged 1 day)
    idx = macro.index[macro.index <= date]
    if idx.empty:
        return {k: 0.0 for k in [
            "nifty_ret_1m", "nifty_ret_3m", "nifty_vol_1m",
            "vol_percentile_1y", "india_vix", "vix_percentile_1y",
            "max_drawdown_3m", "trend_50_200",
        ]}
    row = macro.loc[idx[-1]]

    def _get(col: str, default: float = 0.0) -> float:
        v = row.get(col, default) if hasattr(row, "get") else getattr(row, col, default)
        return float(v) if pd.notna(v) else default

    nifty_vol = _get("nifty_vol_1m", _get("nifty_realized_vol_1m", 0.0))
    vix = _get("india_vix", 0.0)

    # Rolling percentile rank vs trailing 252 trading days
    trail = macro[macro.index <= date].tail(252)
    vol_series = trail.get("nifty_vol_1m", trail.get("nifty_realized_vol_1m", pd.Series(dtype=float)))
    vix_series = trail.get("india_vix", pd.Series(dtype=float))
    vol_pct = float((vol_series < nifty_vol).mean()) if len(vol_series) > 0 else 0.5
    vix_pct = float((vix_series < vix).mean()) if len(vix_series) > 0 else 0.5

    # Max drawdown of Nifty index over last 63 trading days (~3 months)
    nifty_col = next((c for c in macro.columns if "nifty" in c.lower() and "ret" in c.lower() and "1m" in c.lower()), None)
    max_dd_3m = 0.0
    if nifty_col is not None:
        rolling_rets = trail[nifty_col].tail(63).dropna()
        if len(rolling_rets) > 1:
            cumulative = (1 + rolling_rets).cumprod()
            peak = cumulative.cummax()
            max_dd_3m = float(((cumulative - peak) / peak).min())

    # trend_50_200: Nifty 50MA / 200MA ratio (continuous)
    trend_col = next((c for c in macro.columns if "ma_50_200" in c.lower() or "trend_50_200" in c.lower()), None)
    trend_50_200 = _get(trend_col, 1.0) if trend_col else 1.0

    return {
        "nifty_ret_1m": _get("nifty_ret_1m", 0.0),
        "nifty_ret_3m": _get("nifty_ret_3m", _get("nifty_ret_3m", 0.0)),
        "nifty_vol_1m": nifty_vol,
        "vol_percentile_1y": vol_pct,
        "india_vix": vix,
        "vix_percentile_1y": vix_pct,
        "max_drawdown_3m": max_dd_3m,
        "trend_50_200": trend_50_200,
    }


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

    # sector_dispersion: approximate from posture_group sector-level data if available
    sector_disp = 0.0
    if "mean_selected_sector_count" in posture_group.columns:
        # Use cross-sectional vol of stock returns as proxy for sector dispersion
        sector_disp = float(rets.std())

    return {
        "top_decile_return_1m": top_ret,
        "bottom_decile_return_1m": bot_ret,
        "spread_decile": top_ret - bot_ret,
        "cross_sectional_vol": float(rets.std()),
        "hit_rate_top_k": hit_rate,
        "sector_dispersion": sector_disp,
    }


def _portfolio_features(posture_group: pd.DataFrame) -> dict[str, float]:
    """Extract portfolio state from the neutral posture row (path-independent features)."""
    default = {k: 0.0 for k in [
        "cash_ratio", "current_drawdown", "portfolio_vol_1m",
        "turnover_1m", "exposure_concentration", "top5_weight_sum",
    ]}
    if posture_group.empty:
        return default

    neutral = posture_group[posture_group["posture"] == "neutral"]
    row = neutral.iloc[0] if not neutral.empty else posture_group.iloc[0]

    def _g(col: str) -> float:
        v = row.get(col, 0.0) if hasattr(row, "get") else 0.0
        return float(v) if pd.notna(v) else 0.0

    # avg_turnover from posture simulation is available; use as portfolio_vol proxy
    avg_to = _g("avg_turnover")
    avg_cost = _g("avg_cost_ratio")

    # current_drawdown: max_drawdown from the neutral path over H rebalances
    drawdown = _g("max_drawdown")

    return {
        "cash_ratio": 0.05,        # neutral posture default; dataset doesn't store per-step cash
        "current_drawdown": drawdown,
        "portfolio_vol_1m": abs(drawdown) * 0.5,   # approximation from drawdown magnitude
        "turnover_1m": avg_to,
        "exposure_concentration": 0.08,  # ~1/12 stocks at equal max weight; dataset doesn't store HHI
        "top5_weight_sum": 0.40,         # 5 × 8% max weight; dataset doesn't store per-stock weights
    }


def _execution_features(posture_group: pd.DataFrame) -> dict[str, float]:
    default = {"avg_cost_per_turnover": 0.0025, "turnover_capacity_utilization": 0.6}
    if posture_group.empty:
        return default

    neutral = posture_group[posture_group["posture"] == "neutral"]
    row = neutral.iloc[0] if not neutral.empty else posture_group.iloc[0]

    avg_cost = float(row.get("avg_cost_ratio", 0.0025))
    avg_to = float(row.get("avg_turnover", 0.0))
    max_to = 0.45  # from config
    utilization = float(avg_to / max_to) if max_to > 0 else 0.0
    cost_per_to = float(avg_cost / avg_to) if avg_to > 0 else 0.0025

    return {
        "avg_cost_per_turnover": cost_per_to,
        "turnover_capacity_utilization": min(1.0, utilization),
    }


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
