"""
Stock ranking model.

Uses LightGBM LambdaRank to rank stocks within each sector by
expected forward return. Supports optional multi-horizon blending.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    lgb = None  # type: ignore
    HAS_LGB = False


_EXCLUDE_COLS = {"date", "ticker", "sector"}


@dataclass
class _SectorHorizonBlend:
    sector: str
    feature_names: list[str]
    horizon_models: dict[int, object]
    horizon_scalers: dict[int, StandardScaler]
    blend_weights: dict[int, float]

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_raw = X.reindex(columns=self.feature_names, fill_value=0).fillna(0)
        return X_raw.select_dtypes(include=[np.number]).reindex(
            columns=self.feature_names, fill_value=0
        ).fillna(0)

    def _predict_single(self, horizon: int, X: pd.DataFrame) -> np.ndarray:
        model = self.horizon_models.get(horizon)
        scaler = self.horizon_scalers.get(horizon)
        if model is None or scaler is None:
            return np.zeros(len(X), dtype=float)
        X_raw = self._prepare_features(X)
        X_scaled = scaler.transform(X_raw)
        X_named = pd.DataFrame(X_scaled, columns=self.feature_names)
        return np.asarray(model.predict(X_named), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if X.empty:
            return np.array([], dtype=float)
        preds = np.zeros(len(X), dtype=float)
        total_weight = 0.0
        for horizon, weight in self.blend_weights.items():
            if weight <= 0:
                continue
            preds += weight * self._predict_single(horizon, X)
            total_weight += weight
        if total_weight <= 0:
            return np.zeros(len(X), dtype=float)
        return preds / total_weight

    @property
    def feature_importances_(self) -> np.ndarray:
        weighted = np.zeros(len(self.feature_names), dtype=float)
        total_weight = 0.0
        for horizon, weight in self.blend_weights.items():
            model = self.horizon_models.get(horizon)
            if model is None or weight <= 0:
                continue
            importances = getattr(model, "feature_importances_", None)
            if importances is None:
                continue
            weighted += weight * np.asarray(importances, dtype=float)
            total_weight += weight
        if total_weight <= 0:
            return weighted
        return weighted / total_weight


class StockRanker:
    """
    Train a cross-sectional ranking model within sectors.
    Predicts a score where higher → better expected 4-week return.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model_cfg = cfg["stock_model"]
        self.models: dict[str, object] = {}   # one model/blend wrapper per sector
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_names: list[str] = []
        self.horizons: list[int] = []
        self.blend_weights: dict[int, float] = {}
        self.blend_target_horizon: int | None = None
        self.is_fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        stock_features: pd.DataFrame,   # long: date, ticker, sector, features...
        price_matrix: pd.DataFrame,      # date × ticker adj close
        fwd_window: int | Sequence[int] = 28,
    ) -> "StockRanker":
        """Train one ranking model per sector using available history.

        If ``fwd_window`` is a sequence, train a horizon ensemble and calibrate
        blend weights on a holdout slice of the training data.
        """
        if stock_features.empty:
            return self

        feat_cols = [c for c in stock_features.columns if c not in _EXCLUDE_COLS]
        self.feature_names = feat_cols
        self.models = {}
        self.scalers = {}
        self.horizons = []
        self.blend_weights = {}
        self.blend_target_horizon = None

        horizons = self._normalize_horizons(fwd_window)
        if len(horizons) > 1:
            self._fit_blended(stock_features, price_matrix, feat_cols, horizons)
        else:
            self._fit_single(stock_features, price_matrix, feat_cols, horizons[0])

        self.is_fitted = bool(self.models)
        logger.info("StockRanker fitted for sectors: %s", list(self.models.keys()))
        return self

    def _normalize_horizons(self, fwd_window: int | Sequence[int]) -> list[int]:
        if isinstance(fwd_window, Sequence) and not isinstance(fwd_window, (str, bytes)):
            horizons = sorted({int(v) for v in fwd_window if int(v) > 0})
            return horizons or [28]
        return [int(fwd_window)]

    def _fit_single(
        self,
        stock_features: pd.DataFrame,
        price_matrix: pd.DataFrame,
        feat_cols: list[str],
        horizon: int,
    ) -> None:
        self.horizons = [horizon]
        self.blend_weights = {horizon: 1.0}
        self.blend_target_horizon = horizon
        fwd_rets = price_matrix.pct_change(horizon).shift(-horizon)

        sectors = stock_features["sector"].dropna().unique()
        for sector in sectors:
            sec_df = stock_features[stock_features["sector"] == sector].copy()
            if len(sec_df) < 100:
                continue
            model_pack = self._train_sector_model(sec_df, fwd_rets, feat_cols)
            if model_pack is None:
                continue
            model, scaler = model_pack
            self.models[sector] = _SectorHorizonBlend(
                sector=sector,
                feature_names=feat_cols,
                horizon_models={horizon: model},
                horizon_scalers={horizon: scaler},
                blend_weights={horizon: 1.0},
            )
            self.scalers[sector] = scaler

    def _fit_blended(
        self,
        stock_features: pd.DataFrame,
        price_matrix: pd.DataFrame,
        feat_cols: list[str],
        horizons: list[int],
    ) -> None:
        self.horizons = list(horizons)
        self.blend_target_horizon = min(horizons)
        calib_frac = float(self._model_cfg.get("blend_validation_fraction", 0.2))
        calib_frac = float(np.clip(calib_frac, 0.1, 0.4))

        horizon_rets = {
            horizon: price_matrix.pct_change(horizon).shift(-horizon)
            for horizon in horizons
        }
        target_rets = horizon_rets[self.blend_target_horizon]

        sectors = stock_features["sector"].dropna().unique()
        blended_weights = {h: 1.0 / len(horizons) for h in horizons}
        all_eval_frames: list[pd.DataFrame] = []
        per_sector_models: dict[str, dict[int, object]] = {}
        per_sector_scalers: dict[str, dict[int, StandardScaler]] = {}

        for sector in sectors:
            sec_df = stock_features[stock_features["sector"] == sector].copy()
            if len(sec_df) < 100:
                continue

            sec_df = sec_df.sort_values("date")
            unique_dates = pd.Index(sorted(pd.to_datetime(sec_df["date"].dropna().unique())))
            if len(unique_dates) < 8:
                continue
            split_idx = int(len(unique_dates) * (1.0 - calib_frac))
            split_idx = max(1, min(split_idx, len(unique_dates) - 1))
            train_dates = unique_dates[:split_idx]
            calib_dates = unique_dates[split_idx:]

            sector_models: dict[int, object] = {}
            sector_scalers: dict[int, StandardScaler] = {}
            train_sec = sec_df[sec_df["date"].isin(train_dates)].copy()
            for horizon in horizons:
                pack = self._train_sector_model(train_sec, horizon_rets[horizon], feat_cols)
                if pack is None:
                    continue
                model, scaler = pack
                sector_models[horizon] = model
                sector_scalers[horizon] = scaler

            if not sector_models:
                continue
            per_sector_models[sector] = sector_models
            per_sector_scalers[sector] = sector_scalers

            calib_sec = sec_df[sec_df["date"].isin(calib_dates)].copy()
            eval_frame = self._build_eval_frame(calib_sec, target_rets, feat_cols)
            if eval_frame.empty:
                continue

            for horizon in horizons:
                model = sector_models.get(horizon)
                scaler = sector_scalers.get(horizon)
                if model is None or scaler is None:
                    eval_frame[f"pred_{horizon}"] = 0.0
                    continue
                X_raw = eval_frame.reindex(columns=feat_cols, fill_value=0).fillna(0)
                X_scaled = scaler.transform(X_raw)
                X_named = pd.DataFrame(X_scaled, columns=feat_cols)
                eval_frame[f"pred_{horizon}"] = np.asarray(model.predict(X_named), dtype=float)
            all_eval_frames.append(eval_frame)

        if all_eval_frames:
            calib_eval = pd.concat(all_eval_frames, ignore_index=True)
            blended_weights = self._optimize_blend_weights(calib_eval, horizons)

        self.blend_weights = blended_weights

        for sector, sector_models in per_sector_models.items():
            sector_scalers = per_sector_scalers.get(sector, {})
            self.models[sector] = _SectorHorizonBlend(
                sector=sector,
                feature_names=feat_cols,
                horizon_models=sector_models,
                horizon_scalers=sector_scalers,
                blend_weights=blended_weights,
            )
            # Preserve one scaler for compatibility/debugging
            first_horizon = next(iter(sector_scalers.keys()), None)
            if first_horizon is not None:
                self.scalers[sector] = sector_scalers[first_horizon]

        logger.info(
            "StockRanker blended horizons=%s weights=%s target=%s",
            self.horizons,
            {h: round(w, 3) for h, w in self.blend_weights.items()},
            self.blend_target_horizon,
        )

    def _train_sector_model(
        self,
        sec_df: pd.DataFrame,
        fwd_rets: pd.DataFrame,
        feat_cols: list[str],
    ) -> tuple[object, StandardScaler] | None:
        X, y, groups = self._build_rank_dataset(sec_df, fwd_rets, feat_cols)
        if X is None or len(X) < 50:
            return None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(0))

        if HAS_LGB:
            params = {
                "objective": "lambdarank",
                "n_estimators": self._model_cfg["n_estimators"],
                "learning_rate": self._model_cfg["learning_rate"],
                "max_depth": self._model_cfg["max_depth"],
                "num_leaves": self._model_cfg["num_leaves"],
                "subsample": self._model_cfg["subsample"],
                "colsample_bytree": self._model_cfg["colsample_bytree"],
                "min_child_samples": self._model_cfg["min_child_samples"],
                "verbose": -1,
                "n_jobs": 1,
            }
            model = lgb.LGBMRanker(**params)
            model.fit(X_scaled, y.values, group=groups)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=3
            )
            model.fit(X_scaled, y.values)
        return model, scaler

    def _build_eval_frame(
        self,
        sec_df: pd.DataFrame,
        fwd_rets: pd.DataFrame,
        feat_cols: list[str],
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        if sec_df.empty:
            return pd.DataFrame()

        for date_val in sorted(pd.to_datetime(sec_df["date"].dropna().unique())):
            day_df = sec_df[sec_df["date"] == date_val]
            tickers = day_df["ticker"].tolist()
            avail = [t for t in tickers if t in fwd_rets.columns]
            if not avail or date_val not in fwd_rets.index:
                continue

            day_rets = fwd_rets.loc[date_val, avail].dropna()
            if len(day_rets) < 3:
                continue

            for _, row in day_df.iterrows():
                ticker = row["ticker"]
                if ticker not in day_rets.index:
                    continue
                feat_row = row[feat_cols].to_dict()
                feat_row["_ticker"] = ticker
                feat_row["_date"] = date_val
                feat_row["_sector"] = row["sector"]
                feat_row["_return"] = float(day_rets[ticker])
                rows.append(feat_row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def _optimize_blend_weights(
        self,
        eval_frame: pd.DataFrame,
        horizons: list[int],
    ) -> dict[int, float]:
        pred_cols = [f"pred_{h}" for h in horizons if f"pred_{h}" in eval_frame.columns]
        if len(pred_cols) < 2:
            return {h: 1.0 / len(horizons) for h in horizons}

        lambda_ic_std = float(self._model_cfg.get("blend_ic_stability_penalty", 0.5))
        lambda_spread = float(self._model_cfg.get("blend_spread_reward", 0.1))
        step = float(self._model_cfg.get("blend_grid_step", 0.1))
        step = float(np.clip(step, 0.05, 0.25))

        best_weights = {h: 1.0 / len(horizons) for h in horizons}
        best_score = -np.inf

        grid = np.arange(0.0, 1.0 + 1e-9, step)
        for cand in product(grid, repeat=len(horizons)):
            total = float(sum(cand))
            if total <= 0:
                continue
            weights = {h: float(w / total) for h, w in zip(horizons, cand)}
            objective = self._blend_objective(eval_frame, weights, lambda_ic_std, lambda_spread)
            if objective > best_score:
                best_score = objective
                best_weights = weights

        return best_weights

    def _blend_objective(
        self,
        eval_frame: pd.DataFrame,
        weights: dict[int, float],
        lambda_ic_std: float,
        lambda_spread: float,
    ) -> float:
        group_metrics: list[tuple[float, float]] = []
        pred_cols = [f"pred_{h}" for h in weights.keys()]
        if not all(col in eval_frame.columns for col in pred_cols):
            return -np.inf

        for (_, _), group in eval_frame.groupby(["_sector", "_date"], sort=False):
            if len(group) < 3:
                continue
            blended = np.zeros(len(group), dtype=float)
            for horizon, weight in weights.items():
                blended += weight * group[f"pred_{horizon}"].to_numpy(dtype=float)
            rets = group["_return"].to_numpy(dtype=float)
            ic = pd.Series(blended).corr(pd.Series(rets), method="spearman")
            if pd.isna(ic):
                continue
            order = np.argsort(-blended, kind="mergesort")
            k = max(1, min(5, len(group) // 2 if len(group) > 2 else 1))
            top = rets[order[:k]]
            bottom = rets[order[-k:]]
            spread = float(np.mean(top) - np.mean(bottom))
            group_metrics.append((float(ic), spread))

        if not group_metrics:
            return -np.inf

        ics = np.array([x[0] for x in group_metrics], dtype=float)
        spreads = np.array([x[1] for x in group_metrics], dtype=float)
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics, ddof=0))
        mean_spread = float(np.mean(spreads))
        return mean_ic - lambda_ic_std * std_ic + lambda_spread * mean_spread

    def _build_rank_dataset(
        self,
        sec_df: pd.DataFrame,
        fwd_rets: pd.DataFrame,
        feat_cols: list[str],
    ) -> tuple:
        rows = []
        for date_val in sec_df["date"].unique():
            day_df = sec_df[sec_df["date"] == date_val]
            tickers = day_df["ticker"].tolist()
            avail = [t for t in tickers if t in fwd_rets.columns]
            if not avail or date_val not in fwd_rets.index:
                continue

            day_rets = fwd_rets.loc[date_val, avail].dropna()
            if len(day_rets) < 3:
                continue

            # LightGBM LambdaRank requires non-negative integer labels (relevance grades)
            # Convert pct rank → quintile label 0..4
            day_ranks_pct = day_rets.rank(pct=True)
            day_ranks_int = (day_ranks_pct * 4.999).astype(int).clip(0, 4)

            for _, row in day_df.iterrows():
                ticker = row["ticker"]
                if ticker not in day_ranks_int.index:
                    continue
                feat_row = row[feat_cols].to_dict()
                feat_row["_rank"] = day_ranks_int[ticker]
                feat_row["_rank_pct"] = float(day_ranks_pct.get(ticker, 0.5))
                feat_row["_date"] = date_val
                rows.append(feat_row)

        if not rows:
            return None, None, None

        df = pd.DataFrame(rows)
        df = df.sort_values("_date")
        y = df["_rank"]
        X = df[feat_cols].select_dtypes(include=[np.number])

        # Groups: number of stocks per date (for LambdaRank)
        groups = df.groupby("_date").size().values

        return X, y, groups

    # ── Prediction ────────────────────────────────────────────────────────────

    def rank_stocks(
        self,
        stock_features_snap: pd.DataFrame,   # one row per ticker (current date)
        sector: str,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        """
        Score and rank stocks within a sector.
        Returns DataFrame with columns [ticker, score, rank].
        """
        sec_df = stock_features_snap[stock_features_snap["sector"] == sector].copy()

        if sec_df.empty:
            return pd.DataFrame(columns=["ticker", "score", "rank"])

        if sector in self.models:
            model = self.models[sector]
            feat_cols = [c for c in self.feature_names if c in sec_df.columns]
            X_raw = sec_df.reindex(columns=self.feature_names, fill_value=0).fillna(0)
            X_raw = X_raw.select_dtypes(include=[np.number])
            X_raw = X_raw.reindex(columns=self.feature_names, fill_value=0).fillna(0)
            if isinstance(model, _SectorHorizonBlend):
                scores = np.asarray(model.predict(X_raw), dtype=float)
            else:
                scaler = self.scalers.get(sector)
                if scaler is None:
                    scores = np.asarray(model.predict(X_raw), dtype=float)
                else:
                    X_arr = scaler.transform(X_raw)
                    X_named = pd.DataFrame(X_arr, columns=self.feature_names)
                    scores = np.asarray(model.predict(X_named), dtype=float)
        else:
            # fallback: use 3-month momentum
            if "ret_3m" in sec_df.columns:
                scores = sec_df["ret_3m"].fillna(0).values
            elif "mom_3m" in sec_df.columns:
                scores = sec_df["mom_3m"].fillna(0).values
            else:
                scores = np.zeros(len(sec_df))

        result = pd.DataFrame({
            "ticker": sec_df["ticker"].values,
            "score": scores,
        })
        result = result.sort_values(
            ["score", "ticker"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        if top_k:
            result = result.head(top_k)

        return result

    def feature_importance(self, sector: str) -> pd.Series:
        if sector not in self.models:
            return pd.Series(dtype=float)
        model = self.models[sector]
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            return pd.Series(
                np.asarray(importances, dtype=float),
                index=self.feature_names,
            ).sort_values(ascending=False)
        return pd.Series(dtype=float)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "scalers": self.scalers,
                "feature_names": self.feature_names,
                "horizons": self.horizons,
                "blend_weights": self.blend_weights,
                "blend_target_horizon": self.blend_target_horizon,
                "is_fitted": self.is_fitted,
            }, f)
        logger.debug("StockRanker saved → %s", path)

    def load(self, path: str | Path) -> "StockRanker":
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.models = state["models"]
        self.scalers = state["scalers"]
        self.feature_names = state["feature_names"]
        self.horizons = state.get("horizons", [])
        self.blend_weights = state.get("blend_weights", {})
        self.blend_target_horizon = state.get("blend_target_horizon")
        self.is_fitted = state["is_fitted"]
        return self
