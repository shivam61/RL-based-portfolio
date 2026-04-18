"""
Stock ranking model.

Uses LightGBM LambdaRank to rank stocks within each sector by
expected 4-week forward return (cross-sectional).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

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


class StockRanker:
    """
    Train a cross-sectional ranking model within sectors.
    Predicts a score where higher → better expected 4-week return.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model_cfg = cfg["stock_model"]
        self.models: dict[str, object] = {}   # one model per sector
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_names: list[str] = []
        self.is_fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        stock_features: pd.DataFrame,   # long: date, ticker, sector, features...
        price_matrix: pd.DataFrame,      # date × ticker adj close
        fwd_window: int = 28,
    ) -> "StockRanker":
        """Train one ranking model per sector using available history."""
        if stock_features.empty:
            return self

        feat_cols = [c for c in stock_features.columns if c not in _EXCLUDE_COLS]
        self.feature_names = feat_cols

        # Build forward return labels
        fwd_rets = price_matrix.pct_change(fwd_window).shift(-fwd_window)

        sectors = stock_features["sector"].dropna().unique()
        for sector in sectors:
            sec_df = stock_features[stock_features["sector"] == sector].copy()
            if len(sec_df) < 100:
                continue

            X, y, groups = self._build_rank_dataset(sec_df, fwd_rets, feat_cols)
            if X is None or len(X) < 50:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.fillna(0))
            self.scalers[sector] = scaler

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
                    "n_jobs": -1,
                }
                model = lgb.LGBMRanker(**params)
                model.fit(X_scaled, y.values, group=groups)
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.05, max_depth=3
                )
                model.fit(X_scaled, y.values)

            self.models[sector] = model
            logger.debug("StockRanker[%s] trained on %d rows", sector, len(X))

        self.is_fitted = bool(self.models)
        logger.info("StockRanker fitted for sectors: %s", list(self.models.keys()))
        return self

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
            scaler = self.scalers[sector]
            feat_cols = [c for c in self.feature_names if c in sec_df.columns]
            X_raw = sec_df.reindex(columns=self.feature_names, fill_value=0).fillna(0).select_dtypes(include=[np.number])
            X_raw = X_raw.reindex(columns=self.feature_names, fill_value=0).fillna(0)
            X_arr = scaler.transform(X_raw)
            X_named = pd.DataFrame(X_arr, columns=self.feature_names)
            scores = self.models[sector].predict(X_named)
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
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)

        if top_k:
            result = result.head(top_k)

        return result

    def feature_importance(self, sector: str) -> pd.Series:
        if sector not in self.models:
            return pd.Series(dtype=float)
        model = self.models[sector]
        if HAS_LGB and isinstance(model, lgb.LGBMRanker):
            return pd.Series(
                model.feature_importances_,
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
                "is_fitted": self.is_fitted,
            }, f)
        logger.debug("StockRanker saved → %s", path)

    def load(self, path: str | Path) -> "StockRanker":
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.models = state["models"]
        self.scalers = state["scalers"]
        self.feature_names = state["feature_names"]
        self.is_fitted = state["is_fitted"]
        return self
