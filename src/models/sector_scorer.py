"""
Sector scoring model.

Predicts cross-sectional rank of 4-week forward returns across sectors.
Uses LightGBM with walk-forward-safe training.
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
    logger.warning("LightGBM not available; sector scorer will use sklearn fallback")


class SectorScorer:
    """
    Trains on (sector, date) feature rows with a forward-return label.
    Predicts expected sector attractiveness score in [0, 1].
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model_cfg = cfg["sector_model"]
        self.model: Optional[object] = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        sector_features: pd.DataFrame,    # (date, sector, features...)
        sector_returns: pd.DataFrame,      # date × sector, realized returns
        fwd_window: int = 28,
        macro_features: pd.DataFrame | None = None,
    ) -> "SectorScorer":
        """
        Train on all data up to a given cutoff (caller ensures no lookahead).

        Target: cross-sectional rank of forward sector return.
        """
        if sector_features.empty or sector_returns.empty:
            logger.warning("Empty data — sector scorer not trained")
            return self

        # Build training dataset
        X, y = self._build_training_set(
            sector_features, sector_returns, fwd_window, macro_features
        )

        if X is None or len(X) < 50:
            logger.warning("Insufficient data for sector scorer: %d rows", len(X) if X is not None else 0)
            return self

        self.feature_names = list(X.columns)
        X_arr = self.scaler.fit_transform(X.fillna(0))

        if HAS_LGB:
            params = {
                "objective": "regression",
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
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(X_arr, y.values)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3
            )
            self.model.fit(X_arr, y.values)

        self.is_fitted = True
        logger.info(
            "SectorScorer trained on %d rows, %d features",
            len(X), len(self.feature_names),
        )
        return self

    def _build_training_set(
        self,
        sector_features: pd.DataFrame,
        sector_returns: pd.DataFrame,
        fwd_window: int,
        macro_features: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
        # Forward returns (rolling over fwd_window trading days)
        fwd_rets = sector_returns.copy()
        for col in fwd_rets.columns:
            fwd_rets[col] = (
                fwd_rets[col].shift(-fwd_window).rolling(fwd_window).apply(
                    lambda x: (1 + x).prod() - 1, raw=True
                )
            )

        # Cross-sectional rank at each date
        fwd_rank = fwd_rets.rank(axis=1, pct=True)

        rows = []
        for date_val in sector_features.index.unique():
            if isinstance(date_val, tuple):
                continue
            date_mask = sector_features.index == date_val
            sec_feats = sector_features[date_mask]

            if "sector" not in sec_feats.columns:
                continue

            for _, row in sec_feats.iterrows():
                sector = row.get("sector")
                if sector not in fwd_rank.columns:
                    continue
                if date_val not in fwd_rank.index:
                    continue
                target = fwd_rank.loc[date_val, sector]
                if pd.isna(target):
                    continue

                feat_row = row.drop("sector", errors="ignore")
                feat_row = feat_row.to_dict()

                if macro_features is not None:
                    m_hist = macro_features[macro_features.index <= date_val]
                    if not m_hist.empty:
                        m_row = m_hist.iloc[-1].to_dict()
                        for k, v in m_row.items():
                            feat_row[f"macro_{k}"] = v

                feat_row["_date"] = date_val
                feat_row["_sector"] = sector
                feat_row["_target"] = target
                rows.append(feat_row)

        if not rows:
            return None, None

        df = pd.DataFrame(rows)
        y = df["_target"]
        X = df.drop(columns=["_date", "_sector", "_target"], errors="ignore")
        X = X.select_dtypes(include=[np.number])
        return X, y

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        sector_features_row: pd.DataFrame,   # one row per sector
        macro_features_row: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Returns attractiveness score in [0, 1] per sector.
        Higher = more attractive.
        """
        if not self.is_fitted or self.model is None:
            return self._fallback_score(sector_features_row)

        rows = []
        sectors = []
        for _, row in sector_features_row.iterrows():
            sector = row.get("sector", "unknown")
            feat = row.drop("sector", errors="ignore").to_dict()

            if macro_features_row is not None:
                for k, v in macro_features_row.items():
                    feat[f"macro_{k}"] = v

            rows.append(feat)
            sectors.append(sector)

        X = pd.DataFrame(rows)
        X = X.reindex(columns=self.feature_names, fill_value=0).fillna(0)
        X_arr = self.scaler.transform(X)
        # Pass DataFrame with feature names to suppress sklearn/LightGBM warnings
        X_named = pd.DataFrame(X_arr, columns=self.feature_names)
        scores = self.model.predict(X_named)
        scores = np.clip(scores, 0, 1)
        return dict(zip(sectors, scores.tolist()))

    def _fallback_score(self, sector_features_row: pd.DataFrame) -> dict[str, float]:
        """Momentum-based fallback when model is not fitted."""
        scores = {}
        for _, row in sector_features_row.iterrows():
            sector = row.get("sector", "unknown")
            mom = row.get("mom_3m", 0.0)
            if pd.isna(mom):
                mom = 0.0
            scores[sector] = float(np.clip(0.5 + mom * 2, 0, 1))
        return scores

    def feature_importance(self) -> pd.Series:
        if not self.is_fitted or self.model is None:
            return pd.Series(dtype=float)
        if HAS_LGB and isinstance(self.model, lgb.LGBMRegressor):
            imp = self.model.feature_importances_
            return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(dtype=float)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "feature_names": self.feature_names,
                         "is_fitted": self.is_fitted}, f)
        logger.debug("SectorScorer saved → %s", path)

    def load(self, path: str | Path) -> "SectorScorer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self.is_fitted = state["is_fitted"]
        return self
