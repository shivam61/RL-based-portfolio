"""
Stock-level features: returns, momentum, volatility, trend, and relative-to-sector signals.

All features are lagged at least 1 day.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.base import fill_price_gaps

logger = logging.getLogger(__name__)


class StockFeatureBuilder:
    """Compute stock-level features from price and volume matrices."""

    LOGIC_VERSION = "stock_features_v7_minimal_raw_v2"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        feat_cfg = self.cfg["features"]
        self.short = feat_cfg["lookback_short"]    # 21
        self.medium = feat_cfg["lookback_medium"]  # 63
        self.long = feat_cfg["lookback_long"]      # 252
        self.lag = feat_cfg["stock_lag"]

    def build(
        self,
        price_matrix: pd.DataFrame,          # date × ticker
        volume_matrix: pd.DataFrame | None,  # date × ticker (INR crore)
        sector_map: dict[str, str],          # ticker → sector
        benchmark_prices: pd.Series | None = None,
        earnings_panel: pd.DataFrame | None = None,  # ignored; kept for API compatibility
    ) -> pd.DataFrame:
        """
        Returns long-format DataFrame indexed by (date, ticker).
        All features lag-adjusted.
        """
        tickers = [t for t in price_matrix.columns if t in sector_map]
        prices = fill_price_gaps(price_matrix[tickers], limit=5)
        if benchmark_prices is not None:
            benchmark_prices = fill_price_gaps(benchmark_prices, limit=5)
        returns = prices.pct_change(fill_method=None)

        feat_dict: dict[str, pd.DataFrame] = {}

        # ── Minimal raw momentum / risk family ────────────────────────────────
        ret_3m = prices.pct_change(self.medium)
        ret_6m = prices.pct_change(int(self.medium * 2))
        feat_dict["ret_3m"] = ret_3m
        feat_dict["mom_12m_skip1m"] = (
            prices.pct_change(self.long) - prices.pct_change(self.short)
        )
        feat_dict["mom_accel_3m_6m"] = ret_3m - ret_6m

        feat_dict["vol_3m"] = returns.rolling(self.medium).std() * np.sqrt(252)

        # ── Volume / Liquidity ────────────────────────────────────────────────
        if volume_matrix is not None:
            vol_tickers = [t for t in tickers if t in volume_matrix.columns]
            vols = volume_matrix[vol_tickers]
            feat_dict["amihud_1m"] = (
                returns[vol_tickers].abs() / vols.replace(0, np.nan)
            ).rolling(self.short).mean()

        feat_dict["ma_50_200_ratio"] = (
            prices.rolling(50).mean() / prices.rolling(200).mean().replace(0, np.nan) - 1
        )

        # ── Assemble into long format ─────────────────────────────────────────
        all_dfs = []
        for feat_name, df in feat_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                shifted = df.shift(self.lag)
                melted = shifted.stack(future_stack=True).rename(feat_name)
                all_dfs.append(melted)

        if not all_dfs:
            return pd.DataFrame()

        long_df = pd.concat(all_dfs, axis=1)
        long_df.index.names = ["date", "ticker"]
        long_df = long_df.reset_index()
        long_df["sector"] = long_df["ticker"].map(sector_map)

        # drop rows with all NaN features
        feat_cols = [c for c in long_df.columns if c not in ["date", "ticker", "sector"]]
        long_df = long_df.dropna(subset=feat_cols, how="all")

        out_path = Path(self.cfg["paths"]["feature_data"]) / "stock_features.parquet"
        long_df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Stock features: %s → %s", long_df.shape, out_path)
        return long_df

    def get_stock_features_as_of(
        self,
        stock_features: pd.DataFrame,
        as_of: pd.Timestamp,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return stock feature snapshot for a given rebalance date."""
        date_mask = stock_features["date"] <= as_of
        snap = stock_features[date_mask]
        snap = snap.sort_values("date").groupby("ticker").last().reset_index()
        if tickers:
            snap = snap[snap["ticker"].isin(tickers)]
        return snap
