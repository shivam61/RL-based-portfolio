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
from src.features.base import (
    rolling_max_drawdown,
    rolling_vol,
)

logger = logging.getLogger(__name__)


class StockFeatureBuilder:
    """Compute stock-level features from price and volume matrices."""

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
        prices = price_matrix[tickers].ffill()  # fill holiday NaNs so rolling windows work
        if benchmark_prices is not None:
            benchmark_prices = benchmark_prices.ffill()
        returns = prices.pct_change()

        feat_dict: dict[str, pd.DataFrame] = {}

        # ── Returns / Momentum ────────────────────────────────────────────────
        for h, name in [(5, "1w"), (self.short, "1m"), (self.medium, "3m"),
                        (int(self.medium * 2), "6m"), (self.long, "12m")]:
            feat_dict[f"ret_{name}"] = prices.pct_change(h)

        # skip-1-month momentum (standard factor)
        feat_dict["mom_12m_skip1m"] = (
            feat_dict.get("ret_12m", pd.DataFrame()) -
            feat_dict.get("ret_1m", pd.DataFrame())
        )

        # ── Momentum stability ────────────────────────────────────────────────
        feat_dict["mom_stab_3m"] = (returns > 0).rolling(self.medium).mean()
        feat_dict["mom_stab_12m"] = (returns > 0).rolling(self.long).mean()

        # ── Volatility ────────────────────────────────────────────────────────
        feat_dict["vol_1m"] = returns.rolling(self.short).std() * np.sqrt(252)
        feat_dict["vol_3m"] = returns.rolling(self.medium).std() * np.sqrt(252)
        feat_dict["vol_12m"] = returns.rolling(self.long).std() * np.sqrt(252)
        feat_dict["vol_ratio_1m_3m"] = feat_dict["vol_1m"] / feat_dict["vol_3m"].replace(0, np.nan)

        # ── Downside risk ─────────────────────────────────────────────────────
        feat_dict["downside_vol_3m"] = (
            returns.clip(upper=0).rolling(self.medium).std() * np.sqrt(252)
        )
        feat_dict["max_dd_3m"] = prices.rolling(self.medium).apply(
            lambda x: (x / np.maximum.accumulate(x) - 1).min(), raw=True
        )
        feat_dict["max_dd_12m"] = prices.rolling(self.long).apply(
            lambda x: (x / np.maximum.accumulate(x) - 1).min(), raw=True
        )

        # ── Higher moments ────────────────────────────────────────────────────
        feat_dict["skew_3m"] = returns.rolling(self.medium).skew()
        feat_dict["kurt_3m"] = returns.rolling(self.medium).kurt()

        # ── Trend ─────────────────────────────────────────────────────────────
        # above_50ma / above_200ma dropped (run_020): 0% feature importance,
        # redundant vs continuous ma_50_200_ratio
        feat_dict["price_to_52w_high"] = prices / prices.rolling(252).max()
        feat_dict["price_to_52w_low"] = prices / prices.rolling(252).min()

        # ── Mean-reversion ────────────────────────────────────────────────────
        feat_dict["zscore_6m"] = (
            (prices - prices.rolling(int(self.medium * 2)).mean())
            / prices.rolling(int(self.medium * 2)).std().replace(0, np.nan)
        )

        # ── Relative to benchmark ─────────────────────────────────────────────
        if benchmark_prices is not None:
            bm_rets = benchmark_prices.pct_change()
            for col, h in [("ret_1m", self.short), ("ret_3m", self.medium)]:
                if col in feat_dict:
                    bm_h = (1 + bm_rets).rolling(h).apply(lambda x: x.prod() - 1, raw=True)
                    feat_dict[f"alpha_{col}"] = feat_dict[col].sub(bm_h, axis=0)

            bm_var = bm_rets.rolling(self.medium).var()
            feat_dict["beta_3m"] = returns.rolling(self.medium).corr(
                bm_rets
            ) * feat_dict["vol_3m"] / (bm_rets.rolling(self.medium).std() * np.sqrt(252)).clip(lower=1e-6)

        # ── Volume / Liquidity ────────────────────────────────────────────────
        if volume_matrix is not None:
            vol_tickers = [t for t in tickers if t in volume_matrix.columns]
            vols = volume_matrix[vol_tickers]
            feat_dict["avg_vol_1m"] = vols.rolling(self.short).mean()
            feat_dict["avg_vol_3m"] = vols.rolling(self.medium).mean()
            feat_dict["vol_trend"] = (
                feat_dict["avg_vol_1m"] / feat_dict["avg_vol_3m"].replace(0, np.nan)
            )
            feat_dict["amihud_1m"] = (
                returns[vol_tickers].abs() / vols.replace(0, np.nan)
            ).rolling(self.short).mean()

        # ── Additional momentum / trend features ─────────────────────────────
        feat_dict["ma_50_200_ratio"] = (
            prices.rolling(50).mean() / prices.rolling(200).mean().replace(0, np.nan) - 1
        )

        roll_high = prices.rolling(self.long).max()
        roll_low  = prices.rolling(self.long).min()
        feat_dict["price_pctile_1y"] = (
            (prices - roll_low) / (roll_high - roll_low).replace(0, np.nan)
        )

        if volume_matrix is not None and "avg_vol_1m" in feat_dict:
            vol_tickers = [t for t in tickers if t in volume_matrix.columns]
            vols = volume_matrix[vol_tickers]
            vol_20d = vols.rolling(20).mean().replace(0, np.nan)
            feat_dict["vol_spike"] = (vols / vol_20d).reindex(columns=prices.columns)

        gain = returns.clip(lower=0).rolling(14).mean()
        loss = (-returns.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        feat_dict["rsi_14"] = 100 - (100 / (1 + gain / loss))

        # ── Sector-relative features ──────────────────────────────────────────
        sectors = sorted(set(sector_map.values()))
        for feat_name in ["ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_3m"]:
            if feat_name not in feat_dict:
                continue
            sector_means: dict[str, pd.Series] = {}
            for sec in sectors:
                sec_tickers = [t for t in tickers if sector_map.get(t) == sec]
                if not sec_tickers:
                    continue
                avail = [t for t in sec_tickers if t in feat_dict[feat_name].columns]
                if avail:
                    sector_means[sec] = feat_dict[feat_name][avail].mean(axis=1)
            sec_mean_df = pd.DataFrame({
                t: sector_means.get(sector_map[t], pd.Series(np.nan, index=prices.index))
                for t in tickers if t in sector_map
            })
            feat_dict[f"{feat_name}_vs_sector"] = feat_dict[feat_name] - sec_mean_df

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
