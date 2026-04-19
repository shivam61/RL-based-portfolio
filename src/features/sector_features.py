"""
Sector-level features: momentum, relative strength, breadth, volatility,
valuation proxies, macro sensitivities, and news sentiment.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.base import (
    fill_price_gaps,
    lag_series,
    normalized_equal_weight_index,
    rolling_max_drawdown,
    rolling_return,
    rolling_vol,
)

logger = logging.getLogger(__name__)


class SectorFeatureBuilder:
    """Compute sector features from price matrix + macro features."""

    LOGIC_VERSION = "sector_features_v3_normalized_index"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        feat_cfg = self.cfg["features"]
        self.short = feat_cfg["lookback_short"]
        self.medium = feat_cfg["lookback_medium"]
        self.long = feat_cfg["lookback_long"]

    def build(
        self,
        price_matrix: pd.DataFrame,        # date × ticker (adj close)
        sector_map: dict[str, str],         # ticker → sector
        macro_features: pd.DataFrame | None = None,
        benchmark_prices: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Returns long-format DataFrame: (date, sector) × features.
        All features lagged 1 day.
        """
        sectors = sorted(set(sector_map.values()))
        # ffill to fill holiday NaNs before rolling calculations
        price_matrix = fill_price_gaps(price_matrix, limit=5)
        if benchmark_prices is not None:
            benchmark_prices = fill_price_gaps(benchmark_prices, limit=5)
        daily_returns = price_matrix.pct_change(fill_method=None)

        all_rows = []
        for sector in sectors:
            tickers = [t for t, s in sector_map.items() if s == sector and t in price_matrix.columns]
            if not tickers:
                continue

            sec_prices = normalized_equal_weight_index(price_matrix[tickers], max_gap=5)
            sec_rets = daily_returns[tickers].mean(axis=1)

            feat: dict[str, pd.Series] = {}

            # ── Momentum ──────────────────────────────────────────────────────
            feat["mom_1m"] = rolling_return(sec_prices, self.short)
            feat["mom_3m"] = rolling_return(sec_prices, self.medium)
            feat["mom_6m"] = rolling_return(sec_prices, int(self.medium * 2))
            feat["mom_12m"] = rolling_return(sec_prices, self.long)
            feat["mom_12m_ex1m"] = feat["mom_12m"] - feat["mom_1m"]  # skip last month

            # ── Trend strength ────────────────────────────────────────────────
            feat["above_50ma"] = (sec_prices > sec_prices.rolling(50).mean()).astype(float)
            feat["above_200ma"] = (sec_prices > sec_prices.rolling(200).mean()).astype(float)
            feat["price_to_52w_high"] = sec_prices / sec_prices.rolling(252).max()

            # ── Volatility ────────────────────────────────────────────────────
            feat["vol_1m"] = rolling_vol(sec_rets, self.short, 252)
            feat["vol_3m"] = rolling_vol(sec_rets, self.medium, 252)
            feat["vol_ratio"] = feat["vol_1m"] / feat["vol_3m"].replace(0, np.nan)
            feat["max_dd_3m"] = rolling_max_drawdown(sec_prices, self.medium)

            # ── Relative strength vs benchmark ────────────────────────────────
            if benchmark_prices is not None:
                bm_rets = benchmark_prices.pct_change(fill_method=None)
                feat["rel_str_1m"] = feat["mom_1m"] - rolling_return(benchmark_prices, self.short)
                feat["rel_str_3m"] = feat["mom_3m"] - rolling_return(benchmark_prices, self.medium)
                feat["beta_3m"] = (
                    sec_rets.rolling(self.medium).cov(bm_rets)
                    / bm_rets.rolling(self.medium).var().replace(0, np.nan)
                )

            # ── Breadth ───────────────────────────────────────────────────────
            rets_1m = price_matrix[tickers].pct_change(self.short, fill_method=None)
            feat["breadth_1m"] = (rets_1m > 0).sum(axis=1) / len(tickers)
            rets_3m = price_matrix[tickers].pct_change(self.medium, fill_method=None)
            feat["breadth_3m"] = (rets_3m > 0).sum(axis=1) / len(tickers)

            # % stocks in sector above 50 DMA (validates sector trend quality)
            ma50 = price_matrix[tickers].rolling(50).mean()
            feat["pct_above_50ma"] = (
                (price_matrix[tickers] > ma50).sum(axis=1) / len(tickers)
            )

            # % stocks near 52-week high (within 5%) — sector leadership strength
            high_52w = price_matrix[tickers].rolling(self.long).max()
            feat["new_high_ratio"] = (
                (price_matrix[tickers] >= high_52w * 0.95).sum(axis=1) / len(tickers)
            )

            # ── Dispersion (cross-sectional vol within sector) ────────────────
            feat["dispersion_1m"] = (
                price_matrix[tickers].pct_change(self.short, fill_method=None).std(axis=1)
            )

            # ── Macro sensitivity proxies (from sector metadata) ──────────────
            # (encoded from universe.yaml sector metadata, static per sector)
            # Will be joined at model training time

            # ── Mean reversion signal ─────────────────────────────────────────
            feat["zscore_1y"] = (
                (sec_prices - sec_prices.rolling(252).mean())
                / sec_prices.rolling(252).std().replace(0, np.nan)
            )

            # Assemble sector DataFrame
            df_sector = pd.DataFrame(feat)
            df_sector["sector"] = sector
            df_sector = df_sector.shift(1)  # lag 1 to avoid lookahead
            all_rows.append(df_sector)

        if not all_rows:
            return pd.DataFrame()

        combined = pd.concat(all_rows)
        combined.index.name = "date"
        combined = combined.reset_index()   # date becomes a column

        # cross-sectional ranks at each date
        for col in ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "rel_str_1m", "rel_str_3m"]:
            if col not in combined.columns:
                continue
            try:
                pivoted = combined.pivot_table(
                    index="date", columns="sector", values=col, aggfunc="last"
                )
                ranked = pivoted.rank(axis=1, pct=True)
                # melt back to long and merge
                melted = ranked.stack().rename(f"{col}_rank").reset_index()
                combined = combined.merge(melted, on=["date", "sector"], how="left")
            except Exception:
                pass

        combined = combined.set_index("date")

        out_path = Path(self.cfg["paths"]["feature_data"]) / "sector_features.parquet"
        combined.reset_index().to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Sector features: %s → %s", combined.shape, out_path)
        return combined

    def get_sector_features_as_of(
        self, sector_features: pd.DataFrame, as_of: pd.Timestamp, sector: str
    ) -> pd.Series:
        """Point-in-time sector feature vector."""
        hist = sector_features[sector_features.index <= as_of]
        sect = hist[hist["sector"] == sector]
        if sect.empty:
            return pd.Series(dtype=float)
        row = sect.iloc[-1].drop("sector")
        return row.apply(pd.to_numeric, errors="coerce")
