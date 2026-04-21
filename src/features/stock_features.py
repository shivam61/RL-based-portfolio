"""
Stock-level features: block-based momentum, risk, liquidity, trend, and interaction signals.

All features are lagged at least 1 day.
The enabled feature blocks are controlled through `cfg["stock_features"]["blocks"]`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.base import fill_price_gaps

logger = logging.getLogger(__name__)


class StockFeatureBuilder:
    """Compute stock-level features from price and volume matrices."""

    DEFAULT_BLOCKS = ("absolute_momentum", "risk", "liquidity", "trend")

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        feat_cfg = self.cfg["features"]
        self.short = feat_cfg["lookback_short"]    # 21
        self.medium = feat_cfg["lookback_medium"]  # 63
        self.long = feat_cfg["lookback_long"]      # 252
        self.lag = feat_cfg["stock_lag"]
        stock_cfg = self.cfg.get("stock_features", {})
        blocks = stock_cfg.get("blocks", self.DEFAULT_BLOCKS)
        self.blocks = tuple(str(b) for b in blocks)
        self.LOGIC_VERSION = self._logic_version()

    def _logic_version(self) -> str:
        payload = {
            "base": "stock_features_v9_blocks_interactions",
            "blocks": list(self.blocks),
        }
        digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
        return f"stock_features_v9_{digest}"

    def _sector_broadcast(self, frame: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
        for sector in sorted(set(sector_map.values())):
            cols = [t for t, s in sector_map.items() if s == sector and t in frame.columns]
            if not cols:
                continue
            sector_mean = frame[cols].mean(axis=1)
            out.loc[:, cols] = np.repeat(sector_mean.to_numpy()[:, None], len(cols), axis=1)
        return out

    def _sector_zscore(self, frame: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
        for sector in sorted(set(sector_map.values())):
            cols = [t for t, s in sector_map.items() if s == sector and t in frame.columns]
            if not cols:
                continue
            sec = frame[cols]
            mean = sec.mean(axis=1)
            std = sec.std(axis=1).replace(0, np.nan)
            out.loc[:, cols] = sec.sub(mean, axis=0).div(std, axis=0)
        return out

    def _sector_rank(self, frame: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
        for sector in sorted(set(sector_map.values())):
            cols = [t for t, s in sector_map.items() if s == sector and t in frame.columns]
            if not cols:
                continue
            out.loc[:, cols] = frame[cols].rank(axis=1, pct=True)
        return out

    def _smoothed(self, frame: pd.DataFrame, span: int = 3) -> pd.DataFrame:
        return frame.ewm(span=span, adjust=False).mean()

    def _rolling_max_drawdown(self, prices: pd.DataFrame, window: int) -> pd.DataFrame:
        roll_peak = prices.rolling(window, min_periods=max(5, window // 4)).max()
        drawdown = prices / roll_peak.replace(0, np.nan) - 1.0
        return drawdown.rolling(window, min_periods=max(5, window // 4)).min()

    def _market_regime_flag(
        self,
        benchmark_prices: pd.Series,
        index: pd.Index,
        columns: list[str],
    ) -> pd.DataFrame:
        bm = benchmark_prices.reindex(index).ffill(limit=5)
        bm_ma = bm.rolling(200, min_periods=50).mean()
        flag = (bm > bm_ma).astype(float).fillna(0.0)
        data = np.repeat(flag.to_numpy()[:, None], len(columns), axis=1)
        return pd.DataFrame(data, index=index, columns=columns)

    def build(
        self,
        price_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame | None,
        sector_map: dict[str, str],
        benchmark_prices: pd.Series | None = None,
        earnings_panel: pd.DataFrame | None = None,  # ignored; kept for API compatibility
    ) -> pd.DataFrame:
        """Returns long-format DataFrame indexed by (date, ticker)."""
        tickers = [t for t in price_matrix.columns if t in sector_map]
        prices = fill_price_gaps(price_matrix[tickers], limit=5)
        if benchmark_prices is not None:
            benchmark_prices = fill_price_gaps(benchmark_prices, limit=5)
        returns = prices.pct_change(fill_method=None)

        feat_dict: dict[str, pd.DataFrame] = {}
        ret_3m = prices.pct_change(self.medium)
        ret_6m = prices.pct_change(int(self.medium * 2))
        mom_12m_skip1m = prices.pct_change(self.long) - prices.pct_change(self.short)
        vol_3m = returns.rolling(self.medium).std() * np.sqrt(252)
        max_dd_3m = self._rolling_max_drawdown(prices, self.medium)
        smoothed_ret_3m = self._smoothed(ret_3m, span=3)
        smoothed_vol_3m = self._smoothed(vol_3m, span=3)

        vol_tickers: list[str] = []
        amihud_1m = None
        if volume_matrix is not None:
            vol_tickers = [t for t in tickers if t in volume_matrix.columns]
            vols = volume_matrix[vol_tickers]
            amihud_1m = (
                returns[vol_tickers].abs() / vols.replace(0, np.nan)
            ).rolling(self.short).mean()

        if "absolute_momentum" in self.blocks:
            feat_dict["ret_3m"] = ret_3m
            feat_dict["mom_12m_skip1m"] = mom_12m_skip1m
            feat_dict["mom_accel_3m_6m"] = ret_3m - ret_6m

        if "sector_relative_momentum" in self.blocks:
            sector_ret_3m = self._sector_broadcast(ret_3m, sector_map)
            sector_mom_12m = self._sector_broadcast(mom_12m_skip1m, sector_map)
            feat_dict["ret_3m_vs_sector"] = ret_3m - sector_ret_3m
            feat_dict["mom_12m_skip1m_vs_sector"] = mom_12m_skip1m - sector_mom_12m

        if "sector_relative_strength" in self.blocks:
            sector_ret_3m = self._sector_broadcast(ret_3m, sector_map)
            sector_mom_12m = self._sector_broadcast(mom_12m_skip1m, sector_map)
            sector_max_dd_3m = self._sector_broadcast(max_dd_3m, sector_map)
            feat_dict["ret_3m_resid"] = ret_3m - sector_ret_3m
            feat_dict["mom_12m_skip1m_resid"] = mom_12m_skip1m - sector_mom_12m
            feat_dict["ret_3m_sector_rank"] = self._sector_rank(ret_3m, sector_map)
            feat_dict["max_dd_3m_vs_sector"] = max_dd_3m - sector_max_dd_3m
            feat_dict["max_dd_3m_sector_rank"] = self._sector_rank(max_dd_3m, sector_map)

        if "volatility_adjusted_momentum" in self.blocks:
            feat_dict["mom_3m_vol_adj"] = ret_3m / (vol_3m.abs() + 1e-9)

        if "interaction_momentum_volatility" in self.blocks:
            feat_dict["mom_x_vol_3m"] = ret_3m * vol_3m
            feat_dict["mom_x_inv_vol_3m"] = ret_3m / (vol_3m.abs() + 1e-9)

        if "interaction_momentum_drawdown" in self.blocks:
            feat_dict["mom_dd_penalty_3m"] = ret_3m + max_dd_3m.fillna(0.0)

        if "interaction_trend_liquidity" in self.blocks and amihud_1m is not None:
            feat_dict["trend_x_liquidity"] = (
                prices.rolling(50).mean() / prices.rolling(200).mean().replace(0, np.nan) - 1
            ) / (amihud_1m.abs() + 1e-9)

        if "regime_gated_momentum" in self.blocks and benchmark_prices is not None:
            regime_flag = self._market_regime_flag(benchmark_prices, prices.index, tickers)
            feat_dict["mom_regime_gate_3m"] = ret_3m * regime_flag

        if "sector_normalized" in self.blocks:
            feat_dict["ret_3m_sector_z"] = self._sector_zscore(ret_3m, sector_map)
            feat_dict["mom_12m_skip1m_sector_z"] = self._sector_zscore(mom_12m_skip1m, sector_map)
            feat_dict["vol_3m_sector_z"] = self._sector_zscore(vol_3m, sector_map)
            feat_dict["amihud_1m_sector_z"] = self._sector_zscore(amihud_1m, sector_map) if amihud_1m is not None else pd.DataFrame()
            trend_raw = prices.rolling(50).mean() / prices.rolling(200).mean().replace(0, np.nan) - 1
            feat_dict["ma_50_200_ratio_sector_z"] = self._sector_zscore(trend_raw, sector_map)
            feat_dict["ret_3m_sector_rank"] = self._sector_rank(ret_3m, sector_map)

        if "time_smoothing" in self.blocks:
            feat_dict["ret_3m_ema3"] = smoothed_ret_3m
            feat_dict["vol_3m_ema3"] = smoothed_vol_3m

        if "risk" in self.blocks:
            feat_dict["vol_3m"] = vol_3m
            feat_dict["max_dd_3m"] = max_dd_3m

        if "liquidity" in self.blocks and amihud_1m is not None:
            feat_dict["amihud_1m"] = amihud_1m

        if "trend" in self.blocks:
            feat_dict["ma_50_200_ratio"] = (
                prices.rolling(50).mean() / prices.rolling(200).mean().replace(0, np.nan) - 1
            )

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
