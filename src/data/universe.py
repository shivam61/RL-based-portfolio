"""
Universe management: point-in-time membership, liquidity filters,
sector mapping, and blacklist handling.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.config import load_config, load_universe_config
from src.data.contracts import StockMeta, UniverseSnapshot

logger = logging.getLogger(__name__)


class UniverseManager:
    """Manages the investable equity universe with point-in-time correctness."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self._uni_cfg = load_universe_config()
        self._stock_meta: list[StockMeta] = self._load_stock_meta()
        self._sector_meta: dict = self._uni_cfg.get("sectors", {})

    # ── Initialization ────────────────────────────────────────────────────────

    def _load_stock_meta(self) -> list[StockMeta]:
        stocks = []
        for raw in self._uni_cfg.get("stocks", []):
            try:
                stocks.append(StockMeta(**raw))
            except Exception as e:
                logger.warning("Bad stock entry %s: %s", raw.get("ticker"), e)
        logger.info("Loaded %d stocks from universe config", len(stocks))
        return stocks

    # ── Point-in-time universe snapshot ──────────────────────────────────────

    def get_universe(
        self,
        as_of: date,
        price_matrix: pd.DataFrame | None = None,
        volume_matrix: pd.DataFrame | None = None,
        cap_filter: list[str] | None = None,
    ) -> UniverseSnapshot:
        """
        Return the investable universe as of `as_of`.

        Applies:
        - listing/delisting date filter (point-in-time)
        - blacklist filter
        - optional cap-size filter
        - optional liquidity filter (requires volume_matrix)
        """
        eligible = []
        for sm in self._stock_meta:
            if sm.blacklisted:
                continue
            if sm.listed_since and sm.listed_since > as_of:
                continue
            if sm.delisted_on and sm.delisted_on <= as_of:
                continue
            if cap_filter and sm.cap not in cap_filter:
                continue
            eligible.append(sm)

        # data-driven listing filter: only include stocks with price data available
        if price_matrix is not None:
            eligible = self._filter_by_data_availability(
                eligible, price_matrix, as_of
            )

        # optional liquidity filter
        if volume_matrix is not None:
            eligible = self._filter_by_liquidity(
                eligible, volume_matrix, as_of
            )

        return UniverseSnapshot(as_of=as_of, stocks=eligible)

    def _filter_by_data_availability(
        self,
        stocks: list[StockMeta],
        price_matrix: pd.DataFrame,
        as_of: date,
    ) -> list[StockMeta]:
        """Require at least 252 trading days of history before as_of."""
        ts = pd.Timestamp(as_of)
        hist = price_matrix[price_matrix.index <= ts]
        keep = []
        for sm in stocks:
            if sm.ticker not in hist.columns:
                continue
            series = hist[sm.ticker].dropna()
            if len(series) >= 252:
                keep.append(sm)
        return keep

    def _filter_by_liquidity(
        self,
        stocks: list[StockMeta],
        volume_matrix: pd.DataFrame,
        as_of: date,
        lookback_days: int = 63,
        min_avg_vol_cr: float | None = None,
    ) -> list[StockMeta]:
        ts = pd.Timestamp(as_of)
        if min_avg_vol_cr is None:
            min_avg_vol_cr = self.cfg["universe"].get("min_avg_volume_cr", 1.0)
        hist = volume_matrix[volume_matrix.index <= ts].tail(lookback_days)
        keep = []
        for sm in stocks:
            if sm.ticker not in hist.columns:
                keep.append(sm)  # no volume data → keep (conservative)
                continue
            avg_vol = hist[sm.ticker].mean()
            if pd.isna(avg_vol) or avg_vol >= min_avg_vol_cr:
                keep.append(sm)
        return keep

    # ── Sector utilities ──────────────────────────────────────────────────────

    def get_sector_map(self, snapshot: UniverseSnapshot) -> dict[str, str]:
        """Returns {ticker: sector} mapping."""
        return {s.ticker: s.sector for s in snapshot.stocks}

    def get_sector_tickers(
        self, snapshot: UniverseSnapshot, sector: str
    ) -> list[str]:
        return [s.ticker for s in snapshot.stocks if s.sector == sector]

    def get_sector_meta(self, sector: str) -> dict:
        return self._sector_meta.get(sector, {})

    def all_sectors(self, snapshot: UniverseSnapshot) -> list[str]:
        return snapshot.sectors

    # ── Cap-size splits ───────────────────────────────────────────────────────

    def get_cap_tickers(
        self, snapshot: UniverseSnapshot, cap: str
    ) -> list[str]:
        return [s.ticker for s in snapshot.stocks if s.cap == cap]

    # ── Global proxy tickers ──────────────────────────────────────────────────

    def get_global_proxies(self) -> dict[str, str]:
        return self._uni_cfg.get("global_proxies", {})


# ── Standalone helper ─────────────────────────────────────────────────────────

def build_sector_return_matrix(
    price_matrix: pd.DataFrame,
    universe_manager: UniverseManager,
    snapshot: UniverseSnapshot,
    freq: str = "W",          # 'W' weekly, '4W' four-weekly, 'M' monthly
) -> pd.DataFrame:
    """
    Compute sector-level returns (equal-weighted within sector) at given frequency.
    Returns DataFrame: (date × sector).
    """
    sector_map = universe_manager.get_sector_map(snapshot)
    tickers = [t for t in snapshot.tickers if t in price_matrix.columns]
    prices = price_matrix[tickers]

    returns = prices.pct_change()
    sector_returns: dict[str, pd.Series] = {}
    for sector in snapshot.sectors:
        sec_tickers = [t for t in tickers if sector_map.get(t) == sector]
        if not sec_tickers:
            continue
        sector_returns[sector] = returns[sec_tickers].mean(axis=1)

    df = pd.DataFrame(sector_returns)
    if freq != "D":
        df = df.resample(freq).apply(lambda x: (1 + x).prod() - 1)
    return df
