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
from src.universe.eligibility import apply_time_aware_eligibility
from src.universe.historical_sector_universe import HistoricalSectorUniverseStore

logger = logging.getLogger(__name__)


class UniverseManager:
    """Manages the investable equity universe with point-in-time correctness."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self._uni_cfg = load_universe_config()
        self._stock_meta: list[StockMeta] = self._load_stock_meta()
        self._sector_meta: dict = self._uni_cfg.get("sectors", {})
        self._mode = self.cfg.get("universe", {}).get("mode", "static")
        self._hu_cfg = self.cfg.get("universe", {}).get("historical_union", {})
        self._historical_store = HistoricalSectorUniverseStore(self.cfg)
        self._historical_enabled = (
            self._mode == "historical_union_10y" and self._historical_store.is_available
        )
        if self._mode == "historical_union_10y" and not self._historical_enabled:
            logger.warning(
                "Universe mode is historical_union_10y but artifacts are missing. "
                "Falling back to static universe mode."
            )
        if self._historical_enabled:
            self._stock_meta = self._load_historical_stock_meta()
            logger.info(
                "Historical union mode enabled with %d union tickers",
                len(self._stock_meta),
            )

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

    def _load_historical_stock_meta(self) -> list[StockMeta]:
        rows = self._historical_store.union_df
        if rows.empty:
            return self._stock_meta
        stocks: list[StockMeta] = []
        for _, row in rows.iterrows():
            cap = str(row.get("cap", "mid"))
            if cap not in {"large", "mid", "small"}:
                cap = "mid"
            listed_since = (
                pd.Timestamp(row["active_from"]).date()
                if "active_from" in row and pd.notna(row["active_from"])
                else None
            )
            delisted_on = (
                pd.Timestamp(row["active_to"]).date()
                if "active_to" in row and pd.notna(row["active_to"])
                else None
            )
            try:
                stocks.append(
                    StockMeta(
                        ticker=str(row["ticker"]),
                        name=str(row.get("name", row["ticker"])),
                        sector=str(row["sector"]),
                        cap=cap,  # type: ignore[arg-type]
                        listed_since=listed_since,
                        delisted_on=delisted_on,
                        blacklisted=False,
                    )
                )
            except Exception as exc:
                logger.warning("Bad historical union row for %s: %s", row.get("ticker"), exc)
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
        if self._historical_enabled and price_matrix is not None:
            return self._get_historical_universe(
                as_of=as_of,
                price_matrix=price_matrix,
                volume_matrix=volume_matrix,
                cap_filter=cap_filter,
            )
        return self._get_static_universe(
            as_of=as_of,
            price_matrix=price_matrix,
            volume_matrix=volume_matrix,
            cap_filter=cap_filter,
        )

    def _get_static_universe(
        self,
        as_of: date,
        price_matrix: pd.DataFrame | None = None,
        volume_matrix: pd.DataFrame | None = None,
        cap_filter: list[str] | None = None,
    ) -> UniverseSnapshot:
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

        if price_matrix is not None:
            eligible = self._filter_by_data_availability(
                eligible, price_matrix, as_of
            )
        if volume_matrix is not None:
            eligible = self._filter_by_liquidity(
                eligible, volume_matrix, as_of
            )
        return UniverseSnapshot(as_of=as_of, stocks=eligible)

    def _get_historical_universe(
        self,
        as_of: date,
        price_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame | None = None,
        cap_filter: list[str] | None = None,
    ) -> UniverseSnapshot:
        union_df = self._historical_store.union_df
        eligible_rows = apply_time_aware_eligibility(
            union_df=union_df,
            as_of_date=as_of,
            price_matrix=price_matrix,
            volume_matrix=volume_matrix,
            min_price_history_days=int(self._hu_cfg.get("min_price_history_days", 252)),
            min_median_traded_value_cr=float(self._hu_cfg.get("min_median_traded_value_cr", 2.0)),
            liquidity_lookback_days=int(self._hu_cfg.get("liquidity_lookback_days", 63)),
            use_active_window_filter=bool(self._hu_cfg.get("use_active_window_filter", True)),
        )

        by_ticker: dict[str, StockMeta] = {s.ticker: s for s in self._stock_meta}
        stocks: list[StockMeta] = []
        for _, row in eligible_rows.iterrows():
            ticker = str(row["ticker"])
            sm = by_ticker.get(ticker)
            if sm is None:
                # Safety fallback if metadata couldn't be loaded for a row.
                cap = str(row.get("cap", "mid"))
                if cap not in {"large", "mid", "small"}:
                    cap = "mid"
                listed_since = (
                    pd.Timestamp(row["active_from"]).date()
                    if "active_from" in row and pd.notna(row["active_from"])
                    else None
                )
                delisted_on = (
                    pd.Timestamp(row["active_to"]).date()
                    if "active_to" in row and pd.notna(row["active_to"])
                    else None
                )
                sm = StockMeta(
                    ticker=ticker,
                    name=str(row.get("name", ticker)),
                    sector=str(row["sector"]),
                    cap=cap,  # type: ignore[arg-type]
                    listed_since=listed_since,
                    delisted_on=delisted_on,
                    blacklisted=False,
                )
            if cap_filter and sm.cap not in cap_filter:
                continue
            stocks.append(sm)
        return UniverseSnapshot(as_of=as_of, stocks=stocks)

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

    def membership_mask(
        self,
        price_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame | None = None,
        cap_filter: list[str] | None = None,
        min_history_days: int = 252,
        liquidity_lookback_days: int = 63,
        min_avg_vol_cr: float | None = None,
    ) -> pd.DataFrame:
        """
        Return a date x ticker boolean mask for point-in-time universe membership.

        This is the vectorized form of get_universe(...) and is used by the
        feature store / training paths to avoid survivorship bias.
        """
        if price_matrix.empty:
            return pd.DataFrame(dtype=bool)
        if self._historical_enabled:
            return self._historical_membership_mask(
                price_matrix=price_matrix,
                volume_matrix=volume_matrix,
                cap_filter=cap_filter,
                min_history_days=min_history_days,
                liquidity_lookback_days=liquidity_lookback_days,
                min_avg_vol_cr=min_avg_vol_cr,
            )

        prices = price_matrix.sort_index()
        tickers = [
            sm.ticker
            for sm in self._stock_meta
            if not sm.blacklisted
            and sm.ticker in prices.columns
            and (cap_filter is None or sm.cap in cap_filter)
        ]
        if not tickers:
            return pd.DataFrame(index=prices.index, dtype=bool)

        prices = prices[tickers]
        membership = prices.notna().cumsum().ge(min_history_days)

        if volume_matrix is not None and not volume_matrix.empty:
            if min_avg_vol_cr is None:
                min_avg_vol_cr = self.cfg["universe"].get("min_avg_volume_cr", 1.0)
            volumes = volume_matrix.reindex(index=prices.index, columns=tickers)
            avg_vol = volumes.rolling(liquidity_lookback_days, min_periods=1).mean()
            liquid = avg_vol.isna() | avg_vol.ge(min_avg_vol_cr)
            membership &= liquid

        for sm in self._stock_meta:
            if sm.ticker not in membership.columns:
                continue
            if sm.listed_since:
                membership.loc[membership.index < pd.Timestamp(sm.listed_since), sm.ticker] = False
            if sm.delisted_on:
                membership.loc[membership.index >= pd.Timestamp(sm.delisted_on), sm.ticker] = False

        return membership.fillna(False)

    def _historical_membership_mask(
        self,
        price_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame | None = None,
        cap_filter: list[str] | None = None,
        min_history_days: int = 252,
        liquidity_lookback_days: int = 63,
        min_avg_vol_cr: float | None = None,
    ) -> pd.DataFrame:
        union_df = self._historical_store.union_df
        if union_df.empty:
            return pd.DataFrame(index=price_matrix.index, dtype=bool)

        prices = price_matrix.sort_index()
        tickers = [t for t in union_df["ticker"].tolist() if t in prices.columns]
        if not tickers:
            return pd.DataFrame(index=prices.index, dtype=bool)
        if min_avg_vol_cr is None:
            min_avg_vol_cr = float(self._hu_cfg.get("min_median_traded_value_cr", 2.0))
        use_active_window = bool(self._hu_cfg.get("use_active_window_filter", True))

        prices = prices[tickers]
        membership = prices.notna().cumsum().ge(min_history_days)

        if volume_matrix is not None and not volume_matrix.empty:
            vols = volume_matrix.reindex(index=prices.index, columns=tickers)
            med_vol = vols.rolling(liquidity_lookback_days, min_periods=1).median()
            liquid = med_vol.isna() | med_vol.ge(min_avg_vol_cr)
            membership &= liquid

        meta = union_df.set_index("ticker")
        for ticker in tickers:
            if ticker not in membership.columns or ticker not in meta.index:
                continue
            if "added_on" in meta.columns and pd.notna(meta.loc[ticker, "added_on"]):
                membership.loc[membership.index < pd.Timestamp(meta.loc[ticker, "added_on"]), ticker] = False
            if use_active_window:
                if "active_from" in meta.columns and pd.notna(meta.loc[ticker, "active_from"]):
                    membership.loc[membership.index < pd.Timestamp(meta.loc[ticker, "active_from"]), ticker] = False
                if "active_to" in meta.columns and pd.notna(meta.loc[ticker, "active_to"]):
                    membership.loc[membership.index >= pd.Timestamp(meta.loc[ticker, "active_to"]), ticker] = False

        # Apply cap filter only if requested.
        if cap_filter is not None:
            cap_map = {s.ticker: s.cap for s in self._stock_meta}
            for ticker in tickers:
                if cap_map.get(ticker) not in cap_filter:
                    membership.loc[:, ticker] = False

        return membership.fillna(False)


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
