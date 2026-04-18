"""
Macro & global signal data layer.

Downloads global market proxies via yfinance and assembles a single
point-in-time macro DataFrame. India-specific rates (RBI, CPI) are
approximated using public proxy series or hardcoded schedules where
free APIs are unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    yf = None  # type: ignore
    HAS_YFINANCE = False

from src.config import load_config, load_universe_config

logger = logging.getLogger(__name__)


# ── RBI policy rate schedule ──────────────────────────────────────────────────
# Approximate repo rate history (manually curated from RBI public records)
_RBI_REPO_RATE: list[tuple[str, float]] = [
    ("2012-01-01", 8.50),
    ("2012-04-17", 8.00),
    ("2012-10-29", 7.75),
    ("2013-05-03", 7.25),
    ("2013-09-20", 7.50),
    ("2014-01-28", 8.00),
    ("2014-06-03", 8.00),
    ("2015-01-15", 7.75),
    ("2015-03-04", 7.50),
    ("2015-06-02", 7.25),
    ("2015-09-29", 6.75),
    ("2016-04-05", 6.50),
    ("2016-10-04", 6.25),
    ("2017-08-02", 6.00),
    ("2018-06-06", 6.25),
    ("2018-08-01", 6.50),
    ("2019-02-07", 6.25),
    ("2019-04-04", 6.00),
    ("2019-06-06", 5.75),
    ("2019-08-07", 5.40),
    ("2019-10-04", 5.15),
    ("2020-03-27", 4.40),
    ("2020-05-22", 4.00),
    ("2022-05-04", 4.40),
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2025-02-07", 6.25),
    ("2025-04-09", 6.00),
]

# ── RBI meeting dates ─────────────────────────────────────────────────────────
_RBI_MEETING_DATES: list[str] = [
    "2013-01-29", "2013-03-19", "2013-05-03", "2013-06-17",
    "2013-07-30", "2013-09-20", "2013-10-29", "2013-12-18",
    "2014-01-28", "2014-03-06", "2014-04-01", "2014-06-03",
    "2014-07-31", "2014-09-30", "2014-10-28", "2014-12-02",
    "2015-02-03", "2015-04-07", "2015-06-02", "2015-08-04",
    "2015-09-29", "2015-12-01",
    "2016-02-02", "2016-04-05", "2016-06-07", "2016-08-09",
    "2016-10-04", "2016-12-07",
    "2017-02-08", "2017-04-06", "2017-06-07", "2017-08-02",
    "2017-10-04", "2017-12-06",
    "2018-02-07", "2018-04-05", "2018-06-06", "2018-08-01",
    "2018-10-05", "2018-12-05",
    "2019-02-07", "2019-04-04", "2019-06-06", "2019-08-07",
    "2019-10-04", "2019-12-05",
    "2020-02-06", "2020-03-27", "2020-05-22", "2020-08-06",
    "2020-10-09", "2020-12-04",
    "2021-02-05", "2021-04-07", "2021-06-04", "2021-08-06",
    "2021-10-08", "2021-12-08",
    "2022-02-09", "2022-04-08", "2022-05-04", "2022-06-08",
    "2022-08-05", "2022-09-30", "2022-12-07",
    "2023-02-08", "2023-04-06", "2023-06-08", "2023-08-10",
    "2023-10-06", "2023-12-08",
    "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08",
    "2024-10-09", "2024-12-06",
    "2025-02-07", "2025-04-09", "2025-06-06",
]

# ── Indian Budget dates ───────────────────────────────────────────────────────
_BUDGET_DATES: list[str] = [
    "2013-02-28", "2014-07-10", "2015-02-28", "2016-02-29",
    "2017-02-01", "2018-02-01", "2019-02-01", "2020-02-01",
    "2021-02-01", "2022-02-01", "2023-02-01", "2024-02-01",
    "2025-02-01", "2026-02-01",
]

# ── Indian election dates ─────────────────────────────────────────────────────
_ELECTION_WINDOWS: list[tuple[str, str]] = [
    ("2014-04-07", "2014-05-12"),   # general election
    ("2019-04-11", "2019-05-19"),   # general election
    ("2024-04-19", "2024-06-01"),   # general election
]


class MacroDataManager:
    """Assembles and provides macro/global signals."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self._uni_cfg = load_universe_config()
        self._macro_df: Optional[pd.DataFrame] = None

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        start: str | None = None,
        end: str | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        out_path = Path(self.cfg["paths"]["processed_data"]) / "macro.parquet"
        if not force and out_path.exists():
            self._macro_df = pd.read_parquet(out_path)
            self._macro_df.index = pd.to_datetime(self._macro_df.index).normalize()
            logger.info("Loaded cached macro data (%d rows)", len(self._macro_df))
            return self._macro_df

        start = start or self.cfg["backtest"]["start_date"]
        end = end or self.cfg["backtest"]["end_date"]
        proxies = self._uni_cfg.get("global_proxies", {})

        # download global proxies
        price_dfs: dict[str, pd.Series] = {}
        if not HAS_YFINANCE:
            logger.warning("yfinance not available; macro proxy download skipped")
        for name, ticker in proxies.items():
            if not HAS_YFINANCE:
                continue
            try:
                raw = yf.download(
                    ticker, start=start, end=end,
                    auto_adjust=True, progress=False, threads=False,
                )
                if raw is None or raw.empty:
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                col = "Close" if "Close" in raw.columns else raw.columns[0]
                s = raw[col].rename(name)
                s.index = pd.to_datetime(s.index).normalize()
                price_dfs[name] = s
                logger.debug("Downloaded macro proxy: %s (%s)", name, ticker)
            except Exception as e:
                logger.warning("Failed to download %s (%s): %s", name, ticker, e)

        macro = pd.DataFrame(price_dfs)
        macro.index.name = "date"
        macro.sort_index(inplace=True)

        # forward-fill gaps (e.g., weekends, holidays)
        macro = macro.ffill()

        # add RBI repo rate (step function)
        macro["rbi_repo_rate"] = self._build_rbi_rate_series(macro.index)

        # add RBI meeting dummy
        macro["rbi_meeting"] = self._build_event_dummy(macro.index, _RBI_MEETING_DATES)

        # add budget dummy
        macro["budget_day"] = self._build_event_dummy(macro.index, _BUDGET_DATES)

        # add election window dummy
        macro["election_window"] = self._build_election_dummy(macro.index)

        # derived signals
        if "usdinr" in macro.columns:
            macro["usdinr_ret_1m"] = macro["usdinr"].pct_change(21)
            macro["usdinr_ret_3m"] = macro["usdinr"].pct_change(63)

        if "crude_oil" in macro.columns:
            macro["crude_ret_1m"] = macro["crude_oil"].pct_change(21)
            macro["crude_ret_3m"] = macro["crude_oil"].pct_change(63)

        if "vix" in macro.columns:
            macro["vix_pctile_1y"] = (
                macro["vix"]
                .rolling(252)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            )

        if "india_vix" in macro.columns:
            macro["india_vix_ret_1m"] = macro["india_vix"].pct_change(21)
            macro["india_vix_pctile_1y"] = (
                macro["india_vix"]
                .rolling(252)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            )
            # India VIX is more relevant than US VIX for Indian equities; prefer it
            macro["vix_pctile_1y"] = macro["india_vix_pctile_1y"].combine_first(
                macro.get("vix_pctile_1y", pd.Series(dtype=float))
            )

        if "sp500" in macro.columns:
            macro["sp500_ret_1m"] = macro["sp500"].pct_change(21)
            macro["sp500_ret_3m"] = macro["sp500"].pct_change(63)
            macro["sp500_above_200ma"] = (
                macro["sp500"] > macro["sp500"].rolling(200).mean()
            ).astype(float)

        if "us_10y" in macro.columns and "us_2y" in macro.columns:
            macro["yield_curve"] = macro["us_10y"] - macro["us_2y"]

        macro.to_parquet(out_path, engine="pyarrow")
        self._macro_df = macro
        logger.info("Macro data built: %s rows, %d cols → %s",
                    len(macro), macro.shape[1], out_path)
        return macro

    def get_macro_as_of(self, as_of: pd.Timestamp | str) -> pd.Series:
        """Return the macro row valid as of `as_of` (no lookahead)."""
        if self._macro_df is None:
            self.build()
        ts = pd.Timestamp(as_of)
        hist = self._macro_df[self._macro_df.index <= ts]
        if hist.empty:
            raise ValueError(f"No macro data available before {as_of}")
        return hist.iloc[-1]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_rbi_rate_series(index: pd.DatetimeIndex) -> pd.Series:
        rate_df = pd.DataFrame(_RBI_REPO_RATE, columns=["date", "rate"])
        rate_df["date"] = pd.to_datetime(rate_df["date"])
        rate_df = rate_df.set_index("date").reindex(index, method="ffill")
        return rate_df["rate"]

    @staticmethod
    def _build_event_dummy(
        index: pd.DatetimeIndex, dates: list[str], window: int = 3
    ) -> pd.Series:
        ts_dates = pd.to_datetime(dates)
        s = pd.Series(0.0, index=index)
        for d in ts_dates:
            mask = (index >= d) & (index <= d + pd.Timedelta(days=window))
            s[mask] = 1.0
        return s

    @staticmethod
    def _build_election_dummy(index: pd.DatetimeIndex) -> pd.Series:
        s = pd.Series(0.0, index=index)
        for start_str, end_str in _ELECTION_WINDOWS:
            start_ts = pd.Timestamp(start_str)
            end_ts = pd.Timestamp(end_str)
            mask = (index >= start_ts) & (index <= end_ts)
            s[mask] = 1.0
        return s

    def load(self) -> pd.DataFrame:
        """Load cached macro data without rebuilding."""
        out_path = Path(self.cfg["paths"]["processed_data"]) / "macro.parquet"
        if out_path.exists():
            self._macro_df = pd.read_parquet(out_path)
            self._macro_df.index = pd.to_datetime(self._macro_df.index).normalize()
            return self._macro_df
        return self.build()
