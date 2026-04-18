"""
Screener.in quarterly financials scraper.

Provides historical quarterly P&L going back as far as Screener.in holds
(typically 12–16 quarters for most Indian listed companies).

Handles three income-statement formats:
  - Regular companies   (Sales, Operating Profit, OPM %, Net Profit, EPS)
  - Banks               (Revenue, Financing Profit, Financing Margin %)
  - NBFCs / Insurers    (Revenue, similar to banks)

Temporal alignment (CRITICAL — no lookahead):
  BSE Regulation 33 mandates quarterly results within 45 calendar days of
  quarter-end (Q4 within 60 days). We conservatively set:
      available_from_date = quarter_end_date + RESULT_LAG_DAYS
  Default lag = 46 calendar days (just outside the deadline).

Usage:
    scraper = ScreenerScraper(cfg)
    df = scraper.scrape("INFY")      # NSE symbol WITHOUT .NS suffix
    panel = build_screener_panel(tickers, price_index, raw_dir)
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_BASE_URL     = "https://www.screener.in"
_RATE_LIMIT   = 1.2          # seconds between requests (be respectful)
_MAX_RETRIES  = 3
_TIMEOUT      = 20
_RESULT_LAG_Q1Q2Q3 = 46     # calendar days after quarter end (BSE Reg 33)
_RESULT_LAG_Q4     = 61     # Q4 / annual results have 60-day deadline

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.screener.in/",
}

# ── NSE symbol → Screener.in symbol overrides ────────────────────────────────
# Required for mergers, demergers, and renames where the current NSE ticker
# differs from what Screener.in uses internally.
#
# LTIM (LTIMindtree): formed Nov 2022 via merger of LTI + Mindtree.
#   Screener.in retains the pre-merger LTI page with history stitched forward.
#   Pre-merger quarters = LTI standalone; post-merger = combined entity.
#
# Add new entries here whenever a universe ticker gets a corporate action
# that changes its NSE symbol but not its Screener.in slug.
_SCREENER_SYMBOL_OVERRIDES: dict[str, str] = {
    # LTIMindtree: merged Nov 2022 (LTI + Mindtree). Screener.in consolidated
    # page returns 404 for both LTI and LTIM — standalone page is available.
    # Handled via consolidated=False fallback in scrape(); no override needed.
    # Pre-merger quarters will be LTI standalone; post-merger is stitched forward.
    "BAJAJ-AUTO": "BAJAJ-AUTO",  # hyphen is valid on Screener.in
    "M&M": "M-M",   # ampersand not valid in URL; Screener uses M-M
}

# Tickers with known data gaps on Screener.in — will use price-only features.
# Document reason so future maintainers know these aren't bugs.
_KNOWN_MISSING: dict[str, str] = {
    "LTIM": (
        "LTIMindtree (merged Nov 2022 from LTI + Mindtree). "
        "Screener.in consolidated page unavailable; standalone has limited history. "
        "Pre-merger LTI data not stitched. Will use price-only features."
    ),
    "ALKYLAMINE": (
        "Alkyl Amines Chemicals. Screener.in only holds 8 quarters up to Mar 2020 "
        "for this mid-cap. Post-2020 data unavailable via scraper. "
        "Will use price-only features for post-2020 periods."
    ),
}

# Row-label → canonical column name mapping.
# Order matters: first match wins when multiple aliases exist.
_ROW_ALIASES: dict[str, str] = {
    # Revenue / top-line
    "sales":                "revenue",
    "revenue":              "revenue",
    "net sales":            "revenue",
    "total income":         "revenue",
    "interest earned":      "revenue",   # older bank format
    # Operating profit
    "operating profit":     "op_profit",
    "financing profit":     "op_profit", # banks
    "net interest income":  "op_profit", # some NBFCs
    # Operating margin
    "opm %":                "opm",
    "financing margin %":   "opm",       # banks
    "nim %":                "opm",
    # Net profit
    "net profit":           "net_profit",
    "profit after tax":     "net_profit",
    "pat":                  "net_profit",
    # EPS
    "eps in rs":            "eps",
    "eps":                  "eps",
    "diluted eps":          "eps",
}


# ── Date helpers ──────────────────────────────────────────────────────────────

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_screener_date(label: str) -> pd.Timestamp | None:
    """Convert 'Jun 2013' → last day of that month."""
    label = label.strip()
    m = re.match(r"([A-Za-z]{3})\s+(\d{4})", label)
    if not m:
        return None
    mon = _MONTH_MAP.get(m.group(1).lower())
    yr  = int(m.group(2))
    if mon is None:
        return None
    return pd.Timestamp(yr, mon, 1) + pd.offsets.MonthEnd(0)


def _available_from(quarter_end: pd.Timestamp) -> pd.Timestamp:
    """Return the first date a quarterly result can be legitimately used."""
    lag = _RESULT_LAG_Q4 if quarter_end.month == 3 else _RESULT_LAG_Q1Q2Q3
    return quarter_end + pd.Timedelta(days=lag)


# ── Number parser ─────────────────────────────────────────────────────────────

def _parse_number(text: str) -> float | None:
    """Parse '38,318' or '9.45%' or '-1,234' → float. Returns None if blank."""
    text = text.replace(",", "").replace("%", "").strip()
    if not text or text in ("-", "—", ""):
        return None
    try:
        return float(text)
    except ValueError:
        return None


# ── Core scraper ──────────────────────────────────────────────────────────────

class ScreenerScraper:
    """
    Scrapes quarterly results from Screener.in for one ticker at a time.
    Maintains a single requests.Session for connection reuse.
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        self._sess = session or requests.Session()
        self._sess.headers.update(_HEADERS)
        self._last_request_ts: float = 0.0

    def _get(self, url: str) -> requests.Response:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < _RATE_LIMIT:
            time.sleep(_RATE_LIMIT - elapsed)
        for attempt in range(_MAX_RETRIES):
            try:
                r = self._sess.get(url, timeout=_TIMEOUT)
                self._last_request_ts = time.monotonic()
                if r.status_code == 200:
                    return r
                if r.status_code == 429:
                    wait = 10 * (attempt + 1)
                    logger.warning("Rate-limited by Screener.in, sleeping %ds", wait)
                    time.sleep(wait)
                    continue
                logger.debug("HTTP %d for %s", r.status_code, url)
                return r
            except requests.RequestException as e:
                logger.debug("Request error (%s): %s", url, e)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Failed to fetch {url} after {_MAX_RETRIES} attempts")

    def scrape(self, symbol: str, consolidated: bool = True) -> pd.DataFrame | None:
        """
        Scrape quarterly results for a Screener.in symbol.

        Parameters
        ----------
        symbol : str
            NSE symbol WITHOUT the '.NS' suffix (e.g. 'INFY', 'HDFCBANK').
        consolidated : bool
            Try consolidated first; fall back to standalone if not found.

        Returns
        -------
        pd.DataFrame with columns:
            quarter_end, available_from, revenue, op_profit, opm,
            net_profit, eps
        Returns None if the page is unreachable or table not found.
        """
        # Apply symbol override if present (mergers, renames, URL-unsafe chars)
        screener_symbol = _SCREENER_SYMBOL_OVERRIDES.get(symbol, symbol)
        suffix = "consolidated" if consolidated else ""
        url = f"{_BASE_URL}/company/{screener_symbol}/{suffix}/"
        try:
            resp = self._get(url)
        except RuntimeError as e:
            logger.warning("Screener fetch failed for %s: %s", symbol, e)
            return None

        if resp.status_code == 404:
            if consolidated:
                logger.debug("%s: consolidated not found, trying standalone", screener_symbol)
                return self.scrape(symbol, consolidated=False)
            logger.warning("%s (screener slug: %s): not found on Screener.in (404)",
                           symbol, screener_symbol)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        df = self._parse_quarterly_section(soup, symbol)
        if df is None and consolidated:
            logger.debug("%s: no quarterly table in consolidated, trying standalone", symbol)
            return self.scrape(symbol, consolidated=False)
        return df

    def _parse_quarterly_section(
        self,
        soup: BeautifulSoup,
        symbol: str,
    ) -> pd.DataFrame | None:
        """Locate and parse the #quarters section."""
        section = soup.find("section", {"id": "quarters"})
        if section is None:
            logger.debug("%s: no #quarters section found", symbol)
            return None

        tbl = section.find("table")
        if tbl is None:
            logger.debug("%s: no table in #quarters", symbol)
            return None

        thead = tbl.find("thead")
        tbody = tbl.find("tbody")
        if not thead or not tbody:
            return None

        # ── Column headers (dates) ────────────────────────────────────────────
        ths = [th.get_text(strip=True) for th in thead.find_all("th")]
        # ths[0] is always the empty label column; ths[1:] are date strings
        date_strs = ths[1:]
        dates = []
        for ds in date_strs:
            ts = _parse_screener_date(ds)
            if ts is not None:
                dates.append(ts)

        if not dates:
            logger.debug("%s: no parseable date columns", symbol)
            return None

        # ── Row data ──────────────────────────────────────────────────────────
        raw: dict[str, list[float | None]] = {}
        for tr in tbody.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            label = tds[0].get_text(strip=True).lower().rstrip("+").strip()
            # Normalise label
            canonical = _ROW_ALIASES.get(label)
            if canonical is None:
                continue
            values = [_parse_number(td.get_text(strip=True)) for td in tds[1:]]
            # Align to date count
            values = values[: len(dates)]
            while len(values) < len(dates):
                values.append(None)
            # Don't overwrite if already captured (first alias wins)
            if canonical not in raw:
                raw[canonical] = values

        if not raw:
            logger.debug("%s: no recognised rows in quarterly table", symbol)
            return None

        # ── Assemble DataFrame ────────────────────────────────────────────────
        df = pd.DataFrame(raw, index=dates)
        df.index.name = "quarter_end"
        df = df.sort_index()

        # Add available_from_date (CRITICAL for leakage prevention)
        df["available_from"] = df.index.map(_available_from)

        # Ensure canonical columns present (fill missing with NaN)
        for col in ["revenue", "op_profit", "opm", "net_profit", "eps"]:
            if col not in df.columns:
                df[col] = np.nan

        logger.debug(
            "%s: scraped %d quarters (%s → %s)",
            symbol, len(df),
            df.index.min().date(), df.index.max().date(),
        )
        return df


# ── Batch downloader ──────────────────────────────────────────────────────────

def download_screener(
    tickers: list[str],
    save_dir: Path,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download quarterly financials for all tickers from Screener.in.
    Ticker format: NSE symbol with '.NS' suffix (stripped internally).
    Results cached as parquet in save_dir.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    scraper = ScreenerScraper()
    results: dict[str, pd.DataFrame] = {}
    ok = cached = failed = 0

    for ticker in tickers:
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        path = save_dir / f"{symbol}.parquet"

        if path.exists() and not force:
            try:
                results[ticker] = pd.read_parquet(path)
                cached += 1
                continue
            except Exception:
                pass

        df = scraper.scrape(symbol)
        if df is not None and not df.empty:
            df.to_parquet(path)
            results[ticker] = df
            ok += 1
            logger.info(
                "Screener [%s]: %d quarters  %s→%s",
                symbol, len(df),
                df.index.min().date(), df.index.max().date(),
            )
        else:
            failed += 1
            logger.warning("Screener [%s]: no data", symbol)

    logger.info(
        "Screener download: %d ok  %d cached  %d failed  / %d total",
        ok, cached, failed, len(tickers),
    )
    return results


# ── Panel builder ─────────────────────────────────────────────────────────────

def _yoy_growth(s: pd.Series, winsor: float = 5.0) -> pd.Series:
    """YoY = (Q - Q_4_periods_ago) / abs(Q_4_periods_ago), winsorised."""
    shifted = s.shift(4)
    denom = shifted.abs().replace(0, np.nan)
    return ((s - shifted) / denom).clip(-winsor, winsor)


def build_screener_panel(
    tickers: list[str],
    price_index: pd.DatetimeIndex,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Build a daily forward-filled feature panel from cached Screener.in data.

    Temporal rule (CRITICAL):
      Each quarterly value is forward-filled only from its `available_from`
      date (quarter_end + lag), never before. This prevents lookahead.

    Features produced per ticker:
      rev_growth_yoy       — Revenue YoY growth (winsorised ±5×)
      profit_growth_yoy    — Net profit YoY growth (winsorised ±5×)
      opm_level            — Operating margin % (level)
      opm_change_yoy       — OPM change vs same quarter 1 year ago (ppts)
      eps_growth_yoy       — EPS YoY growth (winsorised ±5×)
      rev_vs_trend         — Revenue vs trailing-4Q avg (surprise proxy)
      profit_vs_trend      — Net profit vs trailing-4Q avg (surprise proxy)

    Returns
    -------
    pd.DataFrame with MultiIndex columns (feature, ticker), index = price_index.
    """
    feat_names = [
        "rev_growth_yoy", "profit_growth_yoy", "opm_level",
        "opm_change_yoy", "eps_growth_yoy",
        "rev_vs_trend", "profit_vs_trend",
    ]
    # {feat_name: {ticker: quarterly_series aligned to price_index}}
    feat_frames: dict[str, dict[str, pd.Series]] = {f: {} for f in feat_names}

    for ticker in tickers:
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        path = raw_dir / f"{symbol}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.debug("Cannot read %s: %s", path, e)
            continue

        if df.empty or "available_from" not in df.columns:
            continue

        df = df.sort_index()

        # ── Compute quarterly features ────────────────────────────────────────
        qfeat: dict[str, pd.Series] = {}

        if "revenue" in df.columns and df["revenue"].notna().sum() >= 2:
            rev = df["revenue"].dropna()
            qfeat["rev_growth_yoy"] = _yoy_growth(rev)
            # Surprise proxy: revenue vs trailing 4Q average
            trend = rev.rolling(4, min_periods=2).mean().shift(1)
            qfeat["rev_vs_trend"] = ((rev - trend) / trend.abs().replace(0, np.nan)).clip(-3, 3)

        if "net_profit" in df.columns and df["net_profit"].notna().sum() >= 2:
            np_ = df["net_profit"].dropna()
            qfeat["profit_growth_yoy"] = _yoy_growth(np_)
            trend = np_.rolling(4, min_periods=2).mean().shift(1)
            qfeat["profit_vs_trend"] = ((np_ - trend) / trend.abs().replace(0, np.nan)).clip(-3, 3)

        if "opm" in df.columns and df["opm"].notna().sum() >= 2:
            opm = df["opm"].dropna()
            qfeat["opm_level"] = opm / 100.0  # convert % to decimal
            qfeat["opm_change_yoy"] = (opm - opm.shift(4)).clip(-50, 50) / 100.0

        if "eps" in df.columns and df["eps"].notna().sum() >= 2:
            eps = df["eps"].dropna()
            qfeat["eps_growth_yoy"] = _yoy_growth(eps)

        if not qfeat:
            continue

        # ── Align quarterly features to daily price index ──────────────────
        # CRITICAL: use available_from as the date on which each value appears.
        # Build a daily series by placing each quarterly value on its
        # available_from date, then forward-filling.
        for feat, q_series in qfeat.items():
            # Reindex quarterly values onto available_from dates
            avail_dates = df["available_from"].reindex(q_series.index).dropna()
            if avail_dates.empty:
                continue
            # Daily series: place value on available_from, NaN elsewhere
            daily = pd.Series(np.nan, index=price_index, dtype=float)
            for q_date, avail_date in avail_dates.items():
                if q_date not in q_series.index:
                    continue
                val = q_series.get(q_date)
                if val is None or np.isnan(val):
                    continue
                # Find the first price_index date >= available_from
                idx_pos = price_index.searchsorted(avail_date)
                if idx_pos >= len(price_index):
                    continue
                actual_date = price_index[idx_pos]
                daily.loc[actual_date] = val
            # Forward-fill (each quarterly value holds until next quarter)
            daily = daily.ffill()
            feat_frames[feat][ticker] = daily

    # ── Assemble wide panel ──────────────────────────────────────────────────
    parts: list[pd.DataFrame] = []
    for feat_name in feat_names:
        td = feat_frames[feat_name]
        if not td:
            continue
        wide = pd.DataFrame(td).reindex(price_index)
        wide.columns = pd.MultiIndex.from_product([[feat_name], wide.columns])
        parts.append(wide)

    if not parts:
        logger.warning("Screener panel: no data for any ticker")
        return pd.DataFrame(index=price_index)

    panel = pd.concat(parts, axis=1)
    panel.index.name = "date"

    n_tickers = len(panel.columns.get_level_values(1).unique())
    logger.info(
        "Screener panel built: %d days × %d features × %d tickers",
        len(panel), len(feat_names), n_tickers,
    )
    return panel
