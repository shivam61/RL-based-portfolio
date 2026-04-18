"""
Quarterly earnings / fundamentals data for Indian equities.

Downloads via yfinance quarterly_financials. Coverage is typically
4–5 years back for large-caps; older periods return NaN (ranker falls
back to price momentum).

Outputs a daily forward-filled panel:
  data/processed/earnings_panel.parquet  (date × ticker columns)

Features produced (all lagged 1 quarter = 63 trading days):
  rev_growth_yoy       — revenue vs same quarter 1 year ago
  earnings_growth_yoy  — net income vs same quarter 1 year ago
  ebitda_margin        — EBITDA / Revenue (level, not growth)
  ebitda_margin_chg    — EBITDA margin vs same quarter 1 year ago
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EARNINGS_LAG_DAYS = 63   # 1 quarter reporting delay
_RATE_LIMIT_SEC   = 0.3   # pause between yfinance calls


def download_earnings(
    tickers: list[str],
    save_dir: Path,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download quarterly financials for each ticker and cache as parquet.
    Returns {ticker: raw_quarterly_df}.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return {}

    save_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    ok = err = skip = 0

    for ticker in tickers:
        path = save_dir / f"{ticker.replace('/', '_')}.parquet"
        if path.exists() and not force:
            try:
                results[ticker] = pd.read_parquet(path)
                skip += 1
                continue
            except Exception:
                pass

        try:
            t = yf.Ticker(ticker)
            qf = t.quarterly_financials
            if qf is None or qf.empty:
                err += 1
                continue

            df = qf.T.copy()
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            df.to_parquet(path)
            results[ticker] = df
            ok += 1
        except Exception as e:
            logger.debug("Earnings download failed for %s: %s", ticker, e)
            err += 1

        time.sleep(_RATE_LIMIT_SEC)

    logger.info(
        "Earnings download: %d ok  %d skipped (cached)  %d failed  / %d total",
        ok, skip, err, len(tickers),
    )
    return results


def _safe_growth(series: pd.Series) -> pd.Series:
    """YoY growth: compare each quarter to the one 4 periods prior."""
    shifted = series.shift(4)
    denom = shifted.abs().replace(0, np.nan)
    return (series - shifted) / denom


def build_earnings_panel(
    tickers: list[str],
    price_index: pd.DatetimeIndex,
    raw_dir: Path,
    lag_days: int = _EARNINGS_LAG_DAYS,
) -> pd.DataFrame:
    """
    Build a daily forward-filled panel of earnings features.

    Returns wide DataFrame: index=date, columns=MultiIndex(feature, ticker).
    Caller should call .stack(level=1) to get long format if needed.
    """
    feature_frames: dict[str, dict[str, pd.Series]] = {
        "rev_growth_yoy": {},
        "earnings_growth_yoy": {},
        "ebitda_margin": {},
        "ebitda_margin_chg": {},
    }

    for ticker in tickers:
        path = raw_dir / f"{ticker.replace('/', '_')}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue

        # Normalise column names (yfinance returns verbose names)
        cols_lower = {c: c.lower().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=cols_lower)

        # Revenue
        rev_col = next((c for c in df.columns if "total_revenue" in c or "revenue" == c), None)
        if rev_col:
            rev = df[rev_col].dropna()
            feature_frames["rev_growth_yoy"][ticker] = _safe_growth(rev)

        # Net income
        ni_col = next(
            (c for c in df.columns if c in ("net_income", "net_income_common_stockholders")),
            None,
        )
        if ni_col:
            ni = df[ni_col].dropna()
            feature_frames["earnings_growth_yoy"][ticker] = _safe_growth(ni)

        # EBITDA margin
        ebitda_col = next((c for c in df.columns if "ebitda" in c), None)
        if ebitda_col and rev_col:
            ebitda = df[ebitda_col].dropna()
            rev_align = df[rev_col].reindex(ebitda.index).replace(0, np.nan)
            margin = ebitda / rev_align
            feature_frames["ebitda_margin"][ticker] = margin
            feature_frames["ebitda_margin_chg"][ticker] = _safe_growth(margin)

    # ── Reindex each quarterly series to daily and forward-fill ──────────────
    panel_parts: list[pd.DataFrame] = []

    for feat_name, ticker_series in feature_frames.items():
        if not ticker_series:
            continue
        wide = pd.DataFrame(ticker_series)
        wide = wide.reindex(price_index, method="ffill")
        wide = wide.shift(lag_days)    # point-in-time lag
        wide.columns = pd.MultiIndex.from_product([[feat_name], wide.columns])
        panel_parts.append(wide)

    if not panel_parts:
        logger.warning("No earnings data built — all tickers missing or failed")
        return pd.DataFrame(index=price_index)

    panel = pd.concat(panel_parts, axis=1)
    panel.index.name = "date"
    return panel


def save_earnings_panel(panel: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path)
    logger.info("Earnings panel saved: %s  shape=%s", out_path, panel.shape)


def load_earnings_panel(out_path: Path) -> pd.DataFrame | None:
    if not out_path.exists():
        return None
    try:
        return pd.read_parquet(out_path)
    except Exception as e:
        logger.warning("Could not load earnings panel: %s", e)
        return None
