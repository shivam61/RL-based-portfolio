"""
Historical data ingestion via yfinance.

Point-in-time safe: all joins are done on date index.
No lookahead: callers must slice data using only dates ≤ rebalance_date.
Storage: parquet per ticker in data/raw/.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
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


# ── Parquet helpers ───────────────────────────────────────────────────────────

def _raw_path(cfg: dict, ticker: str) -> Path:
    p = Path(cfg["paths"]["raw_data"]) / "equity"
    p.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace(".", "_").replace("^", "CARET_").replace("=", "EQ_")
    return p / f"{safe}.parquet"


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=True, engine="pyarrow")
    logger.debug("Saved %d rows → %s", len(df), path)


def _load_parquet(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_parquet(path, engine="pyarrow")
    return None


# ── Single ticker download ────────────────────────────────────────────────────

def download_ticker(
    ticker: str,
    start: str,
    end: str,
    cfg: dict,
    force: bool = False,
    retry: int = 3,
) -> Optional[pd.DataFrame]:
    """Download OHLCV for one ticker, cache as parquet, return DataFrame."""
    path = _raw_path(cfg, ticker)

    cached: Optional[pd.DataFrame] = None
    if not force and path.exists():
        cached = _load_parquet(path)
        if cached is not None and not cached.empty:
            last = cached.index[-1].date() if hasattr(cached.index[-1], "date") else cached.index[-1]
            # return cache if already up-to-date (within 1 trading day)
            end_dt = datetime.strptime(end, "%Y-%m-%d").date()
            if (end_dt - last).days <= 1:
                logger.debug("Cache up-to-date for %s (last=%s)", ticker, last)
                return cached
            start = str(last + timedelta(days=1))

    if not HAS_YFINANCE:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return cached  # return whatever we have

    for attempt in range(retry):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                if cached is not None and not cached.empty:
                    logger.debug("No new data for %s; using cache", ticker)
                    return cached
                logger.warning("No data for %s", ticker)
                return None

            # flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw.index = pd.to_datetime(raw.index).normalize()
            raw.index.name = "date"

            # rename to standard schema
            raw = raw.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
                "Adj Close": "adj_close",
            })
            if "adj_close" not in raw.columns:
                raw["adj_close"] = raw["close"]

            raw["ticker"] = ticker
            raw = raw[["open", "high", "low", "close", "adj_close", "volume", "ticker"]]
            raw = raw.dropna(subset=["close"])

            # merge with existing cache
            existing = _load_parquet(path)
            if existing is not None and not existing.empty:
                combined = pd.concat([existing, raw])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined.sort_index(inplace=True)
                raw = combined

            _save_parquet(raw, path)
            return raw

        except Exception as exc:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, ticker, exc)
            time.sleep(2 ** attempt)

    # all retries exhausted — return cached data if available
    if cached is not None and not cached.empty:
        logger.warning("Download failed for %s; using cached data", ticker)
        return cached

    return None


# ── Bulk download ─────────────────────────────────────────────────────────────

def download_universe(
    cfg: dict | None = None,
    force: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download all tickers in universe.yaml + global proxies."""
    if cfg is None:
        cfg = load_config()
    uni = load_universe_config()

    start = start or cfg["backtest"]["start_date"]
    # add 6-month buffer before start for feature calculation warm-up
    start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=365)
    start = start_dt.strftime("%Y-%m-%d")
    end = end or cfg["backtest"]["end_date"]

    tickers: list[str] = [s["ticker"] for s in uni["stocks"]]
    proxies: list[str] = list(uni.get("global_proxies", {}).values())
    all_tickers = list(dict.fromkeys(tickers + proxies))  # dedup, preserve order

    results: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for i, ticker in enumerate(all_tickers):
        logger.info("[%d/%d] Downloading %s ...", i + 1, len(all_tickers), ticker)
        df = download_ticker(ticker, start, end, cfg, force=force)
        if df is not None and not df.empty:
            results[ticker] = df
        else:
            failed.append(ticker)
        # polite delay to avoid rate limiting
        time.sleep(0.3)

    logger.info(
        "Download complete: %d succeeded, %d failed. Failed: %s",
        len(results), len(failed), failed,
    )
    return results


# ── Processed price matrix ────────────────────────────────────────────────────

def build_price_matrix(
    raw_data: dict[str, pd.DataFrame],
    price_col: str = "adj_close",
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Pivot raw per-ticker DataFrames into (date × ticker) price matrix."""
    if cfg is None:
        cfg = load_config()

    frames = {}
    for ticker, df in raw_data.items():
        if price_col in df.columns:
            frames[ticker] = df[price_col].rename(ticker)

    if not frames:
        raise ValueError("No data available to build price matrix")

    price_matrix = pd.DataFrame(frames)
    price_matrix.index = pd.to_datetime(price_matrix.index).normalize()
    price_matrix.sort_index(inplace=True)
    price_matrix.index.name = "date"

    out_path = Path(cfg["paths"]["processed_data"]) / "price_matrix.parquet"
    price_matrix.to_parquet(out_path, engine="pyarrow")
    logger.info("Price matrix: %s → %s", price_matrix.shape, out_path)
    return price_matrix


def build_volume_matrix(
    raw_data: dict[str, pd.DataFrame],
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Build (date × ticker) volume matrix in INR crore."""
    if cfg is None:
        cfg = load_config()

    frames = {}
    for ticker, df in raw_data.items():
        if "volume" in df.columns and "close" in df.columns:
            value = (df["volume"] * df["close"]) / 1e7  # INR crore
            frames[ticker] = value.rename(ticker)

    volume_matrix = pd.DataFrame(frames)
    volume_matrix.index = pd.to_datetime(volume_matrix.index).normalize()
    volume_matrix.sort_index(inplace=True)
    volume_matrix.index.name = "date"

    out_path = Path(cfg["paths"]["processed_data"]) / "volume_matrix.parquet"
    volume_matrix.to_parquet(out_path, engine="pyarrow")
    return volume_matrix


# ── Load cached data ──────────────────────────────────────────────────────────

def load_price_matrix(cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    path = Path(cfg["paths"]["processed_data"]) / "price_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Price matrix not found at {path}. Run download_data.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    df.index = pd.to_datetime(df.index).normalize()
    return df


def load_volume_matrix(cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    path = Path(cfg["paths"]["processed_data"]) / "volume_matrix.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Volume matrix not found at {path}. Run download_data.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    df.index = pd.to_datetime(df.index).normalize()
    return df


def load_raw_ticker(ticker: str, cfg: dict | None = None) -> Optional[pd.DataFrame]:
    if cfg is None:
        cfg = load_config()
    path = _raw_path(cfg, ticker)
    return _load_parquet(path)
