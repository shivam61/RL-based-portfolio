"""
Earnings / fundamental features panel.

Primary source : Screener.in (via src.data.screener)
Fallback       : None — yfinance earnings are too sparse for this backtest.

Canonical processed path: data/processed/screener_panel.parquet

Public API
----------
build_earnings_panel(tickers, price_index, raw_dir)  → pd.DataFrame
save_earnings_panel(panel, path)
load_earnings_panel(path)                             → pd.DataFrame | None
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.screener import build_screener_panel

logger = logging.getLogger(__name__)

_DEFAULT_PANEL_NAME = "screener_panel.parquet"


def build_earnings_panel(
    tickers: list[str],
    price_index: pd.DatetimeIndex,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Build the full earnings feature panel from cached Screener.in data.

    Temporal safety: all features respect available_from_date
    (quarter_end + regulatory lag). No lookahead possible.
    """
    return build_screener_panel(tickers, price_index, raw_dir)


def save_earnings_panel(panel: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path)
    logger.info("Earnings panel saved: %s  shape=%s", out_path, panel.shape)


def load_earnings_panel(path: Path) -> pd.DataFrame | None:
    """Load pre-built screener panel. Returns None if file absent."""
    if not path.exists():
        logger.debug("Earnings panel not found at %s", path)
        return None
    try:
        df = pd.read_parquet(path)
        logger.info("Earnings panel loaded: %s  shape=%s", path, df.shape)
        return df
    except Exception as e:
        logger.warning("Could not load earnings panel at %s: %s", path, e)
        return None
