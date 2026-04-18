#!/usr/bin/env python3
"""
Download quarterly financials for all universe tickers from Screener.in.

Scrapes the #quarters section of each company page (consolidated first,
standalone fallback). Results cached per-ticker in data/raw/screener/.
Then builds the daily forward-filled feature panel.

Usage:
    python scripts/download_screener.py [--force] [--tickers INFY,TCS]

Typical runtime: ~3–5 min for ~100 tickers (rate-limited to 1.2s/ticker).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd

from src.config import load_config, setup_logging
from src.data.earnings import build_earnings_panel, save_earnings_panel
from src.data.screener import download_screener


@click.command()
@click.option("--force", is_flag=True, default=False,
              help="Force re-scrape even if cached")
@click.option("--tickers", default=None,
              help="Comma-separated NSE tickers to scrape (default: full universe)")
@click.option("--config", default=None)
def main(force: bool, tickers: str | None, config: str | None) -> None:
    """Scrape Screener.in and build earnings feature panel."""
    cfg = load_config(config)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    raw_dir       = Path(cfg["paths"]["raw_data"])       / "screener"
    processed_dir = Path(cfg["paths"]["processed_data"])
    panel_path    = processed_dir / "screener_panel.parquet"

    # ── Resolve ticker list ───────────────────────────────────────────────────
    if tickers:
        ticker_list = [t.strip() for t in tickers.split(",")]
        # normalise: add .NS if no suffix
        ticker_list = [
            t if ("." in t) else f"{t}.NS"
            for t in ticker_list
        ]
    else:
        pm_path = processed_dir / "price_matrix.parquet"
        if not pm_path.exists():
            logger.error("Price matrix not found — run download_data.py first")
            sys.exit(1)
        pm = pd.read_parquet(pm_path)
        ticker_list = [
            t for t in pm.columns
            if t.endswith(".NS") or t.endswith(".BO")
        ]

    logger.info("=" * 60)
    logger.info("SCREENER.IN DOWNLOAD")
    logger.info("Tickers: %d", len(ticker_list))
    logger.info("Raw cache: %s", raw_dir)
    logger.info("Force refresh: %s", force)
    logger.info("=" * 60)

    # ── Phase 1: Download raw quarterly data ─────────────────────────────────
    download_screener(ticker_list, raw_dir, force=force)

    # ── Phase 2: Build daily panel ────────────────────────────────────────────
    logger.info("Building daily earnings panel ...")
    pm = pd.read_parquet(processed_dir / "price_matrix.parquet")
    panel = build_earnings_panel(ticker_list, pm.index, raw_dir)

    save_earnings_panel(panel, panel_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    if isinstance(panel.columns, pd.MultiIndex):
        features = panel.columns.get_level_values(0).unique().tolist()
        n_tickers = len(panel.columns.get_level_values(1).unique())
        total_cells = panel.shape[0] * panel.shape[1]
        fill_pct = panel.notna().sum().sum() / total_cells * 100
        logger.info("=" * 60)
        logger.info("SCREENER PANEL BUILT")
        logger.info("Shape:    %s", panel.shape)
        logger.info("Features: %s", features)
        logger.info("Tickers:  %d", n_tickers)
        logger.info("Fill rate (non-NaN): %.1f%%", fill_pct)
        logger.info("Date range: %s → %s",
                    panel.index.min().date(), panel.index.max().date())
        logger.info("=" * 60)
    else:
        logger.warning("Panel built but no data found — check raw_dir or ticker list")


if __name__ == "__main__":
    main()
