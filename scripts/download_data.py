#!/usr/bin/env python3
"""
CLI: Download historical data for the Indian equity universe.

Usage:
    python scripts/download_data.py [--force] [--start 2013-01-01] [--end 2026-04-17]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from src.config import load_config, setup_logging
from src.data.earnings import build_earnings_panel, save_earnings_panel
from src.data.ingestion import (
    build_price_matrix,
    build_volume_matrix,
    download_universe,
)
from src.data.macro import MacroDataManager


@click.command()
@click.option("--force", is_flag=True, default=False, help="Force re-download even if cached")
@click.option("--start", default=None, help="Override start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="Override end date (YYYY-MM-DD)")
@click.option("--config", default=None, help="Path to custom config file")
@click.option("--macro-only", is_flag=True, default=False, help="Only download macro data")
@click.option("--equity-only", is_flag=True, default=False, help="Only download equity data")
@click.option("--earnings-only", is_flag=True, default=False, help="Only download earnings data")
@click.option("--no-earnings", is_flag=True, default=False, help="Skip earnings download")
def main(force, start, end, config, macro_only, equity_only, earnings_only, no_earnings):
    """Download all historical data needed for the backtest."""
    cfg = load_config(config)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("DATA DOWNLOAD")
    logger.info("Force refresh: %s", force)
    logger.info("=" * 60)

    price_matrix = None
    if not macro_only and not earnings_only:
        logger.info("Step 1/4: Downloading equity universe data ...")
        raw_data = download_universe(cfg=cfg, force=force, start=start, end=end)
        logger.info("Downloaded data for %d tickers", len(raw_data))

        logger.info("Step 2/4: Building price matrix ...")
        price_matrix = build_price_matrix(raw_data, cfg=cfg)
        logger.info("Price matrix shape: %s", price_matrix.shape)

        logger.info("Step 2b/4: Building volume matrix ...")
        volume_matrix = build_volume_matrix(raw_data, cfg=cfg)
        logger.info("Volume matrix shape: %s", volume_matrix.shape)

    if not equity_only and not earnings_only:
        logger.info("Step 3/4: Downloading macro / global proxy data ...")
        macro_mgr = MacroDataManager(cfg)
        macro_df = macro_mgr.build(start=start, end=end, force=force)
        logger.info("Macro data shape: %s", macro_df.shape)

    if not no_earnings and not macro_only:
        from pathlib import Path
        import pandas as pd
        logger.info("Step 4/4: Building earnings / fundamentals data from cached Screener data ...")
        raw_dir = Path(cfg["paths"]["raw_data"]) / "earnings"
        processed_path = Path(cfg["paths"]["processed_data"]) / "earnings_panel.parquet"

        # Need price matrix for index; load from cache if not built above
        if price_matrix is None:
            pm_path = Path(cfg["paths"]["processed_data"]) / "price_matrix.parquet"
            if pm_path.exists():
                price_matrix = pd.read_parquet(pm_path)
            else:
                logger.warning("Price matrix not available — skip earnings panel build")
                price_matrix = None

        if price_matrix is not None:
            # Only equity tickers — skip macro proxies (futures, FX, indices)
            tickers = [t for t in price_matrix.columns if t.endswith(".NS") or t.endswith(".BO")]
            panel = build_earnings_panel(tickers, price_matrix.index, raw_dir)
            save_earnings_panel(panel, processed_path)
            logger.info("Earnings panel shape: %s", panel.shape)

    logger.info("=" * 60)
    logger.info("DATA DOWNLOAD COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
