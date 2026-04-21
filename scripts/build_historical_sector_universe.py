#!/usr/bin/env python3
"""
Build survivorship-aware 10-year union sector universe artifacts.

Usage:
    ./.venv/bin/python scripts/build_historical_sector_universe.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd

from src.config import load_config, setup_logging
from src.data.ingestion import load_price_matrix, load_volume_matrix
from src.universe.historical_sector_universe import HistoricalSectorUniverseBuilder


@click.command()
@click.option("--config", default=None, help="Path to custom config file")
@click.option("--as-of", default=None, help="As-of date override (YYYY-MM-DD)")
def main(config: str | None, as_of: str | None) -> None:
    cfg = load_config(config)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    price_matrix = load_price_matrix(cfg)
    volume_matrix = load_volume_matrix(cfg)
    as_of_ts = pd.Timestamp(as_of) if as_of else price_matrix.index.max()

    builder = HistoricalSectorUniverseBuilder(cfg)
    master, union_df, diagnostics = builder.build(
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        as_of=as_of_ts,
    )
    artifacts = builder.persist(master, union_df, diagnostics)

    logger.info("=" * 72)
    logger.info("Historical universe build complete")
    logger.info("As-of date: %s", as_of_ts.date())
    logger.info("Master rows: %d", len(master))
    logger.info("Union rows:  %d", len(union_df))
    logger.info("Sector master: %s", artifacts.sector_master_path)
    logger.info("Union set:     %s", artifacts.union_path)
    logger.info("Diagnostics:   %s", artifacts.diagnostics_path)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
