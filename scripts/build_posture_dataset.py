#!/usr/bin/env python3
"""Build realized forward-outcome posture labels for research."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from src.config import load_config, setup_logging
from src.data.ingestion import load_price_matrix, load_volume_matrix
from src.data.macro import MacroDataManager
from src.rl.posture_dataset import build_posture_dataset, save_posture_dataset


@click.command()
@click.option("--start-date", default=None, help="Optional dataset start date.")
@click.option("--end-date", default=None, help="Optional dataset end date.")
@click.option("--horizon-rebalances", default=2, type=int, show_default=True)
@click.option("--max-samples", default=None, type=int, help="Optional cap for quick research builds.")
@click.option("--config", default=None, help="Path to custom config file")
def main(start_date, end_date, horizon_rebalances, max_samples, config):
    cfg = load_config(config)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("Loading data for posture research dataset ...")
    price_matrix = load_price_matrix(cfg)
    volume_matrix = load_volume_matrix(cfg)
    macro_df = MacroDataManager(cfg).load()

    payload = build_posture_dataset(
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        macro_df=macro_df,
        cfg=cfg,
        start_date=start_date,
        end_date=end_date,
        horizon_rebalances=horizon_rebalances,
        max_samples=max_samples,
    )
    saved = save_posture_dataset(
        payload,
        report_dir=cfg["paths"]["report_dir"],
        prefix="posture_dataset",
    )
    summary_path = Path(saved["summary_path"])
    print(summary_path.read_text())
    logger.info("Posture dataset summary saved → %s", saved["summary_path"])
    logger.info("Posture dataset parquet saved → %s", saved["parquet_path"])


if __name__ == "__main__":
    main()
