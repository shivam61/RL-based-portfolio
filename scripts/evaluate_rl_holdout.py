#!/usr/bin/env python3
"""
Evaluate the fresh causal RL policy on a fixed holdout window.

This trains the RL overlay on history up to the holdout boundary, freezes the
policy, and compares it against a neutral policy on the same holdout period.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

_N_THREADS = str(max(1, os.cpu_count() - 1))
os.environ.setdefault("OMP_NUM_THREADS", _N_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _N_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _N_THREADS)

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from src.config import load_config, setup_logging
from src.data.ingestion import load_price_matrix, load_volume_matrix
from src.data.macro import MacroDataManager
from src.rl.holdout import evaluate_holdout


@click.command()
@click.option("--holdout-start", required=True, help="Holdout start date, e.g. 2024-01-01")
@click.option("--holdout-end", required=True, help="Holdout end date, e.g. 2025-12-31")
@click.option("--timesteps", default=None, type=int, help="Override RL PPO timesteps for holdout training")
@click.option("--config", default=None, help="Path to custom config file")
def main(holdout_start, holdout_end, timesteps, config):
    cfg = load_config(config)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("Loading data for RL holdout evaluation ...")
    price_matrix = load_price_matrix(cfg)
    volume_matrix = load_volume_matrix(cfg)
    macro_df = MacroDataManager(cfg).load()

    result = evaluate_holdout(
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        macro_df=macro_df,
        cfg=cfg,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        total_timesteps=timesteps,
    )

    print(json.dumps(result, indent=2, default=str))

    out_path = Path(cfg["paths"]["report_dir"]) / "rl_holdout_comparison.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("RL holdout comparison saved → %s", out_path)


if __name__ == "__main__":
    main()
