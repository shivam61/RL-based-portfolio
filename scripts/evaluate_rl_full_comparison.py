#!/usr/bin/env python3
"""Compare the latest full-history RL run against full-window baselines."""
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
from src.rl.full_comparison import (
    evaluate_full_backtest_comparison,
    evaluate_full_neutral_policy_comparison,
)


@click.command()
@click.option(
    "--full-rl-metrics",
    default="artifacts/reports/metrics.json",
    show_default=True,
    help="Path to a saved full_rl metrics.json artifact",
)
@click.option(
    "--baseline-mode",
    type=click.Choice(["optimizer_only", "selection_only"], case_sensitive=False),
    default="optimizer_only",
    show_default=True,
    help="Neutral baseline mode to compare against the saved full_rl run",
)
@click.option("--config", default=None, help="Path to custom config file")
def main(full_rl_metrics, baseline_mode, config):
    cfg = load_config(config)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("Loading data for full backtest comparison ...")
    price_matrix = load_price_matrix(cfg)
    volume_matrix = load_volume_matrix(cfg)
    macro_df = MacroDataManager(cfg).load()

    baseline_result = evaluate_full_backtest_comparison(
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        macro_df=macro_df,
        full_rl_metrics_path=full_rl_metrics,
        cfg=cfg,
        baseline_mode=baseline_mode,
    )
    neutral_result = evaluate_full_neutral_policy_comparison(
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        macro_df=macro_df,
        full_rl_metrics_path=full_rl_metrics,
        cfg=cfg,
    )

    print(
        json.dumps(
            {
                "baseline_comparison": baseline_result,
                "neutral_comparison": neutral_result,
            },
            indent=2,
            default=str,
        )
    )

    report_dir = Path(cfg["paths"]["report_dir"])
    baseline_path = report_dir / "rl_full_backtest_comparison.json"
    neutral_path = report_dir / "rl_full_neutral_comparison.json"
    baseline_path.write_text(json.dumps(baseline_result, indent=2, default=str))
    neutral_path.write_text(json.dumps(neutral_result, indent=2, default=str))
    logger.info("Full backtest baseline comparison saved → %s", baseline_path)
    logger.info("Full backtest neutral comparison saved → %s", neutral_path)


if __name__ == "__main__":
    main()
