#!/usr/bin/env python3
"""Build the canonical RL control-evaluation artifact from report outputs."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from src.config import load_config, setup_logging
from src.rl.control_evaluation import evaluate_control_from_artifacts


@click.command()
@click.option("--report-dir", default=None, help="Report directory to read/write. Defaults to config paths.report_dir")
@click.option(
    "--drawdown-threshold",
    default=-0.08,
    type=float,
    show_default=True,
    help="Drawdown threshold used for aggregate drawdown-behavior metrics",
)
@click.option("--config", default=None, help="Path to custom config file")
def main(report_dir, drawdown_threshold, config):
    cfg = load_config(config)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    target_dir = Path(report_dir or cfg["paths"]["report_dir"])
    result = evaluate_control_from_artifacts(
        target_dir,
        drawdown_threshold=drawdown_threshold,
    )

    print(json.dumps(result, indent=2, default=str))
    out_path = target_dir / "rl_control_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("RL control evaluation saved → %s", out_path)


if __name__ == "__main__":
    main()
