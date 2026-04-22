#!/usr/bin/env python3
"""
CLI: Run dedicated RL agent training on collected experience.

Usage:
    python scripts/run_rl.py [--timesteps 200000]

Run this after run_backtest.py to train RL on the full experience buffer.
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from src.config import load_config, setup_logging
from src.rl.agent import RLSectorAgent
from src.rl.contract import CAUSAL_TRAINING_BACKEND


@click.command()
@click.option("--timesteps", default=None, type=int, help="Override total timesteps")
@click.option("--config", default=None, help="Path to custom config file")
@click.option("--eval", "run_eval", is_flag=True, default=False, help="Run evaluation after training")
def main(timesteps, config, run_eval):
    """Train the RL sector allocation agent on collected experience."""
    cfg = load_config(config)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)
    backend = cfg.get("rl", {}).get("training_backend", "disabled")

    if backend != CAUSAL_TRAINING_BACKEND:
        logger.error(
            "Refusing to train RL with backend=%s. Configure rl.training_backend=%s first.",
            backend,
            CAUSAL_TRAINING_BACKEND,
        )
        sys.exit(2)

    model_dir = Path(cfg["paths"]["model_dir"]) / "rl_agent"

    logger.info("Loading RL agent and experience buffer ...")
    agent = RLSectorAgent(cfg)

    if model_dir.exists():
        agent.load(model_dir)
        logger.info("Loaded agent with %d experience steps", agent.buffer_size())
    else:
        logger.warning("No saved agent found at %s; starting fresh", model_dir)
        logger.warning("Run run_backtest.py first to collect experience")

    if agent.buffer_size() < 20:
        logger.error("Insufficient experience (%d steps). Run backtest first.", agent.buffer_size())
        sys.exit(1)

    ts = timesteps or cfg["rl"].get("total_timesteps", 200000)
    logger.info("Training RL agent for %d timesteps on %d experience steps ...",
                ts, agent.buffer_size())

    agent.train(total_timesteps=ts)
    agent.save(model_dir)

    logger.info("RL agent trained and saved → %s", model_dir)

    if run_eval:
        buf = agent._experience_buffer
        if buf:
            logger.warning(
                "Standalone replay-based RL evaluation has been removed. "
                "Use a causal holdout evaluation once the simulator-backed backend is fully implemented."
            )


if __name__ == "__main__":
    main()
