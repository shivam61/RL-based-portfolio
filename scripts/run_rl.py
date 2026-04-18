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


@click.command()
@click.option("--timesteps", default=None, type=int, help="Override total timesteps")
@click.option("--config", default=None, help="Path to custom config file")
@click.option("--eval", "run_eval", is_flag=True, default=False, help="Run evaluation after training")
def main(timesteps, config, run_eval):
    """Train the RL sector allocation agent on collected experience."""
    cfg = load_config(config)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

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
            # Simple evaluation: compare RL actions vs neutral actions
            from src.rl.environment import SectorAllocationEnv
            import numpy as np
            rl_rewards, baseline_rewards = [], []
            for step in buf[-50:]:
                obs_env = SectorAllocationEnv([step], cfg)
                obs = obs_env._get_obs(0)
                if agent.is_trained:
                    action, _ = agent.model.predict(obs.reshape(1, -1), deterministic=True)
                    rl_reward = obs_env._compute_reward(step, action[0])
                    neutral = np.array([1.0] * 15 + [0.05, 1.0, 1.0])
                    baseline_reward = obs_env._compute_reward(step, neutral)
                    rl_rewards.append(rl_reward)
                    baseline_rewards.append(baseline_reward)

            if rl_rewards:
                logger.info("RL avg reward: %.4f vs Baseline: %.4f",
                           np.mean(rl_rewards), np.mean(baseline_rewards))
                improvement = (np.mean(rl_rewards) - np.mean(baseline_rewards))
                logger.info("RL improvement: %.4f (%s)",
                           improvement, "+" if improvement > 0 else "-")


if __name__ == "__main__":
    main()
