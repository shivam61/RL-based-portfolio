"""
RL agent wrapper using stable-baselines3 PPO.

Handles:
- training from experience buffer (offline walk-forward)
- incremental updates as new experience arrives
- safe fallback to neutral action when untrained
- persistence (save/load)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.rl.environment import SectorAllocationEnv, SECTORS, N_SECTORS, ACTION_DIM, STATE_DIM

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    logger.warning("stable-baselines3 not available; RL agent will use rule-based fallback")


class RLSectorAgent:
    """PPO-based sector allocation agent."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._rl_cfg = cfg["rl"]
        self.model: Optional[Any] = None
        self.is_trained: bool = False
        self._experience_buffer: list[dict] = []

    # ── Experience collection ─────────────────────────────────────────────────

    def record_step(self, step_data: dict) -> None:
        """Record a rebalance step for future training."""
        self._experience_buffer.append(step_data)

    def buffer_size(self) -> int:
        return len(self._experience_buffer)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        experience_buffer: list[dict] | None = None,
        total_timesteps: int | None = None,
    ) -> "RLSectorAgent":
        """Train PPO on the collected experience buffer."""
        buf = experience_buffer or self._experience_buffer
        if len(buf) < 20:
            logger.warning(
                "Insufficient experience (%d steps) to train RL agent", len(buf)
            )
            return self

        ts = total_timesteps or self._rl_cfg.get("total_timesteps", 50000)

        if not HAS_SB3:
            logger.info("SB3 not available; skipping RL training")
            return self

        env = SectorAllocationEnv(buf, self.cfg)
        vec_env = DummyVecEnv([lambda: env])

        if self.model is None:
            self.model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=self._rl_cfg.get("learning_rate", 3e-4),
                n_steps=self._rl_cfg.get("n_steps", 512),
                batch_size=self._rl_cfg.get("batch_size", 64),
                n_epochs=self._rl_cfg.get("n_epochs", 10),
                gamma=self._rl_cfg.get("gamma", 0.99),
                gae_lambda=self._rl_cfg.get("gae_lambda", 0.95),
                clip_range=self._rl_cfg.get("clip_range", 0.2),
                ent_coef=self._rl_cfg.get("ent_coef", 0.01),
                verbose=0,
                device="cpu",
            )
        else:
            # continue training with new environment
            self.model.set_env(vec_env)

        self.model.learn(total_timesteps=ts, reset_num_timesteps=False)
        self.is_trained = True
        logger.info("RL agent trained on %d experience steps (%d timesteps)", len(buf), ts)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def decide(
        self,
        macro_state: dict,
        sector_state: dict,
        portfolio_state: dict,
        prev_realized_sector_weights: dict | None = None,  # reserved for future use
    ) -> dict:
        """Given current state, return sector allocation decisions."""
        if not self.is_trained or self.model is None:
            return SectorAllocationEnv.neutral_action()

        obs = self._build_obs(macro_state, sector_state, portfolio_state)
        obs_tensor = np.array(obs, dtype=np.float32).reshape(1, -1)

        try:
            action, _ = self.model.predict(obs_tensor, deterministic=True)
            return SectorAllocationEnv.decode_action(action[0])
        except Exception as e:
            logger.warning("RL predict failed (%s); using neutral action", e)
            return SectorAllocationEnv.neutral_action()

    def _build_obs(
        self,
        macro_state: dict,
        sector_state: dict,
        portfolio_state: dict,
    ) -> np.ndarray:
        """Build observation vector matching environment spec."""
        fake_exp = {
            "macro_state": macro_state,
            "sector_state": sector_state,
            "portfolio_state": portfolio_state,
            "outcome": {},
        }
        env = SectorAllocationEnv([fake_exp], self.cfg)
        return env._get_obs(0)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # save experience buffer
        with open(dir_path / "experience_buffer.pkl", "wb") as f:
            pickle.dump(self._experience_buffer, f)

        # save SB3 model
        if self.model is not None and HAS_SB3:
            self.model.save(str(dir_path / "ppo_model"))

        meta = {"is_trained": self.is_trained, "buffer_size": len(self._experience_buffer)}
        with open(dir_path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info("RL agent saved → %s", dir_path)

    def load(self, dir_path: str | Path) -> "RLSectorAgent":
        dir_path = Path(dir_path)

        buf_path = dir_path / "experience_buffer.pkl"
        if buf_path.exists():
            with open(buf_path, "rb") as f:
                self._experience_buffer = pickle.load(f)

        model_path = dir_path / "ppo_model.zip"
        if model_path.exists() and HAS_SB3:
            env = SectorAllocationEnv(self._experience_buffer or [{}], self.cfg)
            vec_env = DummyVecEnv([lambda: env])
            self.model = PPO.load(str(dir_path / "ppo_model"), env=vec_env)
            self.is_trained = True

        meta_path = dir_path / "meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.is_trained = meta.get("is_trained", False)

        logger.info("RL agent loaded from %s (trained=%s)", dir_path, self.is_trained)
        return self

    # ── Rule-based baseline (non-RL) ─────────────────────────────────────────

    @staticmethod
    def rule_based_action(
        sector_scores: dict[str, float],
        macro_features: dict,
        risk_regime: str = "neutral",
    ) -> dict:
        """
        Deterministic rule-based fallback using sector scores.
        Returns same structure as RL action.
        """
        # Convert sector scores to tilts
        if sector_scores:
            mean_score = np.mean(list(sector_scores.values()))
            tilts = {}
            for sec, score in sector_scores.items():
                if score > mean_score:
                    tilts[sec] = 1.0 + (score - mean_score) * 2
                else:
                    tilts[sec] = 1.0 - (mean_score - score) * 1
                tilts[sec] = float(np.clip(tilts[sec], 0.3, 2.0))
        else:
            tilts = {s: 1.0 for s in SECTORS}

        # cash target based on regime
        cash_target = {
            "bull": 0.03,
            "bear": 0.15,
            "stressed": 0.20,
            "neutral": 0.05,
        }.get(risk_regime, 0.05)

        # aggressiveness based on regime
        agg = {
            "bull": 1.2,
            "bear": 0.7,
            "stressed": 0.5,
            "neutral": 1.0,
        }.get(risk_regime, 1.0)

        return {
            "sector_tilts": tilts,
            "cash_target": cash_target,
            "aggressiveness": agg,
            "should_rebalance": True,
        }
