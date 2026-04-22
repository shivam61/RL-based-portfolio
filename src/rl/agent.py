"""
RL agent wrapper using stable-baselines3 PPO.

Handles:
- transition collection and validation
- safe-disable of invalid legacy replay training
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

from src.rl.contract import (
    CAUSAL_TRAINING_BACKEND,
    canonicalize_transition,
    is_transition_step,
    summarize_buffer,
)
from src.rl.environment import ACTION_DIM, STATE_DIM, HistoricalSectorAllocationEnv, SectorAllocationEnv, SECTORS

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
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
        self.training_backend: str = "disabled"
        self.disable_reason: str | None = (
            "RL policy training is disabled until a causal environment is available."
        )
        self._buffer_summary: dict[str, Any] = summarize_buffer([])

    # ── Experience collection ─────────────────────────────────────────────────

    def record_step(self, step_data: dict) -> None:
        """Record a rebalance step for future training."""
        if is_transition_step(step_data):
            self._experience_buffer.append(canonicalize_transition(step_data))
        else:
            self._experience_buffer.append(step_data)
        self._buffer_summary = summarize_buffer(self._experience_buffer)

    def buffer_size(self) -> int:
        return len(self._experience_buffer)

    def buffer_summary(self) -> dict[str, Any]:
        return dict(self._buffer_summary)

    def _disable_training(self, reason: str) -> None:
        self.model = None
        self.is_trained = False
        self.training_backend = "disabled"
        self.disable_reason = reason
        logger.warning("RL training disabled: %s", reason)

    def build_causal_env(
        self,
        executor,
        *,
        start_idx: int = 0,
        end_idx: int | None = None,
        max_episode_steps: int | None = None,
    ) -> HistoricalSectorAllocationEnv:
        seed = int(self.cfg.get("backtest", {}).get("random_seed", 42))
        return HistoricalSectorAllocationEnv(
            executor,
            start_idx=start_idx,
            end_idx=end_idx,
            max_episode_steps=max_episode_steps,
            seed=seed,
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        experience_buffer: list[dict] | None = None,
        total_timesteps: int | None = None,
        causal_env: HistoricalSectorAllocationEnv | None = None,
    ) -> "RLSectorAgent":
        """Train a fresh PPO policy on the causal backend when available."""
        buf = experience_buffer or self._experience_buffer
        self._buffer_summary = summarize_buffer(buf)
        requested_backend = self._rl_cfg.get("training_backend", CAUSAL_TRAINING_BACKEND)
        if requested_backend != CAUSAL_TRAINING_BACKEND:
            self._disable_training(
                f"Unsupported RL training backend '{requested_backend}'. "
                f"Expected '{CAUSAL_TRAINING_BACKEND}'."
            )
            return self

        if causal_env is not None:
            return self._train_fresh_policy(causal_env, total_timesteps=total_timesteps)

        if len(buf) < 20:
            logger.warning(
                "Insufficient experience (%d steps) to train RL agent", len(buf)
            )
            self._disable_training("Insufficient RL experience to train a causal policy.")
            return self

        if not self._buffer_summary["supports_transition_replay"]:
            self._disable_training(self._buffer_summary["disable_reason"])
            return self

        self._disable_training(
            "Canonical transitions are present, but standalone replay PPO training "
            "is disabled. Use the simulator-backed causal environment instead."
        )
        return self

    def _train_fresh_policy(
        self,
        causal_env: HistoricalSectorAllocationEnv,
        total_timesteps: int | None = None,
    ) -> "RLSectorAgent":
        if not HAS_SB3:
            self._disable_training("stable-baselines3 not available; cannot train RL policy.")
            return self

        total_ts = int(total_timesteps or self._rl_cfg.get("total_timesteps", 20000))
        seed = int(self.cfg.get("backtest", {}).get("random_seed", 42))
        algo = str(self._rl_cfg.get("algorithm", "PPO")).upper()
        if algo != "PPO":
            self._disable_training(f"Unsupported RL algorithm '{algo}'.")
            return self

        try:
            self.model = PPO(
                "MlpPolicy",
                causal_env,
                learning_rate=float(self._rl_cfg.get("learning_rate", 3e-4)),
                n_steps=int(self._rl_cfg.get("n_steps", 36)),
                batch_size=int(self._rl_cfg.get("batch_size", 16)),
                n_epochs=int(self._rl_cfg.get("n_epochs", 10)),
                gamma=float(self._rl_cfg.get("gamma", 0.99)),
                gae_lambda=float(self._rl_cfg.get("gae_lambda", 0.95)),
                clip_range=float(self._rl_cfg.get("clip_range", 0.2)),
                ent_coef=float(self._rl_cfg.get("ent_coef", 0.01)),
                verbose=0,
                seed=seed,
            )
            self.model.learn(total_timesteps=total_ts)
        except Exception as exc:
            self._disable_training(f"Causal RL training failed: {exc}")
            return self

        self.is_trained = True
        self.training_backend = CAUSAL_TRAINING_BACKEND
        self.disable_reason = None
        logger.info(
            "RL policy trained from scratch on causal backend (%d timesteps)",
            total_ts,
        )
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
            return SectorAllocationEnv.neutral_action(self.cfg)

        obs = self._build_obs(macro_state, sector_state, portfolio_state)
        obs_tensor = np.array(obs, dtype=np.float32).reshape(1, -1)

        try:
            action, _ = self.model.predict(obs_tensor, deterministic=True)
            return SectorAllocationEnv.decode_action(action[0], self.cfg)
        except Exception as e:
            logger.warning("RL predict failed (%s); using neutral action", e)
            return SectorAllocationEnv.neutral_action(self.cfg)

    def _build_obs(
        self,
        macro_state: dict,
        sector_state: dict,
        portfolio_state: dict,
    ) -> np.ndarray:
        """Build observation vector matching environment spec."""
        return SectorAllocationEnv.encode_observation(
            macro_state=macro_state,
            sector_state=sector_state,
            portfolio_state=portfolio_state,
        )

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

        meta = {
            "is_trained": self.is_trained,
            "buffer_size": len(self._experience_buffer),
            "training_backend": self.training_backend,
            "disable_reason": self.disable_reason,
            "buffer_summary": self._buffer_summary,
            "observation_dim": int(STATE_DIM),
            "action_dim": int(ACTION_DIM),
        }
        with open(dir_path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info("RL agent saved → %s", dir_path)

    def load(self, dir_path: str | Path) -> "RLSectorAgent":
        dir_path = Path(dir_path)

        buf_path = dir_path / "experience_buffer.pkl"
        if buf_path.exists():
            with open(buf_path, "rb") as f:
                self._experience_buffer = pickle.load(f)
        self._buffer_summary = summarize_buffer(self._experience_buffer)

        meta_path = dir_path / "meta.pkl"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        self.training_backend = meta.get("training_backend", "disabled")
        self.disable_reason = meta.get("disable_reason")
        self.is_trained = bool(meta.get("is_trained", False))
        saved_obs_dim = meta.get("observation_dim")
        saved_action_dim = meta.get("action_dim")

        model_path = dir_path / "ppo_model.zip"
        if not meta_path.exists():
            self._disable_training(
                "Legacy RL artifact is missing causal training metadata; model load refused."
            )
        elif saved_obs_dim is None or saved_action_dim is None:
            self._disable_training(
                "Legacy RL artifact is missing policy-dimension metadata; retrain required."
            )
        elif int(saved_obs_dim) != int(STATE_DIM) or int(saved_action_dim) != int(ACTION_DIM):
            self._disable_training(
                "Saved RL artifact uses incompatible action/state dimensions; retrain required."
            )
        elif self.training_backend != CAUSAL_TRAINING_BACKEND:
            self._disable_training(
                self.disable_reason
                or "RL artifact was not trained with the canonical causal backend."
            )
        elif not model_path.exists():
            self._disable_training("RL artifact metadata exists, but ppo_model.zip is missing.")
        elif not HAS_SB3:
            self._disable_training("stable-baselines3 not available; cannot load RL policy.")
        else:
            self.model = PPO.load(str(dir_path / "ppo_model"))
            self.is_trained = True
            self.disable_reason = None

        logger.info(
            "RL agent loaded from %s (trained=%s, backend=%s)",
            dir_path,
            self.is_trained,
            self.training_backend,
        )
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
            "turnover_cap": None,
            "should_rebalance": True,
        }
