"""
RL replay environment for sector allocation transitions.

This module now enforces the canonical transition contract:
    (state, action, reward, next_state, done, info)

It is intentionally a strict replay surface. It does not make PPO training
causal on its own, so the agent refuses to optimize policies against this
environment until a simulator-backed backend is available.
"""
from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    gym = None  # type: ignore
    spaces = None  # type: ignore

from src.rl.contract import canonicalize_transition, summarize_buffer
from src.rl.historical_executor import HistoricalPeriodExecutor

logger = logging.getLogger(__name__)

# Canonical sector list (must match universe.yaml)
SECTORS = [
    "IT", "Banking", "FinancialServices", "FMCG", "Automobiles",
    "Pharma", "Energy", "Metals", "Telecom", "Cement",
    "CapitalGoods", "ConsumerDiscretionary", "Healthcare", "RealEstate",
    "Chemicals",
]
N_SECTORS = len(SECTORS)

# Action dimensions
# [0:N_SECTORS]        zero-sum sector tilt deltas  [-0.35, 0.35]
# [N_SECTORS]          aggressiveness delta         configurable around 1.0
# [N_SECTORS + 1]      cash control slot            optional bucketed control
# [N_SECTORS + 2]      turnover control slot        optional bucketed control
ACTION_DIM = N_SECTORS + 3
SECTOR_DELTA_MAX = 0.35
CASH_TARGET_NEUTRAL = 0.05
CASH_TARGET_MIN = 0.0
AGGRESSIVENESS_NEUTRAL = 1.0

# State dimensions
MACRO_DIM = 12          # macro features
SECTOR_DIM = N_SECTORS * 4  # per-sector: mom_1m, mom_3m, rel_str_1m, breadth_3m
PORT_DIM = 17           # portfolio state + control features
# REALIZED_SECTOR_DIM kept here for future use when more live experience exists
REALIZED_SECTOR_DIM = N_SECTORS  # not added to STATE_DIM until RL has 500+ training steps
STATE_DIM = MACRO_DIM + SECTOR_DIM + PORT_DIM  # 12+60+17 = 89


_GymBase = gym.Env if HAS_GYM else object


class SectorAllocationEnv(_GymBase):
    """
    Transition replay environment for RL diagnostics.

    The caller must supply canonical transitions. `step()` returns the logged
    next state and reward for the recorded action and reports any action
    mismatch in `info`. This is not a simulator-backed control environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        experience_buffer: list[dict],
        cfg: dict,
        seed: int = 42,
    ):
        if HAS_GYM:
            super().__init__()
        self.cfg = cfg
        self._rl_cfg = cfg["rl"]
        summary = summarize_buffer(experience_buffer)
        if not summary["supports_transition_replay"]:
            raise ValueError(summary["disable_reason"])
        self.experience = [
            canonicalize_transition(step) for step in experience_buffer
        ]
        self.cfg = cfg
        self._rng = np.random.default_rng(seed)
        self._step_idx: int = 0
        self._max_steps: int = max(len(self.experience), 1)

        if HAS_GYM:
            self._setup_gym_spaces()

    def _setup_gym_spaces(self) -> None:
        agg_min, agg_max = _aggressiveness_bounds(self.cfg)
        low_a = np.array(
            [-SECTOR_DELTA_MAX] * N_SECTORS
            + [agg_min - AGGRESSIVENESS_NEUTRAL, -1.0, -1.0],
            dtype=np.float32,
        )
        high_a = np.array(
            [SECTOR_DELTA_MAX] * N_SECTORS
            + [agg_max - AGGRESSIVENESS_NEUTRAL, 1.0, 1.0],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )

    def render(self):
        pass

    def close(self):
        pass

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        self._step_idx = int(self._rng.integers(0, max(1, self._max_steps - 10)))
        obs = self._get_obs(self._step_idx)
        return obs.astype(np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._step_idx >= self._max_steps:
            obs = self._get_obs(self._max_steps - 1)
            return obs.astype(np.float32), 0.0, True, False, {}

        transition = self.experience[self._step_idx]
        reward = self._compute_reward(transition, action)
        recorded_action = self.encode_action(transition["action"])
        mismatch_l1 = float(
            np.mean(
                np.abs(_pad_action(np.asarray(action, dtype=np.float32)) - recorded_action)
            )
        )
        self._step_idx += 1
        obs = self.encode_observation(**transition["next_state"])
        done = bool(transition["done"]) or self._step_idx >= self._max_steps
        info = {
            "replay_only": True,
            "action_mismatch_l1": mismatch_l1,
            "logged_info": dict(transition.get("info", {})),
        }
        return obs.astype(np.float32), float(reward), done, False, info

    # ── State builder ─────────────────────────────────────────────────────────

    def _get_obs(self, idx: int) -> np.ndarray:
        if idx >= self._max_steps:
            return np.zeros(STATE_DIM, dtype=np.float32)

        return self.encode_observation(**self.experience[idx]["state"])

    @staticmethod
    def encode_observation(
        macro_state: dict[str, Any],
        sector_state: dict[str, Any],
        portfolio_state: dict[str, Any],
    ) -> np.ndarray:
        state_vec = np.zeros(STATE_DIM, dtype=np.float32)
        offset = 0

        macro_keys = [
            "vix_level", "usdinr_ret_1m", "crude_ret_1m", "sp500_ret_1m",
            "gold_ret_1m", "risk_on_score", "macro_stress_score",
            "rbi_rate", "rate_cutting_cycle", "election_window",
            "nifty_ret_1m", "nifty_above_200ma",
        ]
        for i, k in enumerate(macro_keys[:MACRO_DIM]):
            v = macro_state.get(k, 0.0)
            state_vec[offset + i] = 0.0 if (v is None or np.isnan(v)) else float(v)
        offset += MACRO_DIM

        for j, sec in enumerate(SECTORS):
            sec_data = sector_state.get(sec, {})
            state_vec[offset + j * 4 + 0] = float(sec_data.get("mom_1m", 0) or 0)
            state_vec[offset + j * 4 + 1] = float(sec_data.get("mom_3m", 0) or 0)
            state_vec[offset + j * 4 + 2] = float(sec_data.get("rel_str_1m", 0) or 0)
            state_vec[offset + j * 4 + 3] = float(sec_data.get("breadth_3m", 0) or 0)
        offset += SECTOR_DIM

        port_keys = [
            "cash_ratio", "ret_1m", "vol_1m", "current_drawdown",
            "max_drawdown", "drawdown_slope_1m", "vol_shock_1m_3m",
            "breadth_deterioration", "recent_turnover_3p", "recent_cost_ratio_3p",
            "risk_cash_floor", "emergency_flag", "hhi", "max_weight",
            "sharpe_3m", "active_ret_1m", "n_stocks",
        ]
        for i, k in enumerate(port_keys[:PORT_DIM]):
            v = portfolio_state.get(k, 0.0)
            state_vec[offset + i] = 0.0 if (v is None or np.isnan(v)) else float(v)

        state_vec = np.clip(state_vec, -10, 10)
        return state_vec

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, transition: dict, action: np.ndarray | None = None) -> float:
        if "reward" not in transition:
            raise ValueError("Reward is only defined for canonical RL transitions.")
        if action is not None and transition.get("action") is not None:
            recorded = self.encode_action(transition["action"])
            mismatch = float(
                np.mean(
                    np.abs(_pad_action(np.asarray(action, dtype=np.float32)) - recorded)
                )
            )
            if mismatch > 1e-6:
                logger.debug(
                    "Replay reward requested for non-recorded action; mismatch_l1=%.6f",
                    mismatch,
                )
        return float(transition["reward"])

    # ── Action decoder ────────────────────────────────────────────────────────

    @staticmethod
    def encode_action(action: dict[str, Any]) -> np.ndarray:
        sector_tilts = action.get("sector_tilts", {})
        raw_sector = np.array(
            [
                float(np.clip(sector_tilts.get(sec, 1.0), 1.0 - SECTOR_DELTA_MAX, 1.0 + SECTOR_DELTA_MAX))
                - 1.0
                for sec in SECTORS
            ],
            dtype=np.float32,
        )
        if raw_sector.size:
            raw_sector = np.clip(
                raw_sector - float(raw_sector.mean()),
                -SECTOR_DELTA_MAX,
                SECTOR_DELTA_MAX,
            )
        agg_min, agg_max = _aggressiveness_bounds(None)
        aggressiveness_slot = float(
            np.clip(
                float(action.get("aggressiveness", AGGRESSIVENESS_NEUTRAL))
                - AGGRESSIVENESS_NEUTRAL,
                agg_min - AGGRESSIVENESS_NEUTRAL,
                agg_max - AGGRESSIVENESS_NEUTRAL,
            )
        )
        cash_slot = _encode_bucket_slot(
            value=float(action.get("cash_target", CASH_TARGET_NEUTRAL)),
            buckets=_cash_buckets(None),
            neutral_value=CASH_TARGET_NEUTRAL,
        )
        turnover_value = action.get("turnover_cap")
        if turnover_value is None:
            turnover_value = _neutral_turnover_cap(None)
        turnover_slot = _encode_bucket_slot(
            value=float(turnover_value),
            buckets=_turnover_buckets(None),
            neutral_value=float(_neutral_turnover_cap(None)),
        )
        return np.array(
            list(raw_sector) + [aggressiveness_slot, cash_slot, turnover_slot],
            dtype=np.float32,
        )

    @staticmethod
    def decode_action(action: np.ndarray, cfg: dict | None = None) -> dict:
        """Convert raw delta action vector into neutral-anchored decisions."""
        raw_action = _pad_action(np.asarray(action, dtype=np.float32))
        sector_delta = np.clip(raw_action[:N_SECTORS], -SECTOR_DELTA_MAX, SECTOR_DELTA_MAX)
        if sector_delta.size:
            sector_delta = np.clip(
                sector_delta - float(np.mean(sector_delta)),
                -SECTOR_DELTA_MAX,
                SECTOR_DELTA_MAX,
            )
        sector_tilts = {
            SECTORS[i]: float(np.clip(1.0 + sector_delta[i], 1.0 - SECTOR_DELTA_MAX, 1.0 + SECTOR_DELTA_MAX))
            for i in range(N_SECTORS)
        }
        agg_min, agg_max = _aggressiveness_bounds(cfg)
        aggressiveness = float(
            np.clip(
                AGGRESSIVENESS_NEUTRAL + raw_action[N_SECTORS],
                agg_min,
                agg_max,
            )
        )
        cash_target = (
            _decode_bucket_slot(
                raw_action[N_SECTORS + 1],
                _cash_buckets(cfg),
                neutral_value=CASH_TARGET_NEUTRAL,
            )
            if _cash_control_enabled(cfg)
            else CASH_TARGET_NEUTRAL
        )
        turnover_cap = (
            _decode_bucket_slot(
                raw_action[N_SECTORS + 2],
                _turnover_buckets(cfg),
                neutral_value=float(_neutral_turnover_cap(cfg) or 0.40),
            )
            if _turnover_control_enabled(cfg)
            else _neutral_turnover_cap(cfg)
        )

        return {
            "sector_tilts": sector_tilts,
            "cash_target": cash_target,
            "aggressiveness": aggressiveness,
            "turnover_cap": turnover_cap,
            "should_rebalance": True,
        }

    @staticmethod
    def neutral_action(cfg: dict | None = None) -> dict:
        """Default no-tilt action for non-RL baseline."""
        return {
            "sector_tilts": {s: 1.0 for s in SECTORS},
            "cash_target": CASH_TARGET_NEUTRAL,
            "aggressiveness": AGGRESSIVENESS_NEUTRAL,
            "turnover_cap": _neutral_turnover_cap(cfg),
            "should_rebalance": True,
        }


class HistoricalSectorAllocationEnv(_GymBase):
    """
    Simulator-backed one-step environment for causal RL integration work.

    This environment executes the supplied action through the historical
    rebalance path instead of replaying a logged reward. Episodes are currently
    one rebalance window long; multi-period training remains a follow-on step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        executor: HistoricalPeriodExecutor,
        start_idx: int = 0,
        end_idx: int | None = None,
        max_episode_steps: int | None = None,
        seed: int = 42,
    ):
        if HAS_GYM:
            super().__init__()
        self.executor = executor
        self.cfg = executor.engine.cfg
        self.start_idx = start_idx
        max_valid_end = len(self.executor.rebalance_dates) - 2
        self.end_idx = max_valid_end if end_idx is None else min(max(end_idx, 0), max_valid_end)
        self.max_episode_steps = max_episode_steps
        self._rng = np.random.default_rng(seed)
        self._idx = start_idx
        self._portfolio = None
        self._nav_points: list[tuple[Any, float]] = []
        self._prepared = None
        self._done = False
        self._steps_taken = 0
        self._warm_start_cache: dict[int, tuple[Any, list[tuple[Any, float]]]] = {}

        if HAS_GYM:
            agg_min, agg_max = _aggressiveness_bounds(self.cfg)
            low_a = np.array(
                [-SECTOR_DELTA_MAX] * N_SECTORS
                + [agg_min - AGGRESSIVENESS_NEUTRAL, -1.0, -1.0],
                dtype=np.float32,
            )
            high_a = np.array(
                [SECTOR_DELTA_MAX] * N_SECTORS
                + [agg_max - AGGRESSIVENESS_NEUTRAL, 1.0, 1.0],
                dtype=np.float32,
            )
            self.action_space = spaces.Box(low=low_a, high=high_a, dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
            )

    def render(self):
        pass

    def close(self):
        pass

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        options = options or {}
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        max_idx = len(self.executor.rebalance_dates) - 2
        allowed_end_idx = min(
            int(options.get("end_idx", self.end_idx)),
            max_idx,
        )
        if "idx" in options:
            self._idx = int(options["idx"])
        elif allowed_end_idx > self.start_idx:
            self._idx = int(self._rng.integers(self.start_idx, allowed_end_idx + 1))
        else:
            self._idx = int(self.start_idx)
        self._idx = min(max(self._idx, 0), allowed_end_idx)
        self._episode_end_idx = allowed_end_idx
        self._episode_max_steps = options.get("max_episode_steps", self.max_episode_steps)
        if options.get("portfolio") is not None and options.get("nav_points") is not None:
            self._portfolio = options["portfolio"]
            self._nav_points = options["nav_points"]
        else:
            self._portfolio, self._nav_points = self._warm_started_state(self._idx)
        self.executor.reset_runtime_state(self._nav_points)
        self._prepared = self.executor.prepare_step(self._idx, self._portfolio, self._nav_points)
        self._done = False
        self._steps_taken = 0

        obs = SectorAllocationEnv.encode_observation(**self._prepared.transition_state)
        info = {
            "replay_only": False,
            "date": str(self._prepared.current_date.date()),
            "next_date": str(self._prepared.next_date.date()),
            "idx": self._idx,
            "end_idx": self._episode_end_idx,
        }
        return obs.astype(np.float32), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._prepared is None or self._portfolio is None:
            raise RuntimeError("HistoricalSectorAllocationEnv.reset() must be called before step().")
        if self._done:
            obs = SectorAllocationEnv.encode_observation(**self._prepared.transition_state)
            return obs.astype(np.float32), 0.0, True, False, {"replay_only": False}

        decision = SectorAllocationEnv.decode_action(np.asarray(action, dtype=np.float32), self.cfg)
        terminated, truncated = self._termination_flags_after_step()
        result = self.executor.execute_prepared_step(
            self._prepared,
            self._portfolio,
            self._nav_points,
            decision,
            done=terminated or truncated,
        )
        self._portfolio = result.next_portfolio
        self._nav_points = result.updated_nav_points
        self._steps_taken += 1
        self._done = terminated or truncated

        obs = SectorAllocationEnv.encode_observation(**result.transition["next_state"])
        info = dict(result.transition["info"])
        info["replay_only"] = False
        info["idx"] = self._idx
        info["steps_taken"] = self._steps_taken

        if not self._done:
            self.executor.engine.risk_engine.update(
                result.post_trade_portfolio.nav,
                result.post_trade_portfolio.date,
            )
            self._idx += 1
            self._prepared = self.executor.prepare_step(
                self._idx,
                self._portfolio,
                self._nav_points,
            )

        return obs.astype(np.float32), float(result.reward), terminated, truncated, info

    def _termination_flags_after_step(self) -> tuple[bool, bool]:
        next_steps_taken = self._steps_taken + 1
        if self._idx >= self._episode_end_idx:
            return True, False
        if self._episode_max_steps is not None and next_steps_taken >= int(self._episode_max_steps):
            return False, True
        return False, False

    def _warm_started_state(self, idx: int) -> tuple[Any, list[tuple[Any, float]]]:
        cached = self._warm_start_cache.get(idx)
        if cached is not None:
            portfolio, nav_points = cached
            return deepcopy(portfolio), list(nav_points)

        portfolio = self.executor.initial_portfolio(self.start_idx)
        nav_points = self.executor.initial_nav_points(self.start_idx)
        self.executor.reset_runtime_state(nav_points)

        for step_idx in range(self.start_idx, idx):
            prepared = self.executor.prepare_step(step_idx, portfolio, nav_points)
            result = self.executor.execute_prepared_step(
                prepared,
                portfolio,
                nav_points,
                SectorAllocationEnv.neutral_action(self.cfg),
                done=False,
            )
            self.executor.engine.risk_engine.update(
                result.post_trade_portfolio.nav,
                result.post_trade_portfolio.date,
            )
            portfolio = result.next_portfolio
            nav_points = result.updated_nav_points

        self._warm_start_cache[idx] = (deepcopy(portfolio), list(nav_points))
        return deepcopy(portfolio), list(nav_points)


def _rl_cfg(cfg: dict | None) -> dict:
    return (cfg or {}).get("rl", {}) if isinstance(cfg, dict) else {}


def _cash_control_enabled(cfg: dict | None) -> bool:
    return bool(_rl_cfg(cfg).get("enable_cash_control", False))


def _turnover_control_enabled(cfg: dict | None) -> bool:
    return bool(_rl_cfg(cfg).get("enable_turnover_control", False))


def _cash_buckets(cfg: dict | None) -> list[float]:
    buckets = _rl_cfg(cfg).get("cash_buckets", [0.0, 0.1, 0.2, 0.3])
    values = sorted({float(np.clip(v, 0.0, 0.30)) for v in buckets})
    return values or [CASH_TARGET_NEUTRAL]


def _turnover_buckets(cfg: dict | None) -> list[float]:
    buckets = _rl_cfg(cfg).get("turnover_buckets", [0.20, 0.30, 0.40])
    values = sorted({float(max(0.05, v)) for v in buckets})
    return values or [0.40]


def _aggressiveness_bounds(cfg: dict | None) -> tuple[float, float]:
    rl_cfg = _rl_cfg(cfg)
    agg_min = float(rl_cfg.get("aggressiveness_min", 0.60))
    agg_max = float(rl_cfg.get("aggressiveness_max", 1.40))
    if agg_min > agg_max:
        agg_min, agg_max = agg_max, agg_min
    return agg_min, agg_max


def _neutral_turnover_cap(cfg: dict | None) -> float | None:
    if cfg is not None and not _turnover_control_enabled(cfg):
        return None
    buckets = _turnover_buckets(cfg)
    cfg_max = (
        float((cfg or {}).get("optimizer", {}).get("max_turnover_per_rebalance", buckets[-1]))
        if isinstance(cfg, dict)
        else float(buckets[-1])
    )
    return float(min(max(buckets), cfg_max))


def _encode_bucket_slot(value: float, buckets: list[float], neutral_value: float) -> float:
    if len(buckets) <= 1:
        return 0.0
    idx = int(np.argmin([abs(float(value) - bucket) for bucket in buckets]))
    neutral_idx = int(np.argmin([abs(float(neutral_value) - bucket) for bucket in buckets]))
    if idx == neutral_idx:
        return 0.0
    if idx > neutral_idx:
        max_up = max(1, len(buckets) - 1 - neutral_idx)
        return float((idx - neutral_idx) / max_up)
    max_down = max(1, neutral_idx)
    return float(-((neutral_idx - idx) / max_down))


def _decode_bucket_slot(raw_value: float, buckets: list[float], neutral_value: float) -> float:
    if len(buckets) <= 1:
        return float(buckets[0])
    normalized = float(np.clip(raw_value, -1.0, 1.0))
    neutral_idx = int(np.argmin([abs(float(neutral_value) - bucket) for bucket in buckets]))
    if normalized >= 0:
        max_up = max(1, len(buckets) - 1 - neutral_idx)
        idx = neutral_idx + int(np.rint(normalized * max_up))
    else:
        max_down = max(1, neutral_idx)
        idx = neutral_idx - int(np.rint(abs(normalized) * max_down))
    idx = int(np.clip(idx, 0, len(buckets) - 1))
    return float(buckets[idx])


def _pad_action(action: np.ndarray) -> np.ndarray:
    raw = np.asarray(action, dtype=np.float32).flatten()
    if raw.size < ACTION_DIM:
        raw = np.pad(raw, (0, ACTION_DIM - raw.size))
    return raw
