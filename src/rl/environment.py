"""
RL Environment for sector allocation.

One step = one 4-week rebalance period.
State  : macro + sector momentum + portfolio state (≈45 dims)
Action : sector tilt multipliers + cash target + aggressiveness + rebalance flag
Reward : portfolio_return - drawdown_pen - turnover_pen - concentration_pen
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    gym = None  # type: ignore
    spaces = None  # type: ignore

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
# [0:N_SECTORS]        sector tilt multipliers  [0.3, 2.0]
# [N_SECTORS]          cash target              [0.0, 0.30]
# [N_SECTORS+1]        aggressiveness           [0.5, 1.5]
# [N_SECTORS+2]        rebalance threshold      [0.0, 1.0]  (>0.5 → rebalance)
ACTION_DIM = N_SECTORS + 3

# State dimensions
MACRO_DIM = 12          # macro features
SECTOR_DIM = N_SECTORS * 4  # per-sector: mom_1m, mom_3m, rel_str_1m, breadth_3m
PORT_DIM = 10           # portfolio state features
REALIZED_SECTOR_DIM = N_SECTORS  # actual post-optimizer sector weights from prior rebalance
STATE_DIM = MACRO_DIM + SECTOR_DIM + PORT_DIM + REALIZED_SECTOR_DIM  # 12+60+10+15 = 97


_GymBase = gym.Env if HAS_GYM else object


class SectorAllocationEnv(_GymBase):
    """
    Gymnasium environment for RL sector allocation.

    Designed for offline training from experience replay collected
    during the walk-forward backtest.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        experience_buffer: list[dict],   # list of step dicts from walk-forward
        cfg: dict,
        seed: int = 42,
    ):
        if HAS_GYM:
            super().__init__()
        self.experience = experience_buffer
        self.cfg = cfg
        self._rl_cfg = cfg["rl"]
        self._rng = np.random.default_rng(seed)
        self._step_idx: int = 0
        self._max_steps: int = max(len(experience_buffer), 1)
        self._return_history: list[float] = []   # rolling window for vol estimation

        if HAS_GYM:
            self._setup_gym_spaces()

    def _setup_gym_spaces(self) -> None:
        low_a = np.array(
            [0.3] * N_SECTORS + [0.0, 0.5, 0.0], dtype=np.float32
        )
        high_a = np.array(
            [2.0] * N_SECTORS + [0.30, 1.5, 1.0], dtype=np.float32
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
        # start from a random point in experience buffer
        self._step_idx = int(self._rng.integers(0, max(1, self._max_steps - 10)))
        self._return_history = []
        obs = self._get_obs(self._step_idx)
        return obs.astype(np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._step_idx >= self._max_steps - 1:
            obs = self._get_obs(self._step_idx)
            return obs.astype(np.float32), 0.0, True, False, {}

        exp = self.experience[self._step_idx]
        reward = self._compute_reward(exp, action)
        self._step_idx += 1
        obs = self._get_obs(self._step_idx)
        done = self._step_idx >= self._max_steps - 1
        return obs.astype(np.float32), float(reward), done, False, {}

    # ── State builder ─────────────────────────────────────────────────────────

    def _get_obs(self, idx: int) -> np.ndarray:
        if idx >= self._max_steps:
            return np.zeros(STATE_DIM, dtype=np.float32)

        exp = self.experience[idx]
        state_vec = np.zeros(STATE_DIM, dtype=np.float32)
        offset = 0

        # macro features
        macro = exp.get("macro_state", {})
        macro_keys = [
            "vix_level", "usdinr_ret_1m", "crude_ret_1m", "sp500_ret_1m",
            "gold_ret_1m", "risk_on_score", "macro_stress_score",
            "rbi_rate", "rate_cutting_cycle", "election_window",
            "fii_flow_zscore", "fii_sell_regime",
        ]
        for i, k in enumerate(macro_keys[:MACRO_DIM]):
            v = macro.get(k, 0.0)
            state_vec[offset + i] = 0.0 if (v is None or np.isnan(v)) else float(v)
        offset += MACRO_DIM

        # sector features
        sector_state = exp.get("sector_state", {})
        for j, sec in enumerate(SECTORS):
            sec_data = sector_state.get(sec, {})
            state_vec[offset + j * 4 + 0] = float(sec_data.get("mom_1m", 0) or 0)
            state_vec[offset + j * 4 + 1] = float(sec_data.get("mom_3m", 0) or 0)
            state_vec[offset + j * 4 + 2] = float(sec_data.get("rel_str_1m", 0) or 0)
            state_vec[offset + j * 4 + 3] = float(sec_data.get("breadth_3m", 0) or 0)
        offset += SECTOR_DIM

        # portfolio state
        port = exp.get("portfolio_state", {})
        port_keys = [
            "cash_ratio", "ret_1m", "vol_1m", "current_drawdown",
            "max_drawdown", "hhi", "max_weight", "sharpe_3m",
            "active_ret_1m", "n_stocks",
        ]
        for i, k in enumerate(port_keys[:PORT_DIM]):
            v = port.get(k, 0.0)
            state_vec[offset + i] = 0.0 if (v is None or np.isnan(v)) else float(v)
        offset += PORT_DIM

        # realized sector weights from previous step's outcome
        # gives RL feedback on what the optimizer actually produced vs intended tilts
        if idx > 0:
            prev_realized = self.experience[idx - 1].get("outcome", {}).get(
                "realized_sector_weights", {}
            )
        else:
            prev_realized = {}
        for j, sec in enumerate(SECTORS):
            state_vec[offset + j] = float(prev_realized.get(sec, 0.0))

        # clip and normalize
        state_vec = np.clip(state_vec, -10, 10)
        return state_vec

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, exp: dict, action: np.ndarray) -> float:
        rl_cfg = self._rl_cfg
        outcome = exp.get("outcome", {})

        port_return = float(outcome.get("portfolio_return", 0.0))
        drawdown = abs(float(outcome.get("max_drawdown_episode", 0.0)))
        turnover = float(outcome.get("turnover", 0.0))
        hhi = float(outcome.get("concentration_hhi", 0.0))
        liq_stress = float(outcome.get("liquidity_stress", 0.0))

        # Sharpe-based reward: scale return by rolling vol estimate (annualized)
        self._return_history.append(port_return)
        window = min(len(self._return_history), 12)  # ~12 periods = 1 year
        if window >= 3:
            vol = float(np.std(self._return_history[-window:]) * np.sqrt(13) + 1e-6)
            # risk-free per 4-week period (6.5% annual RBI rate)
            rf_period = 0.065 / 13
            sharpe_component = (port_return - rf_period) / vol
        else:
            sharpe_component = port_return * 10  # no vol history yet; scale raw return

        reward = (
            sharpe_component
            - rl_cfg["reward_lambda_dd"] * drawdown
            - rl_cfg["reward_lambda_to"] * turnover
            - rl_cfg["reward_lambda_conc"] * hhi
            - rl_cfg["reward_lambda_liq"] * liq_stress
        )
        return float(reward)

    # ── Action decoder ────────────────────────────────────────────────────────

    @staticmethod
    def decode_action(action: np.ndarray) -> dict:
        """Convert raw action vector into structured decisions."""
        sector_tilts = {
            SECTORS[i]: float(np.clip(action[i], 0.3, 2.0))
            for i in range(N_SECTORS)
        }
        cash_target = float(np.clip(action[N_SECTORS], 0.0, 0.30))
        aggressiveness = float(np.clip(action[N_SECTORS + 1], 0.5, 1.5))
        rebalance_signal = float(action[N_SECTORS + 2]) > 0.5

        return {
            "sector_tilts": sector_tilts,
            "cash_target": cash_target,
            "aggressiveness": aggressiveness,
            "should_rebalance": rebalance_signal,
        }

    @staticmethod
    def neutral_action() -> dict:
        """Default no-tilt action for non-RL baseline."""
        return {
            "sector_tilts": {s: 1.0 for s in SECTORS},
            "cash_target": 0.05,
            "aggressiveness": 1.0,
            "should_rebalance": True,
        }
