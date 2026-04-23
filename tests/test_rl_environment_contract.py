from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.contracts import PortfolioState
from src.rl.agent import RLSectorAgent
import src.rl.agent as rl_agent_module
from src.rl.contract import CAUSAL_TRAINING_BACKEND, build_state, build_transition
from src.rl.environment import (
    SECTORS,
    STATE_DIM,
    HistoricalSectorAllocationEnv,
    SectorAllocationEnv,
)
from src.rl.historical_executor import HistoricalPeriodExecutor
from src.rl.policy_utils import apply_posture_policy
from tests.test_backtest import _make_synthetic_data


def _cfg() -> dict:
    return {
        "optimizer": {
            "max_cash": 0.40,
            "max_turnover_per_rebalance": 0.45,
        },
        "rl": {
            "training_backend": "disabled",
            "enable_posture_control": True,
            "posture_neutral_band": 0.25,
            "posture_profiles": {
                "risk_off": {"cash_target": 0.35, "aggressiveness": 0.75, "turnover_cap": 0.15},
                "neutral": {"cash_target": 0.05, "aggressiveness": 1.00, "turnover_cap": 0.35},
                "risk_on": {"cash_target": 0.02, "aggressiveness": 1.30, "turnover_cap": 0.45},
            },
            "aggressiveness_min": 0.60,
            "aggressiveness_max": 1.40,
            "reward_lambda_dd": 2.0,
            "reward_lambda_to": 0.5,
            "reward_lambda_conc": 0.3,
            "reward_lambda_liq": 0.2,
        }
    }


def _transition_step(portfolio_return: float = 0.02) -> dict:
    sector_state = {
        sector: {"mom_1m": 0.1, "mom_3m": 0.2, "rel_str_1m": 0.05, "breadth_3m": 0.6}
        for sector in SECTORS
    }
    state = build_state(
        macro_state={"vix_level": 14.0, "macro_stress_score": 0.2, "risk_on_score": 0.7},
        sector_state=sector_state,
        portfolio_state={
            "cash_ratio": 0.05,
            "ret_1m": 0.01,
            "vol_1m": 0.02,
            "current_drawdown": 0.01,
            "max_drawdown": 0.03,
            "hhi": 0.08,
            "max_weight": 0.07,
            "sharpe_3m": 1.0,
            "active_ret_1m": 0.005,
            "n_stocks": 12,
        },
    )
    next_state = build_state(
        macro_state={"vix_level": 15.0, "macro_stress_score": 0.25, "risk_on_score": 0.65},
        sector_state=sector_state,
        portfolio_state={
            "cash_ratio": 0.08,
            "ret_1m": portfolio_return,
            "vol_1m": 0.03,
            "current_drawdown": 0.02,
            "max_drawdown": 0.04,
            "hhi": 0.09,
            "max_weight": 0.08,
            "sharpe_3m": 0.8,
            "active_ret_1m": 0.002,
            "n_stocks": 10,
        },
    )
    return build_transition(
        state=state,
        action={
            "sector_tilts": {sector: 1.0 for sector in SECTORS},
            "posture": "neutral",
            "cash_target": 0.05,
            "aggressiveness": 1.0,
            "turnover_cap": 0.40,
            "should_rebalance": True,
        },
        reward=portfolio_return,
        next_state=next_state,
        done=False,
        info={"date": "2024-01-01"},
    )


def test_sector_allocation_env_replays_logged_transition_and_reports_action_mismatch():
    experience = [_transition_step(0.02), _transition_step(-0.01), _transition_step(0.01)]

    env = SectorAllocationEnv(experience, _cfg(), seed=7)
    env._step_idx = 0

    neutral = SectorAllocationEnv.encode_action(_transition_step()["action"])
    defensive = np.array(
        [0.25 if idx == 0 else -0.10 for idx in range(len(SECTORS))] + [-0.20],
        dtype=np.float32,
    )

    next_obs, reward, done, truncated, info = env.step(defensive)

    assert done is False
    assert truncated is False
    assert reward == pytest.approx(0.02)
    assert next_obs.shape[0] > 0
    assert info["replay_only"] is True
    assert info["action_mismatch_l1"] > 0

    env._step_idx = 0
    _, neutral_reward, _, _, neutral_info = env.step(neutral)
    assert neutral_reward == pytest.approx(reward)
    assert neutral_info["action_mismatch_l1"] == pytest.approx(0.0)


def test_decode_action_maps_posture_slot_to_risk_off():
    cfg = _cfg()
    action = np.array([0.0] * len(SECTORS) + [-1.0], dtype=np.float32)

    decoded = SectorAllocationEnv.decode_action(action, cfg)

    assert decoded["posture"] == "risk_off"
    assert decoded["cash_target"] == pytest.approx(0.35)
    assert decoded["turnover_cap"] == pytest.approx(0.15)
    assert decoded["aggressiveness"] == pytest.approx(0.75)


def test_decode_action_maps_posture_slot_to_risk_on():
    cfg = _cfg()
    action = np.array([0.0] * len(SECTORS) + [1.0], dtype=np.float32)

    decoded = SectorAllocationEnv.decode_action(action, cfg)

    assert decoded["posture"] == "risk_on"
    assert decoded["cash_target"] == pytest.approx(0.02)
    assert decoded["turnover_cap"] == pytest.approx(0.45)
    assert decoded["aggressiveness"] == pytest.approx(1.30)


def test_apply_posture_policy_freezes_to_neutral_but_preserves_tilts():
    cfg = _cfg()
    cfg["rl"]["force_neutral_posture"] = True

    decision = apply_posture_policy(
        cfg,
        {
            "sector_tilts": {"IT": 1.4, "Banking": 0.8},
            "posture": "risk_on",
            "cash_target": 0.02,
            "aggressiveness": 1.30,
            "turnover_cap": 0.45,
        },
    )

    assert decision["sector_tilts"]["IT"] == pytest.approx(1.4)
    assert decision["posture"] == "neutral"
    assert decision["cash_target"] == pytest.approx(0.05)
    assert decision["aggressiveness"] == pytest.approx(1.0)
    assert decision["turnover_cap"] == pytest.approx(0.35)


def test_apply_posture_policy_respects_explicit_research_opt_out():
    cfg = _cfg()
    cfg["rl"]["force_neutral_posture"] = True

    decision = apply_posture_policy(
        cfg,
        {
            "sector_tilts": {"IT": 1.1},
            "posture": "risk_off",
            "cash_target": 0.35,
            "aggressiveness": 0.75,
            "turnover_cap": 0.15,
            "allow_forced_posture_override": False,
        },
    )

    assert decision["posture"] == "risk_off"
    assert decision["cash_target"] == pytest.approx(0.35)


def test_encode_observation_emits_finite_vector_with_control_features():
    observation = SectorAllocationEnv.encode_observation(**_transition_step()["state"])

    assert observation.shape == (STATE_DIM,)
    assert np.isfinite(observation).all()


def test_historical_executor_stress_signal_rises_with_stress_features():
    high_stress = HistoricalPeriodExecutor._stress_signal(
        {
            "macro_state": {"macro_stress_score": 0.8},
            "portfolio_state": {
                "current_drawdown": -0.10,
                "drawdown_slope_1m": -0.03,
                "vol_shock_1m_3m": 0.4,
                "breadth_deterioration": 0.7,
                "emergency_flag": 1.0,
            },
        }
    )
    low_stress = HistoricalPeriodExecutor._stress_signal(
        {
            "macro_state": {"macro_stress_score": 0.1},
            "portfolio_state": {
                "current_drawdown": -0.01,
                "drawdown_slope_1m": 0.0,
                "vol_shock_1m_3m": 0.0,
                "breadth_deterioration": 0.1,
                "emergency_flag": 0.0,
            },
        }
    )

    assert 0.0 <= low_stress < high_stress <= 1.0


def test_historical_executor_defensive_posture_reflects_cash_risk_and_turnover():
    defensive = HistoricalPeriodExecutor._defensive_posture(
        {"cash_target": 0.15, "aggressiveness": 0.90, "turnover_cap": 0.30}
    )
    neutral = HistoricalPeriodExecutor._defensive_posture(
        {"cash_target": 0.05, "aggressiveness": 1.0, "turnover_cap": 0.40}
    )

    assert defensive > neutral
    assert neutral == pytest.approx(0.0)


def test_historical_executor_target_controls_scale_with_stress():
    cfg = _cfg()
    cfg["rl"].update({"stress_target_moderate": 0.18, "stress_target_high": 0.35})

    low = HistoricalPeriodExecutor._target_controls_for_stress(0.10, cfg)
    moderate = HistoricalPeriodExecutor._target_controls_for_stress(0.20, cfg)
    high = HistoricalPeriodExecutor._target_controls_for_stress(0.40, cfg)

    assert low == {"cash_target": 0.02, "aggressiveness": 1.30, "turnover_cap": 0.45}
    assert moderate == {"cash_target": 0.05, "aggressiveness": 1.0, "turnover_cap": 0.35}
    assert high == {"cash_target": 0.35, "aggressiveness": 0.75, "turnover_cap": 0.15}


def test_historical_executor_guidance_blends_controls_toward_target():
    executor = HistoricalPeriodExecutor.__new__(HistoricalPeriodExecutor)
    executor.engine = type(
        "Engine",
        (),
        {
            "cfg": {
                "rl": {
                    "enable_target_control_blend": True,
                    "target_control_blend_min": 0.15,
                    "target_control_blend_max": 0.85,
                    "stress_target_moderate": 0.18,
                    "stress_target_high": 0.35,
                    "posture_profiles": {
                        "risk_off": {"cash_target": 0.35, "aggressiveness": 0.75, "turnover_cap": 0.15},
                        "neutral": {"cash_target": 0.05, "aggressiveness": 1.0, "turnover_cap": 0.35},
                        "risk_on": {"cash_target": 0.02, "aggressiveness": 1.30, "turnover_cap": 0.45},
                    },
                    "aggressiveness_min": 0.60,
                    "aggressiveness_max": 1.40,
                },
                "optimizer": {"max_turnover_per_rebalance": 0.45, "max_cash": 0.40},
            }
        },
    )()
    executor.sectors = list(SECTORS)

    decision = {
        "sector_tilts": {sector: 1.0 for sector in SECTORS},
        "posture": "neutral",
        "cash_target": 0.15,
        "aggressiveness": 1.0,
        "turnover_cap": 0.30,
        "should_rebalance": True,
    }
    state = {
        "macro_state": {"macro_stress_score": 0.5},
        "portfolio_state": {
            "current_drawdown": -0.12,
            "drawdown_slope_1m": -0.04,
            "vol_shock_1m_3m": 0.4,
            "breadth_deterioration": 0.6,
            "emergency_flag": 1.0,
        },
    }

    guided, guidance = HistoricalPeriodExecutor._apply_target_control_guidance(
        executor,
        decision,
        state,
    )

    assert guidance["enabled"] is True
    assert guidance["blend_weight"] > 0.15
    assert guidance["target_posture"] == "risk_off"
    assert guided["cash_target"] > decision["cash_target"]
    assert guided["aggressiveness"] < decision["aggressiveness"]
    assert guided["turnover_cap"] < decision["turnover_cap"]
    assert guided["posture"] == "risk_off"


def test_historical_executor_posture_guidance_steps_toward_target():
    executor = HistoricalPeriodExecutor.__new__(HistoricalPeriodExecutor)
    executor.engine = type(
        "Engine",
        (),
        {
            "cfg": {
                "rl": {
                    "enable_target_posture_guidance": True,
                    "target_posture_guidance_min_streak": 2,
                    "target_posture_guidance_min_stress": 0.16,
                    "target_posture_guidance_max_step": 1,
                    "enable_target_control_blend": False,
                    "posture_profiles": {
                        "risk_off": {"cash_target": 0.35, "aggressiveness": 0.75, "turnover_cap": 0.15},
                        "neutral": {"cash_target": 0.05, "aggressiveness": 1.0, "turnover_cap": 0.35},
                        "risk_on": {"cash_target": 0.02, "aggressiveness": 1.30, "turnover_cap": 0.45},
                    },
                }
            }
        },
    )()

    decision = {
        "sector_tilts": {sector: 1.0 for sector in SECTORS},
        "posture": "risk_on",
        "cash_target": 0.03,
        "aggressiveness": 1.05,
        "turnover_cap": 0.40,
        "should_rebalance": True,
    }
    state = {
        "macro_state": {"macro_stress_score": 0.8},
        "portfolio_state": {
            "current_drawdown": -0.12,
            "drawdown_slope_1m": -0.04,
            "vol_shock_1m_3m": 0.4,
            "breadth_deterioration": 0.6,
            "emergency_flag": 1.0,
            "target_posture_streak": 3.0,
            "previous_target_posture_score": 1.0,
        },
    }

    guided, guidance = HistoricalPeriodExecutor._apply_target_control_guidance(
        executor,
        decision,
        state,
    )

    assert guidance["applied_posture_guidance"] is True
    assert guidance["target_posture"] == "risk_off"
    assert guided["posture"] == "neutral"
    assert guided["cash_target"] == pytest.approx(0.05)


def test_target_context_from_state_keeps_streak_and_previous_target_consistent():
    state = {
        "portfolio_state": {
            "target_posture_streak": 4.0,
            "previous_target_posture_score": -1.0,
        }
    }

    prev_target, streak = HistoricalPeriodExecutor._target_context_from_state(
        state,
        "neutral",
    )

    assert streak == pytest.approx(4.0)
    assert prev_target == "neutral"


def test_approximate_posture_weights_create_materially_different_portfolios():
    current_weights = {
        "A": 0.22,
        "B": 0.21,
        "C": 0.17,
        "D": 0.15,
        "E": 0.10,
        "F": 0.10,
        "CASH": 0.05,
    }
    base_target_weights = {
        "A": 0.28,
        "B": 0.24,
        "C": 0.18,
        "D": 0.12,
        "E": 0.10,
        "F": 0.08,
    }

    risk_on_weights, risk_on_turnover = HistoricalPeriodExecutor._approximate_posture_weights(
        current_weights=current_weights,
        base_target_weights=base_target_weights,
        candidate_decision={"cash_target": 0.02, "aggressiveness": 1.30, "turnover_cap": 0.45},
    )
    risk_off_weights, risk_off_turnover = HistoricalPeriodExecutor._approximate_posture_weights(
        current_weights=current_weights,
        base_target_weights=base_target_weights,
        candidate_decision={"cash_target": 0.35, "aggressiveness": 0.75, "turnover_cap": 0.15},
    )

    risk_on_cash = risk_on_weights.get("CASH", 0.0)
    risk_off_cash = risk_off_weights.get("CASH", 0.0)
    l1_gap = sum(
        abs(risk_on_weights.get(ticker, 0.0) - risk_off_weights.get(ticker, 0.0))
        for ticker in set(risk_on_weights) | set(risk_off_weights)
    )

    assert risk_on_cash < 0.06
    assert risk_off_cash >= 0.20
    assert risk_off_cash - risk_on_cash > 0.14
    assert risk_off_turnover >= 0.14
    assert risk_on_turnover < 0.10
    assert l1_gap > 0.30


def test_historical_executor_reward_uses_bounded_weights_and_soft_regret(monkeypatch):
    executor = HistoricalPeriodExecutor.__new__(HistoricalPeriodExecutor)
    cfg = _cfg()
    cfg["rl"].update(
        {
            "enable_soft_posture_regret": True,
            "reward_regret_lambda": 0.2,
            "reward_regret_temperature": 0.05,
            "reward_return_weight_low": 1.15,
            "reward_return_weight_high": 0.70,
            "reward_drawdown_weight_low": 0.80,
            "reward_drawdown_weight_high": 1.35,
            "reward_turnover_weight_low": 0.85,
            "reward_turnover_weight_high": 1.20,
            "reward_return_weight_low_reduced": 1.0,
            "reward_return_weight_high_reduced": 0.80,
            "reward_drawdown_weight_low_reduced": 0.75,
            "reward_drawdown_weight_high_reduced": 1.15,
            "reward_turnover_weight_low_reduced": 0.90,
            "reward_turnover_weight_high_reduced": 1.05,
            "reward_regret_variance_threshold": 1.0e-5,
        }
    )
    executor.engine = type("Engine", (), {"cfg": cfg})()
    executor.bm_prices = None
    monkeypatch.setattr(
        HistoricalPeriodExecutor,
        "_compute_posture_value_map",
        lambda self, **kwargs: (
            {"risk_on": 0.03, "neutral": 0.02, "risk_off": 0.01},
            {"horizon_steps": 3, "policy_count": 7},
        ),
    )

    portfolio = PortfolioState(
        date=pd.Timestamp("2024-01-01").date(),
        cash=50000.0,
        holdings={"AAA": 100.0},
        weights={"AAA": 0.90, "CASH": 0.10},
        nav=100000.0,
        sector_weights={"IT": 0.90},
    )
    reward, components = HistoricalPeriodExecutor._compute_reward(
        executor,
        step_idx=0,
        starting_portfolio=portfolio,
        starting_nav_points=[(pd.Timestamp("2024-01-01"), 100000.0)],
        current_weights={"AAA": 0.90, "CASH": 0.10},
        base_target_weights={"AAA": 0.95, "CASH": 0.05},
        realized_asset_returns={"AAA": 0.03},
        observed_turnover=0.15,
        pre_nav=100000.0,
        end_nav=103000.0,
        nav_points=[(pd.Timestamp("2024-01-01"), 100000.0)],
        period_nav=[(pd.Timestamp("2024-01-15"), 99000.0), (pd.Timestamp("2024-01-29"), 103000.0)],
        exec_result=SimpleNamespace(total_cost=200.0, total_turnover=0.15),
        next_portfolio=portfolio,
        liquidity_stress=False,
        current_date=pd.Timestamp("2024-01-01"),
        next_date=pd.Timestamp("2024-01-29"),
        decision={
            "sector_tilts": {sector: 1.0 for sector in SECTORS},
            "posture": "neutral",
            "cash_target": 0.05,
            "aggressiveness": 1.0,
            "turnover_cap": 0.40,
            "should_rebalance": True,
        },
        current_state={
            "macro_state": {"macro_stress_score": 0.4},
            "portfolio_state": {
                "current_drawdown": -0.08,
                "drawdown_slope_1m": -0.02,
                "vol_shock_1m_3m": 0.2,
                "breadth_deterioration": 0.3,
                "emergency_flag": 0.0,
                "previous_posture_score": 0.0,
                "previous_target_posture_score": 0.0,
                "target_posture_streak": 2.0,
            },
        },
        next_state={
            "macro_state": {"macro_stress_score": 0.5},
            "portfolio_state": {
                "current_drawdown": -0.06,
                "drawdown_slope_1m": -0.01,
                "vol_shock_1m_3m": 0.2,
                "breadth_deterioration": 0.3,
                "emergency_flag": 0.0,
            },
        },
        compute_regret=True,
    )

    assert 0.70 <= components["return_weight"] <= 1.15
    assert 0.80 <= components["drawdown_weight"] <= 1.35
    assert 0.85 <= components["turnover_weight"] <= 1.20
    assert components["decision_quality_basis"] == "cached_one_step_soft_regret_v1"
    assert components["soft_regret"] >= 0.0
    assert components["soft_regret_penalty"] >= 0.0
    assert components["posture_utility_variance"] > 0.0
    assert components["regret_horizon_steps"] == 3
    assert components["regret_policy_count"] == 7
    assert reward == pytest.approx(components["reward"])


def test_rl_agent_trains_fresh_policy_on_causal_env(monkeypatch):
    agent = RLSectorAgent(_cfg())
    agent.cfg.setdefault("backtest", {})
    agent.cfg["backtest"]["random_seed"] = 7
    agent._rl_cfg["training_backend"] = CAUSAL_TRAINING_BACKEND
    agent._rl_cfg["algorithm"] = "PPO"
    agent._rl_cfg["learning_rate"] = 3e-4
    agent._rl_cfg["n_steps"] = 8
    agent._rl_cfg["batch_size"] = 4
    agent._rl_cfg["n_epochs"] = 1
    agent._rl_cfg["gamma"] = 0.99
    agent._rl_cfg["gae_lambda"] = 0.95
    agent._rl_cfg["clip_range"] = 0.2
    agent._rl_cfg["ent_coef"] = 0.01

    train_log = {}

    class FakePPO:
        def __init__(self, policy, env, **kwargs):
            train_log["policy"] = policy
            train_log["env"] = env
            train_log["kwargs"] = kwargs

        def learn(self, total_timesteps):
            train_log["total_timesteps"] = total_timesteps
            return self

    monkeypatch.setattr(rl_agent_module, "HAS_SB3", True)
    monkeypatch.setattr(rl_agent_module, "PPO", FakePPO)

    fake_env = SimpleNamespace()
    agent.train(total_timesteps=64, causal_env=fake_env)

    assert agent.is_trained is True
    assert agent.training_backend == CAUSAL_TRAINING_BACKEND
    assert agent.disable_reason is None
    assert train_log["env"] is fake_env
    assert train_log["total_timesteps"] == 64


def test_rl_experience_step_contains_causal_transition_contract():
    cfg = load_config()
    cfg["backtest"]["start_date"] = "2013-01-01"
    cfg["backtest"]["end_date"] = "2014-06-30"
    cfg["backtest"]["min_train_years"] = 1
    cfg["sector_model"]["n_estimators"] = 5
    cfg["stock_model"]["n_estimators"] = 5
    cfg["rl"]["use_rl"] = True
    cfg["rl"]["training_backend"] = "disabled"

    price_matrix, volume_matrix, macro_df = _make_synthetic_data(n_tickers=10, n_days=800)

    from src.backtest.walk_forward import WalkForwardEngine

    engine = WalkForwardEngine(
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        macro_df=macro_df,
        cfg=cfg,
        mode="full_rl",
    )
    engine.run()

    assert engine.rl_agent._experience_buffer
    recorded_step = engine.rl_agent._experience_buffer[0]
    required_keys = {"state", "action", "reward", "next_state", "done"}

    missing = required_keys - set(recorded_step)
    assert not missing, (
        "Recorded RL experience must contain a causal transition tuple. "
        f"Missing keys: {sorted(missing)}"
    )


def _executor_engine(mode: str = "optimizer_only"):
    cfg = load_config()
    cfg["backtest"]["start_date"] = "2013-01-01"
    cfg["backtest"]["end_date"] = "2014-06-30"
    cfg["backtest"]["min_train_years"] = 1
    cfg["sector_model"]["n_estimators"] = 5
    cfg["stock_model"]["n_estimators"] = 5
    cfg["rl"]["use_rl"] = mode == "full_rl"
    cfg["rl"]["training_backend"] = CAUSAL_TRAINING_BACKEND

    price_matrix, volume_matrix, macro_df = _make_synthetic_data(n_tickers=10, n_days=800)

    from src.backtest.walk_forward import WalkForwardEngine

    engine = WalkForwardEngine(
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        macro_df=macro_df,
        cfg=cfg,
        mode=mode,
    )
    return engine, cfg


def test_historical_executor_neutral_action_matches_walk_forward_one_step():
    baseline_engine, _ = _executor_engine(mode="optimizer_only")
    baseline_metrics = baseline_engine.run()
    assert baseline_metrics["total_rebalances"] > 0

    first_date = baseline_engine.rebalance_records[0].rebalance_date
    second_date = baseline_engine.rebalance_records[1].rebalance_date
    expected_end_nav = float(
        baseline_engine.nav_series.loc[:str(second_date)].iloc[-1]
    )
    expected_period_return = (
        expected_end_nav - baseline_engine.rebalance_records[0].pre_nav
    ) / baseline_engine.rebalance_records[0].pre_nav

    step_engine, _ = _executor_engine(mode="optimizer_only")
    executor = HistoricalPeriodExecutor(step_engine, mode="optimizer_only")
    portfolio = executor.initial_portfolio(0)
    nav_points = executor.initial_nav_points(0)
    executor.reset_runtime_state(nav_points)
    prepared = executor.prepare_step(0, portfolio, nav_points)
    result = executor.execute_prepared_step(
        prepared,
        portfolio,
        nav_points,
        rl_decision=None,
        done=True,
    )

    assert str(result.current_date.date()) == str(first_date)
    assert str(result.next_date.date()) == str(second_date)
    assert result.target_weights == pytest.approx(
        baseline_engine.rebalance_records[0].target_weights,
        rel=1e-6,
        abs=1e-6,
    )
    assert result.exec_result.total_turnover == pytest.approx(
        baseline_engine.rebalance_records[0].total_turnover,
        rel=1e-6,
        abs=1e-6,
    )
    assert result.exec_result.total_cost == pytest.approx(
        baseline_engine.rebalance_records[0].total_cost,
        rel=1e-6,
        abs=1e-6,
    )
    reward_components = result.transition["info"]["reward_components"]
    assert reward_components["period_return"] == pytest.approx(
        expected_period_return,
        abs=1e-3,
    )
    assert result.reward == pytest.approx(reward_components["reward"])
    assert result.reward <= reward_components["period_return"] + 1e-9


def test_historical_executor_same_state_different_actions_change_outcome():
    engine, _ = _executor_engine(mode="full_rl")
    executor = HistoricalPeriodExecutor(engine, mode="full_rl")
    current_date = executor.rebalance_dates[1]
    prices_today = engine._get_prices(current_date)
    invested_ticker = next(ticker for ticker in prices_today.index if ticker != "^NSEI")
    entry_price = float(prices_today[invested_ticker])
    invested_value = engine.initial_capital * 0.60
    shares = invested_value / entry_price
    snapshot = engine.universe_mgr.get_universe(
        current_date.date(),
        price_matrix=engine.price_matrix,
        volume_matrix=engine.volume_matrix,
    )
    sector_map = engine.universe_mgr.get_sector_map(snapshot)
    sector = sector_map.get(invested_ticker, "Unknown")
    portfolio = PortfolioState(
        date=current_date.date(),
        cash=float(engine.initial_capital - invested_value),
        holdings={invested_ticker: shares},
        weights={invested_ticker: 0.60, "CASH": 0.40},
        nav=float(engine.initial_capital),
        sector_weights={sector: 0.60},
    )
    nav_points = executor.initial_nav_points(1)
    executor.reset_runtime_state(nav_points)
    prepared = executor.prepare_step(1, portfolio, nav_points)

    hold = executor.execute_prepared_step(
        prepared,
        portfolio,
        nav_points,
        rl_decision=SectorAllocationEnv.neutral_action(),
        done=True,
    )
    executor.reset_runtime_state(nav_points)
    prepared = executor.prepare_step(1, portfolio, nav_points)
    rebalance = executor.execute_prepared_step(
        prepared,
        portfolio,
        nav_points,
        rl_decision={
            "sector_tilts": {
                sector: (0.3 if idx % 2 == 0 else 1.4)
                for idx, sector in enumerate(prepared.snapshot.sectors)
            },
            "cash_target": 0.20,
            "aggressiveness": 0.6,
            "should_rebalance": True,
        },
        done=True,
    )

    assert (
        rebalance.transition["info"]["reward_components"]["action_deviation_penalty"]
        > hold.transition["info"]["reward_components"]["action_deviation_penalty"]
    )
    assert rebalance.reward != pytest.approx(hold.reward)


def test_causal_env_step_returns_executor_transition_not_replay_artifact():
    engine, _ = _executor_engine(mode="full_rl")
    executor = HistoricalPeriodExecutor(engine, mode="full_rl")
    env = HistoricalSectorAllocationEnv(
        executor,
        start_idx=0,
        max_episode_steps=2,
        seed=11,
    )

    obs, info = env.reset(options={"idx": 0})
    assert obs.shape[0] > 0
    assert info["replay_only"] is False

    action = np.array(
        [0.20 if idx % 2 == 0 else -0.10 for idx in range(len(SECTORS))] + [-0.10],
        dtype=np.float32,
    )
    next_obs, reward, done, truncated, step_info = env.step(action)

    assert next_obs.shape[0] > 0
    assert done is False
    assert truncated is False
    assert step_info["replay_only"] is False
    assert "target_weights" in step_info
    assert "turnover" in step_info
    assert "transaction_cost" in step_info
    assert "selected_sectors" in step_info
    assert reward == pytest.approx(step_info["reward_components"]["reward"])
    assert "benchmark_return" in step_info["reward_components"]
    assert step_info["date"] != step_info["next_date"]

    second_obs, _, second_done, second_truncated, second_info = env.step(action)
    assert second_obs.shape[0] > 0
    assert second_done is False
    assert second_truncated is True
    assert second_info["replay_only"] is False


def test_walk_forward_train_rl_uses_causal_env(monkeypatch):
    engine, _ = _executor_engine(mode="full_rl")
    for _ in range(30):
        engine.rl_agent.record_step(_transition_step())

    captured = {}

    def fake_build_causal_env(executor, **kwargs):
        captured["executor"] = executor
        captured["env_kwargs"] = kwargs
        return "causal-env"

    def fake_train(*, total_timesteps=None, causal_env=None, **kwargs):
        captured["total_timesteps"] = total_timesteps
        captured["causal_env"] = causal_env
        engine.rl_agent.is_trained = True
        engine.rl_agent.training_backend = CAUSAL_TRAINING_BACKEND
        return engine.rl_agent

    monkeypatch.setattr(engine.rl_agent, "build_causal_env", fake_build_causal_env)
    monkeypatch.setattr(engine.rl_agent, "train", fake_train)

    engine._train_rl(current_idx=12)

    assert isinstance(captured["executor"], HistoricalPeriodExecutor)
    assert captured["env_kwargs"]["start_idx"] == 0
    assert captured["env_kwargs"]["end_idx"] == 11
    assert captured["causal_env"] == "causal-env"
