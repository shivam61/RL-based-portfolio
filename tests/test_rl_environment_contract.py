from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.contracts import PortfolioState
from src.rl.agent import RLSectorAgent
import src.rl.agent as rl_agent_module
from src.rl.contract import CAUSAL_TRAINING_BACKEND, build_state, build_transition
from src.rl.environment import (
    SECTORS,
    HistoricalSectorAllocationEnv,
    SectorAllocationEnv,
)
from src.rl.historical_executor import HistoricalPeriodExecutor
from tests.test_backtest import _make_synthetic_data


def _cfg() -> dict:
    return {
        "rl": {
            "training_backend": "disabled",
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
            "cash_target": 0.05,
            "aggressiveness": 1.0,
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
