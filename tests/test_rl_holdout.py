from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.rl.contract import CAUSAL_TRAINING_BACKEND
from src.rl.holdout import _summarize_trace, evaluate_holdout
from src.rl.agent import RLSectorAgent
from src.rl.environment import SECTORS
from src.rl.historical_executor import HistoricalPeriodExecutor
from tests.test_backtest import _make_synthetic_data


def test_evaluate_holdout_returns_trained_vs_neutral_comparison(monkeypatch):
    price_matrix, volume_matrix, macro_df = _make_synthetic_data(n_tickers=10, n_days=900)
    cfg = load_config()
    cfg["backtest"]["start_date"] = "2013-01-01"
    cfg["backtest"]["end_date"] = "2015-12-31"
    cfg["backtest"]["min_train_years"] = 1
    cfg["backtest"]["random_seed"] = 7
    cfg["sector_model"]["n_estimators"] = 5
    cfg["stock_model"]["n_estimators"] = 5
    cfg["rl"]["training_backend"] = CAUSAL_TRAINING_BACKEND
    cfg["rl"]["total_timesteps"] = 8
    cfg["rl"]["n_steps"] = 4
    cfg["rl"]["batch_size"] = 4
    cfg["rl"]["n_epochs"] = 1
    cfg["rl"]["min_history_rebalances"] = 3

    def fake_train(self, experience_buffer=None, total_timesteps=None, causal_env=None):
        self.is_trained = True
        self.training_backend = CAUSAL_TRAINING_BACKEND
        self.disable_reason = None
        return self

    def fake_decide(self, macro_state, sector_state, portfolio_state, prev_realized_sector_weights=None):
        return {
            "sector_tilts": {
                sector: (0.3 if idx % 2 == 0 else 1.4)
                for idx, sector in enumerate(SECTORS)
            },
            "posture": "risk_off",
            "cash_target": 0.20,
            "aggressiveness": 0.90,
            "turnover_cap": 0.25,
            "should_rebalance": True,
        }

    monkeypatch.setattr(RLSectorAgent, "train", fake_train)
    monkeypatch.setattr(RLSectorAgent, "decide", fake_decide)
    monkeypatch.setattr(
        HistoricalPeriodExecutor,
        "_compute_posture_value_map",
        lambda self, **kwargs: (
            {"risk_on": 0.03, "neutral": 0.02, "risk_off": 0.01},
            {"horizon_steps": 3, "policy_count": 7},
        ),
    )

    result = evaluate_holdout(
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        macro_df=macro_df,
        cfg=cfg,
        holdout_start="2014-07-01",
        holdout_end="2015-12-31",
        total_timesteps=8,
    )

    assert result["trained_policy_ready"] is True
    assert result["training_backend"] == CAUSAL_TRAINING_BACKEND
    assert result["holdout_windows"] > 0
    assert "cagr" in result["trained_policy"]
    assert "cagr" in result["neutral_policy"]
    assert "cagr" in result["uplift"]
    assert (
        result["trained_policy_diagnostics"]["mean_cash_target"]
        > result["neutral_policy_diagnostics"]["mean_cash_target"]
    )
    assert result["trained_policy_diagnostics"]["cash_usage_rate"] > 0.0
    assert result["trained_policy_diagnostics"]["aggressiveness_usage_rate"] > 0.0
    assert result["trained_policy_diagnostics"]["posture_usage_rate"] > 0.0
    assert "risk_off" in result["trained_policy_diagnostics"]["unique_postures"]
    assert result["trained_policy_diagnostics"]["posture_counts"]["risk_off"] > 0
    assert result["trained_policy_diagnostics"]["target_posture_counts"]
    assert result["trained_policy_diagnostics"]["posture_by_stress_bucket"]
    assert result["trained_policy_diagnostics"]["target_posture_by_stress_bucket"]
    assert (
        result["trained_policy_diagnostics"]["decision_quality_basis"]
        == "cached_one_step_soft_regret_v1"
    )
    assert "mean_regret" in result["trained_policy_diagnostics"]
    assert "regret_by_stress_bucket" in result["trained_policy_diagnostics"]
    assert "posture_optimality_rate_by_stress_bucket" in result["trained_policy_diagnostics"]
    assert "mean_posture_utility_dispersion" in result["trained_policy_diagnostics"]
    assert "posture_utility_dispersion_by_stress_bucket" in result["trained_policy_diagnostics"]
    assert "mean_requested_vs_realized_cash_gap" in result["trained_policy_diagnostics"]
    assert "optimizer_reason_counts" in result["trained_policy_diagnostics"]
    assert "optimizer_fallback_counts" in result["trained_policy_diagnostics"]
    assert "optimizer_relaxation_tier_counts" in result["trained_policy_diagnostics"]
    assert "control_realization_by_posture" in result["trained_policy_diagnostics"]
    assert "control_realization_by_stress_bucket" in result["trained_policy_diagnostics"]
    assert result["trained_policy_behavior_flags"]["advisory_only"] is True
    assert "unique_posture_count" in result["trained_policy_behavior_flags"]
    assert "posture_change_rate" in result["trained_policy_behavior_flags"]


def test_summarize_trace_reports_dispersion_and_bucketed_decision_quality():
    trace = [
        {
            "cash_target": 0.20,
            "turnover_cap": 0.25,
            "aggressiveness": 0.90,
            "turnover": 0.20,
            "reward": 0.01,
            "should_rebalance": True,
            "posture": "risk_off",
            "selected_sectors": ["IT"],
            "selected_stock_count": 10,
            "sector_tilts": {"IT": 1.0},
            "reward_components": {
                "stress_signal": 0.50,
                "target_posture": "risk_off",
                "best_posture": "risk_off",
                "soft_regret": 0.0,
                "posture_utility_variance": 0.02,
                "defensive_posture": 0.6,
                "target_defensive_posture": 0.7,
                "target_posture_penalty": 0.0,
                "posture_progress_bonus": 0.0,
                "posture_stale_penalty": 0.0,
                "posture_flip_penalty": 0.0,
                "posture_distance_to_target": 0.0,
                "active_return": 0.01,
                "drawdown_penalty": 0.02,
                "turnover_penalty": 0.01,
                "concentration_penalty": 0.0,
                "liquidity_penalty": 0.0,
            },
        },
        {
            "cash_target": 0.05,
            "turnover_cap": 0.40,
            "aggressiveness": 1.0,
            "turnover": 0.30,
            "reward": -0.01,
            "should_rebalance": True,
            "posture": "neutral",
            "selected_sectors": ["IT"],
            "selected_stock_count": 11,
            "sector_tilts": {"IT": 1.0},
            "reward_components": {
                "stress_signal": 0.10,
                "target_posture": "risk_on",
                "best_posture": "risk_on",
                "soft_regret": 0.5,
                "posture_utility_variance": 0.0,
                "defensive_posture": 0.0,
                "target_defensive_posture": 0.1,
                "target_posture_penalty": 0.0,
                "posture_progress_bonus": 0.0,
                "posture_stale_penalty": 0.0,
                "posture_flip_penalty": 0.0,
                "posture_distance_to_target": 0.5,
                "active_return": -0.01,
                "drawdown_penalty": 0.01,
                "turnover_penalty": 0.01,
                "concentration_penalty": 0.0,
                "liquidity_penalty": 0.0,
            },
        },
    ]

    summary = _summarize_trace(trace)

    assert summary["decision_quality_basis"] == "cached_one_step_soft_regret_v1"
    assert summary["posture_counts"] == {"neutral": 1, "risk_off": 1}
    assert summary["target_posture_by_stress_bucket"]["high"]["risk_off"] == 1
    assert summary["posture_optimality_rate"] == pytest.approx(0.5)
    assert summary["mean_regret"] == pytest.approx(0.25)
    assert summary["regret_by_stress_bucket"]["high"] == pytest.approx(0.0)
    assert summary["mean_posture_utility_dispersion"] == pytest.approx(0.01)
    assert summary["posture_utility_dispersion_by_stress_bucket"]["low"] == pytest.approx(0.0)
    assert "mean_requested_vs_realized_cash_gap" in summary
    assert "optimizer_reason_counts" in summary
    assert "optimizer_fallback_counts" in summary
    assert "optimizer_relaxation_tier_counts" in summary
