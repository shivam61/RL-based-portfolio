from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.posture_dataset import (
    _fixed_posture_decision,
    _horizon_utility,
    _restore_model_snapshot,
    _snapshot_model_state,
    _summarize_dataset,
)


def test_fixed_posture_decision_sets_research_opt_out():
    cfg = {
        "rl": {
            "posture_profiles": {
                "risk_off": {
                    "cash_target": 0.35,
                    "aggressiveness": 0.75,
                    "turnover_cap": 0.15,
                }
            }
        }
    }

    decision = _fixed_posture_decision(cfg, ["IT", "Banking"], "risk_off")

    assert decision["posture"] == "risk_off"
    assert decision["cash_target"] == pytest.approx(0.35)
    assert decision["allow_forced_posture_override"] is False


def test_summarize_dataset_reports_best_posture_distribution():
    rows = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "posture": "risk_on",
                "utility": 0.10,
                "utility_return_only": 0.12,
                "utility_return_minus_drawdown": 0.1075,
                "utility_full_utility": 0.10,
                "total_return": 0.12,
                "max_drawdown": -0.05,
                "avg_turnover": 0.20,
                "avg_cost_ratio": 0.01,
                "fallback_count": 0,
                "mean_selected_sector_count": 12,
                "mean_selected_stock_count": 60,
            },
            {
                "date": "2024-01-01",
                "posture": "neutral",
                "utility": 0.08,
                "utility_return_only": 0.10,
                "utility_return_minus_drawdown": 0.09,
                "utility_full_utility": 0.08,
                "total_return": 0.10,
                "max_drawdown": -0.04,
                "avg_turnover": 0.18,
                "avg_cost_ratio": 0.01,
                "fallback_count": 0,
                "mean_selected_sector_count": 11,
                "mean_selected_stock_count": 50,
            },
            {
                "date": "2024-01-01",
                "posture": "risk_off",
                "utility": 0.02,
                "utility_return_only": 0.03,
                "utility_return_minus_drawdown": 0.025,
                "utility_full_utility": 0.02,
                "total_return": 0.03,
                "max_drawdown": -0.02,
                "avg_turnover": 0.10,
                "avg_cost_ratio": 0.005,
                "fallback_count": 1,
                "mean_selected_sector_count": 8,
                "mean_selected_stock_count": 38,
            },
            {
                "date": "2024-02-01",
                "posture": "risk_on",
                "utility": -0.03,
                "utility_return_only": -0.01,
                "utility_return_minus_drawdown": -0.03,
                "utility_full_utility": -0.03,
                "total_return": -0.01,
                "max_drawdown": -0.08,
                "avg_turnover": 0.24,
                "avg_cost_ratio": 0.01,
                "fallback_count": 0,
                "mean_selected_sector_count": 12,
                "mean_selected_stock_count": 61,
            },
            {
                "date": "2024-02-01",
                "posture": "neutral",
                "utility": 0.04,
                "utility_return_only": 0.05,
                "utility_return_minus_drawdown": 0.04,
                "utility_full_utility": 0.04,
                "total_return": 0.05,
                "max_drawdown": -0.04,
                "avg_turnover": 0.16,
                "avg_cost_ratio": 0.01,
                "fallback_count": 0,
                "mean_selected_sector_count": 11,
                "mean_selected_stock_count": 52,
            },
            {
                "date": "2024-02-01",
                "posture": "risk_off",
                "utility": 0.01,
                "utility_return_only": 0.02,
                "utility_return_minus_drawdown": 0.015,
                "utility_full_utility": 0.01,
                "total_return": 0.02,
                "max_drawdown": -0.02,
                "avg_turnover": 0.09,
                "avg_cost_ratio": 0.005,
                "fallback_count": 1,
                "mean_selected_sector_count": 8,
                "mean_selected_stock_count": 39,
            },
        ]
    )
    samples = [
        {
            "date": "2024-01-01",
            "stress_bucket": "low",
            "stress_signal": 0.10,
            "best_posture": "risk_on",
            "utility_margin": 0.02,
            "winner_by_utility_mode": {
                "return_only": "risk_on",
                "return_minus_drawdown": "risk_on",
                "full_utility": "risk_on",
            },
            "margin_by_utility_mode": {
                "return_only": 0.02,
                "return_minus_drawdown": 0.0175,
                "full_utility": 0.02,
            },
            "posture_outcomes": {
                "risk_on": {"fallback_count": 0},
                "neutral": {"fallback_count": 0},
                "risk_off": {"fallback_count": 1},
            },
        },
        {
            "date": "2024-02-01",
            "stress_bucket": "high",
            "stress_signal": 0.40,
            "best_posture": "neutral",
            "utility_margin": 0.03,
            "winner_by_utility_mode": {
                "return_only": "neutral",
                "return_minus_drawdown": "neutral",
                "full_utility": "neutral",
            },
            "margin_by_utility_mode": {
                "return_only": 0.03,
                "return_minus_drawdown": 0.025,
                "full_utility": 0.03,
            },
            "posture_outcomes": {
                "risk_on": {"fallback_count": 0},
                "neutral": {"fallback_count": 0},
                "risk_off": {"fallback_count": 1},
            },
        },
    ]

    summary = _summarize_dataset(rows, samples, horizon_rebalances=2, utility_mode="full_utility")

    assert summary["sample_count"] == 2
    assert summary["primary_utility_mode"] == "full_utility"
    assert summary["best_posture_counts"] == {"risk_on": 1, "neutral": 1}
    assert summary["best_posture_by_stress_bucket"]["low"]["risk_on"] == 1
    assert summary["best_posture_by_stress_bucket"]["high"]["neutral"] == 1
    assert summary["utility_mode_summaries"]["return_only"]["best_posture_counts"]["risk_on"] == 1
    assert summary["winner_by_metric"]["avg_turnover"]["risk_off"] == 2
    assert summary["posture_outcome_stats"]["risk_off"]["mean_fallback_count"] == pytest.approx(1.0)
    assert summary["execution_clean_subset"]["sample_count"] == 0


def test_horizon_utility_supports_multiple_modes():
    cfg = {"rl": {"reward_lambda_dd": 0.25, "reward_lambda_to": 0.5, "reward_lambda_liq": 0.2}}

    assert _horizon_utility(cfg, total_return=0.10, max_drawdown=-0.04, avg_turnover=0.20, avg_cost_ratio=0.01, utility_mode="return_only") == pytest.approx(0.10)
    assert _horizon_utility(cfg, total_return=0.10, max_drawdown=-0.04, avg_turnover=0.20, avg_cost_ratio=0.01, utility_mode="return_minus_drawdown") == pytest.approx(0.09)
    assert _horizon_utility(cfg, total_return=0.10, max_drawdown=-0.04, avg_turnover=0.20, avg_cost_ratio=0.01, utility_mode="full_utility") == pytest.approx(-0.012)


def test_model_snapshot_round_trip_restores_trained_state():
    engine = SimpleNamespace(
        sector_scorer=SimpleNamespace(
            model={"name": "sector_v1"},
            scaler={"kind": "sector_scaler"},
            feature_names=["a", "b"],
            is_fitted=True,
        ),
        stock_ranker=SimpleNamespace(
            models={"IT": {"name": "stock_v1"}},
            scalers={"IT": {"kind": "stock_scaler"}},
            feature_names=["x", "y"],
            is_fitted=True,
        ),
    )

    snapshot = _snapshot_model_state(engine)

    engine.sector_scorer.model = None
    engine.sector_scorer.scaler = None
    engine.sector_scorer.feature_names = []
    engine.sector_scorer.is_fitted = False
    engine.stock_ranker.models = {}
    engine.stock_ranker.scalers = {}
    engine.stock_ranker.feature_names = []
    engine.stock_ranker.is_fitted = False

    _restore_model_snapshot(engine, snapshot)

    assert engine.sector_scorer.model == {"name": "sector_v1"}
    assert engine.sector_scorer.scaler == {"kind": "sector_scaler"}
    assert engine.sector_scorer.feature_names == ["a", "b"]
    assert engine.sector_scorer.is_fitted is True
    assert engine.stock_ranker.models == {"IT": {"name": "stock_v1"}}
    assert engine.stock_ranker.scalers == {"IT": {"kind": "stock_scaler"}}
    assert engine.stock_ranker.feature_names == ["x", "y"]
    assert engine.stock_ranker.is_fitted is True
