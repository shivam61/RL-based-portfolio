from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.rl.contract import CAUSAL_TRAINING_BACKEND
from src.rl.holdout import evaluate_holdout
from src.rl.agent import RLSectorAgent
from src.rl.environment import SECTORS
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
    assert (
        result["trained_policy_diagnostics"]["decision_quality_basis"]
        == "target_posture_proxy"
    )
    assert "mean_regret" in result["trained_policy_diagnostics"]
    assert "regret_by_stress_bucket" in result["trained_policy_diagnostics"]
    assert "control_realization_by_posture" in result["trained_policy_diagnostics"]
    assert "control_realization_by_stress_bucket" in result["trained_policy_diagnostics"]
    assert result["trained_policy_behavior_flags"]["advisory_only"] is True
    assert result["trained_policy_behavior_flags"]["warnings"]
