from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.posture_dataset import (
    _fixed_posture_decision,
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
                "total_return": 0.12,
                "max_drawdown": -0.05,
                "avg_turnover": 0.20,
                "fallback_count": 0,
                "mean_selected_sector_count": 12,
                "mean_selected_stock_count": 60,
            },
            {
                "date": "2024-01-01",
                "posture": "neutral",
                "utility": 0.08,
                "total_return": 0.10,
                "max_drawdown": -0.04,
                "avg_turnover": 0.18,
                "fallback_count": 0,
                "mean_selected_sector_count": 11,
                "mean_selected_stock_count": 50,
            },
            {
                "date": "2024-01-01",
                "posture": "risk_off",
                "utility": 0.02,
                "total_return": 0.03,
                "max_drawdown": -0.02,
                "avg_turnover": 0.10,
                "fallback_count": 1,
                "mean_selected_sector_count": 8,
                "mean_selected_stock_count": 38,
            },
            {
                "date": "2024-02-01",
                "posture": "risk_on",
                "utility": -0.03,
                "total_return": -0.01,
                "max_drawdown": -0.08,
                "avg_turnover": 0.24,
                "fallback_count": 0,
                "mean_selected_sector_count": 12,
                "mean_selected_stock_count": 61,
            },
            {
                "date": "2024-02-01",
                "posture": "neutral",
                "utility": 0.04,
                "total_return": 0.05,
                "max_drawdown": -0.04,
                "avg_turnover": 0.16,
                "fallback_count": 0,
                "mean_selected_sector_count": 11,
                "mean_selected_stock_count": 52,
            },
            {
                "date": "2024-02-01",
                "posture": "risk_off",
                "utility": 0.01,
                "total_return": 0.02,
                "max_drawdown": -0.02,
                "avg_turnover": 0.09,
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
        },
        {
            "date": "2024-02-01",
            "stress_bucket": "high",
            "stress_signal": 0.40,
            "best_posture": "neutral",
            "utility_margin": 0.03,
        },
    ]

    summary = _summarize_dataset(rows, samples, horizon_rebalances=2)

    assert summary["sample_count"] == 2
    assert summary["best_posture_counts"] == {"risk_on": 1, "neutral": 1}
    assert summary["best_posture_by_stress_bucket"]["low"]["risk_on"] == 1
    assert summary["best_posture_by_stress_bucket"]["high"]["neutral"] == 1
    assert summary["posture_outcome_stats"]["risk_off"]["mean_fallback_count"] == pytest.approx(1.0)
