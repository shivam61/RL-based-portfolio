from __future__ import annotations

import json

import pandas as pd

from src.rl.control_evaluation import evaluate_control_from_artifacts


def test_evaluate_control_from_artifacts_builds_canonical_stage0_report(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    (report_dir / "metrics.json").write_text(
        json.dumps(
            {
                "mode": "full_rl",
                "start_date": "2013-01-01",
                "end_date": "2026-04-17",
                "cagr": 0.1827,
                "sharpe": 0.75,
                "sortino": 0.97,
                "calmar": 0.56,
                "max_drawdown": -0.326,
                "avg_turnover": 0.2803,
                "total_return": 5.62,
                "final_nav": 3308406.58,
                "information_ratio": 0.70,
            }
        )
    )
    (report_dir / "rl_full_neutral_comparison.json").write_text(
        json.dumps(
            {
                "window": {"start_date": "2013-01-01", "end_date": "2026-04-17"},
                "trained_policy": {
                    "cagr": 0.1827,
                    "sharpe": 0.75,
                    "sortino": 0.97,
                    "calmar": 0.56,
                    "max_drawdown": -0.326,
                    "avg_turnover": 0.2803,
                    "total_return": 5.62,
                    "final_nav": 3308406.58,
                    "information_ratio": 0.70,
                },
                "neutral_policy": {
                    "initial_nav": 500000.0,
                    "cagr": 0.1785,
                    "sharpe": 0.72,
                    "sortino": 0.94,
                    "calmar": 0.54,
                    "max_drawdown": -0.328,
                    "avg_turnover": 0.2743,
                    "total_return": 5.36,
                    "final_nav": 3182911.37,
                    "information_ratio": 0.66,
                },
                "uplift": {
                    "cagr": 0.0042,
                    "sharpe": 0.029,
                    "avg_turnover": 0.006,
                },
                "neutral_policy_trace": [
                    {
                        "date": "2020-03-19",
                        "period_return": -0.12,
                        "turnover": 0.40,
                        "cash_target": 0.12,
                        "aggressiveness": 1.0,
                        "selected_sectors": ["IT", "FMCG"],
                        "selected_sector_count": 2,
                        "selected_stock_count": 20,
                        "should_rebalance": True,
                    },
                    {
                        "date": "2020-04-16",
                        "period_return": 0.02,
                        "turnover": 0.35,
                        "cash_target": 0.15,
                        "aggressiveness": 1.0,
                        "selected_sectors": ["IT", "FMCG"],
                        "selected_sector_count": 2,
                        "selected_stock_count": 19,
                        "should_rebalance": True,
                    },
                    {
                        "date": "2020-05-14",
                        "period_return": 0.08,
                        "turnover": 0.20,
                        "cash_target": 0.15,
                        "aggressiveness": 1.0,
                        "selected_sectors": ["IT", "FMCG"],
                        "selected_sector_count": 2,
                        "selected_stock_count": 18,
                        "should_rebalance": True,
                    },
                ],
            }
        )
    )
    (report_dir / "rl_full_backtest_comparison.json").write_text(
        json.dumps(
            {
                "full_rl": {"cagr": 0.1827, "sharpe": 0.75},
                "baseline": {"cagr": 0.0948, "sharpe": 0.23},
                "uplift": {"cagr": 0.0879, "sharpe": 0.52},
            }
        )
    )
    (report_dir / "rl_holdout_comparison.json").write_text(
        json.dumps(
            {
                "train_end_rebalance": "2015-12-31",
                "holdout_start_rebalance": "2016-01-28",
                "holdout_end_rebalance": "2016-12-29",
                "holdout_windows": 12,
                "trained_policy": {"cagr": 0.3268, "sharpe": 1.51},
                "neutral_policy": {"cagr": 0.3239, "sharpe": 1.46},
                "uplift": {"cagr": 0.0029, "sharpe": 0.044},
                "trained_policy_diagnostics": {
                    "mean_cash_target": 0.05,
                    "mean_aggressiveness": 1.08,
                    "mean_turnover": 0.24,
                    "cash_usage_rate": 0.25,
                    "turnover_cap_usage_rate": 0.10,
                },
                "neutral_policy_diagnostics": {
                    "mean_cash_target": 0.05,
                    "mean_aggressiveness": 1.0,
                    "mean_turnover": 0.25,
                    "cash_usage_rate": 0.0,
                    "turnover_cap_usage_rate": 0.0,
                },
            }
        )
    )

    pd.DataFrame(
        [
            {
                "date": "2020-03-19",
                "pre_nav": 1128935.01,
                "post_nav": 1100000.0,
                "cash_pct": 12.0,
                "turnover_pct": 43.27,
                "n_stocks": 19,
                "selected_sector_count": 15,
                "selected_stock_count": 60,
                "aggressiveness": 1.03,
                "emergency": False,
                "tilt_IT": 1.1,
                "tilt_FMCG": 1.0,
            },
            {
                "date": "2020-04-16",
                "pre_nav": 918778.60,
                "post_nav": 920000.0,
                "cash_pct": 12.0,
                "turnover_pct": 44.16,
                "n_stocks": 18,
                "selected_sector_count": 15,
                "selected_stock_count": 58,
                "aggressiveness": 0.99,
                "emergency": False,
                "tilt_IT": 1.0,
                "tilt_FMCG": 1.1,
            },
            {
                "date": "2020-05-14",
                "pre_nav": 926202.38,
                "post_nav": 950000.0,
                "cash_pct": 12.0,
                "turnover_pct": 23.57,
                "n_stocks": 17,
                "selected_sector_count": 15,
                "selected_stock_count": 56,
                "aggressiveness": 0.99,
                "emergency": False,
                "tilt_IT": 1.0,
                "tilt_FMCG": 1.2,
            },
        ]
    ).to_csv(report_dir / "rebalance_log.csv", index=False)

    result = evaluate_control_from_artifacts(report_dir)

    assert result["reference_modes"]["current_rl"]["cagr"] == 0.1827
    assert result["reference_modes"]["neutral_full_stack"]["cagr"] == 0.1785
    assert result["reference_modes"]["optimizer_only"]["cagr"] == 0.0948
    assert result["holdout"]["current_rl_vs_neutral"]["uplift"]["cagr"] == 0.0029
    assert result["holdout"]["drawdown_behavior"]["current_rl"]["cash_usage_rate"] == 0.25
    covid = result["stress_windows"]["2020_covid"]
    assert covid["current_rl"]["observations"] == 3
    assert covid["neutral_full_stack"]["observations"] == 3
    assert covid["current_rl"]["avg_cash_pct"] == 12.0
    assert round(covid["neutral_full_stack"]["avg_cash_pct"], 2) == 14.0
    assert covid["delta_rl_minus_neutral"]["avg_cash_pct"] == -2.0
    assert result["drawdown_behavior"]["current_rl"]["observations"] >= 1
