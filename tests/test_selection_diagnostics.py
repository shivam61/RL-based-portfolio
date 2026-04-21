from __future__ import annotations

from datetime import date

import pandas as pd

from src.reporting.report import ReportGenerator
from src.reporting.selection_diagnostics import (
    compute_selection_diagnostics,
    prepare_selection_diagnostics,
)


def _sample_selection_records():
    return [
        {
            "rebalance_date": date(2024, 1, 31),
            "selected_stocks": [
                {"ticker": "AAA.NS", "sector": "IT"},
                {"ticker": "BBB.NS", "sector": "IT"},
            ],
            "candidate_stock_scores": {"AAA.NS": 0.9, "BBB.NS": 0.2, "CCC.NS": 0.4},
            "candidate_stock_sectors": {"AAA.NS": "IT", "BBB.NS": "IT", "CCC.NS": "IT"},
            "raw_stock_scores": {"AAA.NS": 0.9, "BBB.NS": 0.2, "CCC.NS": 0.4},
            "next_period_returns": {"AAA.NS": 0.10, "BBB.NS": -0.02},
            "universe_forward_returns": {
                "AAA.NS": 0.10,
                "BBB.NS": -0.02,
                "CCC.NS": 0.03,
            },
            "sector_forward_returns": {
                "IT": {"AAA.NS": 0.10, "BBB.NS": -0.02, "CCC.NS": 0.03}
            },
        },
        {
            "rebalance_date": date(2024, 2, 29),
            "selected_stocks": [
                {"ticker": "AAA.NS", "sector": "IT"},
                {"ticker": "CCC.NS", "sector": "IT"},
            ],
            "candidate_stock_scores": {"AAA.NS": 0.7, "BBB.NS": 0.1, "CCC.NS": 0.6},
            "candidate_stock_sectors": {"AAA.NS": "IT", "BBB.NS": "IT", "CCC.NS": "IT"},
            "raw_stock_scores": {"AAA.NS": 0.7, "BBB.NS": 0.1, "CCC.NS": 0.6},
            "next_period_returns": {"AAA.NS": 0.04, "CCC.NS": 0.01},
            "universe_forward_returns": {
                "AAA.NS": 0.04,
                "BBB.NS": -0.03,
                "CCC.NS": 0.01,
            },
            "sector_forward_returns": {
                "IT": {"AAA.NS": 0.04, "BBB.NS": -0.03, "CCC.NS": 0.01}
            },
        },
        {
            "rebalance_date": date(2024, 3, 28),
            "selected_stocks": [
                {"ticker": "DDD.NS", "sector": "Banking"},
                {"ticker": "EEE.NS", "sector": "Banking"},
            ],
            "candidate_stock_scores": {"DDD.NS": 0.8, "EEE.NS": 0.3, "FFF.NS": 0.5},
            "candidate_stock_sectors": {
                "DDD.NS": "Banking",
                "EEE.NS": "Banking",
                "FFF.NS": "Banking",
            },
            "raw_stock_scores": {"DDD.NS": 0.8, "EEE.NS": 0.3},
            "next_period_returns": {"DDD.NS": 0.05, "EEE.NS": 0.02},
            "universe_forward_returns": {"DDD.NS": 0.05, "EEE.NS": 0.02},
            "sector_forward_returns": {
                "Banking": {"DDD.NS": 0.05, "EEE.NS": 0.02, "FFF.NS": 0.08}
            },
        },
    ]


def test_compute_selection_diagnostics_returns_expected_summary():
    diagnostics = compute_selection_diagnostics(_sample_selection_records())

    assert diagnostics is not None
    summary = diagnostics["summary"]
    frame = diagnostics["per_rebalance"]

    assert summary["periods"] == 3
    assert round(summary["avg_selected_count"], 2) == 2.00
    assert round(summary["top_k_avg_forward_return"], 4) == 0.0333
    assert round(summary["top_k_minus_universe"], 4) == 0.0072
    assert round(summary["top_k_minus_sector_median"], 4) == 0.0033
    assert round(summary["precision_at_k"], 2) == 0.83
    assert round(summary["rank_ic"], 2) == 1.00
    assert round(summary["stability"], 2) == 0.17
    assert round(summary["intra_sector_dispersion"], 4) == 0.0341
    assert round(summary["top_bottom_spread"], 4) == 0.06
    assert round(summary["within_sector_ic"], 2) == 0.83
    assert round(summary["within_sector_ic_weighted"], 2) == 0.83
    assert round(summary["within_sector_top_bottom_spread"], 4) == 0.0733
    assert round(summary["within_sector_top_bottom_spread_weighted"], 4) == 0.0733
    assert round(summary["within_sector_top_k_minus_sector_median"], 4) == 0.0333
    assert round(summary["within_sector_top_k_minus_sector_median_weighted"], 4) == 0.0333
    assert list(frame["selected_stocks"]) == ["AAA.NS|BBB.NS", "AAA.NS|CCC.NS", "DDD.NS|EEE.NS"]


def test_prepare_selection_diagnostics_accepts_precomputed_payload():
    payload = {
        "summary": {"precision_at_k": 0.6},
        "per_rebalance": [{"date": "2024-01-31", "precision_at_k": 0.6}],
    }

    prepared = prepare_selection_diagnostics(payload)

    assert prepared is not None
    assert prepared["summary"]["precision_at_k"] == 0.6
    assert isinstance(prepared["per_rebalance"], pd.DataFrame)
    assert len(prepared["per_rebalance"]) == 1


def test_report_generator_saves_selection_diagnostics_when_present(tmp_path):
    cfg = {"paths": {"report_dir": str(tmp_path / "reports")}}
    generator = ReportGenerator(cfg)
    nav = pd.Series([100.0, 102.0], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))

    class _FakeStockRanker:
        def __init__(self):
            self.is_fitted = True
            self.models = {"IT": object(), "Banking": object()}

        def feature_importance(self, sector: str) -> pd.Series:
            if sector == "IT":
                return pd.Series({"mom_12m": 10.0, "ret_3m": 4.0})
            return pd.Series({"vol_3m": 7.0, "max_dd_12m": 3.0})

    report_dir = generator.generate_full_report(
        metrics={"start_date": "2024-01-31", "end_date": "2024-02-29"},
        nav_series=nav,
        rebalance_records=[],
        selection_diagnostics=_sample_selection_records(),
        stock_ranker=_FakeStockRanker(),
    )

    assert (report_dir / "selection_diagnostics.json").exists()
    assert (report_dir / "selection_rebalance_log.csv").exists()
    assert (report_dir / "stock_ranker_feature_importance.csv").exists()
    assert (report_dir / "stock_ranker_feature_importance.json").exists()
    saved = pd.read_csv(report_dir / "selection_rebalance_log.csv")
    assert len(saved) == 3
    assert "precision_at_k" in saved.columns

    stock_imp = pd.read_csv(report_dir / "stock_ranker_feature_importance.csv")
    assert set(stock_imp["sector"]) == {"Banking", "IT"}
    assert {"feature", "importance", "importance_share", "rank"} <= set(stock_imp.columns)


def test_report_generator_skips_selection_diagnostics_when_absent(tmp_path):
    cfg = {"paths": {"report_dir": str(tmp_path / "reports")}}
    generator = ReportGenerator(cfg)
    nav = pd.Series([100.0, 101.0], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))

    report_dir = generator.generate_full_report(
        metrics={"start_date": "2024-01-31", "end_date": "2024-02-29"},
        nav_series=nav,
        rebalance_records=[],
    )

    assert not (report_dir / "selection_diagnostics.json").exists()
    assert not (report_dir / "selection_rebalance_log.csv").exists()
