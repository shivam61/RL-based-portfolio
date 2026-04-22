from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone

import pandas as pd

from src.reporting.report import ReportGenerator


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
    ]


class _FakeStockRanker:
    def __init__(self):
        self.is_fitted = True
        self.models = {"IT": object()}

    def feature_importance(self, sector: str) -> pd.Series:
        assert sector == "IT"
        return pd.Series({"mom_12m": 10.0, "ret_3m": 4.0})


def test_report_generator_writes_manifest_and_freshness_metadata(tmp_path):
    report_dir = tmp_path / "reports"
    model_dir = tmp_path / "models"
    report_dir.mkdir()
    model_dir.mkdir()
    (model_dir / "sector_scorer.pkl").write_bytes(b"sector-model")
    (model_dir / "stock_ranker.pkl").write_bytes(b"stock-model")
    rl_dir = model_dir / "rl_agent"
    rl_dir.mkdir()
    (rl_dir / "policy.bin").write_bytes(b"policy")

    stale_file = report_dir / "stale_note.txt"
    stale_file.write_text("older artifact")
    old_ts = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(stale_file, (old_ts, old_ts))

    cfg = {
        "paths": {
            "report_dir": str(report_dir),
            "model_dir": str(model_dir),
        },
        "backtest": {
            "start_date": "2024-01-31",
            "end_date": "2024-02-29",
            "initial_capital": 500000,
        },
        "stock_features": {"blocks": ["absolute_momentum", "risk"]},
        "rl": {"use_rl": False},
    }
    nav = pd.Series([100.0, 102.0], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
    generator = ReportGenerator(cfg)

    generated_dir = generator.generate_full_report(
        metrics={
            "mode": "selection_only",
            "start_date": "2024-01-31",
            "end_date": "2024-02-29",
            "cagr": 0.12,
            "total_return": 0.02,
            "year_returns": {2024: 0.02},
        },
        nav_series=nav,
        rebalance_records=[],
        selection_diagnostics=_sample_selection_records(),
        stock_ranker=_FakeStockRanker(),
    )

    metrics_payload = json.loads((generated_dir / "metrics.json").read_text())
    metrics_meta = metrics_payload["_report_metadata"]
    assert metrics_payload["run_mode"] == "selection_only"
    assert metrics_payload["backtest_start_date"] == "2024-01-31"
    assert metrics_payload["backtest_end_date"] == "2024-02-29"
    assert metrics_meta["artifact_type"] == "metrics"
    assert metrics_meta["run_mode"] == "selection_only"
    assert metrics_meta["manifest_file"] == "run_manifest.json"
    assert metrics_meta["run_id"].startswith("selection_only_2024_01_31_2024_02_29_")
    datetime.fromisoformat(metrics_payload["report_generated_at_utc"].replace("Z", "+00:00"))

    diagnostics_payload = json.loads((generated_dir / "selection_diagnostics.json").read_text())
    assert diagnostics_payload["_report_metadata"]["artifact_type"] == "selection_diagnostics"
    assert diagnostics_payload["_report_metadata"]["run_id"] == metrics_meta["run_id"]

    selection_log = pd.read_csv(generated_dir / "selection_rebalance_log.csv")
    assert {
        "run_id",
        "run_mode",
        "backtest_start_date",
        "backtest_end_date",
        "report_generated_at_utc",
    } <= set(selection_log.columns)
    assert selection_log["run_mode"].nunique() == 1
    assert selection_log["run_mode"].iloc[0] == "selection_only"

    importance_csv = pd.read_csv(generated_dir / "stock_ranker_feature_importance.csv")
    assert "run_id" in importance_csv.columns
    assert importance_csv["run_id"].iloc[0] == metrics_meta["run_id"]

    manifest = json.loads((generated_dir / "run_manifest.json").read_text())
    assert manifest["run"]["mode"] == "selection_only"
    assert manifest["run"]["backtest_start_date"] == "2024-01-31"
    assert manifest["run"]["backtest_end_date"] == "2024-02-29"
    assert len(manifest["config"]["sha256"]) == 64

    report_artifacts = {item["name"]: item for item in manifest["reports"]["artifacts"]}
    assert report_artifacts["metrics.json"]["generated_in_current_run"] is True
    assert report_artifacts["metrics.json"]["artifact_type"] == "metrics"
    assert report_artifacts["stale_note.txt"]["generated_in_current_run"] is False
    assert report_artifacts["stale_note.txt"]["stale_relative_to_current_run"] is True

    model_artifacts = {item["name"]: item for item in manifest["models"]["artifacts"]}
    assert {"sector_scorer.pkl", "stock_ranker.pkl", "rl_agent"} <= set(model_artifacts)
    assert model_artifacts["sector_scorer.pkl"]["sha256"]
    assert model_artifacts["rl_agent"]["is_dir"] is True


def test_report_generator_writes_minimal_manifest_without_model_dir(tmp_path):
    cfg = {
        "paths": {"report_dir": str(tmp_path / "reports")},
        "rl": {"use_rl": False},
    }
    nav = pd.Series([100.0, 101.0], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
    generator = ReportGenerator(cfg)

    report_dir = generator.generate_full_report(
        metrics={
            "mode": "optimizer_only",
            "start_date": "2024-01-31",
            "end_date": "2024-02-29",
        },
        nav_series=nav,
        rebalance_records=[],
    )

    manifest = json.loads((report_dir / "run_manifest.json").read_text())
    assert manifest["run"]["mode"] == "optimizer_only"
    assert manifest["models"]["artifacts"] == []

    rebalance_log = pd.read_csv(report_dir / "rebalance_log.csv")
    assert len(rebalance_log) == 0
    assert {
        "run_id",
        "run_mode",
        "backtest_start_date",
        "backtest_end_date",
        "report_generated_at_utc",
        "selected_sector_count",
        "selected_stock_count",
        "turnover_cap_pct",
    } <= set(rebalance_log.columns)
