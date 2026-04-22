"""Compare a saved full-history RL run against fresh full-history baselines."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.simulator import PortfolioSimulator
from src.backtest.walk_forward import WalkForwardEngine
from src.config import load_config
from src.rl.environment import SectorAllocationEnv
from src.rl.historical_executor import HistoricalPeriodExecutor


def evaluate_full_backtest_comparison(
    price_matrix: pd.DataFrame,
    volume_matrix: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    full_rl_metrics_path: str | Path,
    cfg: dict | None = None,
    baseline_mode: str = "optimizer_only",
) -> dict[str, Any]:
    """Compare a saved `full_rl` report against a fresh same-window neutral baseline."""
    if baseline_mode not in {"optimizer_only", "selection_only"}:
        raise ValueError("baseline_mode must be 'optimizer_only' or 'selection_only'.")

    cfg_eval = load_config() if cfg is None else cfg
    cfg_eval = _copy_cfg(cfg_eval)
    full_rl_payload = _load_json(Path(full_rl_metrics_path))
    if str(full_rl_payload.get("mode")) != "full_rl":
        raise ValueError("full_rl_metrics_path must point to a full_rl metrics artifact.")

    start_date = str(full_rl_payload["start_date"])
    end_date = str(full_rl_payload["end_date"])
    cfg_eval["backtest"]["start_date"] = start_date
    cfg_eval["backtest"]["end_date"] = end_date

    end_ts = pd.Timestamp(end_date)
    engine = WalkForwardEngine(
        price_matrix=price_matrix.loc[:end_ts],
        volume_matrix=volume_matrix.loc[:end_ts],
        macro_df=macro_df.loc[:end_ts],
        cfg=cfg_eval,
        mode=baseline_mode,
    )
    baseline_metrics = engine.run()
    baseline_metrics["mode"] = baseline_mode
    baseline_metrics["start_date"] = start_date
    baseline_metrics["end_date"] = end_date
    baseline_metrics["random_seed"] = int(cfg_eval.get("backtest", {}).get("random_seed", 42))
    baseline_metrics["stock_feature_blocks"] = cfg_eval.get("stock_features", {}).get("blocks", [])
    baseline_metrics["stock_fwd_window_days"] = int(cfg_eval["stock_model"].get("fwd_window_days", 56))
    baseline_metrics["sector_fwd_window_days"] = int(cfg_eval["sector_model"].get("fwd_window_days", 28))

    full_rl_metrics = _strip_report_metadata(full_rl_payload)
    return {
        "comparison_generated_at_utc": _isoformat_utc(datetime.now(timezone.utc)),
        "comparison_type": "full_backtest_rl_vs_baseline_mode",
        "window": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "full_rl_source": {
            "metrics_path": str(Path(full_rl_metrics_path).resolve()),
            "run_id": full_rl_payload.get("_report_metadata", {}).get("run_id"),
            "report_generated_at_utc": full_rl_payload.get("report_generated_at_utc"),
            "training_backend": full_rl_payload.get("rl_training_backend"),
        },
        "baseline_mode": baseline_mode,
        "full_rl": full_rl_metrics,
        "baseline": baseline_metrics,
        "uplift": _compute_uplift(full_rl_metrics, baseline_metrics),
    }


def evaluate_full_neutral_policy_comparison(
    price_matrix: pd.DataFrame,
    volume_matrix: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    full_rl_metrics_path: str | Path,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """
    Compare a saved `full_rl` run against a true neutral-policy execution path.

    The neutral policy keeps all sector tilts at 1.0, neutral aggressiveness,
    and the default neutral cash target, but still runs through the same
    `full_rl` optimizer/risk/execution stack.
    """
    cfg_eval = load_config() if cfg is None else cfg
    cfg_eval = _copy_cfg(cfg_eval)
    full_rl_payload = _load_json(Path(full_rl_metrics_path))
    if str(full_rl_payload.get("mode")) != "full_rl":
        raise ValueError("full_rl_metrics_path must point to a full_rl metrics artifact.")

    start_date = str(full_rl_payload["start_date"])
    end_date = str(full_rl_payload["end_date"])
    cfg_eval["backtest"]["start_date"] = start_date
    cfg_eval["backtest"]["end_date"] = end_date

    end_ts = pd.Timestamp(end_date)
    engine = WalkForwardEngine(
        price_matrix=price_matrix.loc[:end_ts],
        volume_matrix=volume_matrix.loc[:end_ts],
        macro_df=macro_df.loc[:end_ts],
        cfg=cfg_eval,
        mode="full_rl",
        use_rl=False,
    )
    neutral = _run_full_window_policy(
        executor=HistoricalPeriodExecutor(engine, mode="full_rl", allow_model_retraining=True),
        benchmark=_benchmark_series(
            price_matrix,
            cfg_eval,
            pd.Timestamp(start_date),
            end_ts,
        ),
    )

    full_rl_metrics = _strip_report_metadata(full_rl_payload)
    return {
        "comparison_generated_at_utc": _isoformat_utc(datetime.now(timezone.utc)),
        "comparison_type": "full_backtest_rl_vs_true_neutral_policy",
        "window": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "full_rl_source": {
            "metrics_path": str(Path(full_rl_metrics_path).resolve()),
            "run_id": full_rl_payload.get("_report_metadata", {}).get("run_id"),
            "report_generated_at_utc": full_rl_payload.get("report_generated_at_utc"),
            "training_backend": full_rl_payload.get("rl_training_backend"),
        },
        "neutral_policy_definition": {
            "sector_tilts": "all 1.0",
            "cash_target": 0.05,
            "aggressiveness": 1.0,
            "should_rebalance": True,
            "execution_mode": "full_rl_stack_with_fixed_neutral_action",
        },
        "trained_policy": full_rl_metrics,
        "neutral_policy": neutral["metrics"],
        "neutral_policy_diagnostics": neutral["diagnostics"],
        "neutral_policy_trace": neutral["trace"],
        "uplift": _compute_uplift(full_rl_metrics, neutral["metrics"]),
    }


def _compute_uplift(full_rl: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "avg_turnover",
        "total_return",
        "final_nav",
        "information_ratio",
    )
    uplift: dict[str, Any] = {}
    for key in keys:
        if key in full_rl and key in baseline:
            uplift[key] = float(full_rl[key]) - float(baseline[key])
    return uplift


def _run_full_window_policy(
    *,
    executor: HistoricalPeriodExecutor,
    benchmark: pd.Series | None,
) -> dict[str, Any]:
    portfolio = executor.initial_portfolio(0)
    nav_points = executor.initial_nav_points(0)
    executor.reset_runtime_state(nav_points)
    trace: list[dict[str, Any]] = []

    end_idx = len(executor.rebalance_dates) - 2
    for idx in range(0, end_idx + 1):
        prepared = executor.prepare_step(idx, portfolio, nav_points)
        decision = SectorAllocationEnv.neutral_action(executor.engine.cfg)
        result = executor.execute_prepared_step(
            prepared,
            portfolio,
            nav_points,
            rl_decision=decision,
            done=idx == end_idx,
        )
        trace.append(
            {
                "date": str(result.current_date.date()),
                "reward": float(result.reward),
                "period_return": float(
                    result.transition.get("info", {})
                    .get("reward_components", {})
                    .get("period_return", 0.0)
                ),
                "turnover": float(result.exec_result.total_turnover),
                "transaction_cost": float(result.exec_result.total_cost),
                "cash_target": float(result.cash_target),
                "turnover_cap": (
                    float(decision.get("turnover_cap"))
                    if decision.get("turnover_cap") is not None
                    else None
                ),
                "aggressiveness": float(decision.get("aggressiveness", 1.0)),
                "should_rebalance": bool(decision.get("should_rebalance", True)),
                "selected_sectors": list(result.selected_sectors),
                "selected_sector_count": int(len(result.selected_sectors)),
                "selected_stock_count": int(len(result.selected_stock_rows)),
                "sector_tilts": {
                    str(sector): float(tilt)
                    for sector, tilt in decision.get("sector_tilts", {}).items()
                },
                "reward_components": dict(
                    result.transition.get("info", {}).get("reward_components", {})
                ),
            }
        )
        executor.engine.risk_engine.update(
            result.post_trade_portfolio.nav,
            result.post_trade_portfolio.date,
        )
        portfolio = result.next_portfolio
        nav_points = result.updated_nav_points

    nav_series = pd.Series(
        [float(nav) for _, nav in nav_points],
        index=pd.DatetimeIndex([pd.Timestamp(ts) for ts, _ in nav_points]),
        name="portfolio_nav",
    )
    nav_series = nav_series[~nav_series.index.duplicated(keep="last")].sort_index()
    metrics = PortfolioSimulator.compute_metrics(nav_series, benchmark)
    metrics["avg_turnover"] = (
        float(sum(entry["turnover"] for entry in trace) / len(trace)) if trace else 0.0
    )
    metrics["total_rebalances"] = len(trace)
    metrics["mode"] = "full_rl_neutral_policy"
    metrics["start_date"] = str(executor.engine.start_date.date())
    metrics["end_date"] = str(executor.engine.end_date.date())
    metrics["random_seed"] = int(executor.engine.cfg.get("backtest", {}).get("random_seed", 42))
    metrics["stock_feature_blocks"] = executor.engine.cfg.get("stock_features", {}).get("blocks", [])
    metrics["stock_fwd_window_days"] = int(executor.engine.cfg["stock_model"].get("fwd_window_days", 56))
    metrics["sector_fwd_window_days"] = int(executor.engine.cfg["sector_model"].get("fwd_window_days", 28))
    return {
        "metrics": metrics,
        "diagnostics": _summarize_trace(trace),
        "trace": trace,
    }


def _strip_report_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(payload)
    stripped.pop("_report_metadata", None)
    return stripped


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _benchmark_series(
    price_matrix: pd.DataFrame,
    cfg: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    bm_ticker = cfg["backtest"].get("benchmark_ticker", "^NSEI")
    if bm_ticker not in price_matrix.columns:
        return None
    return price_matrix[bm_ticker].loc[start:end].dropna()


def _summarize_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace:
        return {}
    def mean(key: str) -> float | None:
        values = [entry.get(key) for entry in trace if entry.get(key) is not None]
        if not values:
            return None
        return float(sum(float(value) for value in values) / len(values))
    return {
        "mean_cash_target": mean("cash_target"),
        "mean_turnover_cap": mean("turnover_cap"),
        "mean_aggressiveness": mean("aggressiveness"),
        "mean_turnover": mean("turnover"),
        "mean_selected_sector_count": float(
            sum(len(entry["selected_sectors"]) for entry in trace) / len(trace)
        ),
        "mean_selected_stock_count": float(
            sum(entry["selected_stock_count"] for entry in trace) / len(trace)
        ),
        "mean_reward": mean("reward"),
        "rebalance_rate": float(
            sum(1.0 if entry["should_rebalance"] else 0.0 for entry in trace) / len(trace)
        ),
    }


def _copy_cfg(cfg: dict) -> dict:
    copied: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            copied[key] = _copy_cfg(value)
        elif isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
