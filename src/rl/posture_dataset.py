"""Build realized forward-outcome datasets for posture research."""
from __future__ import annotations

from copy import deepcopy
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.walk_forward import WalkForwardEngine
from src.config import load_config
from src.data.contracts import PortfolioState
from src.rl.environment import SectorAllocationEnv
from src.rl.historical_executor import HistoricalPeriodExecutor, PreparedHistoricalStep
from src.rl.policy_utils import default_decision

UTILITY_MODES = (
    "return_only",
    "return_minus_drawdown",
    "full_utility",
)


class _ModelStateTimeline:
    """Incrementally cache trained model state by rebalance index."""

    def __init__(self, engine: WalkForwardEngine):
        self._engine = engine
        self._trainer = HistoricalPeriodExecutor(engine, mode="full_rl", allow_model_retraining=True)
        self._cached_snapshots: dict[int, dict[str, bytes]] = {}
        self._last_idx = -1

    def restore(self, idx: int, engine: WalkForwardEngine) -> None:
        snapshot = self._ensure(idx)
        _restore_model_snapshot(engine, snapshot)

    def _ensure(self, idx: int) -> dict[str, bytes]:
        idx = int(idx)
        if idx in self._cached_snapshots:
            return self._cached_snapshots[idx]

        for step_idx in range(self._last_idx + 1, idx + 1):
            current_date = self._trainer.rebalance_dates[step_idx]
            if self._engine._should_retrain_models(step_idx, current_date):
                self._engine._train_models(current_date, idx=step_idx)
            self._cached_snapshots[step_idx] = _snapshot_model_state(self._engine)
            self._last_idx = step_idx
        return self._cached_snapshots[idx]


def build_posture_dataset(
    price_matrix: pd.DataFrame,
    volume_matrix: pd.DataFrame,
    macro_df: pd.DataFrame,
    cfg: dict | None = None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    horizon_rebalances: int = 2,
    max_samples: int | None = None,
    utility_mode: str = "full_utility",
    near_tie_threshold: float = 0.01,
) -> dict[str, Any]:
    """Build a realized forward posture-outcome dataset from a neutral reference path."""
    utility_mode = _normalize_utility_mode(utility_mode)
    cfg_eval = load_config() if cfg is None else deepcopy(cfg)
    if start_date is not None:
        cfg_eval["backtest"]["start_date"] = str(pd.Timestamp(start_date).date())
    if end_date is not None:
        cfg_eval["backtest"]["end_date"] = str(pd.Timestamp(end_date).date())

    end_ts = pd.Timestamp(cfg_eval["backtest"]["end_date"])
    engine = WalkForwardEngine(
        price_matrix=price_matrix.loc[:end_ts],
        volume_matrix=volume_matrix.loc[:end_ts],
        macro_df=macro_df.loc[:end_ts],
        cfg=cfg_eval,
        mode="full_rl",
        use_rl=False,
    )
    executor = HistoricalPeriodExecutor(engine, mode="full_rl", allow_model_retraining=True)
    portfolio = executor.initial_portfolio(0)
    nav_points = executor.initial_nav_points(0)
    executor.reset_runtime_state(nav_points)
    timeline_engine = WalkForwardEngine(
        price_matrix=price_matrix.loc[:end_ts],
        volume_matrix=volume_matrix.loc[:end_ts],
        macro_df=macro_df.loc[:end_ts],
        cfg=deepcopy(cfg_eval),
        mode="full_rl",
        use_rl=False,
    )
    model_timeline = _ModelStateTimeline(timeline_engine)

    end_idx = len(executor.rebalance_dates) - 2
    horizon = max(1, int(horizon_rebalances))
    rows: list[dict[str, Any]] = []
    sample_payloads: list[dict[str, Any]] = []

    for idx in range(0, end_idx + 1):
        prepared = executor.prepare_step(idx, portfolio, nav_points)
        if idx + horizon - 1 <= end_idx:
            sample_date = prepared.current_date
            sample = _build_sample(
                engine=engine,
                reference_executor=executor,
                prepared=prepared,
                start_idx=idx,
                horizon_rebalances=horizon,
                portfolio=portfolio,
                nav_points=nav_points,
                model_timeline=model_timeline,
                utility_mode=utility_mode,
            )
            rows.extend(sample.pop("rows"))
            sample_payloads.append(sample)
            if max_samples is not None and len(sample_payloads) >= int(max_samples):
                break

        neutral_decision = SectorAllocationEnv.neutral_action(cfg_eval)
        result = executor.execute_prepared_step(
            prepared,
            portfolio,
            nav_points,
            rl_decision=neutral_decision,
            done=idx == end_idx,
        )
        executor.engine.risk_engine.update(
            result.post_trade_portfolio.nav,
            result.post_trade_portfolio.date,
        )
        portfolio = result.next_portfolio
        nav_points = result.updated_nav_points
        if max_samples is not None and len(sample_payloads) >= int(max_samples):
            break

    frame = pd.DataFrame(rows)
    summary = _summarize_dataset(
        frame,
        sample_payloads,
        horizon,
        utility_mode=utility_mode,
        near_tie_threshold=float(near_tie_threshold),
    )
    return {
        "generated_at_utc": _isoformat_utc(datetime.now(timezone.utc)),
        "dataset_type": "realized_posture_outcomes_v1",
        "window": {
            "start_date": str(engine.start_date.date()),
            "end_date": str(engine.end_date.date()),
        },
        "horizon_rebalances": horizon,
        "primary_utility_mode": utility_mode,
        "near_tie_threshold": float(near_tie_threshold),
        "reference_policy": {
            "name": "neutral_full_stack",
            "posture": "neutral",
            "execution_mode": "full_rl_stack",
        },
        "samples": sample_payloads,
        "summary": summary,
        "rows": frame.to_dict(orient="records"),
    }


def save_posture_dataset(
    payload: dict[str, Any],
    *,
    report_dir: str | Path,
    prefix: str = "posture_dataset",
) -> dict[str, str]:
    """Persist posture dataset rows and summary to the report directory."""
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(payload.get("rows", []))

    parquet_path = report_path / f"{prefix}.parquet"
    json_path = report_path / f"{prefix}_summary.json"

    rows.to_parquet(parquet_path, index=False)
    summary_payload = dict(payload)
    summary_payload.pop("rows", None)
    json_path.write_text(_json_dumps(summary_payload))
    return {
        "parquet_path": str(parquet_path.resolve()),
        "summary_path": str(json_path.resolve()),
    }


def _build_sample(
    *,
    engine: WalkForwardEngine,
    reference_executor: HistoricalPeriodExecutor,
    prepared: PreparedHistoricalStep,
    start_idx: int,
    horizon_rebalances: int,
    portfolio: PortfolioState,
    nav_points: list[tuple[pd.Timestamp, float]],
    model_timeline: _ModelStateTimeline,
    utility_mode: str,
) -> dict[str, Any]:
    state = deepcopy(prepared.transition_state)
    stress_signal = float(reference_executor._stress_signal(state))
    stress_bucket = _stress_bucket(stress_signal)
    posture_outcomes: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for posture in ("risk_on", "neutral", "risk_off"):
        outcome = _simulate_fixed_posture_horizon(
            engine=engine,
            reference_executor=reference_executor,
            start_idx=start_idx,
            horizon_rebalances=horizon_rebalances,
            portfolio=portfolio,
            nav_points=nav_points,
            posture=posture,
            model_timeline=model_timeline,
        )
        posture_outcomes[posture] = outcome
        rows.append(
            {
                "date": str(prepared.current_date.date()),
                "rebalance_idx": int(start_idx),
                "horizon_rebalances": int(horizon_rebalances),
                "stress_signal": stress_signal,
                "stress_bucket": stress_bucket,
                "posture": posture,
                "utility": float(outcome["utility_by_mode"][utility_mode]),
                "utility_mode": str(utility_mode),
                "utility_return_only": float(outcome["utility_by_mode"]["return_only"]),
                "utility_return_minus_drawdown": float(outcome["utility_by_mode"]["return_minus_drawdown"]),
                "utility_full_utility": float(outcome["utility_by_mode"]["full_utility"]),
                "total_return": float(outcome["total_return"]),
                "max_drawdown": float(outcome["max_drawdown"]),
                "avg_turnover": float(outcome["avg_turnover"]),
                "avg_cost_ratio": float(outcome["avg_cost_ratio"]),
                "fallback_count": int(outcome["fallback_count"]),
                "mean_selected_sector_count": float(outcome["mean_selected_sector_count"]),
                "mean_selected_stock_count": float(outcome["mean_selected_stock_count"]),
                "ending_nav": float(outcome["ending_nav"]),
            }
        )

    winner_by_mode: dict[str, str] = {}
    margin_by_mode: dict[str, float] = {}
    for mode in UTILITY_MODES:
        ordered = sorted(
            posture_outcomes.items(),
            key=lambda item: (
                float(item[1]["utility_by_mode"][mode]),
                float(item[1]["total_return"]),
            ),
            reverse=True,
        )
        best_posture, best_metrics = ordered[0]
        second_metrics = ordered[1][1] if len(ordered) > 1 else ordered[0][1]
        winner_by_mode[mode] = str(best_posture)
        margin_by_mode[mode] = float(
            best_metrics["utility_by_mode"][mode] - second_metrics["utility_by_mode"][mode]
        )

    return {
        "date": str(prepared.current_date.date()),
        "rebalance_idx": int(start_idx),
        "horizon_rebalances": int(horizon_rebalances),
        "state": state,
        "stress_signal": stress_signal,
        "stress_bucket": stress_bucket,
        "best_posture": winner_by_mode[utility_mode],
        "utility_margin": margin_by_mode[utility_mode],
        "winner_by_utility_mode": winner_by_mode,
        "margin_by_utility_mode": margin_by_mode,
        "posture_outcomes": posture_outcomes,
        "rows": rows,
    }


def _simulate_fixed_posture_horizon(
    *,
    engine: WalkForwardEngine,
    reference_executor: HistoricalPeriodExecutor,
    start_idx: int,
    horizon_rebalances: int,
    portfolio: PortfolioState,
    nav_points: list[tuple[pd.Timestamp, float]],
    posture: str,
    model_timeline: _ModelStateTimeline,
) -> dict[str, Any]:
    executor = HistoricalPeriodExecutor(engine, mode="full_rl", allow_model_retraining=False)
    _copy_runtime_state(executor, reference_executor, nav_points)
    sim_portfolio = _clone_portfolio(portfolio)
    sim_nav_points = _clone_nav_points(nav_points)
    trace: list[dict[str, Any]] = []

    final_idx = min(len(executor.rebalance_dates) - 2, start_idx + horizon_rebalances - 1)
    for idx in range(start_idx, final_idx + 1):
        model_timeline.restore(idx, executor.engine)
        prepared = executor.prepare_step(idx, sim_portfolio, sim_nav_points)
        decision = _fixed_posture_decision(executor.engine.cfg, list(prepared.snapshot.sectors), posture)
        result = executor.execute_prepared_step(
            prepared,
            sim_portfolio,
            sim_nav_points,
            rl_decision=decision,
            done=idx == final_idx,
        )
        info = result.transition.get("info", {})
        optimizer_diag = dict(info.get("optimizer_diagnostics", {}))
        trace.append(
            {
                "turnover": float(result.exec_result.total_turnover),
                "cost_ratio": float(info.get("transaction_cost_ratio", 0.0)),
                "fallback_mode": str(info.get("optimizer_fallback_mode", "none")),
                "selected_sector_count": int(len(result.selected_sectors)),
                "selected_stock_count": int(len(result.selected_stock_rows)),
            }
        )
        executor.engine.risk_engine.update(
            result.post_trade_portfolio.nav,
            result.post_trade_portfolio.date,
        )
        sim_portfolio = result.next_portfolio
        sim_nav_points = result.updated_nav_points
        _ = optimizer_diag

    nav_series = pd.Series(
        [float(nav) for _, nav in sim_nav_points],
        index=pd.DatetimeIndex([pd.Timestamp(ts) for ts, _ in sim_nav_points]),
        name="portfolio_nav",
    )
    nav_series = nav_series[~nav_series.index.duplicated(keep="last")].sort_index()
    total_return = 0.0
    if len(nav_series) >= 2 and float(nav_series.iloc[0]) > 0:
        total_return = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1.0)
    max_drawdown = _nav_max_drawdown(nav_series)
    avg_turnover = float(np.mean([entry["turnover"] for entry in trace])) if trace else 0.0
    avg_cost_ratio = float(np.mean([entry["cost_ratio"] for entry in trace])) if trace else 0.0
    utility_by_mode = {
        mode: _horizon_utility(
            engine.cfg,
            total_return=total_return,
            max_drawdown=max_drawdown,
            avg_turnover=avg_turnover,
            avg_cost_ratio=avg_cost_ratio,
            utility_mode=mode,
        )
        for mode in UTILITY_MODES
    }
    return {
        "utility": float(utility_by_mode["full_utility"]),
        "utility_by_mode": utility_by_mode,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
        "avg_cost_ratio": avg_cost_ratio,
        "fallback_count": int(sum(1 for entry in trace if entry["fallback_mode"] != "none")),
        "mean_selected_sector_count": float(np.mean([entry["selected_sector_count"] for entry in trace])) if trace else 0.0,
        "mean_selected_stock_count": float(np.mean([entry["selected_stock_count"] for entry in trace])) if trace else 0.0,
        "ending_nav": float(nav_series.iloc[-1]) if not nav_series.empty else float(sim_portfolio.nav),
    }


def _fixed_posture_decision(cfg: dict, sectors: list[str], posture: str) -> dict[str, Any]:
    decision = default_decision(sectors)
    controls = HistoricalPeriodExecutor._target_controls_for_posture(posture, cfg)
    decision["posture"] = str(posture)
    decision["cash_target"] = float(controls["cash_target"])
    decision["aggressiveness"] = float(controls["aggressiveness"])
    decision["turnover_cap"] = float(controls["turnover_cap"])
    decision["allow_forced_posture_override"] = False
    return decision


def _copy_runtime_state(
    dst: HistoricalPeriodExecutor,
    src: HistoricalPeriodExecutor,
    nav_points: list[tuple[pd.Timestamp, float]],
) -> None:
    dst.reset_runtime_state(_clone_nav_points(nav_points))
    dst._recent_turnovers = list(src._recent_turnovers)
    dst._recent_cost_ratios = list(src._recent_cost_ratios)
    dst._prev_posture = str(src._prev_posture)
    dst._prev_target_posture = str(src._prev_target_posture)
    dst._prev_stress_signal = float(src._prev_stress_signal)
    dst._target_posture_streak = int(src._target_posture_streak)
    dst._prev_posture_mismatch = float(src._prev_posture_mismatch)


def _snapshot_model_state(engine: WalkForwardEngine) -> dict[str, bytes]:
    return {
        "sector_scorer": pickle.dumps(
            {
                "model": engine.sector_scorer.model,
                "scaler": engine.sector_scorer.scaler,
                "feature_names": list(engine.sector_scorer.feature_names),
                "is_fitted": bool(engine.sector_scorer.is_fitted),
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        ),
        "stock_ranker": pickle.dumps(
            {
                "models": engine.stock_ranker.models,
                "scalers": engine.stock_ranker.scalers,
                "feature_names": list(engine.stock_ranker.feature_names),
                "is_fitted": bool(engine.stock_ranker.is_fitted),
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        ),
    }


def _restore_model_snapshot(engine: WalkForwardEngine, snapshot: dict[str, bytes]) -> None:
    sector_state = pickle.loads(snapshot["sector_scorer"])
    engine.sector_scorer.model = sector_state["model"]
    engine.sector_scorer.scaler = sector_state["scaler"]
    engine.sector_scorer.feature_names = list(sector_state["feature_names"])
    engine.sector_scorer.is_fitted = bool(sector_state["is_fitted"])

    stock_state = pickle.loads(snapshot["stock_ranker"])
    engine.stock_ranker.models = stock_state["models"]
    engine.stock_ranker.scalers = stock_state["scalers"]
    engine.stock_ranker.feature_names = list(stock_state["feature_names"])
    engine.stock_ranker.is_fitted = bool(stock_state["is_fitted"])


def _clone_portfolio(portfolio: PortfolioState) -> PortfolioState:
    return PortfolioState(
        date=portfolio.date,
        cash=float(portfolio.cash),
        holdings=dict(portfolio.holdings),
        weights=dict(portfolio.weights),
        nav=float(portfolio.nav),
        sector_weights=dict(portfolio.sector_weights),
    )


def _clone_nav_points(nav_points: list[tuple[pd.Timestamp, float]]) -> list[tuple[pd.Timestamp, float]]:
    return [(pd.Timestamp(ts), float(nav)) for ts, nav in nav_points]


def _nav_max_drawdown(nav_series: pd.Series) -> float:
    if nav_series.empty:
        return 0.0
    running_max = nav_series.cummax()
    drawdowns = nav_series / running_max - 1.0
    return float(drawdowns.min())


def _horizon_utility(
    cfg: dict,
    *,
    total_return: float,
    max_drawdown: float,
    avg_turnover: float,
    avg_cost_ratio: float,
    utility_mode: str = "full_utility",
) -> float:
    utility_mode = _normalize_utility_mode(utility_mode)
    rl_cfg = cfg.get("rl", {}) if isinstance(cfg, dict) else {}
    dd_weight = float(rl_cfg.get("reward_lambda_dd", 0.25))
    to_weight = float(rl_cfg.get("reward_lambda_to", 0.5))
    cost_weight = float(rl_cfg.get("reward_lambda_liq", 0.2))
    if utility_mode == "return_only":
        return float(total_return)
    if utility_mode == "return_minus_drawdown":
        return float(total_return - dd_weight * abs(float(max_drawdown)))
    return float(
        total_return
        - dd_weight * abs(float(max_drawdown))
        - to_weight * float(avg_turnover)
        - cost_weight * float(avg_cost_ratio)
    )


def _stress_bucket(stress_signal: float) -> str:
    if stress_signal >= 0.35:
        return "high"
    if stress_signal >= 0.18:
        return "medium"
    return "low"


def _summarize_dataset(
    rows: pd.DataFrame,
    samples: list[dict[str, Any]],
    horizon_rebalances: int,
    *,
    utility_mode: str = "full_utility",
    near_tie_threshold: float = 0.01,
) -> dict[str, Any]:
    utility_mode = _normalize_utility_mode(utility_mode)
    if rows.empty:
        return {
            "sample_count": 0,
            "horizon_rebalances": int(horizon_rebalances),
            "primary_utility_mode": utility_mode,
            "near_tie_threshold": float(near_tie_threshold),
            "best_posture_counts": {},
            "best_posture_by_stress_bucket": {},
            "mean_utility_margin": None,
            "mean_utility_margin_by_stress_bucket": {},
            "near_tie_rate": None,
            "utility_mode_summaries": {},
            "posture_outcome_stats": {},
            "winner_by_metric": {},
            "execution_clean_subset": {},
        }

    sample_frame = pd.DataFrame(
        {
            "date": [sample["date"] for sample in samples],
            "stress_bucket": [sample["stress_bucket"] for sample in samples],
            "stress_signal": [float(sample["stress_signal"]) for sample in samples],
            "best_posture": [sample["best_posture"] for sample in samples],
            "utility_margin": [float(sample["utility_margin"]) for sample in samples],
            "winner_return_only": [sample["winner_by_utility_mode"]["return_only"] for sample in samples],
            "winner_return_minus_drawdown": [
                sample["winner_by_utility_mode"]["return_minus_drawdown"] for sample in samples
            ],
            "winner_full_utility": [sample["winner_by_utility_mode"]["full_utility"] for sample in samples],
            "margin_return_only": [float(sample["margin_by_utility_mode"]["return_only"]) for sample in samples],
            "margin_return_minus_drawdown": [
                float(sample["margin_by_utility_mode"]["return_minus_drawdown"]) for sample in samples
            ],
            "margin_full_utility": [float(sample["margin_by_utility_mode"]["full_utility"]) for sample in samples],
            "execution_clean": [
                bool(all(
                    int(metrics.get("fallback_count", 0)) == 0
                    for metrics in sample.get("posture_outcomes", {}).values()
                ))
                for sample in samples
            ],
        }
    )
    outcome_stats: dict[str, Any] = {}
    for posture, posture_rows in rows.groupby("posture", sort=False):
        outcome_stats[str(posture)] = {
            "mean_utility": float(posture_rows["utility"].mean()),
            "mean_total_return": float(posture_rows["total_return"].mean()),
            "mean_max_drawdown": float(posture_rows["max_drawdown"].mean()),
            "mean_avg_turnover": float(posture_rows["avg_turnover"].mean()),
            "mean_avg_cost_ratio": float(posture_rows["avg_cost_ratio"].mean()),
            "mean_fallback_count": float(posture_rows["fallback_count"].mean()),
            "mean_selected_sector_count": float(posture_rows["mean_selected_sector_count"].mean()),
            "mean_selected_stock_count": float(posture_rows["mean_selected_stock_count"].mean()),
        }

    utility_mode_summaries = {
        mode: _utility_mode_summary(
            sample_frame=sample_frame,
            winner_col=f"winner_{mode}",
            margin_col=f"margin_{mode}",
            near_tie_threshold=float(near_tie_threshold),
        )
        for mode in UTILITY_MODES
    }

    winner_by_metric = {
        "total_return": _winner_counts_from_rows(rows, metric="total_return", higher_is_better=True),
        "max_drawdown": _winner_counts_from_rows(rows, metric="max_drawdown", higher_is_better=True),
        "avg_turnover": _winner_counts_from_rows(rows, metric="avg_turnover", higher_is_better=False),
        "avg_cost_ratio": _winner_counts_from_rows(rows, metric="avg_cost_ratio", higher_is_better=False),
    }

    return {
        "sample_count": int(len(sample_frame)),
        "horizon_rebalances": int(horizon_rebalances),
        "primary_utility_mode": utility_mode,
        "near_tie_threshold": float(near_tie_threshold),
        "best_posture_counts": dict(utility_mode_summaries[utility_mode]["best_posture_counts"]),
        "best_posture_by_stress_bucket": dict(utility_mode_summaries[utility_mode]["best_posture_by_stress_bucket"]),
        "mean_utility_margin": float(utility_mode_summaries[utility_mode]["mean_utility_margin"]),
        "mean_utility_margin_by_stress_bucket": dict(utility_mode_summaries[utility_mode]["mean_utility_margin_by_stress_bucket"]),
        "near_tie_rate": float(utility_mode_summaries[utility_mode]["near_tie_rate"]),
        "utility_mode_summaries": utility_mode_summaries,
        "posture_outcome_stats": outcome_stats,
        "winner_by_metric": winner_by_metric,
        "execution_clean_subset": _utility_mode_summary(
            sample_frame=sample_frame[sample_frame["execution_clean"]].copy(),
            winner_col=f"winner_{utility_mode}",
            margin_col=f"margin_{utility_mode}",
            near_tie_threshold=float(near_tie_threshold),
        ),
        "label_stability_across_utility_modes": _label_stability_summary(sample_frame),
    }


def _utility_mode_summary(
    *,
    sample_frame: pd.DataFrame,
    winner_col: str,
    margin_col: str,
    near_tie_threshold: float,
) -> dict[str, Any]:
    if sample_frame.empty:
        return {
            "sample_count": 0,
            "best_posture_counts": {},
            "best_posture_by_stress_bucket": {},
            "mean_utility_margin": None,
            "mean_utility_margin_by_stress_bucket": {},
            "near_tie_rate": None,
        }
    return {
        "sample_count": int(len(sample_frame)),
        "best_posture_counts": {
            str(key): int(value) for key, value in sample_frame[winner_col].value_counts().to_dict().items()
        },
        "best_posture_by_stress_bucket": {
            str(bucket): {
                str(key): int(value)
                for key, value in bucket_frame[winner_col].value_counts().to_dict().items()
            }
            for bucket, bucket_frame in sample_frame.groupby("stress_bucket", sort=False)
        },
        "mean_utility_margin": float(sample_frame[margin_col].mean()),
        "mean_utility_margin_by_stress_bucket": {
            str(bucket): float(bucket_frame[margin_col].mean())
            for bucket, bucket_frame in sample_frame.groupby("stress_bucket", sort=False)
        },
        "near_tie_rate": float((sample_frame[margin_col] <= float(near_tie_threshold)).mean()),
    }


def _winner_counts_from_rows(rows: pd.DataFrame, *, metric: str, higher_is_better: bool) -> dict[str, int]:
    if rows.empty or metric not in rows.columns:
        return {}
    counts: dict[str, int] = {}
    for _, frame in rows.groupby("date", sort=False):
        ordered = frame.sort_values(
            [metric, "posture"],
            ascending=[not higher_is_better, True],
            kind="mergesort",
        )
        winner = str(ordered.iloc[0]["posture"])
        counts[winner] = counts.get(winner, 0) + 1
    return counts


def _label_stability_summary(sample_frame: pd.DataFrame) -> dict[str, Any]:
    if sample_frame.empty:
        return {
            "all_modes_agree_rate": None,
            "return_only_vs_full_utility_agree_rate": None,
            "return_minus_drawdown_vs_full_utility_agree_rate": None,
        }
    all_agree = (
        (sample_frame["winner_return_only"] == sample_frame["winner_return_minus_drawdown"])
        & (sample_frame["winner_return_only"] == sample_frame["winner_full_utility"])
    )
    return {
        "all_modes_agree_rate": float(all_agree.mean()),
        "return_only_vs_full_utility_agree_rate": float(
            (sample_frame["winner_return_only"] == sample_frame["winner_full_utility"]).mean()
        ),
        "return_minus_drawdown_vs_full_utility_agree_rate": float(
            (sample_frame["winner_return_minus_drawdown"] == sample_frame["winner_full_utility"]).mean()
        ),
    }


def _normalize_utility_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in UTILITY_MODES:
        raise ValueError(f"Unsupported utility mode: {value}")
    return mode


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
