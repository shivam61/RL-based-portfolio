"""
Fixed-policy holdout evaluation for the causal RL overlay.

This module trains a fresh RL policy on history up to a holdout boundary and
then compares the frozen trained policy against a neutral policy on the same
post-training rebalance windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtest.simulator import PortfolioSimulator
from src.backtest.walk_forward import WalkForwardEngine
from src.config import load_config
from src.rl.contract import CAUSAL_TRAINING_BACKEND
from src.rl.environment import SECTORS, SectorAllocationEnv
from src.rl.historical_executor import HistoricalPeriodExecutor
from src.rl.policy_utils import default_decision


@dataclass
class HoldoutRunResult:
    metrics: dict[str, Any]
    nav_series: pd.Series
    trace: list[dict[str, Any]]


def evaluate_holdout(
    price_matrix: pd.DataFrame,
    volume_matrix: pd.DataFrame,
    macro_df: pd.DataFrame,
    cfg: dict | None = None,
    *,
    holdout_start: str,
    holdout_end: str,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    """
    Train a fresh causal RL policy through the holdout boundary, then compare
    the frozen trained policy against a neutral policy on the same holdout.
    """
    cfg_eval = load_config() if cfg is None else cfg
    cfg_eval = _copy_cfg(cfg_eval)
    cfg_eval["rl"]["training_backend"] = CAUSAL_TRAINING_BACKEND
    cfg_eval["rl"]["use_rl"] = True
    if total_timesteps is not None:
        cfg_eval["rl"]["total_timesteps"] = int(total_timesteps)

    holdout_start_ts = pd.Timestamp(holdout_start)
    holdout_end_ts = pd.Timestamp(holdout_end)
    cfg_eval["backtest"]["end_date"] = str(holdout_end_ts.date())

    engine = WalkForwardEngine(
        price_matrix=price_matrix.loc[:holdout_end_ts],
        volume_matrix=volume_matrix.loc[:holdout_end_ts],
        macro_df=macro_df.loc[:holdout_end_ts],
        cfg=cfg_eval,
        mode="full_rl",
    )
    rebalance_dates = engine._generate_rebalance_dates()
    holdout_start_idx = _find_holdout_start_idx(rebalance_dates, holdout_start_ts)
    holdout_end_idx = len(rebalance_dates) - 2
    if holdout_start_idx <= 0:
        raise ValueError("Holdout start must leave at least one prior rebalance window for training.")
    if holdout_start_idx > holdout_end_idx:
        raise ValueError("Holdout start is after the last available rebalance window.")

    train_end_rebalance = rebalance_dates[holdout_start_idx - 1]
    holdout_rebalance = rebalance_dates[holdout_start_idx]
    engine._train_models(holdout_rebalance, idx=holdout_start_idx)

    train_executor = HistoricalPeriodExecutor(
        engine,
        mode="full_rl",
        allow_model_retraining=False,
    )
    train_env = engine.rl_agent.build_causal_env(
        train_executor,
        start_idx=0,
        end_idx=holdout_start_idx - 1,
        max_episode_steps=holdout_start_idx,
    )
    engine.rl_agent.train(
        total_timesteps=int(cfg_eval["rl"].get("total_timesteps", 20000)),
        causal_env=train_env,
    )

    eval_executor = HistoricalPeriodExecutor(
        engine,
        mode="full_rl",
        allow_model_retraining=False,
    )
    neutral = _run_holdout_policy(
        executor=eval_executor,
        start_idx=holdout_start_idx,
        end_idx=holdout_end_idx,
        decision_fn=lambda prepared: SectorAllocationEnv.neutral_action(cfg_eval),
        benchmark=_benchmark_series(price_matrix, cfg_eval, holdout_rebalance, holdout_end_ts),
    )
    trained = _run_holdout_policy(
        executor=eval_executor,
        start_idx=holdout_start_idx,
        end_idx=holdout_end_idx,
        decision_fn=lambda prepared: engine.rl_agent.decide(
            macro_state=prepared.transition_state["macro_state"],
            sector_state=prepared.transition_state["sector_state"],
            portfolio_state=prepared.transition_state["portfolio_state"],
        ),
        benchmark=_benchmark_series(price_matrix, cfg_eval, holdout_rebalance, holdout_end_ts),
    )
    fixed_posture_runs = {
        posture: _run_holdout_policy(
            executor=eval_executor,
            start_idx=holdout_start_idx,
            end_idx=holdout_end_idx,
            decision_fn=lambda prepared, posture=posture: _fixed_posture_decision(
                cfg_eval,
                list(prepared.snapshot.sectors),
                posture,
            ),
            benchmark=_benchmark_series(price_matrix, cfg_eval, holdout_rebalance, holdout_end_ts),
        )
        for posture in ("risk_on", "neutral", "risk_off")
    }

    return {
        "training_backend": engine.rl_agent.training_backend,
        "trained_policy_ready": bool(engine.rl_agent.is_trained),
        "train_end_rebalance": str(train_end_rebalance.date()),
        "holdout_start_rebalance": str(holdout_rebalance.date()),
        "holdout_end_rebalance": str(rebalance_dates[holdout_end_idx + 1].date()),
        "holdout_windows": int(holdout_end_idx - holdout_start_idx + 1),
        "trained_policy": trained.metrics,
        "neutral_policy": neutral.metrics,
        "uplift": _compute_uplift(trained.metrics, neutral.metrics),
        "trained_policy_diagnostics": _summarize_trace(trained.trace, cfg_eval),
        "neutral_policy_diagnostics": _summarize_trace(neutral.trace, cfg_eval),
        "fixed_posture_policies": {
            posture: {
                "metrics": run.metrics,
                "diagnostics": _summarize_trace(run.trace, cfg_eval),
            }
            for posture, run in fixed_posture_runs.items()
        },
        "trained_policy_behavior_flags": _posture_behavior_flags(
            _summarize_trace(trained.trace, cfg_eval),
        ),
        "trained_policy_trace": trained.trace,
        "neutral_policy_trace": neutral.trace,
    }


def _run_holdout_policy(
    *,
    executor: HistoricalPeriodExecutor,
    start_idx: int,
    end_idx: int,
    decision_fn: Callable[[Any], dict[str, Any]],
    benchmark: pd.Series | None,
) -> HoldoutRunResult:
    portfolio = executor.initial_portfolio(start_idx)
    nav_points = executor.initial_nav_points(start_idx)
    executor.reset_runtime_state(nav_points)
    trace: list[dict[str, Any]] = []

    for idx in range(start_idx, end_idx + 1):
        prepared = executor.prepare_step(idx, portfolio, nav_points)
        decision = decision_fn(prepared)
        result = executor.execute_prepared_step(
            prepared,
            portfolio,
            nav_points,
            rl_decision=decision,
            done=idx == end_idx,
        )
        info = result.transition.get("info", {})
        reward_components = dict(info.get("reward_components", {}))
        trace.append(
            {
                "date": str(result.current_date.date()),
                "reward": float(result.reward),
                "period_return": float(
                    reward_components.get("period_return", 0.0)
                ),
                "turnover": float(result.exec_result.total_turnover),
                "transaction_cost": float(result.exec_result.total_cost),
                "cash_target": float(info.get("cash_target", result.cash_target)),
                "turnover_cap": (
                    float(info.get("turnover_cap"))
                    if info.get("turnover_cap") is not None
                    else None
                ),
                "requested_cash_target": float(info.get("requested_cash_target", info.get("cash_target", result.cash_target))),
                "requested_turnover_cap": (
                    float(info.get("requested_turnover_cap"))
                    if info.get("requested_turnover_cap") is not None
                    else None
                ),
                "realized_cash_weight": float(info.get("realized_cash_weight", result.next_portfolio.weights.get("CASH", 0.0))),
                "optimizer_reason_code": str(info.get("optimizer_reason_code", "unknown")),
                "optimizer_fallback_mode": str(info.get("optimizer_fallback_mode", "none")),
                "optimizer_diagnostics": dict(info.get("optimizer_diagnostics", {})),
                "posture": str(info.get("posture", decision.get("posture", "neutral"))),
                "aggressiveness": float(info.get("aggressiveness", decision.get("aggressiveness", 1.0))),
                "should_rebalance": bool(decision.get("should_rebalance", True)),
                "selected_sectors": list(result.selected_sectors),
                "selected_sector_count": int(len(result.selected_sectors)),
                "selected_stock_count": int(len(result.selected_stock_rows)),
                "sector_tilts": {
                    str(sector): float(tilt)
                    for sector, tilt in decision.get("sector_tilts", {}).items()
                },
                "reward_components": reward_components,
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
        float(np.mean([entry["turnover"] for entry in trace])) if trace else 0.0
    )
    metrics["total_rebalances"] = len(trace)
    return HoldoutRunResult(metrics=metrics, nav_series=nav_series, trace=trace)


def _compute_uplift(trained: dict[str, Any], neutral: dict[str, Any]) -> dict[str, Any]:
    keys = ("cagr", "sharpe", "max_drawdown", "avg_turnover", "total_return")
    uplift: dict[str, Any] = {}
    for key in keys:
        if key in trained and key in neutral:
            uplift[key] = float(trained[key]) - float(neutral[key])
    return uplift


def _fixed_posture_decision(
    cfg: dict,
    sectors: list[str],
    posture: str,
) -> dict[str, Any]:
    decision = default_decision(sectors)
    controls = HistoricalPeriodExecutor._target_controls_for_posture(posture, cfg)
    decision["posture"] = str(posture)
    decision["cash_target"] = float(controls["cash_target"])
    decision["aggressiveness"] = float(controls["aggressiveness"])
    decision["turnover_cap"] = float(controls["turnover_cap"])
    decision["allow_forced_posture_override"] = False
    return decision


def _summarize_trace(trace: list[dict[str, Any]], cfg: dict | None = None) -> dict[str, Any]:
    if not trace:
        return {}
    rl_cfg = (cfg or {}).get("rl", {}) if isinstance(cfg, dict) else {}
    tilt_vectors = []
    for entry in trace:
        tilts = entry.get("sector_tilts", {})
        tilt_vectors.append(
            np.array(
                [float(tilts.get(sector, 1.0)) for sector in SECTORS],
                dtype=float,
            )
        )
    arr = np.vstack(tilt_vectors)
    change_rate = 0.0
    if len(arr) > 1:
        diffs = np.abs(np.diff(arr, axis=0)).sum(axis=1)
        change_rate = float(np.mean(diffs > 1e-6))
    neutral_cash = float(SectorAllocationEnv.neutral_action(cfg).get("cash_target", 0.05))
    neutral_turnover_cap = SectorAllocationEnv.neutral_action(cfg).get("turnover_cap")
    neutral_aggressiveness = float(
        SectorAllocationEnv.neutral_action(cfg).get("aggressiveness", 1.0)
    )
    neutral_posture = str(SectorAllocationEnv.neutral_action(cfg).get("posture", "neutral"))

    def usage_rate(key: str, neutral_value: float | None) -> float:
        values = [entry.get(key) for entry in trace]
        if not values:
            return 0.0
        if neutral_value is None:
            return float(
                np.mean([1.0 if value is not None else 0.0 for value in values])
            )
        return float(
            np.mean(
                [
                    1.0 if value is not None and abs(float(value) - float(neutral_value)) > 1e-6 else 0.0
                    for value in values
                ]
            )
        )

    def unique_values(key: str) -> list[float]:
        values = sorted(
            {
                round(float(value), 6)
                for value in [entry.get(key) for entry in trace]
                if value is not None
            }
        )
        return values

    def posture_counts(values: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return counts

    def stress_bucket(stress_signal: float) -> str:
        moderate = float(rl_cfg.get("stress_target_moderate", 0.18))
        high = float(rl_cfg.get("stress_target_high", 0.35))
        if stress_signal >= high:
            return "high"
        if stress_signal >= moderate:
            return "medium"
        return "low"

    def mean_of(entries: list[dict[str, Any]], key: str) -> float | None:
        values = [entry.get(key) for entry in entries if entry.get(key) is not None]
        if not values:
            return None
        return float(np.mean([float(value) for value in values]))

    def mean_opt_diag(entries: list[dict[str, Any]], key: str) -> float | None:
        values = [
            (entry.get("optimizer_diagnostics") or {}).get(key)
            for entry in entries
            if (entry.get("optimizer_diagnostics") or {}).get(key) is not None
        ]
        if not values:
            return None
        return float(np.mean([float(value) for value in values]))

    postures = [str(entry.get("posture", "neutral")) for entry in trace]
    target_postures = [
        str(entry.get("reward_components", {}).get("target_posture", "neutral"))
        for entry in trace
    ]
    posture_change_rate = 0.0
    if len(postures) > 1:
        posture_change_rate = float(
            np.mean(
                [
                    1.0 if postures[idx] != postures[idx - 1] else 0.0
                    for idx in range(1, len(postures))
                ]
            )
        )

    bucket_rows: dict[str, list[dict[str, Any]]] = {"low": [], "medium": [], "high": []}
    posture_rows: dict[str, list[dict[str, Any]]] = {}
    posture_by_bucket: dict[str, dict[str, int]] = {"low": {}, "medium": {}, "high": {}}
    target_by_bucket: dict[str, dict[str, int]] = {"low": {}, "medium": {}, "high": {}}
    regrets: list[float] = []
    optimal_hits: list[float] = []
    utility_dispersion: list[float] = []
    for entry, posture, target_posture in zip(trace, postures, target_postures):
        reward_components = entry.get("reward_components", {})
        stress_signal = float(reward_components.get("stress_signal", 0.0))
        bucket = stress_bucket(stress_signal)
        bucket_rows[bucket].append(entry)
        posture_rows.setdefault(posture, []).append(entry)
        posture_by_bucket[bucket][posture] = posture_by_bucket[bucket].get(posture, 0) + 1
        target_by_bucket[bucket][target_posture] = (
            target_by_bucket[bucket].get(target_posture, 0) + 1
        )
        regret = float(
            reward_components.get(
                "soft_regret",
                reward_components.get("posture_distance_to_target", 0.0),
            )
        )
        regrets.append(regret)
        best_posture = str(reward_components.get("best_posture", target_posture))
        optimal_hits.append(1.0 if posture == best_posture else 0.0)
        utility_dispersion.append(
            float(
                reward_components.get(
                    "posture_utility_variance",
                    0.0,
                )
            )
        )

    control_by_posture = {
        posture: {
            "observations": len(entries),
            "avg_cash_target": mean_of(entries, "cash_target"),
            "avg_aggressiveness": mean_of(entries, "aggressiveness"),
            "avg_turnover": mean_of(entries, "turnover"),
            "avg_selected_stock_count": mean_of(entries, "selected_stock_count"),
            "avg_selected_sector_count": mean_of(entries, "selected_sector_count"),
            "avg_weight_before_cap": mean_opt_diag(entries, "avg_weight_before_cap"),
            "avg_weight_after_cap": mean_opt_diag(entries, "avg_weight_after_cap"),
            "avg_cap_clipping_ratio": mean_opt_diag(entries, "cap_clipping_ratio"),
            "avg_top5_weight_sum": mean_opt_diag(entries, "top5_weight_sum"),
            "avg_top10_weight_sum": mean_opt_diag(entries, "top10_weight_sum"),
            "avg_hhi": mean_opt_diag(entries, "realized_hhi"),
        }
        for posture, entries in sorted(posture_rows.items())
    }
    control_by_stress_bucket = {
        bucket: {
            "observations": len(entries),
            "avg_cash_target": mean_of(entries, "cash_target"),
            "avg_aggressiveness": mean_of(entries, "aggressiveness"),
            "avg_turnover": mean_of(entries, "turnover"),
            "avg_selected_stock_count": mean_of(entries, "selected_stock_count"),
            "avg_selected_sector_count": mean_of(entries, "selected_sector_count"),
            "avg_weight_before_cap": mean_opt_diag(entries, "avg_weight_before_cap"),
            "avg_weight_after_cap": mean_opt_diag(entries, "avg_weight_after_cap"),
            "avg_cap_clipping_ratio": mean_opt_diag(entries, "cap_clipping_ratio"),
            "avg_top5_weight_sum": mean_opt_diag(entries, "top5_weight_sum"),
            "avg_top10_weight_sum": mean_opt_diag(entries, "top10_weight_sum"),
            "avg_hhi": mean_opt_diag(entries, "realized_hhi"),
        }
        for bucket, entries in bucket_rows.items()
    }
    regret_by_stress_bucket = {
        bucket: (
            float(
                np.mean(
                    [
                        float(
                            entry.get("reward_components", {}).get(
                                "soft_regret",
                                entry.get("reward_components", {}).get(
                                    "posture_distance_to_target", 0.0
                                ),
                            )
                        )
                        for entry in entries
                    ]
                )
            )
            if entries
            else None
        )
        for bucket, entries in bucket_rows.items()
    }
    utility_dispersion_by_stress_bucket = {
        bucket: (
            float(
                np.mean(
                    [
                        float(entry.get("reward_components", {}).get("posture_utility_variance", 0.0))
                        for entry in entries
                    ]
                )
            )
            if entries
            else None
        )
        for bucket, entries in bucket_rows.items()
    }
    posture_optimality_by_stress_bucket = {
        bucket: (
            float(
                np.mean(
                    [
                        1.0
                        if str(entry.get("posture", "neutral"))
                        == str(
                            entry.get("reward_components", {}).get(
                                "best_posture",
                                entry.get("reward_components", {}).get("target_posture", "neutral"),
                            )
                        )
                        else 0.0
                        for entry in entries
                    ]
                )
            )
            if entries
            else None
        )
        for bucket, entries in bucket_rows.items()
    }

    return {
        "mean_abs_tilt_deviation": float(np.mean(np.abs(arr - 1.0))),
        "mean_min_tilt": float(np.mean(np.min(arr, axis=1))),
        "mean_max_tilt": float(np.mean(np.max(arr, axis=1))),
        "mean_cash_target": float(np.mean([entry["cash_target"] for entry in trace])),
        "mean_turnover_cap": float(
            np.mean(
                [
                    entry["turnover_cap"]
                    for entry in trace
                    if entry.get("turnover_cap") is not None
                ]
            )
        ) if any(entry.get("turnover_cap") is not None for entry in trace) else None,
        "mean_aggressiveness": float(
            np.mean([entry["aggressiveness"] for entry in trace])
        ),
        "mean_turnover": float(np.mean([entry["turnover"] for entry in trace])),
        "cash_usage_rate": usage_rate("cash_target", neutral_cash),
        "turnover_cap_usage_rate": usage_rate("turnover_cap", neutral_turnover_cap),
        "mean_requested_vs_realized_cash_gap": float(
            np.mean(
                [
                    abs(float(entry.get("requested_cash_target", entry["cash_target"])) - float(entry.get("realized_cash_weight", 0.0)))
                    for entry in trace
                ]
            )
        ),
        "optimizer_reason_counts": {
            reason: int(sum(1 for entry in trace if str(entry.get("optimizer_reason_code", "unknown")) == reason))
            for reason in sorted({str(entry.get("optimizer_reason_code", "unknown")) for entry in trace})
        },
        "optimizer_fallback_counts": {
            mode: int(sum(1 for entry in trace if str(entry.get("optimizer_fallback_mode", "none")) == mode))
            for mode in sorted({str(entry.get("optimizer_fallback_mode", "none")) for entry in trace})
        },
        "optimizer_relaxation_tier_counts": {
            tier: int(
                sum(
                    1
                    for entry in trace
                    if str((entry.get("optimizer_diagnostics") or {}).get("relaxation_tier", "unknown")) == tier
                )
            )
            for tier in sorted(
                {
                    str((entry.get("optimizer_diagnostics") or {}).get("relaxation_tier", "unknown"))
                    for entry in trace
                }
            )
        },
        "mean_fallback_cash_target_gap": float(np.mean([
            float((entry.get("optimizer_diagnostics") or {}).get("fallback_cash_target_gap", 0.0))
            for entry in trace
            if str(entry.get("optimizer_fallback_mode", "none")) == "risk_off_de_risk"
        ])) if any(str(e.get("optimizer_fallback_mode", "none")) == "risk_off_de_risk" for e in trace) else None,
        "fallback_cash_delta_hit_rate": float(np.mean([
            1.0 if float((entry.get("optimizer_diagnostics") or {}).get("fallback_cash_delta", 0.0)) >= 0.01 else 0.0
            for entry in trace
            if str(entry.get("optimizer_fallback_mode", "none")) == "risk_off_de_risk"
        ])) if any(str(e.get("optimizer_fallback_mode", "none")) == "risk_off_de_risk" for e in trace) else None,
        "mean_fallback_turnover": float(np.mean([
            float((entry.get("optimizer_diagnostics") or {}).get("fallback_turnover", 0.0))
            for entry in trace
            if str(entry.get("optimizer_fallback_mode", "none")) == "risk_off_de_risk"
        ])) if any(str(e.get("optimizer_fallback_mode", "none")) == "risk_off_de_risk" for e in trace) else None,
        "aggressiveness_usage_rate": usage_rate("aggressiveness", neutral_aggressiveness),
        "posture_usage_rate": float(
            np.mean([1.0 if posture != neutral_posture else 0.0 for posture in postures])
        ),
        "unique_cash_targets": unique_values("cash_target"),
        "unique_turnover_caps": unique_values("turnover_cap"),
        "unique_postures": sorted(set(postures)),
        "posture_counts": posture_counts(postures),
        "target_posture_counts": posture_counts(target_postures),
        "posture_by_stress_bucket": posture_by_bucket,
        "target_posture_by_stress_bucket": target_by_bucket,
        "rebalance_rate": float(
            np.mean([1.0 if entry["should_rebalance"] else 0.0 for entry in trace])
        ),
        "mean_selected_sector_count": float(
            np.mean(
                [
                    entry.get("selected_sector_count", len(entry.get("selected_sectors", [])))
                    for entry in trace
                ]
            )
        ),
        "mean_selected_stock_count": float(
            np.mean([entry["selected_stock_count"] for entry in trace])
        ),
        "policy_change_rate": change_rate,
        "posture_change_rate": posture_change_rate,
        "unique_selected_sector_sets": int(
            len({tuple(sorted(entry["selected_sectors"])) for entry in trace})
        ),
        "mean_reward": float(np.mean([entry["reward"] for entry in trace])),
        "mean_stress_signal": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("stress_signal", 0.0))
                    for entry in trace
                ]
            )
        ),
        "mean_defensive_posture": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("defensive_posture", 0.0))
                    for entry in trace
                ]
            )
        ),
        "mean_target_defensive_posture": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("target_defensive_posture", 0.0))
                    for entry in trace
                ]
            )
        ),
        "mean_posture_progress_bonus": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("posture_progress_bonus", 0.0))
                    for entry in trace
                ]
            )
        ),
        "mean_posture_stale_penalty": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("posture_stale_penalty", 0.0))
                    for entry in trace
                ]
            )
        ),
        "mean_posture_flip_penalty": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("posture_flip_penalty", 0.0))
                    for entry in trace
                ]
            )
        ),
        "mean_posture_distance_to_target": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("posture_distance_to_target", 0.0))
                    for entry in trace
                ]
            )
        ),
        "decision_quality_basis": (
            "cached_one_step_soft_regret_v1"
            if any(
                "soft_regret" in entry.get("reward_components", {})
                for entry in trace
            )
            else "target_posture_proxy"
        ),
        "posture_optimality_rate": float(np.mean(optimal_hits)),
        "mean_regret": float(np.mean(regrets)),
        "regret_by_stress_bucket": regret_by_stress_bucket,
        "posture_optimality_rate_by_stress_bucket": posture_optimality_by_stress_bucket,
        "mean_posture_utility_dispersion": float(np.mean(utility_dispersion)),
        "posture_utility_dispersion_by_stress_bucket": utility_dispersion_by_stress_bucket,
        "stress_posture_correlation": _correlation(
            [
                float(entry.get("reward_components", {}).get("stress_signal", 0.0))
                for entry in trace
            ],
            [
                float(entry.get("reward_components", {}).get("defensive_posture", 0.0))
                for entry in trace
            ],
        ),
        "mean_active_return": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("active_return", 0.0))
                    for entry in trace
                ]
            )
        ),
        "mean_drawdown_penalty": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("drawdown_penalty", 0.0))
                    for entry in trace
                ]
            )
        ),
        "mean_turnover_penalty": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("turnover_penalty", 0.0))
                    for entry in trace
                ]
            )
        ),
        "mean_concentration_penalty": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("concentration_penalty", 0.0))
                    for entry in trace
                ]
            )
        ),
        "mean_liquidity_penalty": float(
            np.mean(
                [
                    float(entry.get("reward_components", {}).get("liquidity_penalty", 0.0))
                    for entry in trace
                ]
            )
        ),
        "control_realization_by_posture": control_by_posture,
        "control_realization_by_stress_bucket": control_by_stress_bucket,
    }


def _posture_behavior_flags(diagnostics: dict[str, Any]) -> dict[str, Any]:
    unique_postures = diagnostics.get("unique_postures") or []
    posture_change_rate = float(diagnostics.get("posture_change_rate") or 0.0)
    warnings: list[str] = []
    if len(unique_postures) <= 1:
        warnings.append("static posture across holdout")
    if posture_change_rate <= 0.0:
        warnings.append("no posture transitions observed")
    return {
        "advisory_only": True,
        "warnings": warnings,
        "unique_posture_count": len(unique_postures),
        "posture_change_rate": posture_change_rate,
    }


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _find_holdout_start_idx(rebalance_dates: list[pd.Timestamp], holdout_start: pd.Timestamp) -> int:
    for idx, ts in enumerate(rebalance_dates[:-1]):
        if ts >= holdout_start:
            return idx
    raise ValueError("No holdout rebalance date found on or after the requested start.")


def _benchmark_series(
    price_matrix: pd.DataFrame,
    cfg: dict,
    holdout_start: pd.Timestamp,
    holdout_end: pd.Timestamp,
) -> pd.Series | None:
    bm_ticker = cfg["backtest"].get("benchmark_ticker", "^NSEI")
    if bm_ticker not in price_matrix.columns:
        return None
    return price_matrix[bm_ticker].loc[holdout_start:holdout_end].dropna()


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
