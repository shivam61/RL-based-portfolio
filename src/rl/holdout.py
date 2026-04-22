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


def _summarize_trace(trace: list[dict[str, Any]], cfg: dict | None = None) -> dict[str, Any]:
    if not trace:
        return {}
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
        "aggressiveness_usage_rate": usage_rate("aggressiveness", neutral_aggressiveness),
        "unique_cash_targets": unique_values("cash_target"),
        "unique_turnover_caps": unique_values("turnover_cap"),
        "rebalance_rate": float(
            np.mean([1.0 if entry["should_rebalance"] else 0.0 for entry in trace])
        ),
        "mean_selected_sector_count": float(
            np.mean([len(entry["selected_sectors"]) for entry in trace])
        ),
        "mean_selected_stock_count": float(
            np.mean([entry["selected_stock_count"] for entry in trace])
        ),
        "policy_change_rate": change_rate,
        "unique_selected_sector_sets": int(
            len({tuple(sorted(entry["selected_sectors"])) for entry in trace})
        ),
        "mean_reward": float(np.mean([entry["reward"] for entry in trace])),
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
    }


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
