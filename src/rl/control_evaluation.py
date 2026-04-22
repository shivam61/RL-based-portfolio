"""Canonical control evaluation for RL-vs-neutral portfolio behavior."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NAMED_STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "2018_q4": ("2018-10-04", "2018-11-01"),
    "2020_covid": ("2020-03-19", "2020-05-14"),
    "2022_rate_shock": ("2022-05-12", "2022-07-07"),
    "2024_late_drawdown": ("2024-10-24", "2024-11-21"),
    "2025_prolonged_drawdown": ("2025-01-16", "2025-09-25"),
    "2026_early_weakness": ("2025-12-18", "2026-03-12"),
}


def evaluate_control_from_artifacts(
    report_dir: str | Path,
    *,
    drawdown_threshold: float = -0.08,
) -> dict[str, Any]:
    """
    Build one canonical RL control-evaluation artifact from existing report files.

    Expected inputs in ``report_dir``:
    - ``metrics.json``
    - ``rl_full_neutral_comparison.json``
    - ``rebalance_log.csv``

    Optional:
    - ``rl_full_backtest_comparison.json``
    - ``rl_holdout_comparison.json``
    """
    report_path = Path(report_dir)
    metrics_payload = _load_json(report_path / "metrics.json")
    neutral_payload = _load_json(report_path / "rl_full_neutral_comparison.json")
    baseline_payload = _load_json_if_exists(report_path / "rl_full_backtest_comparison.json")
    holdout_payload = _load_json_if_exists(report_path / "rl_holdout_comparison.json")

    trained_df = _load_trained_rebalance_log(report_path / "rebalance_log.csv")
    neutral_df = _neutral_trace_to_frame(
        neutral_payload["neutral_policy_trace"],
        initial_nav=float(neutral_payload["neutral_policy"].get("initial_nav", 0.0)),
    )

    aligned_trained, aligned_neutral = _align_frames(trained_df, neutral_df)

    control_eval = {
        "evaluation_generated_at_utc": _isoformat_utc(datetime.now(timezone.utc)),
        "evaluation_type": "rl_control_evaluation",
        "window": dict(neutral_payload.get("window", {})),
        "sources": {
            "report_dir": str(report_path.resolve()),
            "metrics_path": str((report_path / "metrics.json").resolve()),
            "full_neutral_comparison_path": str((report_path / "rl_full_neutral_comparison.json").resolve()),
            "full_baseline_comparison_path": (
                str((report_path / "rl_full_backtest_comparison.json").resolve())
                if baseline_payload is not None
                else None
            ),
            "holdout_comparison_path": (
                str((report_path / "rl_holdout_comparison.json").resolve())
                if holdout_payload is not None
                else None
            ),
            "rebalance_log_path": str((report_path / "rebalance_log.csv").resolve()),
        },
        "reference_modes": {
            "current_rl": _extract_summary_metrics(metrics_payload),
            "neutral_full_stack": _extract_summary_metrics(neutral_payload["neutral_policy"]),
            "optimizer_only": (
                _extract_summary_metrics(baseline_payload["baseline"])
                if baseline_payload is not None
                else None
            ),
        },
        "full_window": {
            "current_rl_vs_neutral": {
                "trained_policy": _extract_summary_metrics(neutral_payload["trained_policy"]),
                "neutral_policy": _extract_summary_metrics(neutral_payload["neutral_policy"]),
                "uplift": _extract_summary_metrics(neutral_payload["uplift"]),
            },
            "current_rl_vs_baseline": (
                {
                    "trained_policy": _extract_summary_metrics(baseline_payload["full_rl"]),
                    "baseline_policy": _extract_summary_metrics(baseline_payload["baseline"]),
                    "uplift": _extract_summary_metrics(baseline_payload["uplift"]),
                }
                if baseline_payload is not None
                else None
            ),
        },
        "holdout": _build_holdout_section(holdout_payload),
        "drawdown_behavior": {
            "threshold": float(drawdown_threshold),
            "current_rl": _behavior_summary(
                aligned_trained[aligned_trained["drawdown_pct"] <= drawdown_threshold]
            ),
            "neutral_full_stack": _behavior_summary(
                aligned_neutral[aligned_trained["drawdown_pct"] <= drawdown_threshold]
            ),
            "delta_rl_minus_neutral": _metric_deltas(
                _behavior_summary(aligned_trained[aligned_trained["drawdown_pct"] <= drawdown_threshold]),
                _behavior_summary(aligned_neutral[aligned_trained["drawdown_pct"] <= drawdown_threshold]),
            ),
        },
        "stress_windows": _evaluate_named_windows(aligned_trained, aligned_neutral),
        "notes": _build_notes(trained_df),
    }
    return control_eval


def _load_trained_rebalance_log(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["date"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["pre_nav"] = pd.to_numeric(frame["pre_nav"], errors="coerce")
    frame["cash_pct"] = pd.to_numeric(frame.get("cash_pct"), errors="coerce")
    frame["aggressiveness"] = pd.to_numeric(frame.get("aggressiveness"), errors="coerce")
    frame["turnover_pct"] = pd.to_numeric(frame.get("turnover_pct"), errors="coerce")
    frame["stock_count"] = pd.to_numeric(frame.get("n_stocks"), errors="coerce")
    frame["sector_count"] = _trained_sector_count(frame)
    frame["next_pre_nav"] = frame["pre_nav"].shift(-1)
    frame["peak_nav"] = frame["pre_nav"].cummax()
    frame["drawdown_pct"] = frame["pre_nav"] / frame["peak_nav"] - 1.0
    return frame


def _neutral_trace_to_frame(trace: list[dict[str, Any]], *, initial_nav: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nav = float(initial_nav)
    for entry in trace:
        period_return = float(
            entry.get("period_return", entry.get("reward_components", {}).get("period_return", 0.0))
        )
        next_nav = nav * (1.0 + period_return)
        rows.append(
            {
                "date": pd.Timestamp(entry["date"]),
                "pre_nav": float(nav),
                "next_pre_nav": float(next_nav),
                "cash_pct": float(entry.get("cash_target", 0.0)) * 100.0,
                "aggressiveness": float(entry.get("aggressiveness", 1.0)),
                "turnover_pct": float(entry.get("turnover", 0.0)) * 100.0,
                "stock_count": float(entry.get("selected_stock_count", 0.0)),
                "sector_count": float(
                    entry.get(
                        "selected_sector_count",
                        len(entry.get("selected_sectors", [])),
                    )
                ),
            }
        )
        nav = next_nav
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["peak_nav"] = frame["pre_nav"].cummax()
    frame["drawdown_pct"] = frame["pre_nav"] / frame["peak_nav"] - 1.0
    return frame


def _align_frames(
    trained: pd.DataFrame,
    neutral: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_dates = sorted(set(trained["date"]) & set(neutral["date"]))
    trained_aligned = trained[trained["date"].isin(common_dates)].copy().reset_index(drop=True)
    neutral_aligned = neutral[neutral["date"].isin(common_dates)].copy().reset_index(drop=True)
    return trained_aligned, neutral_aligned


def _behavior_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"observations": 0}
    return {
        "observations": int(len(frame)),
        "avg_cash_pct": _mean(frame["cash_pct"]),
        "avg_aggressiveness": _mean(frame["aggressiveness"]),
        "avg_turnover_pct": _mean(frame["turnover_pct"]),
        "avg_stock_count": _mean(frame["stock_count"]),
        "avg_sector_count": _mean(frame["sector_count"]),
        "avg_drawdown_pct": _mean(frame["drawdown_pct"] * 100.0),
        "max_drawdown_pct": _min_pct(frame["drawdown_pct"]),
    }


def _evaluate_named_windows(
    trained: pd.DataFrame,
    neutral: pd.DataFrame,
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for name, (start, end) in NAMED_STRESS_WINDOWS.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        trained_window = trained[(trained["date"] >= start_ts) & (trained["date"] <= end_ts)].copy()
        neutral_window = neutral[(neutral["date"] >= start_ts) & (neutral["date"] <= end_ts)].copy()
        windows[name] = {
            "start_date": start,
            "end_date": end,
            "current_rl": {
                **_behavior_summary(trained_window),
                **_window_nav_summary(trained, trained_window),
            },
            "neutral_full_stack": {
                **_behavior_summary(neutral_window),
                **_window_nav_summary(neutral, neutral_window),
            },
            "delta_rl_minus_neutral": _metric_deltas(
                {**_behavior_summary(trained_window), **_window_nav_summary(trained, trained_window)},
                {**_behavior_summary(neutral_window), **_window_nav_summary(neutral, neutral_window)},
            ),
        }
    return windows


def _window_nav_summary(full_df: pd.DataFrame, window_df: pd.DataFrame) -> dict[str, Any]:
    if window_df.empty:
        return {"window_loss_pct": None, "recovery_rebalances": None}
    start_nav = float(window_df["pre_nav"].iloc[0])
    nav_points = pd.concat(
        [window_df["pre_nav"], window_df["next_pre_nav"].dropna()],
        ignore_index=True,
    ).astype(float)
    min_nav = float(nav_points.min()) if not nav_points.empty else start_nav
    loss_pct = ((min_nav / start_nav) - 1.0) * 100.0 if start_nav > 0 else None

    end_idx = int(window_df.index.max())
    future = full_df.loc[end_idx + 1 :, "pre_nav"].astype(float)
    recovery = future[future >= start_nav]
    recovery_rebalances = (
        int(recovery.index[0] - end_idx) if not recovery.empty else None
    )
    return {
        "window_start_nav": start_nav,
        "window_min_nav": min_nav,
        "window_loss_pct": loss_pct,
        "recovery_rebalances": recovery_rebalances,
    }


def _build_holdout_section(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        "window": {
            "train_end_rebalance": payload.get("train_end_rebalance"),
            "holdout_start_rebalance": payload.get("holdout_start_rebalance"),
            "holdout_end_rebalance": payload.get("holdout_end_rebalance"),
            "holdout_windows": payload.get("holdout_windows"),
        },
        "current_rl_vs_neutral": {
            "trained_policy": _extract_summary_metrics(payload.get("trained_policy", {})),
            "neutral_policy": _extract_summary_metrics(payload.get("neutral_policy", {})),
            "uplift": _extract_summary_metrics(payload.get("uplift", {})),
        },
        "drawdown_behavior": {
            "current_rl": _extract_summary_metrics(payload.get("trained_policy_diagnostics", {})),
            "neutral_full_stack": _extract_summary_metrics(payload.get("neutral_policy_diagnostics", {})),
            "delta_rl_minus_neutral": _metric_deltas(
                payload.get("trained_policy_diagnostics", {}),
                payload.get("neutral_policy_diagnostics", {}),
            ),
        },
    }


def _extract_summary_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "avg_turnover",
        "total_return",
        "final_nav",
        "information_ratio",
        "mean_cash_target",
        "mean_aggressiveness",
        "mean_turnover",
        "mean_selected_stock_count",
        "mean_selected_sector_count",
        "mean_reward",
        "mean_stress_signal",
        "mean_defensive_posture",
        "mean_target_defensive_posture",
        "mean_posture_progress_bonus",
        "mean_posture_stale_penalty",
        "mean_posture_flip_penalty",
        "mean_posture_distance_to_target",
        "stress_posture_correlation",
        "cash_usage_rate",
        "turnover_cap_usage_rate",
        "aggressiveness_usage_rate",
        "posture_usage_rate",
        "posture_change_rate",
        "unique_postures",
        "posture_counts",
        "target_posture_counts",
        "posture_by_stress_bucket",
        "target_posture_by_stress_bucket",
        "decision_quality_basis",
        "posture_optimality_rate",
        "posture_optimality_rate_by_stress_bucket",
        "mean_regret",
        "regret_by_stress_bucket",
        "mean_posture_utility_dispersion",
        "posture_utility_dispersion_by_stress_bucket",
        "control_realization_by_posture",
        "control_realization_by_stress_bucket",
        "observations",
        "avg_cash_pct",
        "avg_aggressiveness",
        "avg_turnover_pct",
        "avg_stock_count",
        "avg_sector_count",
        "avg_drawdown_pct",
        "max_drawdown_pct",
        "window_loss_pct",
        "recovery_rebalances",
    ]
    extracted: dict[str, Any] = {}
    for key in keys:
        if key in payload:
            extracted[key] = _jsonable_scalar(payload[key])
    return extracted


def _metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, left_value in left.items():
        right_value = right.get(key)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            out[key] = float(left_value) - float(right_value)
    return out


def _trained_sector_count(frame: pd.DataFrame) -> pd.Series:
    if "selected_sector_count" in frame.columns:
        values = pd.to_numeric(frame["selected_sector_count"], errors="coerce")
        if values.notna().any():
            return values
    tilt_cols = [col for col in frame.columns if col.startswith("tilt_")]
    if not tilt_cols:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[tilt_cols].notna().sum(axis=1).astype(float)


def _build_notes(trained_df: pd.DataFrame) -> list[str]:
    notes = []
    if "selected_sector_count" not in trained_df.columns:
        notes.append(
            "trained current_rl sector count is estimated from visible tilt columns because older rebalance logs did not persist selected_sector_count"
        )
    return notes


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    return _load_json(path) if path.exists() else None


def _mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else None


def _min_pct(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.min() * 100.0) if not clean.empty else None


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
