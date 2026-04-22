"""Shared policy helpers for backtest-aligned allocation decisions."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_sector_state(sector_feats: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build the sector-state payload expected by the RL policy."""
    state: dict[str, dict[str, float]] = {}
    if sector_feats is None or sector_feats.empty:
        return state

    for _, row in sector_feats.iterrows():
        sector = str(row.get("sector", "unknown"))
        state[sector] = {
            "mom_1m": float(row.get("mom_1m", 0.0) or 0.0),
            "mom_3m": float(row.get("mom_3m", 0.0) or 0.0),
            "rel_str_1m": float(row.get("rel_str_1m", 0.0) or 0.0),
            "breadth_3m": float(row.get("breadth_3m", 0.0) or 0.0),
        }
    return state


def build_control_context(
    sector_feats: pd.DataFrame | None,
    *,
    risk_signal: object | None = None,
    risk_action: object | None = None,
    recent_turnovers: Sequence[float] | None = None,
    recent_cost_ratios: Sequence[float] | None = None,
) -> dict[str, float]:
    """Build validated control-state features used by the RL overlay."""
    breadth = 1.0
    if sector_feats is not None and not sector_feats.empty and "breadth_3m" in sector_feats.columns:
        breadth_values = pd.to_numeric(sector_feats["breadth_3m"], errors="coerce").dropna()
        if not breadth_values.empty:
            breadth = float(np.clip(breadth_values.mean(), 0.0, 1.0))

    turnovers = [float(v) for v in (recent_turnovers or []) if np.isfinite(float(v))]
    costs = [float(v) for v in (recent_cost_ratios or []) if np.isfinite(float(v))]

    return {
        "market_breadth_3m": breadth,
        "recent_turnover_3p": float(np.mean(turnovers[-3:])) if turnovers else 0.0,
        "recent_cost_ratio_3p": float(np.mean(costs[-3:])) if costs else 0.0,
        "risk_cash_floor": float(getattr(risk_action, "cash_floor", 0.0) or 0.0),
        "emergency_rebalance": float(bool(getattr(risk_signal, "emergency_rebalance", False))),
    }


def default_decision(sectors: list[str]) -> dict[str, object]:
    """Neutral allocation decision used when RL is disabled or unavailable."""
    return {
        "sector_tilts": {sector: 1.0 for sector in sectors},
        "cash_target": 0.05,
        "aggressiveness": 1.0,
        "turnover_cap": None,
        "should_rebalance": True,
    }


def select_sectors(
    sectors: list[str],
    sector_scores: dict[str, float],
    rl_decision: dict[str, object],
    *,
    full_rl: bool,
) -> list[str]:
    """Mirror backtest sector-selection semantics."""
    ordered = sorted(
        ((sector, float(sector_scores.get(sector, 0.0))) for sector in sectors),
        key=lambda item: (-item[1], item[0]),
    )
    if full_rl:
        return [sector for sector, _ in ordered]

    top_n = min(len(ordered), 5)
    return [sector for sector, _ in ordered[:top_n]]
