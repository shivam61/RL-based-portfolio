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
    posture_context: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build validated control-state features used by the RL overlay."""
    breadth = 1.0
    if sector_feats is not None and not sector_feats.empty and "breadth_3m" in sector_feats.columns:
        breadth_values = pd.to_numeric(sector_feats["breadth_3m"], errors="coerce").dropna()
        if not breadth_values.empty:
            breadth = float(np.clip(breadth_values.mean(), 0.0, 1.0))

    turnovers = [float(v) for v in (recent_turnovers or []) if np.isfinite(float(v))]
    costs = [float(v) for v in (recent_cost_ratios or []) if np.isfinite(float(v))]

    context = {
        "market_breadth_3m": breadth,
        "recent_turnover_3p": float(np.mean(turnovers[-3:])) if turnovers else 0.0,
        "recent_cost_ratio_3p": float(np.mean(costs[-3:])) if costs else 0.0,
        "risk_cash_floor": float(getattr(risk_action, "cash_floor", 0.0) or 0.0),
        "emergency_rebalance": float(bool(getattr(risk_signal, "emergency_rebalance", False))),
    }
    if posture_context:
        for key, value in posture_context.items():
            if value is None:
                continue
            context[key] = float(value)
    return context


def default_decision(sectors: list[str]) -> dict[str, object]:
    """Neutral allocation decision used when RL is disabled or unavailable."""
    return {
        "sector_tilts": {sector: 1.0 for sector in sectors},
        "posture": "neutral",
        "cash_target": 0.05,
        "aggressiveness": 1.0,
        "turnover_cap": 0.40,
        "should_rebalance": True,
    }


def apply_posture_policy(
    cfg: dict | None,
    decision: dict[str, object],
) -> dict[str, object]:
    """Apply production posture policy while preserving sector tilts."""
    normalized = dict(decision)
    if not bool(normalized.get("allow_forced_posture_override", True)):
        normalized.pop("allow_forced_posture_override", None)
        return normalized

    rl_cfg = (cfg or {}).get("rl", {}) if isinstance(cfg, dict) else {}
    if not bool(rl_cfg.get("force_neutral_posture", False)):
        normalized.pop("allow_forced_posture_override", None)
        return normalized

    posture_profiles = rl_cfg.get("posture_profiles", {}) if isinstance(rl_cfg, dict) else {}
    neutral_profile = (
        posture_profiles.get("neutral", {}) if isinstance(posture_profiles, dict) else {}
    )
    if not isinstance(neutral_profile, dict):
        neutral_profile = {}
    normalized["posture"] = "neutral"
    normalized["cash_target"] = float(neutral_profile.get("cash_target", 0.05))
    normalized["aggressiveness"] = float(neutral_profile.get("aggressiveness", 1.0))
    normalized["turnover_cap"] = float(neutral_profile.get("turnover_cap", 0.35))
    normalized.pop("allow_forced_posture_override", None)
    return normalized


def posture_selection_profile(
    cfg: dict | None,
    posture: str,
) -> dict[str, int | None]:
    rl_cfg = (cfg or {}).get("rl", {}) if isinstance(cfg, dict) else {}
    stock_cfg = (cfg or {}).get("stock_model", {}) if isinstance(cfg, dict) else {}
    posture_profiles = rl_cfg.get("posture_profiles", {}) if isinstance(rl_cfg, dict) else {}
    profile = posture_profiles.get(str(posture), {}) if isinstance(posture_profiles, dict) else {}
    sector_top_n = profile.get("sector_top_n")
    stock_top_k = profile.get("stock_top_k_per_sector", stock_cfg.get("top_k_per_sector", 5))
    return {
        "sector_top_n": int(sector_top_n) if sector_top_n is not None else None,
        "stock_top_k_per_sector": int(stock_top_k) if stock_top_k is not None else None,
    }


def select_sectors(
    sectors: list[str],
    sector_scores: dict[str, float],
    rl_decision: dict[str, object],
    *,
    full_rl: bool,
    cfg: dict | None = None,
) -> list[str]:
    """Mirror backtest sector-selection semantics."""
    ordered = sorted(
        ((sector, float(sector_scores.get(sector, 0.0))) for sector in sectors),
        key=lambda item: (-item[1], item[0]),
    )
    if full_rl:
        posture = str(rl_decision.get("posture", "neutral"))
        sector_top_n = posture_selection_profile(cfg, posture).get("sector_top_n")
        if sector_top_n is None:
            return [sector for sector, _ in ordered]
        top_n = min(len(ordered), int(sector_top_n))
        return [sector for sector, _ in ordered[:top_n]]

    top_n = min(len(ordered), 5)
    return [sector for sector, _ in ordered[:top_n]]
