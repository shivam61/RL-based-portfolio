"""Portfolio state features used by the RL agent and sector scorer."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data.contracts import PortfolioState


def compute_portfolio_features(
    state: PortfolioState,
    recent_returns: pd.Series,           # recent portfolio daily returns
    benchmark_returns: pd.Series | None = None,
    control_context: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute a fixed-length portfolio state feature vector.
    Used as part of RL environment state and sector model input.
    """
    feats: dict[str, float] = {}

    # ── Capital / cash ────────────────────────────────────────────────────────
    feats["cash_ratio"] = state.cash / state.nav if state.nav > 0 else 0.0
    feats["nav_log"] = np.log(state.nav) if state.nav > 0 else 0.0

    # ── Concentration ─────────────────────────────────────────────────────────
    weights = np.array(list(state.weights.values())) if state.weights else np.array([])
    if len(weights) > 0:
        feats["n_stocks"] = float(len(weights))
        feats["max_weight"] = float(weights.max())
        feats["hhi"] = float(np.sum(weights ** 2))
        feats["effective_n"] = float(1.0 / feats["hhi"]) if feats["hhi"] > 0 else 0.0
    else:
        feats["n_stocks"] = 0.0
        feats["max_weight"] = 0.0
        feats["hhi"] = 1.0
        feats["effective_n"] = 0.0

    # ── Sector concentration ──────────────────────────────────────────────────
    sec_weights = np.array(list(state.sector_weights.values())) if state.sector_weights else np.array([])
    if len(sec_weights) > 0:
        feats["max_sector_weight"] = float(sec_weights.max())
        feats["sector_hhi"] = float(np.sum(sec_weights ** 2))
    else:
        feats["max_sector_weight"] = 0.0
        feats["sector_hhi"] = 1.0

    # ── Recent returns ────────────────────────────────────────────────────────
    if len(recent_returns) >= 5:
        feats["ret_1w"] = float(recent_returns.iloc[-5:].add(1).prod() - 1) if len(recent_returns) >= 5 else 0.0
    else:
        feats["ret_1w"] = 0.0

    if len(recent_returns) >= 21:
        feats["ret_1m"] = float(recent_returns.iloc[-21:].add(1).prod() - 1)
        feats["vol_1m"] = float(recent_returns.iloc[-21:].std() * np.sqrt(252))
    else:
        feats["ret_1m"] = 0.0
        feats["vol_1m"] = 0.0

    if len(recent_returns) >= 63:
        feats["ret_3m"] = float(recent_returns.iloc[-63:].add(1).prod() - 1)
        feats["vol_3m"] = float(recent_returns.iloc[-63:].std() * np.sqrt(252))
        feats["sharpe_3m"] = (
            feats["ret_3m"] / (feats["vol_3m"] / 2)
            if feats["vol_3m"] > 0 else 0.0
        )
    else:
        feats["ret_3m"] = 0.0
        feats["vol_3m"] = 0.0
        feats["sharpe_3m"] = 0.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    if len(recent_returns) >= 10:
        cumulative = (1 + recent_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max.replace(0, np.nan)
        feats["current_drawdown"] = float(drawdown.iloc[-1]) if not drawdown.empty else 0.0
        feats["max_drawdown"] = float(drawdown.min()) if not drawdown.empty else 0.0
    else:
        feats["current_drawdown"] = 0.0
        feats["max_drawdown"] = 0.0

    # ── Active return vs benchmark ────────────────────────────────────────────
    if benchmark_returns is not None and len(benchmark_returns) >= 21 and len(recent_returns) >= 21:
        # deduplicate indices before intersection
        r_dedup = recent_returns[~recent_returns.index.duplicated(keep="last")]
        b_dedup = benchmark_returns[~benchmark_returns.index.duplicated(keep="last")]
        common_idx = r_dedup.index.intersection(b_dedup.index)
        if len(common_idx) >= 5:
            pr = r_dedup.reindex(common_idx)
            br = b_dedup.reindex(common_idx)
            active = pr - br
            feats["active_ret_1m"] = float(active.iloc[-21:].sum())
            feats["tracking_error_1m"] = float(active.iloc[-21:].std() * np.sqrt(252))
        else:
            feats["active_ret_1m"] = 0.0
            feats["tracking_error_1m"] = 0.0
    else:
        feats["active_ret_1m"] = 0.0
        feats["tracking_error_1m"] = 0.0

    # ── Control-state features ───────────────────────────────────────────────
    ctx = control_context or {}
    if len(recent_returns) >= 21:
        recent_window = recent_returns.iloc[-21:]
        cumulative = (1 + recent_window).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max.replace(0, np.nan)
        if len(drawdown) >= 6:
            feats["drawdown_slope_1m"] = float(drawdown.iloc[-1] - drawdown.iloc[-6])
        else:
            feats["drawdown_slope_1m"] = float(drawdown.iloc[-1])
    else:
        feats["drawdown_slope_1m"] = 0.0

    vol_3m = feats.get("vol_3m", 0.0)
    vol_1m = feats.get("vol_1m", 0.0)
    feats["vol_shock_1m_3m"] = (
        float(vol_1m / vol_3m - 1.0)
        if vol_3m and np.isfinite(vol_3m) and abs(vol_3m) > 1e-9
        else 0.0
    )

    breadth_level = float(ctx.get("market_breadth_3m", 1.0) or 0.0)
    breadth_level = float(np.clip(breadth_level, 0.0, 1.0))
    feats["breadth_deterioration"] = float(1.0 - breadth_level)
    feats["recent_turnover_3p"] = float(max(0.0, ctx.get("recent_turnover_3p", 0.0) or 0.0))
    feats["recent_cost_ratio_3p"] = float(max(0.0, ctx.get("recent_cost_ratio_3p", 0.0) or 0.0))
    feats["risk_cash_floor"] = float(max(0.0, ctx.get("risk_cash_floor", 0.0) or 0.0))
    feats["emergency_flag"] = float(1.0 if ctx.get("emergency_rebalance", 0.0) else 0.0)

    return feats
