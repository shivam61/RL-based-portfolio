"""
Constrained long-only portfolio optimizer.

Objective:
    maximize  α'w  -  λ_risk * w'Σw  -  λ_to * ||w - w_prev||_1
              -  λ_conc * sum(w_i^2)

Subject to:
    sum(w) + cash = 1
    w >= 0
    w_i <= max_stock_weight
    sector_w_j <= max_sector_weight
    ||w - w_prev||_1 <= max_turnover
    cash in [min_cash, max_cash]
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    logger.warning("CVXPY not available; optimizer will use simple rank-based weights")


class PortfolioOptimizer:
    """CVXPY-based constrained long-only optimizer."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._opt_cfg = cfg["optimizer"]

    def optimize(
        self,
        alpha_scores: dict[str, float],          # ticker → expected return/score
        cov_matrix: pd.DataFrame | None,         # ticker × ticker covariance
        sector_map: dict[str, str],              # ticker → sector
        current_weights: dict[str, float] | None = None,
        sector_tilts: dict[str, float] | None = None,  # RL sector multipliers
        aggressiveness: float = 1.0,
        cash_target: float | None = None,
        forced_exclude: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Returns portfolio weights (including 'CASH' key).

        Parameters
        ----------
        alpha_scores : ticker → score (higher = more attractive)
        cov_matrix   : estimated return covariance (None → use identity)
        sector_map   : ticker → sector
        current_weights : previous portfolio weights (for turnover penalty)
        sector_tilts : RL-provided multipliers per sector [0.5, 2.0]
        aggressiveness : scales overall risk tolerance [0.5, 1.5]
        cash_target  : desired cash fraction (overrides optimizer if set)
        forced_exclude : tickers to exclude from portfolio
        """
        opt_cfg = self._opt_cfg
        tickers = [t for t in alpha_scores if t in sector_map]
        if forced_exclude:
            tickers = [t for t in tickers if t not in forced_exclude]

        if not tickers:
            return {"CASH": 1.0}

        n = len(tickers)
        alpha = np.array([alpha_scores.get(t, 0.0) for t in tickers])

        # Apply sector tilts to alpha scores
        if sector_tilts:
            tilt_arr = np.array([
                sector_tilts.get(sector_map.get(t, ""), 1.0) for t in tickers
            ])
            alpha = alpha * tilt_arr

        # Normalize alpha to [0, 1]
        alpha_min, alpha_max = alpha.min(), alpha.max()
        if alpha_max > alpha_min:
            alpha = (alpha - alpha_min) / (alpha_max - alpha_min)

        # Covariance matrix
        if cov_matrix is not None and not cov_matrix.empty:
            avail = [t for t in tickers if t in cov_matrix.index]
            if len(avail) == n:
                Sigma = cov_matrix.loc[tickers, tickers].values
            else:
                Sigma = np.eye(n) * 0.04  # default 20% vol proxy
        else:
            Sigma = np.eye(n) * 0.04

        # Previous weights (for turnover)
        w_prev = np.array([
            (current_weights or {}).get(t, 0.0) for t in tickers
        ])
        w_prev_cash = (current_weights or {}).get("CASH", 0.0)

        # Constraints config
        max_stock = opt_cfg["max_stock_weight"]
        max_sector = opt_cfg["max_sector_weight"] * aggressiveness
        max_sector = min(max_sector, 0.50)
        max_to = opt_cfg["max_turnover_per_rebalance"]
        risk_aversion = cfg_risk_aversion = self.cfg["optimizer"]["risk_aversion"] / aggressiveness

        cash_min = opt_cfg["min_cash"]
        cash_max = opt_cfg["max_cash"]
        if cash_target is not None:
            cash_min = max(cash_min, cash_target - 0.05)
            cash_max = min(cash_max, cash_target + 0.05)

        if not HAS_CVXPY:
            return self._rank_based_weights(
                alpha, tickers, sector_map, n,
                max_stock, max_sector, cash_target or cash_min
            )

        # ── CVXPY problem ─────────────────────────────────────────────────────
        w = cp.Variable(n, nonneg=True)
        cash = cp.Variable(nonneg=True)

        objective_terms = [
            alpha @ w,                           # maximize alpha
            -risk_aversion * cp.quad_form(w, Sigma),  # risk penalty
            -0.5 * cp.sum_squares(w),            # concentration penalty
        ]

        # turnover penalty
        if current_weights:
            turnover = cp.norm1(w - w_prev)
            objective_terms.append(-opt_cfg.get("turnover_cost", 0.3) * turnover)

        objective = cp.Maximize(sum(objective_terms))

        constraints = [
            cp.sum(w) + cash == 1.0,
            cash >= cash_min,
            cash <= cash_max,
            w <= max_stock,
        ]

        # sector constraints
        sector_constraints = self._build_sector_constraints(
            w, tickers, sector_map, max_sector
        )
        constraints.extend(sector_constraints)

        # turnover constraint
        if current_weights:
            constraints.append(cp.norm1(w - w_prev) <= max_to)

        prob = cp.Problem(objective, constraints)
        try:
            solver = opt_cfg.get("solver", "CLARABEL")
            prob.solve(solver=solver, verbose=False)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                raise ValueError(f"Solver status: {prob.status}")
        except Exception as e:
            logger.warning("CVXPY solver failed (%s); using rank fallback", e)
            return self._rank_based_weights(
                alpha, tickers, sector_map, n,
                max_stock, max_sector, cash_target or cash_min
            )

        w_val = np.array(w.value).clip(0)
        cash_val = float(cash.value) if cash.value is not None else cash_min

        # apply no-trade band
        band = opt_cfg.get("no_trade_band", 0.005)
        for i, t in enumerate(tickers):
            if abs(w_val[i] - w_prev[i]) < band:
                w_val[i] = w_prev[i]

        # re-normalize
        total = w_val.sum() + cash_val
        if total > 0:
            w_val /= total
            cash_val /= total

        result = {t: float(w) for t, w in zip(tickers, w_val) if w > 1e-5}
        result["CASH"] = float(max(cash_val, 0))
        return result

    def _build_sector_constraints(
        self,
        w,
        tickers: list[str],
        sector_map: dict[str, str],
        max_sector: float,
    ):
        import cvxpy as cp
        constraints = []
        sectors = set(sector_map.values())
        for sec in sectors:
            indices = [i for i, t in enumerate(tickers) if sector_map.get(t) == sec]
            if not indices:
                continue
            sec_weight = sum(w[i] for i in indices)
            constraints.append(sec_weight <= max_sector)
        return constraints

    def _rank_based_weights(
        self,
        alpha: np.ndarray,
        tickers: list[str],
        sector_map: dict[str, str],
        n: int,
        max_stock: float,
        max_sector: float,
        cash: float,
    ) -> dict[str, float]:
        """Simple rank-based weight assignment as fallback."""
        ranks = alpha.argsort()[::-1]
        top_k = max(5, int(n * 0.4))  # top 40% or at least 5
        selected = ranks[:top_k]

        w = np.zeros(n)
        w[selected] = 1.0 / top_k

        # clip stock weights
        w = np.minimum(w, max_stock)

        # clip sector weights iteratively
        sectors = set(sector_map.values())
        for sec in sectors:
            idx = [i for i, t in enumerate(tickers) if sector_map.get(t) == sec]
            sec_total = w[idx].sum()
            if sec_total > max_sector:
                scale = max_sector / sec_total
                w[np.array(idx)] *= scale

        # scale to leave room for cash
        equity_budget = 1.0 - cash
        total = w.sum()
        if total > 0:
            w = w * equity_budget / total

        result = {t: float(v) for t, v in zip(tickers, w) if v > 1e-5}
        result["CASH"] = float(cash)
        return result

    @staticmethod
    def estimate_covariance(
        price_matrix: pd.DataFrame,
        tickers: list[str],
        window: int = 252,
        shrinkage: float = 0.1,
    ) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage covariance estimate."""
        avail = [t for t in tickers if t in price_matrix.columns]
        if not avail:
            return pd.DataFrame()

        rets = price_matrix[avail].pct_change().iloc[-window:].dropna()
        if len(rets) < 20:
            return pd.DataFrame(np.eye(len(avail)) * 0.04, index=avail, columns=avail)

        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(rets.values)
            cov_arr = lw.covariance_ * 252  # annualize
        except Exception:
            cov_arr = rets.cov().values * 252

        cov_df = pd.DataFrame(cov_arr, index=avail, columns=avail)
        return cov_df
