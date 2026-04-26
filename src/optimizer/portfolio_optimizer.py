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
    0.5 * (||w - w_prev||_1 + liquidation_cost + |cash - cash_prev|) <= effective_max_to
    cash in [min_cash, max_cash]

Turnover accounting (full-portfolio, one-way):
  - w_prev covers only the tickers in the current candidate set.
  - Positions being fully liquidated (in prev portfolio but NOT in candidates)
    contribute a fixed constant `liquidation_cost` to the turnover budget.
  - Cash changes are included via |cash - cash_prev| (modeled as cp.abs).
  - effective_max_to = max(max_to, minimum_feasible_to) so the problem
    is never made infeasible by the turnover constraint alone.
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
        self.last_optimize_diagnostics: dict[str, object] = {}

    def optimize(
        self,
        alpha_scores: dict[str, float],          # ticker → expected return/score
        cov_matrix: pd.DataFrame | None,         # ticker × ticker covariance
        sector_map: dict[str, str],              # ticker → sector
        current_weights: dict[str, float] | None = None,
        sector_tilts: dict[str, float] | None = None,  # RL sector multipliers
        aggressiveness: float = 1.0,
        cash_target: float | None = None,
        max_turnover_override: float | None = None,
        forced_exclude: list[str] | None = None,
        posture: str | None = None,
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

        # Fixed turnover from positions being fully liquidated:
        # tickers in the prev portfolio that are NOT in the current candidate set.
        # Their sale is predetermined — subtract from the remaining budget.
        if current_weights:
            liquidation_cost = sum(
                v for k, v in current_weights.items()
                if k != "CASH" and k not in set(tickers)
            )
        else:
            liquidation_cost = 0.0

        # Constraints config
        max_stock = opt_cfg["max_stock_weight"]
        if posture:
            posture_profiles = self.cfg.get("rl", {}).get("posture_profiles", {})
            posture_profile = posture_profiles.get(str(posture), {})
            if isinstance(posture_profile, dict) and posture_profile.get("max_stock_weight") is not None:
                max_stock = float(posture_profile["max_stock_weight"])
        aggressiveness_effect_scale = float(opt_cfg.get("aggressiveness_effect_scale", 1.5))
        effective_aggressiveness = max(0.25, float(aggressiveness)) ** aggressiveness_effect_scale
        max_sector = opt_cfg["max_sector_weight"] * effective_aggressiveness
        max_sector = min(max_sector, 0.50)
        max_to = (
            float(max_turnover_override)
            if max_turnover_override is not None
            else float(opt_cfg["max_turnover_per_rebalance"])
        )
        risk_aversion = cfg_risk_aversion = self.cfg["optimizer"]["risk_aversion"] / effective_aggressiveness

        cash_min = opt_cfg["min_cash"]
        cash_max = opt_cfg["max_cash"]
        cash_target_tolerance = float(opt_cfg.get("cash_target_tolerance", 0.05))
        if cash_target is not None:
            cash_min = max(cash_min, cash_target - cash_target_tolerance)
            cash_max = min(cash_max, cash_target + cash_target_tolerance)
        target_cash_weight = float(cash_target if cash_target is not None else cash_min)
        reference_weights = self._reference_pre_cap_weights(
            alpha=alpha,
            target_cash_weight=target_cash_weight,
        )

        diagnostics: dict[str, object] = {
            "posture": str(posture or "neutral"),
            "requested_cash_target": float(cash_target if cash_target is not None else cash_min),
            "cash_target_tolerance": float(cash_target_tolerance if cash_target is not None else 0.0),
            "previous_cash_weight": float(w_prev_cash),
            "required_liquidation": float(liquidation_cost),
            "requested_turnover_cap": float(max_to),
            "effective_turnover_budget": float(max_to),
            "effective_turnover_needed": 0.0,
            "turnover_shortfall": 0.0,
            "liquidation_cost": float(liquidation_cost),
            "candidate_stock_count": int(n),
            "candidate_sector_count": int(len(set(sector_map.get(t, "") for t in tickers))),
            "max_stock_weight": float(max_stock),
            "max_sector_weight": float(max_sector),
            "aggressiveness": float(aggressiveness),
            "status": "init",
            "solver_status": "init",
            "fallback_mode": "none",
            "solver_retry_without_turnover": False,
            "turnover_repair_applied": False,
            "relaxation_tier": "A_full",
            "binding_constraints": [],
            "cash_gap": None,
        }
        diagnostics.update(
            self._compression_metrics(
                reference_weights=reference_weights,
                result_weights=None,
                max_stock=float(max_stock),
            )
        )

        if not HAS_CVXPY:
            diagnostics["status"] = "no_cvxpy"
            result = self._fallback_weights(
                alpha=alpha,
                reference_weights=reference_weights,
                alpha_scores=alpha_scores,
                tickers=tickers,
                sector_map=sector_map,
                n=n,
                max_stock=max_stock,
                max_sector=max_sector,
                cash=float(cash_target or cash_min),
                current_weights=current_weights,
                max_turnover=max_to,
                liquidation_cost=liquidation_cost,
                posture=str(posture or "neutral"),
                diagnostics=diagnostics,
            )
            self.last_optimize_diagnostics = diagnostics
            return result

        # ── CVXPY problem ─────────────────────────────────────────────────────
        w = cp.Variable(n, nonneg=True)
        cash = cp.Variable(nonneg=True)

        objective_terms = [
            alpha @ w,                           # maximize alpha
            -risk_aversion * cp.quad_form(w, Sigma),  # risk penalty
            -0.5 * cp.sum_squares(w),            # concentration penalty
        ]
        if cash_target is not None:
            objective_terms.append(
                -float(opt_cfg.get("cash_target_penalty", 0.0)) * cp.abs(cash - float(cash_target))
            )

        # Build full-portfolio turnover expression once; reuse in objective + constraint.
        # effective_max_to: floored so the constraint is never infeasible by itself.
        separate_cash_budget = bool(opt_cfg.get("separate_cash_turnover_budget", False))
        max_cash_delta = float(opt_cfg.get("max_cash_delta_per_rebalance", 0.20))

        if current_weights:
            equity_to = cp.norm1(w - w_prev)
            cash_to   = cp.abs(cash - w_prev_cash)
            one_way_turnover = 0.5 * (equity_to + cash_to + liquidation_cost)
            # When cash budget is separate, floor equity turnover without the cash component.
            if separate_cash_budget:
                cash_move_floor = 0.0
            else:
                cash_move_floor = abs(w_prev_cash - (cash_target or cash_min))
            min_feasible = (
                0.5 * liquidation_cost
                + cash_move_floor
                + 0.02   # small solver headroom
            )
            effective_max_to = max(max_to, min_feasible)
            diagnostics["effective_turnover_budget"] = float(effective_max_to)
            diagnostics["effective_turnover_needed"] = float(min_feasible)
            diagnostics["turnover_shortfall"] = float(max(0.0, min_feasible - max_to))
            diagnostics["separate_cash_budget"] = separate_cash_budget

            objective_terms.append(
                -opt_cfg.get("turnover_cost", 0.3) * one_way_turnover
            )
        else:
            effective_max_to = max_to  # no previous portfolio → no constraint
            diagnostics["effective_turnover_needed"] = float(max_to)

        objective = cp.Maximize(sum(objective_terms))

        solver = opt_cfg.get("solver", "CLARABEL")

        def _solve(constrs):
            p = cp.Problem(objective, constrs)
            p.solve(solver=solver, verbose=False)
            if p.status not in ("optimal", "optimal_inaccurate"):
                raise ValueError(f"Solver status: {p.status}")
            if w.value is None:
                raise ValueError("Solver returned None solution")
            return p.status

        def _build_constraints(
            *,
            turnover_budget: float,
            max_stock_limit: float,
            max_sector_limit: float,
        ):
            constrs = [
                cp.sum(w) + cash == 1.0,
                cash >= cash_min,
                cash <= cash_max,
                w <= max_stock_limit,
            ]
            constrs.extend(
                self._build_sector_constraints(
                    w, tickers, sector_map, max_sector_limit
                )
            )
            if current_weights:
                if separate_cash_budget:
                    # Equity and cash turnover are budgeted independently.
                    # Equity: half the one-way norm of equity weight changes + liquidation.
                    # Cash: absolute cash change capped separately (no ×0.5 — it is already one-way).
                    equity_only_to = 0.5 * (equity_to + liquidation_cost)
                    constrs.append(equity_only_to <= turnover_budget)
                    constrs.append(cash_to <= max_cash_delta)
                else:
                    constrs.append(one_way_turnover <= turnover_budget)
            return constrs

        relaxation_tiers = [
            {
                "label": "A_full",
                "turnover_budget": float(effective_max_to),
                "max_stock_limit": float(max_stock),
                "max_sector_limit": float(max_sector),
                "status": "optimal",
            },
        ]
        if current_weights:
            relaxation_tiers.extend(
                [
                    {
                        "label": "B_relax_turnover",
                        "turnover_budget": float(min(opt_cfg["max_turnover_per_rebalance"], effective_max_to * 1.20)),
                        "max_stock_limit": float(max_stock),
                        "max_sector_limit": float(max_sector),
                        "status": "optimal_relaxed_turnover",
                    },
                    {
                        "label": "C_relax_caps",
                        "turnover_budget": float(effective_max_to),
                        "max_stock_limit": float(min(max_stock * 1.20, 1.0)),
                        "max_sector_limit": float(min(max_sector * 1.20, 1.0)),
                        "status": "optimal_relaxed_caps",
                    },
                    {
                        "label": "D_relax_both",
                        "turnover_budget": float(min(opt_cfg["max_turnover_per_rebalance"], effective_max_to * 1.20)),
                        "max_stock_limit": float(min(max_stock * 1.20, 1.0)),
                        "max_sector_limit": float(min(max_sector * 1.20, 1.0)),
                        "status": "optimal_relaxed_turnover_and_caps",
                    },
                ]
            )

        last_error: Exception | None = None
        for tier in relaxation_tiers:
            try:
                constraints = _build_constraints(
                    turnover_budget=float(tier["turnover_budget"]),
                    max_stock_limit=float(tier["max_stock_limit"]),
                    max_sector_limit=float(tier["max_sector_limit"]),
                )
                raw_status = _solve(constraints)
                diagnostics["status"] = str(tier["status"])
                diagnostics["solver_status"] = str(raw_status)
                diagnostics["relaxation_tier"] = str(tier["label"])
                diagnostics["effective_turnover_budget"] = float(tier["turnover_budget"])
                diagnostics["max_stock_weight"] = float(tier["max_stock_limit"])
                diagnostics["max_sector_weight"] = float(tier["max_sector_limit"])
                diagnostics["solver_retry_without_turnover"] = str(tier["label"]) in {"B_relax_turnover", "D_relax_both"}
                break
            except Exception as e:
                last_error = e
                diagnostics.setdefault("tier_failures", []).append(
                    {"tier": str(tier["label"]), "error": str(e)}
                )
        else:
            logger.warning("CVXPY solver failed across relaxation tiers (%s); using rank fallback", last_error)
            diagnostics["status"] = f"solver_failed:{type(last_error).__name__ if last_error else 'Unknown'}"
            diagnostics["solver_status"] = "fallback"
            diagnostics["relaxation_tier"] = "fallback"
            result = self._fallback_weights(
                alpha=alpha,
                reference_weights=reference_weights,
                alpha_scores=alpha_scores,
                tickers=tickers,
                sector_map=sector_map,
                n=n,
                max_stock=max_stock,
                max_sector=max_sector,
                cash=float(cash_target or cash_min),
                current_weights=current_weights,
                max_turnover=effective_max_to if current_weights else max_to,
                liquidation_cost=liquidation_cost,
                posture=str(posture or "neutral"),
                diagnostics=diagnostics,
            )
            self.last_optimize_diagnostics = diagnostics
            return result

        w_val = np.array(w.value).clip(0)
        cash_val = float(cash.value) if cash.value is not None else cash_min

        # Verify the turnover constraint was actually satisfied (guards against
        # optimal_inaccurate solutions that violate it meaningfully).
        if current_weights:
            actual_total_to = self._compute_one_way_turnover(
                w_val, w_prev, cash_val, w_prev_cash, liquidation_cost
            )
            if actual_total_to > effective_max_to * 1.10:   # 10% tolerance
                repaired = self._repair_turnover_violation(
                    w_val=w_val,
                    cash_val=cash_val,
                    w_prev=w_prev,
                    w_prev_cash=w_prev_cash,
                    liquidation_cost=liquidation_cost,
                    max_turnover=effective_max_to,
                    max_stock=max_stock,
                    tickers=tickers,
                    sector_map=sector_map,
                    max_sector=max_sector,
                )
                if repaired is not None:
                    w_val, cash_val = repaired
                    diagnostics["turnover_repair_applied"] = True
                    actual_total_to = self._compute_one_way_turnover(
                        w_val, w_prev, cash_val, w_prev_cash, liquidation_cost
                    )
                logger.warning(
                    "Turnover constraint violated after solve: actual=%.2f budget=%.2f",
                    actual_total_to, effective_max_to,
                )
            diagnostics["actual_turnover_after_repair"] = float(actual_total_to)

        # apply no-trade band
        band = opt_cfg.get("no_trade_band", 0.005)
        for i, t in enumerate(tickers):
            if abs(w_val[i] - w_prev[i]) < band:
                w_val[i] = w_prev[i]

        # Preserve the solver's cash choice through post-processing. If snapping
        # small equity trades back to previous weights increases equity exposure,
        # scale equities down and leave the excess in cash rather than
        # renormalizing cash away.
        equity_total = float(w_val.sum())
        max_equity_budget = max(0.0, 1.0 - float(cash_val))
        if equity_total > max_equity_budget and equity_total > 0.0:
            w_val *= max_equity_budget / equity_total
        cash_val = max(float(cash_val), 1.0 - float(w_val.sum()))

        result = {t: float(w) for t, w in zip(tickers, w_val) if w > 1e-5}
        result["CASH"] = float(max(cash_val, 0))
        diagnostics["realized_cash_weight"] = float(result["CASH"])
        diagnostics["cash_gap"] = float(result["CASH"] - diagnostics["requested_cash_target"])
        diagnostics.update(
            self._compression_metrics(
                reference_weights=reference_weights,
                result_weights=result,
                max_stock=float(diagnostics["max_stock_weight"]),
            )
        )
        diagnostics["binding_constraints"] = self._binding_constraints(
            result=result,
            tickers=tickers,
            sector_map=sector_map,
            max_stock=float(diagnostics["max_stock_weight"]),
            max_sector=float(diagnostics["max_sector_weight"]),
            current_weights=current_weights,
            turnover_budget=float(diagnostics["effective_turnover_budget"]),
            liquidation_cost=float(liquidation_cost),
            requested_cash_target=float(diagnostics["requested_cash_target"]),
            cash_tolerance=float(diagnostics.get("cash_target_tolerance", 0.0)),
        )
        diagnostics["used_turnover_retry"] = bool(diagnostics.get("solver_retry_without_turnover", False))
        diagnostics["status"] = str(diagnostics.get("status") or "optimal")
        self.last_optimize_diagnostics = diagnostics
        return result

    def _fallback_weights(
        self,
        *,
        alpha: np.ndarray,
        reference_weights: dict[int, float],
        alpha_scores: dict[str, float],
        tickers: list[str],
        sector_map: dict[str, str],
        n: int,
        max_stock: float,
        max_sector: float,
        cash: float,
        current_weights: dict[str, float] | None,
        max_turnover: float | None,
        liquidation_cost: float,
        posture: str,
        diagnostics: dict[str, object],
    ) -> dict[str, float]:
        if posture == "risk_off" and current_weights:
            result = self._risk_off_de_risk_fallback(
                alpha_scores=alpha_scores,
                current_weights=current_weights,
                cash_target=float(cash),
                max_turnover=float(max_turnover if max_turnover is not None else self._opt_cfg["max_turnover_per_rebalance"]),
                liquidation_cost=float(liquidation_cost),
            )
            diagnostics["fallback_mode"] = "risk_off_de_risk"
        else:
            result = self._rank_based_weights(
                alpha,
                tickers,
                sector_map,
                n,
                max_stock,
                max_sector,
                cash,
                current_weights=current_weights,
                max_turnover=max_turnover,
                liquidation_cost=liquidation_cost,
            )
            diagnostics["fallback_mode"] = "rank"
        diagnostics["realized_cash_weight"] = float(result.get("CASH", 0.0))
        diagnostics["cash_gap"] = float(result.get("CASH", 0.0) - diagnostics["requested_cash_target"])
        diagnostics.update(
            self._compression_metrics(
                reference_weights=reference_weights,
                result_weights=result,
                max_stock=float(diagnostics["max_stock_weight"]),
            )
        )
        diagnostics["binding_constraints"] = ["fallback"]
        diagnostics["used_turnover_retry"] = bool(diagnostics.get("solver_retry_without_turnover", False))
        diagnostics["status"] = "fallback"
        diagnostics["solver_status"] = "fallback"
        return result

    @staticmethod
    def _reference_pre_cap_weights(
        *,
        alpha: np.ndarray,
        target_cash_weight: float,
    ) -> dict[int, float]:
        investable = max(0.0, 1.0 - float(target_cash_weight))
        if investable <= 0.0 or len(alpha) == 0:
            return {}
        positives = np.clip(alpha.astype(float), 0.0, None)
        if positives.sum() <= 1e-12:
            positives = np.ones(len(alpha), dtype=float)
        weights = investable * positives / positives.sum()
        return {idx: float(weight) for idx, weight in enumerate(weights) if weight > 1e-10}

    @staticmethod
    def _compression_metrics(
        *,
        reference_weights: dict[int, float],
        result_weights: dict[str, float] | None,
        max_stock: float,
    ) -> dict[str, float]:
        before = np.array(list(reference_weights.values()), dtype=float)
        total_mass = float(before.sum()) if before.size else 0.0
        clipped_mass = float(np.maximum(before - float(max_stock), 0.0).sum()) if before.size else 0.0
        after_values = []
        top_weights = []
        if result_weights:
            after_values = [
                float(weight)
                for ticker, weight in result_weights.items()
                if ticker != "CASH" and float(weight) > 0.0
            ]
            top_weights = sorted(after_values, reverse=True)
        after = np.array(after_values, dtype=float) if after_values else np.array([], dtype=float)
        return {
            "avg_weight_before_cap": float(before.mean()) if before.size else 0.0,
            "avg_weight_after_cap": float(after.mean()) if after.size else 0.0,
            "cap_clipping_ratio": float(clipped_mass / total_mass) if total_mass > 1e-12 else 0.0,
            "top5_weight_sum": float(sum(top_weights[:5])) if top_weights else 0.0,
            "top10_weight_sum": float(sum(top_weights[:10])) if top_weights else 0.0,
            "realized_hhi": float(sum(weight * weight for weight in top_weights)) if top_weights else 0.0,
        }

    def _binding_constraints(
        self,
        *,
        result: dict[str, float],
        tickers: list[str],
        sector_map: dict[str, str],
        max_stock: float,
        max_sector: float,
        current_weights: dict[str, float] | None,
        turnover_budget: float,
        liquidation_cost: float,
        requested_cash_target: float,
        cash_tolerance: float,
    ) -> list[str]:
        bindings: list[str] = []
        tol = 1e-3
        cash_weight = float(result.get("CASH", 0.0))
        if cash_tolerance > 0.0 and abs(cash_weight - requested_cash_target) >= max(cash_tolerance - tol, tol):
            bindings.append("cash_target")

        stock_weights = [float(result.get(t, 0.0)) for t in tickers]
        if stock_weights and max(stock_weights) >= max_stock - tol:
            bindings.append("position_cap")

        sector_totals: dict[str, float] = {}
        for ticker in tickers:
            sector = sector_map.get(ticker, "")
            sector_totals[sector] = sector_totals.get(sector, 0.0) + float(result.get(ticker, 0.0))
        if sector_totals and max(sector_totals.values()) >= max_sector - tol:
            bindings.append("sector_cap")

        if current_weights:
            w_prev = np.array([float((current_weights or {}).get(t, 0.0)) for t in tickers], dtype=float)
            w_new = np.array([float(result.get(t, 0.0)) for t in tickers], dtype=float)
            turnover = self._compute_one_way_turnover(
                w_new,
                w_prev,
                cash_weight,
                float((current_weights or {}).get("CASH", 0.0)),
                float(liquidation_cost),
            )
            if turnover >= turnover_budget - tol:
                bindings.append("turnover_cap")
        return bindings

    @staticmethod
    def _compute_one_way_turnover(
        w_val: np.ndarray,
        w_prev: np.ndarray,
        cash_val: float,
        w_prev_cash: float,
        liquidation_cost: float,
    ) -> float:
        equity_to = float(np.sum(np.abs(w_val - w_prev)))
        cash_to = abs(float(cash_val) - float(w_prev_cash))
        return 0.5 * (equity_to + cash_to + float(liquidation_cost))

    def _repair_turnover_violation(
        self,
        w_val: np.ndarray,
        cash_val: float,
        w_prev: np.ndarray,
        w_prev_cash: float,
        liquidation_cost: float,
        max_turnover: float,
        max_stock: float,
        tickers: list[str],
        sector_map: dict[str, str],
        max_sector: float,
    ) -> tuple[np.ndarray, float] | None:
        """
        Blend toward previous weights until one-way turnover fits the budget.
        """
        actual = self._compute_one_way_turnover(
            w_val, w_prev, cash_val, w_prev_cash, liquidation_cost
        )
        if actual <= max_turnover:
            return w_val, cash_val

        baseline = 0.5 * float(liquidation_cost)
        if actual <= baseline + 1e-9 or max_turnover <= baseline:
            return None

        blend = (max_turnover - baseline) / (actual - baseline)
        blend = float(np.clip(blend, 0.0, 1.0))

        repaired_w = w_prev + blend * (w_val - w_prev)
        repaired_cash = w_prev_cash + blend * (cash_val - w_prev_cash)

        repaired_w = np.clip(repaired_w, 0.0, max_stock)
        repaired_cash = max(float(repaired_cash), 0.0)

        for sec in set(sector_map.values()):
            idx = np.array([i for i, t in enumerate(tickers) if sector_map.get(t) == sec], dtype=int)
            if len(idx) == 0:
                continue
            sec_total = repaired_w[idx].sum()
            if sec_total > max_sector and sec_total > 0:
                repaired_w[idx] *= max_sector / sec_total

        total = repaired_w.sum() + repaired_cash
        if total <= 0:
            return None

        repaired_w /= total
        repaired_cash /= total
        return repaired_w, repaired_cash

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
        *,
        current_weights: dict[str, float] | None = None,
        max_turnover: float | None = None,
        liquidation_cost: float = 0.0,
    ) -> dict[str, float]:
        """Simple rank-based weight assignment as fallback.

        This path is intentionally conservative: it keeps leftover capital in
        cash rather than scaling holdings back up and violating hard caps.
        """
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

        equity_budget = 1.0 - cash
        total = w.sum()
        if total > equity_budget and total > 0:
            w = w * equity_budget / total

        # Preserve hard caps in fallback mode; leave any residual uninvested.
        realized_cash = max(cash, 1.0 - float(w.sum()))

        if current_weights and max_turnover is not None:
            w_prev = np.array([(current_weights or {}).get(t, 0.0) for t in tickers], dtype=float)
            w_prev_cash = float((current_weights or {}).get("CASH", 0.0))
            repaired = self._repair_turnover_violation(
                w_val=w,
                cash_val=float(realized_cash),
                w_prev=w_prev,
                w_prev_cash=w_prev_cash,
                liquidation_cost=float(liquidation_cost),
                max_turnover=float(max_turnover),
                max_stock=max_stock,
                tickers=tickers,
                sector_map=sector_map,
                max_sector=max_sector,
            )
            if repaired is not None:
                w, realized_cash = repaired

        result = {t: float(v) for t, v in zip(tickers, w) if v > 1e-5}
        result["CASH"] = float(realized_cash)
        return result

    @staticmethod
    def _risk_off_de_risk_fallback(
        *,
        alpha_scores: dict[str, float],
        current_weights: dict[str, float],
        cash_target: float,
        max_turnover: float,
        liquidation_cost: float,
    ) -> dict[str, float]:
        current_cash = float(current_weights.get("CASH", 0.0))
        result = {
            ticker: float(weight)
            for ticker, weight in current_weights.items()
            if ticker != "CASH" and float(weight) > 1.0e-8
        }
        if not result:
            return {"CASH": 1.0}

        baseline = 0.5 * max(0.0, float(liquidation_cost))
        available_turnover = max(0.0, float(max_turnover) - baseline)
        achievable_cash = min(float(cash_target), current_cash + available_turnover)
        to_raise = max(0.0, achievable_cash - current_cash)
        if to_raise <= 1.0e-8:
            out = dict(result)
            out["CASH"] = current_cash
            return out

        ordered = sorted(
            result,
            key=lambda ticker: (float(alpha_scores.get(ticker, 0.0)), float(result.get(ticker, 0.0))),
        )
        remaining = to_raise
        for ticker in ordered:
            if remaining <= 1.0e-8:
                break
            sell_weight = min(float(result.get(ticker, 0.0)), remaining)
            result[ticker] = float(result.get(ticker, 0.0)) - sell_weight
            remaining -= sell_weight
            if result[ticker] <= 1.0e-8:
                result.pop(ticker, None)

        realized_cash = 1.0 - sum(result.values())
        out = {ticker: float(weight) for ticker, weight in result.items() if weight > 1.0e-8}
        out["CASH"] = float(max(realized_cash, current_cash))
        return out

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
