"""
Independent risk engine.

Monitors portfolio risk in real-time and provides hard-limit actions:
- drawdown breaches → cut equity exposure
- vol spike → raise cash
- concentration breach → force rebalance
- liquidity stress → avoid small caps
- macro stress → defensive tilt
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.data.contracts import PortfolioState

logger = logging.getLogger(__name__)


@dataclass
class RiskSignal:
    date: date
    drawdown: float = 0.0
    realized_vol: float = 0.0
    hhi: float = 0.0
    liquidity_stress: bool = False
    macro_stress: bool = False
    emergency_rebalance: bool = False
    suggested_cash_floor: float = 0.0
    risk_messages: list[str] = field(default_factory=list)


@dataclass
class RiskAction:
    force_rebalance: bool = False
    cap_small_cap: bool = False
    cash_floor: float = 0.0
    max_stock_weight_override: Optional[float] = None
    max_sector_weight_override: Optional[float] = None
    exclude_tickers: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


class RiskEngine:
    """Real-time risk monitor with hard limit enforcement."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._risk_cfg = cfg["risk"]
        self._peak_nav: float = 0.0
        self._nav_history: list[float] = []
        self._date_history: list[date] = []

    def update(self, nav: float, date_val: date) -> None:
        """Update peak NAV tracker with current portfolio value."""
        self._nav_history.append(nav)
        self._date_history.append(date_val)
        if nav > self._peak_nav:
            self._peak_nav = nav

    def current_drawdown(self) -> float:
        if self._peak_nav <= 0 or not self._nav_history:
            return 0.0
        current_nav = self._nav_history[-1]
        return (current_nav - self._peak_nav) / self._peak_nav

    def evaluate(
        self,
        portfolio_state: PortfolioState,
        recent_returns: pd.Series,
        macro_features: pd.Series | None = None,
        volume_matrix: pd.DataFrame | None = None,
    ) -> tuple[RiskSignal, RiskAction]:
        """
        Evaluate all risk dimensions and return signal + action.
        """
        risk_cfg = self._risk_cfg
        signal = RiskSignal(date=portfolio_state.date)
        action = RiskAction()

        # ── 1. Drawdown ───────────────────────────────────────────────────────
        dd = self.current_drawdown()
        signal.drawdown = dd

        if dd <= -risk_cfg["max_drawdown_hard"]:
            signal.emergency_rebalance = True
            action.force_rebalance = True
            action.cash_floor = risk_cfg["stress_cash_floor"]
            action.messages.append(
                f"HARD DRAWDOWN BREACH: {dd:.1%} < -{risk_cfg['max_drawdown_hard']:.1%}"
            )
        elif dd <= -risk_cfg["max_drawdown_soft"]:
            action.cash_floor = max(action.cash_floor, 0.10)
            action.messages.append(
                f"Soft drawdown alert: {dd:.1%} < -{risk_cfg['max_drawdown_soft']:.1%}"
            )

        # ── 2. Realized volatility ────────────────────────────────────────────
        if len(recent_returns) >= 21:
            vol_1m = float(recent_returns.iloc[-21:].std() * np.sqrt(252))
            signal.realized_vol = vol_1m
            vol_thresh = risk_cfg["max_realized_vol_ann"]
            if vol_1m > vol_thresh:
                action.cash_floor = max(action.cash_floor, 0.08)
                action.cap_small_cap = True
                action.messages.append(
                    f"Elevated vol: {vol_1m:.1%} > {vol_thresh:.1%}"
                )

        # ── 3. Concentration ──────────────────────────────────────────────────
        weights = np.array(list(portfolio_state.weights.values()))
        if len(weights) > 0:
            hhi = float(np.sum(weights ** 2))
            signal.hhi = hhi
            if hhi > risk_cfg["max_concentration_hhi"]:
                action.messages.append(
                    f"High concentration HHI={hhi:.3f} > {risk_cfg['max_concentration_hhi']}"
                )

        # ── 4. Liquidity stress ───────────────────────────────────────────────
        if volume_matrix is not None and not volume_matrix.empty:
            tickers = list(portfolio_state.weights.keys())
            avail = [t for t in tickers if t in volume_matrix.columns and t != "CASH"]
            if avail:
                recent_vol = volume_matrix[avail].iloc[-21:]
                avg_vol = recent_vol.mean()
                portfolio_value = portfolio_state.nav
                for ticker in avail:
                    w = portfolio_state.weights.get(ticker, 0)
                    position_value = w * portfolio_value / 1e7  # in crore
                    avg_daily_vol = avg_vol.get(ticker, float("inf"))
                    if avg_daily_vol > 0:
                        days_to_liquidate = position_value / avg_daily_vol
                        if days_to_liquidate > risk_cfg["min_liquidity_days"]:
                            action.exclude_tickers.append(ticker)
                            signal.liquidity_stress = True
                            action.messages.append(
                                f"{ticker}: {days_to_liquidate:.1f} days to liquidate"
                            )

        # ── 5. Macro stress ───────────────────────────────────────────────────
        if macro_features is not None:
            macro_stress = macro_features.get("macro_stress_score", 0.0)
            if not pd.isna(macro_stress) and macro_stress > 0.6:
                signal.macro_stress = True
                action.cash_floor = max(action.cash_floor, risk_cfg["stress_cash_floor"])
                action.cap_small_cap = True
                action.messages.append(
                    f"Macro stress score: {macro_stress:.2f} > 0.6"
                )

        # ── 6. Vol regime ─────────────────────────────────────────────────────
        if macro_features is not None:
            vix = macro_features.get("vix_level", 0)
            if not pd.isna(vix) and vix > 30:
                action.cap_small_cap = True
                action.cash_floor = max(action.cash_floor, 0.12)
                action.messages.append(f"High VIX: {vix:.1f}")

        signal.suggested_cash_floor = action.cash_floor
        if action.messages:
            logger.info("RiskEngine [%s]: %s", portfolio_state.date, " | ".join(action.messages))
            signal.risk_messages = action.messages

        return signal, action

    def check_pre_trade(
        self,
        target_weights: dict[str, float],
        sector_map: dict[str, str],
        cap_map: dict[str, str],
        action: RiskAction,
    ) -> dict[str, float]:
        """
        Apply risk overrides to target_weights before execution.
        Returns cleaned target_weights.
        """
        result = dict(target_weights)

        # Remove excluded tickers
        for t in action.exclude_tickers:
            if t in result and t != "CASH":
                freed = result.pop(t)
                result["CASH"] = result.get("CASH", 0) + freed

        # Cap small-cap exposure
        if action.cap_small_cap:
            small_tickers = [t for t, cap in cap_map.items() if cap == "small"]
            small_total = sum(result.get(t, 0) for t in small_tickers)
            small_cap_max = 0.05
            if small_total > small_cap_max:
                scale = small_cap_max / small_total
                for t in small_tickers:
                    if t in result:
                        freed = result[t] * (1 - scale)
                        result[t] *= scale
                        result["CASH"] = result.get("CASH", 0) + freed

        # Enforce cash floor
        current_cash = result.get("CASH", 0)
        if current_cash < action.cash_floor:
            deficit = action.cash_floor - current_cash
            equity_tickers = [t for t in result if t != "CASH"]
            equity_total = sum(result.get(t, 0) for t in equity_tickers)
            if equity_total > 0:
                for t in equity_tickers:
                    result[t] *= (1 - deficit / equity_total)
            result["CASH"] = action.cash_floor

        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    def regime(self, recent_returns: pd.Series) -> str:
        """Classify current market regime: bull | bear | stressed | neutral."""
        if len(recent_returns) < 21:
            return "neutral"
        vol = recent_returns.iloc[-21:].std() * np.sqrt(252)
        ret_3m = (1 + recent_returns.iloc[-63:]).prod() - 1 if len(recent_returns) >= 63 else 0
        dd = self.current_drawdown()

        if vol > self._risk_cfg["regime_vol_threshold"] or dd < -0.15:
            return "stressed"
        if ret_3m > 0.05 and vol < 0.20:
            return "bull"
        if ret_3m < -0.05:
            return "bear"
        return "neutral"
