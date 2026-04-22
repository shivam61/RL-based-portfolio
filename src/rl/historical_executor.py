"""
Historical one-period executor for causal RL experiments.

This module extracts the rebalance-and-hold path from the walk-forward engine
into a reusable surface that can be shared by tests and simulator-backed RL
environments. It currently executes a single rebalance window faithfully; it is
not yet a full multi-period trainer on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.simulator import ExecutionResult
from src.data.contracts import PortfolioState
from src.features.portfolio_features import compute_portfolio_features
from src.optimizer.portfolio_optimizer import PortfolioOptimizer
from src.rl.contract import build_state, build_transition
from src.rl.policy_utils import (
    build_control_context,
    build_sector_state,
    default_decision,
    select_sectors,
)
from src.risk.risk_engine import RiskAction, RiskEngine, RiskSignal


@dataclass
class PreparedHistoricalStep:
    idx: int
    current_date: pd.Timestamp
    next_date: pd.Timestamp
    snapshot: Any
    prices_today: pd.Series
    sector_map: dict[str, str]
    cap_map: dict[str, str]
    macro_now: dict[str, Any]
    sector_feats_now: pd.DataFrame
    stock_feats_now: pd.DataFrame
    sector_scores: dict[str, float]
    recent_returns: pd.Series
    risk_regime: str
    risk_signal: RiskSignal
    risk_action: RiskAction
    transition_state: dict[str, Any]


@dataclass
class HistoricalStepResult:
    current_date: pd.Timestamp
    next_date: pd.Timestamp
    pre_nav: float
    end_nav: float
    reward: float
    cash_target: float
    selected_sectors: list[str]
    alpha_scores: dict[str, float]
    selected_stock_rows: list[dict[str, object]]
    pre_risk_target_weights: dict[str, float]
    target_weights: dict[str, float]
    exec_result: ExecutionResult
    post_trade_portfolio: PortfolioState
    next_portfolio: PortfolioState
    period_nav: list[tuple[pd.Timestamp, float]]
    updated_nav_points: list[tuple[pd.Timestamp, float]]
    transition: dict[str, Any]


class HistoricalPeriodExecutor:
    """Reusable one-period historical executor built on top of WalkForwardEngine."""

    def __init__(
        self,
        engine: Any,
        mode: str | None = None,
        *,
        allow_model_retraining: bool = True,
    ):
        self.engine = engine
        self.mode = mode or engine.mode
        self.allow_model_retraining = allow_model_retraining
        self.rebalance_dates = engine._generate_rebalance_dates()
        if len(self.rebalance_dates) < 2:
            raise ValueError("Historical executor requires at least two rebalance dates.")
        bm_ticker = self.engine.cfg["backtest"].get("benchmark_ticker", "^NSEI")
        self.bm_prices = (
            self.engine.price_matrix[bm_ticker]
            if bm_ticker in self.engine.price_matrix.columns
            else None
        )
        self._recent_turnovers: list[float] = []
        self._recent_cost_ratios: list[float] = []
        self._prev_posture: str = "neutral"
        self._prev_target_posture: str = "neutral"
        self._prev_stress_signal: float = 0.0
        self._target_posture_streak: int = 0
        self._prev_posture_mismatch: float = 0.0

    def initial_portfolio(self, idx: int = 0) -> PortfolioState:
        start = self.rebalance_dates[idx]
        return PortfolioState(
            date=start.date(),
            cash=float(self.engine.initial_capital),
            holdings={},
            weights={"CASH": 1.0},
            nav=float(self.engine.initial_capital),
            sector_weights={},
        )

    def initial_nav_points(self, idx: int = 0) -> list[tuple[pd.Timestamp, float]]:
        return [(self.rebalance_dates[idx], float(self.engine.initial_capital))]

    def reset_runtime_state(
        self,
        nav_points: list[tuple[pd.Timestamp, float]] | None = None,
    ) -> None:
        self.engine.risk_engine = RiskEngine(self.engine.cfg)
        self._recent_turnovers = []
        self._recent_cost_ratios = []
        self._prev_posture = "neutral"
        self._prev_target_posture = "neutral"
        self._prev_stress_signal = 0.0
        self._target_posture_streak = 0
        self._prev_posture_mismatch = 0.0
        for ts, nav in nav_points or self.initial_nav_points():
            self.engine.risk_engine.update(float(nav), pd.Timestamp(ts).date())

    def prepare_step(
        self,
        idx: int,
        portfolio: PortfolioState,
        nav_points: list[tuple[pd.Timestamp, float]],
    ) -> PreparedHistoricalStep:
        if idx < 0 or idx >= len(self.rebalance_dates) - 1:
            raise IndexError("Historical step index out of range.")

        current_date = self.rebalance_dates[idx]
        next_date = self.rebalance_dates[idx + 1]

        if self.allow_model_retraining and self.engine._should_retrain_models(idx, current_date):
            self.engine._train_models(current_date, idx=idx)

        prices_today = self.engine._get_prices(current_date)
        if prices_today.empty:
            raise ValueError(f"No prices available on {current_date.date()}")

        snapshot = self.engine.universe_mgr.get_universe(
            current_date.date(),
            price_matrix=self.engine.price_matrix,
            volume_matrix=self.engine.volume_matrix,
        )
        sector_map = self.engine.universe_mgr.get_sector_map(snapshot)
        cap_map = {stock.ticker: stock.cap for stock in snapshot.stocks}

        macro_now_raw = self.engine._get_macro_features(current_date)
        macro_now = (
            macro_now_raw.to_dict()
            if isinstance(macro_now_raw, pd.Series)
            else dict(macro_now_raw)
        )
        sector_feats_now = self.engine._get_sector_features_now(
            current_date, snapshot, prices_today, self.bm_prices
        )
        stock_feats_now = self.engine._get_stock_features_now(
            current_date, snapshot, prices_today, self.bm_prices
        )
        sector_scores = self.engine.sector_scorer.predict(sector_feats_now, macro_now_raw)

        recent_returns = self.engine._get_recent_portfolio_returns(nav_points, current_date)
        bm_rets_recent = (
            self.bm_prices.pct_change(fill_method=None)
            if self.bm_prices is not None
            else None
        )
        risk_regime = self.engine.risk_engine.regime(recent_returns)
        risk_signal, risk_action = self.engine.risk_engine.evaluate(
            portfolio,
            recent_returns,
            macro_features=(
                macro_now_raw
                if isinstance(macro_now_raw, pd.Series)
                else pd.Series(macro_now)
            ),
            volume_matrix=self.engine.volume_matrix.loc[:current_date],
        )
        portfolio_features = self._build_portfolio_features(
            portfolio=portfolio,
            recent_returns=recent_returns,
            benchmark_returns=bm_rets_recent,
            sector_feats=sector_feats_now,
            risk_signal=risk_signal,
            risk_action=risk_action,
            macro_state=macro_now,
        )

        transition_state = build_state(
            macro_state=macro_now,
            sector_state=build_sector_state(sector_feats_now),
            portfolio_state=dict(portfolio_features),
        )
        return PreparedHistoricalStep(
            idx=idx,
            current_date=current_date,
            next_date=next_date,
            snapshot=snapshot,
            prices_today=prices_today,
            sector_map=sector_map,
            cap_map=cap_map,
            macro_now=macro_now,
            sector_feats_now=sector_feats_now,
            stock_feats_now=stock_feats_now,
            sector_scores=sector_scores,
            recent_returns=recent_returns,
            risk_regime=risk_regime,
            risk_signal=risk_signal,
            risk_action=risk_action,
            transition_state=transition_state,
        )

    def execute_prepared_step(
        self,
        prepared: PreparedHistoricalStep,
        portfolio: PortfolioState,
        nav_points: list[tuple[pd.Timestamp, float]],
        rl_decision: dict[str, Any] | None = None,
        *,
        done: bool = False,
    ) -> HistoricalStepResult:
        decision = self._normalize_decision(prepared.snapshot.sectors, rl_decision)
        decision, control_guidance = self._apply_target_control_guidance(
            decision,
            prepared.transition_state,
        )
        portfolio_mtm = self.engine.simulator.value_portfolio(
            portfolio, prepared.prices_today, prepared.current_date.date()
        )
        pre_nav = self._compute_pre_nav(portfolio, prepared.prices_today)

        cash_target = (
            0.0
            if self.mode == "selection_only"
            else max(
                float(decision.get("cash_target", 0.05)),
                float(prepared.risk_action.cash_floor),
            )
        )

        selected_sectors = select_sectors(
            prepared.snapshot.sectors,
            prepared.sector_scores,
            decision,
            full_rl=self.mode == "full_rl",
        )

        alpha_scores, selected_stock_rows = self._build_alpha_scores(prepared, decision)
        if self.mode == "selection_only":
            target_weights = self.engine._build_equal_weight_targets(
                alpha_scores=alpha_scores,
                cash_target=cash_target,
            )
            pre_risk_target_weights = dict(target_weights)
        else:
            cov_matrix = PortfolioOptimizer.estimate_covariance(
                self.engine.price_matrix.loc[:prepared.current_date],
                list(alpha_scores.keys()),
            )
            target_weights = self.engine.optimizer.optimize(
                alpha_scores=alpha_scores,
                cov_matrix=cov_matrix,
                sector_map=prepared.sector_map,
                current_weights=portfolio.weights,
                sector_tilts=(
                    decision["sector_tilts"]
                    if self.mode == "full_rl"
                    else {sector: 1.0 for sector in prepared.snapshot.sectors}
                ),
                aggressiveness=(
                    float(decision.get("aggressiveness", 1.0))
                    if self.mode == "full_rl"
                    else 1.0
                ),
                cash_target=cash_target,
                max_turnover_override=(
                    float(decision.get("turnover_cap"))
                    if decision.get("turnover_cap") is not None
                    else None
                ),
                forced_exclude=prepared.risk_action.exclude_tickers,
            )
            pre_risk_target_weights = dict(target_weights)

        if self.mode != "selection_only":
            target_weights = self.engine.risk_engine.check_pre_trade(
                target_weights,
                prepared.sector_map,
                prepared.cap_map,
                prepared.risk_action,
            )

        exec_result = self.engine.simulator.execute_rebalance(
            target_weights,
            portfolio_mtm,
            prepared.prices_today,
            prepared.current_date.date(),
        )

        post_trade_portfolio = self._with_sector_weights(
            exec_result.new_portfolio,
            prepared.sector_map,
        )
        period_nav = self.engine._compute_period_nav(
            post_trade_portfolio,
            prepared.prices_today,
            prepared.current_date,
            prepared.next_date,
        )
        end_nav = period_nav[-1][1] if period_nav else post_trade_portfolio.nav

        updated_nav_points = list(nav_points)
        updated_nav_points.extend(
            (ts, float(nav))
            for ts, nav in period_nav
            if np.isfinite(nav) and nav > 0
        )
        if not period_nav:
            updated_nav_points.append((prepared.next_date, float(end_nav)))

        current_target_posture = self._target_posture_for_stress(
            self._stress_signal(prepared.transition_state),
            self.engine.cfg,
        )
        self._target_posture_streak = (
            self._target_posture_streak + 1
            if current_target_posture == self._prev_target_posture
            else 1
        )
        self._prev_posture_mismatch = self._posture_distance(
            str(decision.get("posture", "neutral")),
            current_target_posture,
        )
        self._prev_posture = str(decision.get("posture", "neutral"))
        self._prev_target_posture = current_target_posture
        self._prev_stress_signal = float(self._stress_signal(prepared.transition_state))

        next_prices = self.engine._get_prices(prepared.next_date)
        next_portfolio = self._with_sector_weights(
            self.engine.simulator.value_portfolio(
                post_trade_portfolio,
                next_prices,
                prepared.next_date.date(),
            ),
            prepared.sector_map,
        )
        next_state = self._build_state_for_date(
            prepared.next_date,
            next_portfolio,
            updated_nav_points,
        )
        reward, reward_components = self._compute_reward(
            pre_nav=pre_nav,
            end_nav=end_nav,
            nav_points=nav_points,
            period_nav=period_nav,
            exec_result=exec_result,
            next_portfolio=next_portfolio,
            liquidity_stress=prepared.risk_signal.liquidity_stress,
            current_date=prepared.current_date,
            next_date=prepared.next_date,
            decision=decision,
            current_state=prepared.transition_state,
            next_state=next_state,
        )
        self._recent_turnovers.append(float(exec_result.total_turnover))
        if pre_nav > 0:
            self._recent_cost_ratios.append(float(exec_result.total_cost) / float(pre_nav))

        transition = build_transition(
            state=prepared.transition_state,
            action=decision,
            reward=float(reward),
            next_state=next_state,
            done=done,
            info={
                "date": str(prepared.current_date.date()),
                "next_date": str(prepared.next_date.date()),
                "mode": self.mode,
                "selected_sectors": [str(sector) for sector in selected_sectors],
                "pre_risk_target_weights": {
                    ticker: float(weight)
                    for ticker, weight in pre_risk_target_weights.items()
                },
                "target_weights": {
                    ticker: float(weight)
                    for ticker, weight in target_weights.items()
                },
                "executed_weights": {
                    ticker: float(weight)
                    for ticker, weight in next_portfolio.weights.items()
                },
                "executed_sector_weights": {
                    sector: float(weight)
                    for sector, weight in next_portfolio.sector_weights.items()
                },
                "turnover": float(exec_result.total_turnover),
                "transaction_cost": float(exec_result.total_cost),
                "cash_target": float(cash_target),
                "turnover_cap": (
                    float(decision.get("turnover_cap"))
                    if decision.get("turnover_cap") is not None
                    else None
                ),
                "posture": str(decision.get("posture", "neutral")),
                "control_guidance": control_guidance,
                "reward_components": reward_components,
            },
        )
        return HistoricalStepResult(
            current_date=prepared.current_date,
            next_date=prepared.next_date,
            pre_nav=float(pre_nav),
            end_nav=float(end_nav),
            reward=float(reward),
            cash_target=float(cash_target),
            selected_sectors=selected_sectors,
            alpha_scores=alpha_scores,
            selected_stock_rows=selected_stock_rows,
            pre_risk_target_weights=pre_risk_target_weights,
            target_weights=target_weights,
            exec_result=exec_result,
            post_trade_portfolio=post_trade_portfolio,
            next_portfolio=next_portfolio,
            period_nav=period_nav,
            updated_nav_points=updated_nav_points,
            transition=transition,
        )

    def _compute_reward(
        self,
        *,
        pre_nav: float,
        end_nav: float,
        nav_points: list[tuple[pd.Timestamp, float]],
        period_nav: list[tuple[pd.Timestamp, float]],
        exec_result: ExecutionResult,
        next_portfolio: PortfolioState,
        liquidity_stress: bool,
        current_date: pd.Timestamp,
        next_date: pd.Timestamp,
        decision: dict[str, Any],
        current_state: dict[str, Any],
        next_state: dict[str, Any],
    ) -> tuple[float, dict[str, float | bool | None]]:
        rl_cfg = self.engine.cfg.get("rl", {})
        risk_cfg = self.engine.cfg.get("risk", {})

        period_return = (
            (end_nav - pre_nav) / pre_nav
            if pre_nav > 0 and np.isfinite(end_nav) and np.isfinite(pre_nav)
            else 0.0
        )
        benchmark_return = self._benchmark_period_return(current_date, next_date)
        active_return = (
            period_return - benchmark_return
            if benchmark_return is not None
            else period_return
        )

        prev_peak = max(
            [float(nav) for _, nav in nav_points] + [float(pre_nav)],
            default=float(pre_nav),
        )
        min_nav = min(
            [float(nav) for _, nav in period_nav] + [float(end_nav)],
            default=float(end_nav),
        )
        realized_drawdown = (
            min(0.0, (min_nav - prev_peak) / prev_peak)
            if prev_peak > 0
            else 0.0
        )
        drawdown_penalty = float(rl_cfg.get("reward_lambda_dd", 0.0)) * abs(realized_drawdown)

        cost_ratio = (
            float(exec_result.total_cost) / float(pre_nav)
            if pre_nav > 0
            else 0.0
        )
        turnover_penalty = float(rl_cfg.get("reward_lambda_to", 0.0)) * cost_ratio

        weights = np.array(
            [float(weight) for weight in next_portfolio.weights.values() if float(weight) > 0.0],
            dtype=float,
        )
        concentration_hhi = float(np.sum(weights ** 2)) if len(weights) else 1.0
        hhi_threshold = float(risk_cfg.get("max_concentration_hhi", 0.15))
        concentration_penalty = float(rl_cfg.get("reward_lambda_conc", 0.0)) * max(
            0.0,
            concentration_hhi - hhi_threshold,
        )
        liquidity_penalty = (
            float(rl_cfg.get("reward_lambda_liq", 0.0)) if liquidity_stress else 0.0
        )
        action_deviation_penalty = float(
            rl_cfg.get("reward_lambda_action_dev", 0.0)
        ) * self._action_deviation(decision)
        current_stress = self._stress_signal(current_state)
        next_stress = self._stress_signal(next_state)
        stress_signal = float(max(current_stress, next_stress))
        portfolio_state = current_state.get("portfolio_state", {}) if isinstance(current_state, dict) else {}
        defensive_posture = self._defensive_posture(decision)
        target_posture_label = self._target_posture_for_stress(stress_signal, self.engine.cfg)
        target_controls = self._target_controls_for_posture(target_posture_label, self.engine.cfg)
        target_posture = self._defensive_posture(target_controls)
        decision_posture = str(decision.get("posture", "neutral"))
        prev_posture_label = self._label_from_score(
            float(portfolio_state.get("previous_posture_score", 0.0) or 0.0)
        )
        prev_target_posture_label = self._label_from_score(
            float(portfolio_state.get("previous_target_posture_score", 0.0) or 0.0)
        )
        target_streak = max(0.0, float(portfolio_state.get("target_posture_streak", 0.0) or 0.0))
        prev_distance = self._posture_distance(prev_posture_label, target_posture_label)
        current_distance = self._posture_distance(decision_posture, target_posture_label)
        defense_gap_penalty = float(
            rl_cfg.get("reward_lambda_defense_gap", 0.0)
        ) * max(0.0, stress_signal - defensive_posture)
        overdefense_penalty = float(
            rl_cfg.get("reward_lambda_overdefense", 0.0)
        ) * max(0.0, defensive_posture - stress_signal)
        stress_turnover_penalty = float(
            rl_cfg.get("reward_lambda_stress_turnover", 0.0)
        ) * stress_signal * max(0.0, float(exec_result.total_turnover) - 0.20)
        target_posture_penalty = float(
            rl_cfg.get("reward_lambda_target_posture", 0.0)
        ) * self._target_control_gap(decision, target_controls)
        posture_progress_bonus = float(
            rl_cfg.get("reward_lambda_posture_progress", 0.0)
        ) * (prev_distance - current_distance)
        posture_stale_penalty = float(
            rl_cfg.get("reward_lambda_posture_stale", 0.0)
        ) * (
            current_distance
            * max(0.0, target_streak - 1.0)
            * (1.0 if decision_posture == prev_posture_label and current_distance > 0.0 else 0.0)
        )
        posture_flip_penalty = float(
            rl_cfg.get("reward_lambda_posture_flip", 0.0)
        ) * (
            1.0
            if decision_posture != prev_posture_label
            and target_posture_label == prev_target_posture_label
            and current_distance >= prev_distance
            else 0.0
        )

        reward = (
            active_return
            - drawdown_penalty
            - turnover_penalty
            - concentration_penalty
            - liquidity_penalty
            - action_deviation_penalty
            - defense_gap_penalty
            - overdefense_penalty
            - stress_turnover_penalty
            - target_posture_penalty
            + posture_progress_bonus
            - posture_stale_penalty
            - posture_flip_penalty
        )
        return float(reward), {
            "period_return": float(period_return),
            "benchmark_return": (
                float(benchmark_return) if benchmark_return is not None else None
            ),
            "active_return": float(active_return),
            "drawdown_penalty": float(drawdown_penalty),
            "realized_drawdown": float(realized_drawdown),
            "transaction_cost": float(exec_result.total_cost),
            "transaction_cost_ratio": float(cost_ratio),
            "realized_turnover": float(exec_result.total_turnover),
            "turnover_penalty": float(turnover_penalty),
            "concentration_hhi": float(concentration_hhi),
            "concentration_penalty": float(concentration_penalty),
            "liquidity_penalty": float(liquidity_penalty),
            "liquidity_stress": bool(liquidity_stress),
            "action_deviation_penalty": float(action_deviation_penalty),
            "stress_signal": float(stress_signal),
            "defensive_posture": float(defensive_posture),
            "posture": str(decision_posture),
            "target_posture": str(target_posture_label),
            "prev_posture": str(prev_posture_label),
            "prev_target_posture": str(prev_target_posture_label),
            "target_posture_streak": float(target_streak),
            "posture_distance_to_target": float(current_distance),
            "prev_posture_distance_to_target": float(prev_distance),
            "target_defensive_posture": float(target_posture),
            "target_cash_target": float(target_controls["cash_target"]),
            "target_aggressiveness": float(target_controls["aggressiveness"]),
            "target_turnover_cap": float(target_controls["turnover_cap"]),
            "defense_gap_penalty": float(defense_gap_penalty),
            "overdefense_penalty": float(overdefense_penalty),
            "stress_turnover_penalty": float(stress_turnover_penalty),
            "target_posture_penalty": float(target_posture_penalty),
            "posture_progress_bonus": float(posture_progress_bonus),
            "posture_stale_penalty": float(posture_stale_penalty),
            "posture_flip_penalty": float(posture_flip_penalty),
            "reward": float(reward),
        }

    def _benchmark_period_return(
        self,
        current_date: pd.Timestamp,
        next_date: pd.Timestamp,
    ) -> float | None:
        if self.bm_prices is None:
            return None

        current_prices = self.bm_prices.loc[self.bm_prices.index <= current_date]
        next_prices = self.bm_prices.loc[self.bm_prices.index <= next_date]
        if current_prices.empty or next_prices.empty:
            return None

        start_price = float(current_prices.iloc[-1])
        end_price = float(next_prices.iloc[-1])
        if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price <= 0:
            return None
        return float((end_price - start_price) / start_price)

    @staticmethod
    def _action_deviation(decision: dict[str, Any]) -> float:
        sector_tilts = decision.get("sector_tilts", {})
        sector_dev = [
            abs(float(sector_tilts.get(sector, 1.0)) - 1.0)
            for sector in sector_tilts
        ]
        cash_dev = abs(float(decision.get("cash_target", 0.05)) - 0.05)
        aggressiveness_dev = abs(float(decision.get("aggressiveness", 1.0)) - 1.0)
        turnover_cap = decision.get("turnover_cap")
        turnover_dev = abs(float(turnover_cap) - 0.40) if turnover_cap is not None else 0.0
        components = sector_dev + [cash_dev, aggressiveness_dev, turnover_dev]
        if not components:
            return 0.0
        return float(np.mean(components))

    @staticmethod
    def _stress_signal(state: dict[str, Any]) -> float:
        macro_state = state.get("macro_state", {}) if isinstance(state, dict) else {}
        portfolio_state = state.get("portfolio_state", {}) if isinstance(state, dict) else {}
        drawdown_component = np.clip(
            abs(min(0.0, float(portfolio_state.get("current_drawdown", 0.0) or 0.0))) / 0.12,
            0.0,
            1.0,
        )
        slope_component = np.clip(
            abs(min(0.0, float(portfolio_state.get("drawdown_slope_1m", 0.0) or 0.0))) / 0.05,
            0.0,
            1.0,
        )
        vol_component = np.clip(
            max(0.0, float(portfolio_state.get("vol_shock_1m_3m", 0.0) or 0.0)) / 0.5,
            0.0,
            1.0,
        )
        breadth_component = np.clip(
            float(portfolio_state.get("breadth_deterioration", 0.0) or 0.0),
            0.0,
            1.0,
        )
        macro_component = np.clip(
            float(macro_state.get("macro_stress_score", 0.0) or 0.0),
            0.0,
            1.0,
        )
        emergency_component = np.clip(
            float(portfolio_state.get("emergency_flag", 0.0) or 0.0),
            0.0,
            1.0,
        )
        signal = (
            0.30 * drawdown_component
            + 0.15 * slope_component
            + 0.15 * vol_component
            + 0.15 * breadth_component
            + 0.15 * macro_component
            + 0.10 * emergency_component
        )
        return float(np.clip(signal, 0.0, 1.0))

    @staticmethod
    def _defensive_posture(decision: dict[str, Any]) -> float:
        cash_target = float(decision.get("cash_target", 0.05) or 0.05)
        aggressiveness = float(decision.get("aggressiveness", 1.0) or 1.0)
        turnover_cap = decision.get("turnover_cap")
        cash_component = np.clip((cash_target - 0.05) / 0.25, 0.0, 1.0)
        aggressiveness_component = np.clip((1.0 - aggressiveness) / 0.40, 0.0, 1.0)
        turnover_component = (
            np.clip((0.40 - float(turnover_cap)) / 0.20, 0.0, 1.0)
            if turnover_cap is not None
            else 0.0
        )
        posture = (
            0.45 * cash_component
            + 0.35 * aggressiveness_component
            + 0.20 * turnover_component
        )
        return float(np.clip(posture, 0.0, 1.0))

    @staticmethod
    def _target_controls_for_stress(stress_signal: float, cfg: dict | None = None) -> dict[str, float]:
        target_posture = HistoricalPeriodExecutor._target_posture_for_stress(stress_signal, cfg)
        return HistoricalPeriodExecutor._target_controls_for_posture(target_posture, cfg)

    @staticmethod
    def _target_posture_for_stress(stress_signal: float, cfg: dict | None = None) -> str:
        rl_cfg = (cfg or {}).get("rl", {}) if isinstance(cfg, dict) else {}
        moderate_threshold = float(rl_cfg.get("stress_target_moderate", 0.18))
        high_threshold = float(rl_cfg.get("stress_target_high", 0.35))
        if high_threshold < moderate_threshold:
            moderate_threshold, high_threshold = high_threshold, moderate_threshold

        if stress_signal >= high_threshold:
            return "risk_off"
        if stress_signal >= moderate_threshold:
            return "neutral"
        return "risk_on"

    @staticmethod
    def _target_controls_for_posture(posture: str, cfg: dict | None = None) -> dict[str, float]:
        rl_cfg = (cfg or {}).get("rl", {}) if isinstance(cfg, dict) else {}
        profiles = rl_cfg.get("posture_profiles", {}) if isinstance(rl_cfg, dict) else {}
        defaults = {
            "risk_off": {
                "cash_target": float(rl_cfg.get("target_cash_high", 0.20)),
                "aggressiveness": float(rl_cfg.get("target_aggressiveness_high", 0.90)),
                "turnover_cap": float(rl_cfg.get("target_turnover_cap_high", 0.25)),
            },
            "neutral": {
                "cash_target": 0.05,
                "aggressiveness": 1.0,
                "turnover_cap": 0.40,
            },
            "risk_on": {
                "cash_target": float(rl_cfg.get("target_cash_risk_on", 0.03)),
                "aggressiveness": float(rl_cfg.get("target_aggressiveness_risk_on", 1.05)),
                "turnover_cap": float(rl_cfg.get("target_turnover_cap_risk_on", 0.40)),
            },
        }
        profile = defaults.get(posture, defaults["neutral"]).copy()
        configured = profiles.get(posture, {}) if isinstance(profiles, dict) else {}
        if isinstance(configured, dict):
            profile.update({k: float(v) for k, v in configured.items() if k in profile})
        return profile

    @staticmethod
    def _target_control_gap(decision: dict[str, Any], target_controls: dict[str, float]) -> float:
        decision_posture = str(decision.get("posture", "neutral"))
        target_posture = HistoricalPeriodExecutor._nearest_posture_label(target_controls)
        posture_gap = 0.0 if decision_posture == target_posture else 1.0
        cash_gap = abs(
            float(decision.get("cash_target", 0.05) or 0.05)
            - float(target_controls.get("cash_target", 0.05))
        ) / 0.25
        aggressiveness_gap = abs(
            float(decision.get("aggressiveness", 1.0) or 1.0)
            - float(target_controls.get("aggressiveness", 1.0))
        ) / 0.40
        turnover_cap = decision.get("turnover_cap")
        turnover_gap = (
            abs(float(turnover_cap) - float(target_controls.get("turnover_cap", 0.40))) / 0.20
            if turnover_cap is not None
            else 0.0
        )
        return float(np.mean([posture_gap, cash_gap, aggressiveness_gap, turnover_gap]))

    @staticmethod
    def _nearest_posture_label(controls: dict[str, float]) -> str:
        candidates = {
            "risk_off": {"cash_target": 0.20, "aggressiveness": 0.90, "turnover_cap": 0.25},
            "neutral": {"cash_target": 0.05, "aggressiveness": 1.0, "turnover_cap": 0.40},
            "risk_on": {"cash_target": 0.03, "aggressiveness": 1.05, "turnover_cap": 0.40},
        }
        best_label = "neutral"
        best_distance = float("inf")
        for label, target in candidates.items():
            distance = (
                abs(float(controls.get("cash_target", 0.05)) - target["cash_target"]) / 0.25
                + abs(float(controls.get("aggressiveness", 1.0)) - target["aggressiveness"]) / 0.40
                + abs(float(controls.get("turnover_cap", 0.40)) - target["turnover_cap"]) / 0.20
            )
            if distance < best_distance:
                best_label = label
                best_distance = distance
        return best_label

    @staticmethod
    def _posture_score(posture: str) -> float:
        mapping = {"risk_on": -1.0, "neutral": 0.0, "risk_off": 1.0}
        return float(mapping.get(posture, 0.0))

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score <= -0.5:
            return "risk_on"
        if score >= 0.5:
            return "risk_off"
        return "neutral"

    @staticmethod
    def _posture_distance(left: str, right: str) -> float:
        return float(
            abs(
                HistoricalPeriodExecutor._posture_score(left)
                - HistoricalPeriodExecutor._posture_score(right)
            )
            / 2.0
        )

    @staticmethod
    def _step_toward_posture(current: str, target: str, *, max_step: int = 1) -> str:
        order = ["risk_on", "neutral", "risk_off"]
        try:
            current_idx = order.index(current)
        except ValueError:
            current_idx = order.index("neutral")
        try:
            target_idx = order.index(target)
        except ValueError:
            target_idx = order.index("neutral")
        if current_idx == target_idx:
            return order[current_idx]
        step = min(max_step, abs(target_idx - current_idx))
        if target_idx > current_idx:
            return order[current_idx + step]
        return order[current_idx - step]

    def _apply_target_control_guidance(
        self,
        decision: dict[str, Any],
        current_state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float | bool]]:
        rl_cfg = self.engine.cfg.get("rl", {})
        portfolio_state = current_state.get("portfolio_state", {}) if isinstance(current_state, dict) else {}
        stress_signal = self._stress_signal(current_state)
        target_posture = self._target_posture_for_stress(stress_signal, self.engine.cfg)
        target_controls = self._target_controls_for_posture(target_posture, self.engine.cfg)
        target_streak = max(0.0, float(portfolio_state.get("target_posture_streak", 0.0) or 0.0))
        prev_target_posture = self._label_from_score(
            float(portfolio_state.get("previous_target_posture_score", 0.0) or 0.0)
        )
        posture_guidance_enabled = bool(rl_cfg.get("enable_target_posture_guidance", False))
        guided = dict(decision)
        applied_posture_guidance = False
        posture_guidance_step = 0.0
        if posture_guidance_enabled:
            min_streak = float(rl_cfg.get("target_posture_guidance_min_streak", 2))
            min_stress = float(rl_cfg.get("target_posture_guidance_min_stress", 0.16))
            max_step = int(max(1, rl_cfg.get("target_posture_guidance_max_step", 1)))
            current_posture = str(decision.get("posture", "neutral"))
            if (
                current_posture != target_posture
                and target_streak >= min_streak
                and stress_signal >= min_stress
                and target_posture == prev_target_posture
            ):
                guided["posture"] = self._step_toward_posture(
                    current_posture,
                    target_posture,
                    max_step=max_step,
                )
                applied_posture_guidance = guided["posture"] != current_posture
                posture_guidance_step = self._posture_distance(
                    current_posture,
                    str(guided["posture"]),
                )
                posture_controls = self._target_controls_for_posture(str(guided["posture"]), self.engine.cfg)
                guided["cash_target"] = float(posture_controls["cash_target"])
                guided["aggressiveness"] = float(posture_controls["aggressiveness"])
                guided["turnover_cap"] = float(posture_controls["turnover_cap"])

        if not bool(rl_cfg.get("enable_target_control_blend", False)):
            return guided, {
                "enabled": False,
                "blend_weight": 0.0,
                "stress_signal": float(stress_signal),
                "target_posture": str(target_posture),
                "target_posture_streak": float(target_streak),
                "posture_guidance_enabled": posture_guidance_enabled,
                "applied_posture_guidance": applied_posture_guidance,
                "posture_guidance_step": float(posture_guidance_step),
            }

        min_blend = float(rl_cfg.get("target_control_blend_min", 0.15))
        max_blend = float(rl_cfg.get("target_control_blend_max", 0.85))
        if min_blend > max_blend:
            min_blend, max_blend = max_blend, min_blend
        blend_weight = float(np.clip(min_blend + (max_blend - min_blend) * stress_signal, 0.0, 1.0))

        guided["cash_target"] = float(
            (1.0 - blend_weight) * float(guided.get("cash_target", 0.05))
            + blend_weight * float(target_controls["cash_target"])
        )
        guided["aggressiveness"] = float(
            (1.0 - blend_weight) * float(guided.get("aggressiveness", 1.0))
            + blend_weight * float(target_controls["aggressiveness"])
        )
        turnover_cap = guided.get("turnover_cap")
        if turnover_cap is not None:
            guided["turnover_cap"] = float(
                (1.0 - blend_weight) * float(turnover_cap)
                + blend_weight * float(target_controls["turnover_cap"])
            )
        guided = self._normalize_decision(
            list(decision.get("sector_tilts", {}).keys()),
            guided,
        )
        guided["posture"] = self._nearest_posture_label(guided)
        return guided, {
            "enabled": True,
            "blend_weight": float(blend_weight),
            "stress_signal": float(stress_signal),
            "target_posture": str(target_posture),
            "target_posture_streak": float(target_streak),
            "posture_guidance_enabled": posture_guidance_enabled,
            "applied_posture_guidance": applied_posture_guidance,
            "posture_guidance_step": float(posture_guidance_step),
            "target_cash_target": float(target_controls["cash_target"]),
            "target_aggressiveness": float(target_controls["aggressiveness"]),
            "target_turnover_cap": float(target_controls["turnover_cap"]),
        }

    def _build_alpha_scores(
        self,
        prepared: PreparedHistoricalStep,
        decision: dict[str, Any],
    ) -> tuple[dict[str, float], list[dict[str, object]]]:
        top_k = self.engine.cfg["stock_model"]["top_k_per_sector"]
        alpha_scores: dict[str, float] = {}
        selected_stock_rows: list[dict[str, object]] = []
        selected_sectors = select_sectors(
            prepared.snapshot.sectors,
            prepared.sector_scores,
            decision,
            full_rl=self.mode == "full_rl",
        )
        for sector in selected_sectors:
            ranking = self.engine.stock_ranker.rank_stocks(
                prepared.stock_feats_now, sector, top_k=top_k
            )
            for _, row in ranking.iterrows():
                ticker = str(row["ticker"])
                raw_score = float(row["score"])
                alpha_scores[ticker] = raw_score
                selected_stock_rows.append(
                    {"ticker": ticker, "sector": sector, "score": raw_score}
                )

        if not alpha_scores and not prepared.stock_feats_now.empty:
            for ticker in prepared.snapshot.tickers[:30]:
                if ticker in prepared.prices_today.index:
                    alpha_scores[ticker] = 0.5
                    selected_stock_rows.append(
                        {
                            "ticker": ticker,
                            "sector": prepared.sector_map.get(ticker, "Unknown"),
                            "score": 0.5,
                        }
                    )
        return alpha_scores, selected_stock_rows

    def _build_state_for_date(
        self,
        as_of: pd.Timestamp,
        portfolio: PortfolioState,
        nav_points: list[tuple[pd.Timestamp, float]],
    ) -> dict[str, Any]:
        snapshot = self.engine.universe_mgr.get_universe(
            as_of.date(),
            price_matrix=self.engine.price_matrix,
            volume_matrix=self.engine.volume_matrix,
        )
        prices_now = self.engine._get_prices(as_of)
        macro_now_raw = self.engine._get_macro_features(as_of)
        macro_now = (
            macro_now_raw.to_dict()
            if isinstance(macro_now_raw, pd.Series)
            else dict(macro_now_raw)
        )
        sector_feats_now = self.engine._get_sector_features_now(
            as_of, snapshot, prices_now, self.bm_prices
        )
        recent_returns = self.engine._get_recent_portfolio_returns(nav_points, as_of)
        bm_rets_recent = (
            self.bm_prices.pct_change(fill_method=None)
            if self.bm_prices is not None
            else None
        )
        risk_signal, risk_action = self.engine.risk_engine.evaluate(
            portfolio,
            recent_returns,
            macro_features=(
                macro_now_raw
                if isinstance(macro_now_raw, pd.Series)
                else pd.Series(macro_now)
            ),
            volume_matrix=self.engine.volume_matrix.loc[:as_of],
        )
        portfolio_state = self._build_portfolio_features(
            portfolio=portfolio,
            recent_returns=recent_returns,
            benchmark_returns=bm_rets_recent,
            sector_feats=sector_feats_now,
            risk_signal=risk_signal,
            risk_action=risk_action,
            macro_state=macro_now,
        )
        return build_state(
            macro_state=macro_now,
            sector_state=build_sector_state(sector_feats_now),
            portfolio_state=dict(portfolio_state),
        )

    def _normalize_decision(
        self,
        sectors: list[str],
        rl_decision: dict[str, Any] | None,
    ) -> dict[str, Any]:
        decision = default_decision(sectors)
        if not rl_decision:
            return decision
        sector_tilts = {
            str(sector): float(tilt)
            for sector, tilt in rl_decision.get("sector_tilts", {}).items()
        }
        decision["sector_tilts"].update(sector_tilts)
        posture = str(rl_decision.get("posture", decision.get("posture", "neutral")))
        if posture not in {"risk_off", "neutral", "risk_on"}:
            posture = "neutral"
        decision["posture"] = posture
        if "cash_target" in rl_decision:
            decision["cash_target"] = float(np.clip(rl_decision["cash_target"], 0.0, 0.30))
        if "aggressiveness" in rl_decision:
            agg_min = float(self.engine.cfg.get("rl", {}).get("aggressiveness_min", 0.60))
            agg_max = float(self.engine.cfg.get("rl", {}).get("aggressiveness_max", 1.40))
            if agg_min > agg_max:
                agg_min, agg_max = agg_max, agg_min
            decision["aggressiveness"] = float(np.clip(rl_decision["aggressiveness"], agg_min, agg_max))
        if "turnover_cap" in rl_decision and rl_decision.get("turnover_cap") is not None:
            max_turnover = float(
                self.engine.cfg.get("optimizer", {}).get("max_turnover_per_rebalance", 0.40)
            )
            decision["turnover_cap"] = float(np.clip(rl_decision["turnover_cap"], 0.05, max_turnover))
        posture_controls = self._target_controls_for_posture(decision["posture"], self.engine.cfg)
        if "cash_target" not in rl_decision:
            decision["cash_target"] = float(posture_controls["cash_target"])
        if "aggressiveness" not in rl_decision:
            decision["aggressiveness"] = float(posture_controls["aggressiveness"])
        if "turnover_cap" not in rl_decision or rl_decision.get("turnover_cap") is None:
            decision["turnover_cap"] = float(posture_controls["turnover_cap"])
        return decision

    def _build_portfolio_features(
        self,
        *,
        portfolio: PortfolioState,
        recent_returns: pd.Series,
        benchmark_returns: pd.Series | None,
        sector_feats: pd.DataFrame,
        risk_signal: RiskSignal,
        risk_action: RiskAction,
        macro_state: dict[str, Any],
    ) -> dict[str, float]:
        base_context = self._build_control_context(
            sector_feats,
            risk_signal=risk_signal,
            risk_action=risk_action,
        )
        features = compute_portfolio_features(
            portfolio,
            recent_returns,
            benchmark_returns,
            control_context=base_context,
        )
        provisional_state = build_state(
            macro_state=macro_state,
            sector_state=build_sector_state(sector_feats),
            portfolio_state=dict(features),
        )
        current_stress = self._stress_signal(provisional_state)
        target_posture = self._target_posture_for_stress(current_stress, self.engine.cfg)
        target_streak = (
            self._target_posture_streak + 1
            if target_posture == self._prev_target_posture
            else 1
        )
        enriched_context = self._build_control_context(
            sector_feats,
            risk_signal=risk_signal,
            risk_action=risk_action,
            posture_context={
                "current_stress_signal": current_stress,
                "previous_stress_signal": self._prev_stress_signal,
                "target_posture_score": self._posture_score(target_posture),
                "previous_posture_score": self._posture_score(self._prev_posture),
                "previous_target_posture_score": self._posture_score(self._prev_target_posture),
                "target_posture_streak": float(target_streak),
                "previous_posture_mismatch": self._prev_posture_mismatch,
            },
        )
        return compute_portfolio_features(
            portfolio,
            recent_returns,
            benchmark_returns,
            control_context=enriched_context,
        )

    def _build_control_context(
        self,
        sector_feats: pd.DataFrame,
        *,
        risk_signal: RiskSignal,
        risk_action: RiskAction,
        posture_context: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return build_control_context(
            sector_feats,
            risk_signal=risk_signal,
            risk_action=risk_action,
            recent_turnovers=self._recent_turnovers,
            recent_cost_ratios=self._recent_cost_ratios,
            posture_context=posture_context,
        )

    def _with_sector_weights(
        self,
        portfolio: PortfolioState,
        sector_map: dict[str, str],
    ) -> PortfolioState:
        sector_weights: dict[str, float] = {}
        for ticker, weight in portfolio.weights.items():
            if ticker == "CASH":
                continue
            sector = sector_map.get(ticker, "Unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + float(weight)
        return PortfolioState(
            date=portfolio.date,
            cash=portfolio.cash,
            holdings=dict(portfolio.holdings),
            weights=dict(portfolio.weights),
            nav=portfolio.nav,
            sector_weights=sector_weights,
        )

    def _compute_pre_nav(self, portfolio: PortfolioState, prices_today: pd.Series) -> float:
        mtm_value = sum(
            portfolio.holdings.get(ticker, 0.0) * float(prices_today.get(ticker, 0.0) or 0.0)
            for ticker in portfolio.holdings
            if ticker in prices_today.index
            and np.isfinite(float(prices_today.get(ticker, 0.0) or 0.0))
        ) + portfolio.cash
        if np.isfinite(mtm_value) and mtm_value > 0:
            return max(float(mtm_value), float(portfolio.nav))
        return float(portfolio.nav)
