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
        compute_regret: bool = True,
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
            step_idx=prepared.idx,
            starting_portfolio=portfolio,
            starting_nav_points=nav_points,
            current_weights=dict(portfolio_mtm.weights),
            base_target_weights=dict(target_weights),
            realized_asset_returns=self._realized_asset_returns(
                prepared.prices_today,
                next_prices,
            ),
            observed_turnover=float(exec_result.total_turnover),
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
            compute_regret=compute_regret,
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
                "aggressiveness": float(decision.get("aggressiveness", 1.0)),
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
        step_idx: int,
        starting_portfolio: PortfolioState,
        starting_nav_points: list[tuple[pd.Timestamp, float]],
        current_weights: dict[str, float],
        base_target_weights: dict[str, float],
        realized_asset_returns: dict[str, float],
        observed_turnover: float,
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
        compute_regret: bool = True,
    ) -> tuple[float, dict[str, Any]]:
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
        concentration_excess = max(0.0, concentration_hhi - hhi_threshold)
        liquidity_penalty = float(rl_cfg.get("reward_lambda_liq", 0.0)) if liquidity_stress else 0.0
        action_deviation_penalty = float(
            rl_cfg.get("reward_lambda_action_dev", 0.0)
        ) * self._action_deviation(decision)
        current_stress = self._stress_signal(current_state)
        next_stress = self._stress_signal(next_state)
        stress_signal = float(max(current_stress, next_stress))
        full_utility, full_utility_components = self._compose_utility(
            active_return=active_return,
            realized_drawdown=realized_drawdown,
            cost_ratio=cost_ratio,
            concentration_excess=concentration_excess,
            liquidity_stress=liquidity_stress,
            stress_signal=stress_signal,
            reduced=False,
        )
        reduced_utility, reduced_utility_components = self._compose_utility(
            active_return=active_return,
            realized_drawdown=realized_drawdown,
            cost_ratio=cost_ratio,
            concentration_excess=concentration_excess,
            liquidity_stress=liquidity_stress,
            stress_signal=stress_signal,
            reduced=True,
        )
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
        posture_value_map: dict[str, float] = {}
        chosen_posture_utility = float(reduced_utility)
        soft_regret = 0.0
        soft_regret_penalty = 0.0
        soft_regret_baseline = float(reduced_utility)
        best_posture = decision_posture
        posture_utility_variance = 0.0
        posture_utility_spread = 0.0
        posture_utility_variance_threshold = float(
            rl_cfg.get("reward_regret_variance_threshold", 1.0e-5)
        )
        regret_horizon = 1
        regret_policy_count = 0
        if compute_regret and bool(rl_cfg.get("enable_soft_posture_regret", True)):
            posture_value_map, regret_meta = self._compute_posture_value_map(
                current_weights=current_weights,
                base_target_weights=base_target_weights,
                realized_asset_returns=realized_asset_returns,
                observed_cost_ratio=cost_ratio,
                observed_turnover=observed_turnover,
                pre_nav=pre_nav,
                nav_points=starting_nav_points,
                base_decision=decision,
                stress_signal=stress_signal,
                benchmark_return=benchmark_return,
                liquidity_stress=liquidity_stress,
            )
            if posture_value_map:
                values = np.asarray(list(posture_value_map.values()), dtype=float)
                chosen_posture_utility = float(
                    posture_value_map.get(decision_posture, float(np.mean(values)))
                )
                soft_regret_baseline = self._soft_utility_baseline(
                    values,
                    float(rl_cfg.get("reward_regret_temperature", 0.05)),
                )
                soft_regret = float(max(0.0, soft_regret_baseline - chosen_posture_utility))
                soft_regret_penalty = float(
                    rl_cfg.get("reward_regret_lambda", 0.15)
                ) * soft_regret
                best_posture = max(posture_value_map.items(), key=lambda item: item[1])[0]
                posture_utility_variance = float(np.var(values)) if len(values) else 0.0
                posture_utility_spread = float(np.max(values) - np.min(values)) if len(values) else 0.0
                regret_horizon = int(regret_meta.get("horizon_steps", 1))
                regret_policy_count = int(regret_meta.get("policy_count", 0))

        reward = (
            full_utility
            - action_deviation_penalty
            - soft_regret_penalty
        )
        return float(reward), {
            "period_return": float(period_return),
            "benchmark_return": (
                float(benchmark_return) if benchmark_return is not None else None
            ),
            "active_return": float(active_return),
            "utility_full": float(full_utility),
            "utility_reduced": float(reduced_utility),
            "regime_weighted_return": float(full_utility_components["weighted_return"]),
            "drawdown_penalty": float(full_utility_components["drawdown_penalty"]),
            "realized_drawdown": float(realized_drawdown),
            "transaction_cost": float(exec_result.total_cost),
            "transaction_cost_ratio": float(cost_ratio),
            "realized_turnover": float(exec_result.total_turnover),
            "turnover_penalty": float(full_utility_components["turnover_penalty"]),
            "concentration_hhi": float(concentration_hhi),
            "concentration_penalty": float(full_utility_components["concentration_penalty"]),
            "liquidity_penalty": float(full_utility_components["liquidity_penalty"]),
            "liquidity_stress": bool(liquidity_stress),
            "action_deviation_penalty": float(action_deviation_penalty),
            "stress_signal": float(stress_signal),
            "return_weight": float(full_utility_components["return_weight"]),
            "drawdown_weight": float(full_utility_components["drawdown_weight"]),
            "turnover_weight": float(full_utility_components["turnover_weight"]),
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
            "target_posture_penalty": float(target_posture_penalty),
            "posture_progress_bonus": float(posture_progress_bonus),
            "posture_stale_penalty": float(posture_stale_penalty),
            "posture_flip_penalty": float(posture_flip_penalty),
            "decision_quality_basis": "cached_one_step_soft_regret_v1",
            "soft_regret": float(soft_regret),
            "soft_regret_penalty": float(soft_regret_penalty),
            "soft_regret_baseline": float(soft_regret_baseline),
            "chosen_posture_utility": float(chosen_posture_utility),
            "best_posture": str(best_posture),
            "posture_optimality": float(1.0 if decision_posture == best_posture else 0.0),
            "posture_value_map": {
                posture: float(value) for posture, value in posture_value_map.items()
            },
            "posture_utility_variance": float(posture_utility_variance),
            "posture_utility_spread": float(posture_utility_spread),
            "posture_utility_variance_threshold": float(posture_utility_variance_threshold),
            "posture_utility_variance_above_threshold": bool(
                posture_utility_variance > posture_utility_variance_threshold
            ),
            "regret_horizon_steps": int(regret_horizon),
            "regret_policy_count": int(regret_policy_count),
            "reward": float(reward),
        }

    def _compose_utility(
        self,
        *,
        active_return: float,
        realized_drawdown: float,
        cost_ratio: float,
        concentration_excess: float,
        liquidity_stress: bool,
        stress_signal: float,
        reduced: bool,
    ) -> tuple[float, dict[str, float]]:
        rl_cfg = self.engine.cfg.get("rl", {})
        return_weight, drawdown_weight, turnover_weight = self._regime_weights(
            stress_signal,
            reduced=reduced,
        )
        concentration_scale = float(
            rl_cfg.get(
                "reward_regret_concentration_scale" if reduced else "reward_concentration_scale",
                0.50 if reduced else 1.0,
            )
        )
        liquidity_scale = float(
            rl_cfg.get(
                "reward_regret_liquidity_scale" if reduced else "reward_liquidity_scale",
                0.50 if reduced else 1.0,
            )
        )
        drawdown_penalty = (
            drawdown_weight
            * float(rl_cfg.get("reward_lambda_dd", 0.0))
            * abs(realized_drawdown)
        )
        turnover_penalty = (
            turnover_weight
            * float(rl_cfg.get("reward_lambda_to", 0.0))
            * cost_ratio
        )
        concentration_penalty = (
            concentration_scale
            * float(rl_cfg.get("reward_lambda_conc", 0.0))
            * concentration_excess
        )
        liquidity_penalty = (
            liquidity_scale * float(rl_cfg.get("reward_lambda_liq", 0.0))
            if liquidity_stress
            else 0.0
        )
        weighted_return = return_weight * active_return
        utility = (
            weighted_return
            - drawdown_penalty
            - turnover_penalty
            - concentration_penalty
            - liquidity_penalty
        )
        return float(utility), {
            "weighted_return": float(weighted_return),
            "drawdown_penalty": float(drawdown_penalty),
            "turnover_penalty": float(turnover_penalty),
            "concentration_penalty": float(concentration_penalty),
            "liquidity_penalty": float(liquidity_penalty),
            "return_weight": float(return_weight),
            "drawdown_weight": float(drawdown_weight),
            "turnover_weight": float(turnover_weight),
        }

    def _regime_weights(
        self,
        stress_signal: float,
        *,
        reduced: bool,
    ) -> tuple[float, float, float]:
        rl_cfg = self.engine.cfg.get("rl", {})
        suffix = "_reduced" if reduced else ""
        clipped = float(np.clip(stress_signal, 0.0, 1.0))
        return (
            self._interpolate_weight(
                clipped,
                low=float(rl_cfg.get(f"reward_return_weight_low{suffix}", 1.15 if not reduced else 1.0)),
                high=float(rl_cfg.get(f"reward_return_weight_high{suffix}", 0.70 if not reduced else 0.80)),
            ),
            self._interpolate_weight(
                clipped,
                low=float(rl_cfg.get(f"reward_drawdown_weight_low{suffix}", 0.80 if not reduced else 0.75)),
                high=float(rl_cfg.get(f"reward_drawdown_weight_high{suffix}", 1.35 if not reduced else 1.15)),
            ),
            self._interpolate_weight(
                clipped,
                low=float(rl_cfg.get(f"reward_turnover_weight_low{suffix}", 0.85 if not reduced else 0.90)),
                high=float(rl_cfg.get(f"reward_turnover_weight_high{suffix}", 1.20 if not reduced else 1.05)),
            ),
        )

    @staticmethod
    def _interpolate_weight(stress_signal: float, *, low: float, high: float) -> float:
        return float(low + (high - low) * np.clip(stress_signal, 0.0, 1.0))

    def _compute_posture_value_map(
        self,
        *,
        current_weights: dict[str, float],
        base_target_weights: dict[str, float],
        realized_asset_returns: dict[str, float],
        observed_cost_ratio: float,
        observed_turnover: float,
        pre_nav: float,
        nav_points: list[tuple[pd.Timestamp, float]],
        base_decision: dict[str, Any],
        stress_signal: float,
        benchmark_return: float | None,
        liquidity_stress: bool,
    ) -> tuple[dict[str, float], dict[str, int]]:
        postures = ("risk_on", "neutral", "risk_off")
        posture_values: dict[str, float] = {}
        for posture in postures:
            candidate_decision = self._counterfactual_decision(base_decision, posture)
            candidate_utility = self._approximate_counterfactual_utility(
                current_weights=current_weights,
                base_target_weights=base_target_weights,
                realized_asset_returns=realized_asset_returns,
                observed_cost_ratio=observed_cost_ratio,
                observed_turnover=observed_turnover,
                pre_nav=pre_nav,
                nav_points=nav_points,
                candidate_decision=candidate_decision,
                stress_signal=stress_signal,
                benchmark_return=benchmark_return,
                liquidity_stress=liquidity_stress,
            )
            posture_values[posture] = float(candidate_utility)
        return posture_values, {
            "horizon_steps": 1,
            "policy_count": len(postures),
        }

    def _approximate_counterfactual_utility(
        self,
        *,
        current_weights: dict[str, float],
        base_target_weights: dict[str, float],
        realized_asset_returns: dict[str, float],
        observed_cost_ratio: float,
        observed_turnover: float,
        pre_nav: float,
        nav_points: list[tuple[pd.Timestamp, float]],
        candidate_decision: dict[str, Any],
        stress_signal: float,
        benchmark_return: float | None,
        liquidity_stress: bool,
    ) -> float:
        candidate_weights, candidate_turnover = self._approximate_posture_weights(
            current_weights=current_weights,
            base_target_weights=base_target_weights,
            candidate_decision=candidate_decision,
        )
        period_return = float(
            sum(
                float(weight) * float(realized_asset_returns.get(ticker, 0.0))
                for ticker, weight in candidate_weights.items()
                if ticker != "CASH"
            )
        )
        active_return = (
            period_return - float(benchmark_return)
            if benchmark_return is not None
            else period_return
        )
        candidate_cost_ratio = self._approximate_cost_ratio(
            observed_cost_ratio=observed_cost_ratio,
            observed_turnover=observed_turnover,
            candidate_turnover=candidate_turnover,
        )
        prev_peak = max([float(nav) for _, nav in nav_points] + [float(pre_nav)], default=float(pre_nav))
        approx_end_nav = float(pre_nav) * (1.0 + period_return - candidate_cost_ratio)
        approx_drawdown = min(0.0, (approx_end_nav - prev_peak) / prev_peak) if prev_peak > 0 else 0.0
        concentration_hhi = float(
            sum(float(weight) ** 2 for ticker, weight in candidate_weights.items() if ticker != "CASH" and float(weight) > 0.0)
        )
        reduced_utility, _ = self._compose_utility(
            active_return=active_return,
            realized_drawdown=approx_drawdown,
            cost_ratio=candidate_cost_ratio,
            concentration_excess=max(0.0, concentration_hhi - float(self.engine.cfg.get("risk", {}).get("max_concentration_hhi", 0.15))),
            liquidity_stress=liquidity_stress,
            stress_signal=stress_signal,
            reduced=True,
        )
        return float(reduced_utility)

    def _counterfactual_decision(
        self,
        base_decision: dict[str, Any],
        posture: str,
    ) -> dict[str, Any]:
        decision = dict(base_decision)
        posture_controls = self._target_controls_for_posture(posture, self.engine.cfg)
        decision["posture"] = posture
        decision["cash_target"] = float(posture_controls["cash_target"])
        decision["aggressiveness"] = float(posture_controls["aggressiveness"])
        decision["turnover_cap"] = float(posture_controls["turnover_cap"])
        return decision

    @staticmethod
    def _approximate_posture_weights(
        *,
        current_weights: dict[str, float],
        base_target_weights: dict[str, float],
        candidate_decision: dict[str, Any],
    ) -> tuple[dict[str, float], float]:
        cash_target = float(candidate_decision.get("cash_target", 0.05))
        aggressiveness = float(candidate_decision.get("aggressiveness", 1.0))
        turnover_cap = float(candidate_decision.get("turnover_cap", 0.40) or 0.40)
        current = {ticker: float(weight) for ticker, weight in current_weights.items()}
        current.setdefault("CASH", 0.0)
        base_equity = {
            ticker: float(weight)
            for ticker, weight in base_target_weights.items()
            if ticker != "CASH" and not str(ticker).startswith("__") and float(weight) > 0.0
        }
        current_equity = {
            ticker: float(weight)
            for ticker, weight in current.items()
            if ticker != "CASH" and float(weight) > 0.0
        }
        base_total = sum(base_equity.values())
        current_total = sum(current_equity.values())
        if base_total <= 0:
            desired = {"CASH": 1.0}
        else:
            target_mix = {ticker: weight / base_total for ticker, weight in base_equity.items()}
            current_mix = (
                {ticker: weight / current_total for ticker, weight in current_equity.items()}
                if current_total > 0
                else target_mix
            )
            aggressive_mode = aggressiveness >= 1.0
            if aggressive_mode:
                posture_mix = target_mix
                mix_strength = float(np.clip((aggressiveness - 1.0) / 0.40, 0.0, 1.0))
            else:
                overlap = sorted(set(target_mix) | set(current_mix))
                blended = {
                    ticker: 0.65 * current_mix.get(ticker, 0.0) + 0.35 * target_mix.get(ticker, 0.0)
                    for ticker in overlap
                }
                ranked = sorted(
                    blended.items(),
                    key=lambda item: (target_mix.get(item[0], 0.0), current_mix.get(item[0], 0.0)),
                    reverse=True,
                )
                focus_ratio = float(np.clip((1.0 - aggressiveness) / 0.40, 0.0, 1.0))
                keep_count = max(1, int(round(len(ranked) * (1.0 - 0.55 * focus_ratio))))
                kept = dict(ranked[:keep_count])
                kept_total = sum(kept.values())
                posture_mix = (
                    {ticker: weight / kept_total for ticker, weight in kept.items()}
                    if kept_total > 0
                    else target_mix
                )
                mix_strength = float(np.clip(0.45 + 0.55 * focus_ratio, 0.0, 1.0))
            equity_budget = float(np.clip(1.0 - cash_target, 0.0, 1.0))
            tickers = sorted(set(posture_mix) | set(current_mix))
            desired = {
                ticker: equity_budget * (
                    (1.0 - mix_strength) * current_mix.get(ticker, 0.0)
                    + mix_strength * posture_mix.get(ticker, 0.0)
                )
                for ticker in tickers
            }
            desired["CASH"] = cash_target
        current_cash = float(current.get("CASH", 0.0))
        candidate = dict(current)
        desired_cash = float(desired.get("CASH", cash_target))
        cash_shift = float(np.clip(desired_cash - current_cash, -turnover_cap, turnover_cap))
        candidate["CASH"] = current_cash + cash_shift
        current_equity_budget = max(1.0e-12, 1.0 - current_cash)
        candidate_equity_budget = max(0.0, 1.0 - candidate["CASH"])
        for ticker in list(candidate):
            if ticker == "CASH":
                continue
            candidate[ticker] = max(0.0, candidate.get(ticker, 0.0)) * (candidate_equity_budget / current_equity_budget)
        all_tickers = sorted(set(candidate) | set(desired))
        residual_turnover_budget = max(0.0, turnover_cap - abs(cash_shift))
        raw_turnover = 0.5 * sum(abs(desired.get(ticker, 0.0) - candidate.get(ticker, 0.0)) for ticker in all_tickers)
        move_fraction = (
            1.0
            if raw_turnover <= 1e-12
            else float(np.clip(residual_turnover_budget / raw_turnover, 0.0, 1.0))
        )
        candidate = {
            ticker: candidate.get(ticker, 0.0) + move_fraction * (desired.get(ticker, 0.0) - candidate.get(ticker, 0.0))
            for ticker in all_tickers
        }
        total = sum(max(0.0, weight) for weight in candidate.values())
        if total > 0:
            candidate = {ticker: max(0.0, weight) / total for ticker, weight in candidate.items()}
        candidate_turnover = 0.5 * sum(abs(candidate.get(ticker, 0.0) - current.get(ticker, 0.0)) for ticker in all_tickers)
        return candidate, float(candidate_turnover)

    def _approximate_cost_ratio(
        self,
        *,
        observed_cost_ratio: float,
        observed_turnover: float,
        candidate_turnover: float,
    ) -> float:
        if observed_turnover > 1.0e-8:
            unit_cost = observed_cost_ratio / observed_turnover
        else:
            simulator = getattr(self.engine, "simulator", None)
            unit_cost = (
                float(getattr(simulator, "total_cost_bps", 0.0)) / 10000.0
                if simulator is not None
                else 0.0
            )
        return float(max(0.0, unit_cost) * max(0.0, candidate_turnover))

    @staticmethod
    def _soft_utility_baseline(values: np.ndarray, temperature: float) -> float:
        if values.size == 0:
            return 0.0
        temp = max(float(temperature), 1.0e-6)
        scaled = values / temp
        max_scaled = float(np.max(scaled))
        return float(temp * (np.log(np.sum(np.exp(scaled - max_scaled))) + max_scaled))

    @staticmethod
    def _stress_bucket_label(stress_signal: float, cfg: dict | None = None) -> str:
        rl_cfg = (cfg or {}).get("rl", {}) if isinstance(cfg, dict) else {}
        moderate_threshold = float(rl_cfg.get("stress_target_moderate", 0.18))
        high_threshold = float(rl_cfg.get("stress_target_high", 0.35))
        if stress_signal >= high_threshold:
            return "high"
        if stress_signal >= moderate_threshold:
            return "medium"
        return "low"

    @staticmethod
    def _copy_portfolio_state(portfolio: PortfolioState) -> PortfolioState:
        return PortfolioState(
            date=portfolio.date,
            cash=float(portfolio.cash),
            holdings=dict(portfolio.holdings),
            weights=dict(portfolio.weights),
            nav=float(portfolio.nav),
            sector_weights=dict(portfolio.sector_weights),
        )

    @staticmethod
    def _realized_asset_returns(
        current_prices: pd.Series,
        next_prices: pd.Series,
    ) -> dict[str, float]:
        returns: dict[str, float] = {}
        tickers = set(current_prices.index) | set(next_prices.index)
        for ticker in tickers:
            start = float(current_prices.get(ticker, np.nan))
            end = float(next_prices.get(ticker, np.nan))
            if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
                continue
            returns[str(ticker)] = float((end - start) / start)
        return returns

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
        optimizer_cfg = (cfg or {}).get("optimizer", {}) if isinstance(cfg, dict) else {}
        profiles = rl_cfg.get("posture_profiles", {}) if isinstance(rl_cfg, dict) else {}
        defaults = {
            "risk_off": {
                "cash_target": float(rl_cfg.get("target_cash_high", 0.35)),
                "aggressiveness": float(rl_cfg.get("target_aggressiveness_high", 0.75)),
                "turnover_cap": float(rl_cfg.get("target_turnover_cap_high", 0.15)),
            },
            "neutral": {
                "cash_target": 0.05,
                "aggressiveness": 1.0,
                "turnover_cap": 0.35,
            },
            "risk_on": {
                "cash_target": float(rl_cfg.get("target_cash_risk_on", 0.02)),
                "aggressiveness": float(rl_cfg.get("target_aggressiveness_risk_on", 1.30)),
                "turnover_cap": float(rl_cfg.get("target_turnover_cap_risk_on", 0.45)),
            },
        }
        profile = defaults.get(posture, defaults["neutral"]).copy()
        configured = profiles.get(posture, {}) if isinstance(profiles, dict) else {}
        if isinstance(configured, dict):
            profile.update({k: float(v) for k, v in configured.items() if k in profile})
        agg_min = float(rl_cfg.get("aggressiveness_min", 0.60))
        agg_max = float(rl_cfg.get("aggressiveness_max", 1.40))
        if agg_min > agg_max:
            agg_min, agg_max = agg_max, agg_min
        max_cash = float(optimizer_cfg.get("max_cash", 0.40))
        max_turnover = float(optimizer_cfg.get("max_turnover_per_rebalance", 0.45))
        profile["cash_target"] = float(np.clip(profile["cash_target"], 0.0, max_cash))
        profile["aggressiveness"] = float(np.clip(profile["aggressiveness"], agg_min, agg_max))
        profile["turnover_cap"] = float(np.clip(profile["turnover_cap"], 0.05, max_turnover))
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
            "risk_off": {"cash_target": 0.35, "aggressiveness": 0.75, "turnover_cap": 0.15},
            "neutral": {"cash_target": 0.05, "aggressiveness": 1.0, "turnover_cap": 0.35},
            "risk_on": {"cash_target": 0.02, "aggressiveness": 1.30, "turnover_cap": 0.45},
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
            max_cash = float(self.engine.cfg.get("optimizer", {}).get("max_cash", 0.40))
            decision["cash_target"] = float(np.clip(rl_decision["cash_target"], 0.0, max_cash))
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
