"""
Walk-forward training and simulation engine.

The core loop:
    For each 4-week rebalance date from 2013→2026:
        1. Build training data [backtest_start, rebalance_date)
        2. Train/refresh sector scorer + stock ranker (if needed)
        3. Score sectors → RL/rule-based tilts → target sector weights
        4. Rank stocks within chosen sectors
        5. Portfolio optimizer → target weights
        6. Risk engine → final weights
        7. Execute trades with costs
        8. Hold for 4 weeks, compute daily NAV
        9. Record outcome
        10. Feed outcome to RL buffer
    End: compute full performance metrics
"""
from __future__ import annotations

import logging
import pickle
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtest.simulator import ExecutionResult, PortfolioSimulator
from src.config import load_config
from src.data.contracts import PortfolioState, RebalanceRecord
from src.data.macro import MacroDataManager
from src.data.universe import UniverseManager
from src.features.feature_store import FeatureStore
from src.features.base import fill_price_gaps
from src.features.macro_features import MacroFeatureBuilder
from src.features.portfolio_features import compute_portfolio_features
from src.features.sector_features import SectorFeatureBuilder
from src.features.stock_features import StockFeatureBuilder
from src.models.sector_scorer import SectorScorer
from src.models.stock_ranker import StockRanker
from src.optimizer.portfolio_optimizer import PortfolioOptimizer
from src.rl.agent import RLSectorAgent
from src.rl.contract import CAUSAL_TRAINING_BACKEND
from src.rl.policy_utils import apply_posture_policy, posture_selection_profile
from src.rl.environment import SectorAllocationEnv, SECTORS
from src.rl.historical_executor import HistoricalPeriodExecutor
from src.rl.policy_utils import build_control_context
from src.rl.retrain_triggers import EventDetector
from src.risk.risk_engine import RiskEngine

logger = logging.getLogger(__name__)


class WalkForwardEngine:
    """
    Orchestrates the full 10-year walk-forward backtest.
    """

    def __init__(
        self,
        price_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame,
        macro_df: pd.DataFrame,
        cfg: dict | None = None,
        use_rl: bool | None = None,
        mode: str = "full_rl",
    ):
        self.cfg = cfg or load_config()
        self.price_matrix = price_matrix
        self.volume_matrix = volume_matrix
        self.macro_df = macro_df
        valid_modes = {"selection_only", "optimizer_only", "full_rl"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported backtest mode: {mode}")
        self.mode = mode
        self.use_rl = (
            use_rl if use_rl is not None else self.cfg["rl"].get("use_rl", True)
        ) and self.mode == "full_rl"

        bt_cfg = self.cfg["backtest"]
        self.initial_capital = bt_cfg["initial_capital"]
        self.rebalance_weeks = bt_cfg["rebalance_freq_weeks"]
        self.min_train_years = bt_cfg["min_train_years"]
        self.tc_bps = bt_cfg["transaction_cost_bps"]
        self.start_date = pd.Timestamp(bt_cfg["start_date"])
        end_str = bt_cfg["end_date"]
        self.end_date = pd.Timestamp(end_str) if end_str != "latest" else pd.Timestamp.today()

        # Managers
        self.universe_mgr = UniverseManager(self.cfg)
        self.macro_mgr = MacroDataManager(self.cfg)
        self.macro_fb = MacroFeatureBuilder(self.cfg)
        self.sector_fb = SectorFeatureBuilder(self.cfg)
        self.stock_fb = StockFeatureBuilder(self.cfg)
        self.optimizer = PortfolioOptimizer(self.cfg)
        self.risk_engine = RiskEngine(self.cfg)
        self.simulator = PortfolioSimulator(self.cfg)
        self.sector_fwd_window_days = int(
            self.cfg["sector_model"].get("fwd_window_days", 28)
        )
        self.stock_fwd_window_days = int(self.cfg["stock_model"].get("fwd_window_days", 56))

        # Precompute macro features
        logger.info("Building macro features ...")
        self.macro_features = self.macro_fb.build(macro_df)

        # Feature store (compute-once, partitioned parquet)
        store_dir = Path(self.cfg["paths"]["artifact_dir"]) / "feature_store"
        self.feature_store = FeatureStore(store_dir, self.cfg)

        # Models (trained incrementally)
        self.sector_scorer = SectorScorer(self.cfg)
        self.stock_ranker = StockRanker(self.cfg)
        self.rl_agent = RLSectorAgent(self.cfg)
        self.rl_training_backend = str(
            self.cfg.get("rl", {}).get("training_backend", "disabled")
        ).strip()
        self.rl_model_enabled = (
            self.use_rl and self.rl_training_backend == CAUSAL_TRAINING_BACKEND
        )

        # Adaptive retrain trigger detector
        self.event_detector = EventDetector(self.cfg)

        # Results
        self.rebalance_records: list[RebalanceRecord] = []
        self.nav_series: pd.Series = pd.Series(dtype=float)
        self.daily_log: list[dict] = []
        self.selection_diagnostics: list[dict] = []
        self._trigger_log: list[dict] = []  # records every event that fired
        self._rl_retrain_count: int = 0
        self._recent_turnovers: list[float] = []
        self._recent_cost_ratios: list[float] = []

    # ── Main simulation ───────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute the full walk-forward backtest. Returns performance metrics."""
        logger.info("=" * 70)
        logger.info("Starting walk-forward backtest")
        logger.info("Period: %s → %s", self.start_date.date(), self.end_date.date())
        logger.info("Initial capital: INR %.0f", self.initial_capital)
        logger.info("Mode: %s", self.mode)
        logger.info("RL overlay: %s", "ENABLED" if self.use_rl else "DISABLED")
        logger.info("RL training backend: %s", self.rl_training_backend)
        if self.use_rl and not self.rl_model_enabled:
            logger.info("RL model execution disabled until a causal backend is configured")
        logger.info("=" * 70)

        # ── Feature store: compute-once backfill (skipped on re-runs) ────────────
        logger.info("Checking feature store ...")
        self.feature_store.build_or_append(
            price_matrix=self.price_matrix,
            volume_matrix=self.volume_matrix,
            macro_df=None,
            macro_features_df=self.macro_features,
            sector_fb=self.sector_fb,
            stock_fb=self.stock_fb,
            universe_mgr=self.universe_mgr,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        logger.info("Feature store ready.")

        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates()
        if not rebalance_dates:
            raise ValueError("No rebalance dates generated")

        # Initial portfolio state
        first_rebal = rebalance_dates[0]
        portfolio = PortfolioState(
            date=first_rebal.date(),
            cash=float(self.initial_capital),
            holdings={},
            weights={"CASH": 1.0},
            nav=float(self.initial_capital),
            sector_weights={},
        )
        self.risk_engine.update(portfolio.nav, portfolio.date)

        nav_points: list[tuple] = [(first_rebal, float(self.initial_capital))]
        portfolio_returns: list[float] = []
        self._recent_turnovers = []
        self._recent_cost_ratios = []

        # Precompute benchmark return series
        bm_ticker = self.cfg["backtest"].get("benchmark_ticker", "^NSEI")
        bm_prices = (
            self.price_matrix[bm_ticker]
            if bm_ticker in self.price_matrix.columns
            else None
        )

        logger.info("Running %d rebalance periods ...", len(rebalance_dates) - 1)

        for i in tqdm(range(len(rebalance_dates) - 1), desc="Walk-forward"):
            current_date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]
            _t_period_start = time.perf_counter()

            # ── A. Train/refresh models ───────────────────────────────────────
            if self._should_retrain_models(i, current_date):
                self._train_models(current_date, idx=i)

            if self.rl_model_enabled and self._should_retrain_rl(i):
                self._train_rl(current_idx=i)

            # ── B. Get current universe ───────────────────────────────────────
            _t = time.perf_counter()
            prices_today = self._get_prices(current_date)
            if prices_today.empty:
                logger.warning("No prices on %s; skipping rebalance", current_date.date())
                continue

            snapshot = self.universe_mgr.get_universe(
                current_date.date(),
                price_matrix=self.price_matrix,
                volume_matrix=self.volume_matrix,
            )
            sector_map = self.universe_mgr.get_sector_map(snapshot)
            cap_map = {s.ticker: s.cap for s in snapshot.stocks}
            logger.debug("  [timing] universe+prices     → %.3fs", time.perf_counter() - _t)

            # ── C. Build current features ─────────────────────────────────────
            _t = time.perf_counter()
            macro_now = self._get_macro_features(current_date)
            sector_feats_now = self._get_sector_features_now(
                current_date, snapshot, prices_today, bm_prices
            )
            stock_feats_now = self._get_stock_features_now(
                current_date, snapshot, prices_today, bm_prices
            )
            logger.debug("  [timing] feature snapshots   → %.3fs", time.perf_counter() - _t)

            # ── D. Score sectors ──────────────────────────────────────────────
            _t = time.perf_counter()
            sector_scores = self.sector_scorer.predict(sector_feats_now, macro_now)
            logger.debug("  [timing] sector_scorer.pred  → %.3fs", time.perf_counter() - _t)

            # ── E. RL / rule-based decisions ──────────────────────────────────
            recent_rets = self._get_recent_portfolio_returns(nav_points, current_date)
            bm_rets_recent = bm_prices.pct_change() if bm_prices is not None else None
            risk_regime = self.risk_engine.regime(recent_rets)

            # ── F. Risk evaluation ────────────────────────────────────────────
            risk_signal, risk_action = self.risk_engine.evaluate(
                portfolio,
                recent_rets,
                macro_features=macro_now if isinstance(macro_now, pd.Series) else pd.Series(macro_now),
                volume_matrix=self.volume_matrix,
            )
            port_feats = compute_portfolio_features(
                portfolio,
                recent_rets,
                bm_rets_recent,
                control_context=build_control_context(
                    sector_feats_now,
                    risk_signal=risk_signal,
                    risk_action=risk_action,
                    recent_turnovers=self._recent_turnovers,
                    recent_cost_ratios=self._recent_cost_ratios,
                ),
            )

            sector_state = self._build_sector_state(sector_feats_now)

            transition_state = {
                "macro_state": macro_now.to_dict() if isinstance(macro_now, pd.Series) else dict(macro_now),
                "sector_state": sector_state,
                "portfolio_state": dict(port_feats),
            }

            if self.rl_model_enabled and self.rl_agent.is_trained:
                rl_decision = self.rl_agent.decide(
                    macro_state=transition_state["macro_state"],
                    sector_state=sector_state,
                    portfolio_state=port_feats,
                    prev_realized_sector_weights=portfolio.sector_weights,
                )
            else:
                if self.mode == "full_rl":
                    rl_decision = RLSectorAgent.rule_based_action(
                        sector_scores,
                        macro_now.to_dict() if isinstance(macro_now, pd.Series) else {},
                        risk_regime=risk_regime,
                    )
                else:
                    rl_decision = self._default_decision(snapshot.sectors)
            rl_decision = apply_posture_policy(self.cfg, rl_decision)

            # Keep selection_only as a pure selection signal; other modes still honor risk cash floors.
            if self.mode == "selection_only":
                cash_target = 0.0
            else:
                cash_target = max(rl_decision.get("cash_target", 0.05), risk_action.cash_floor)

            # ── G. Select stocks ──────────────────────────────────────────────
            top_k = self.cfg["stock_model"]["top_k_per_sector"]
            alpha_scores: dict[str, float] = {}
            selected_stock_rows: list[dict[str, object]] = []
            selected_sectors = self._select_sectors(snapshot.sectors, sector_scores, rl_decision)
            posture = str(rl_decision.get("posture", "neutral"))
            selection_profile = posture_selection_profile(self.cfg, posture)
            top_k = int(selection_profile.get("stock_top_k_per_sector") or top_k)
            for sector in selected_sectors:
                ranking = self.stock_ranker.rank_stocks(
                    stock_feats_now, sector, top_k=top_k
                )
                for _, row in ranking.iterrows():
                    ticker = row["ticker"]
                    raw_score = float(row["score"])
                    alpha_scores[ticker] = raw_score  # tilt applied once in optimizer
                    selected_stock_rows.append(
                        {"ticker": ticker, "sector": sector, "score": raw_score}
                    )

            # fallback: if no alpha scores, use sector-momentum top stocks
            if not alpha_scores and not stock_feats_now.empty:
                for t in snapshot.tickers[:30]:
                    if t in prices_today.index:
                        alpha_scores[t] = 0.5
                        selected_stock_rows.append(
                            {"ticker": t, "sector": sector_map.get(t, "Unknown"), "score": 0.5}
                        )

            # ── H. Optimize portfolio ─────────────────────────────────────────
            optimizer_diagnostics = {}
            if self.mode == "selection_only":
                target_weights = self._build_equal_weight_targets(
                    alpha_scores=alpha_scores,
                    cash_target=cash_target,
                )
                pre_risk_target_weights = dict(target_weights)
            else:
                cov_matrix = PortfolioOptimizer.estimate_covariance(
                    self.price_matrix.loc[:current_date],
                    list(alpha_scores.keys()),
                )
                target_weights = self.optimizer.optimize(
                    alpha_scores=alpha_scores,
                    cov_matrix=cov_matrix,
                    sector_map=sector_map,
                    current_weights=portfolio.weights,
                    sector_tilts=(
                        rl_decision["sector_tilts"]
                        if self.mode == "full_rl"
                        else {sector: 1.0 for sector in snapshot.sectors}
                    ),
                    aggressiveness=(
                        rl_decision.get("aggressiveness", 1.0)
                        if self.mode == "full_rl"
                        else 1.0
                    ),
                    cash_target=cash_target,
                    max_turnover_override=(
                        float(rl_decision.get("turnover_cap"))
                        if rl_decision.get("turnover_cap") is not None
                        else None
                    ),
                    forced_exclude=risk_action.exclude_tickers,
                    posture=str(rl_decision.get("posture", "neutral")),
                )
                pre_risk_target_weights = dict(target_weights)
                optimizer_diagnostics = dict(getattr(self.optimizer, "last_optimize_diagnostics", {}) or {})

            # ── I. Pre-trade risk checks ──────────────────────────────────────
            if self.mode != "selection_only":
                target_weights = self.risk_engine.check_pre_trade(
                    target_weights, sector_map, cap_map, risk_action
                )

            # ── J. Execute trades ─────────────────────────────────────────────
            # Mark-to-market before trade to get accurate period return
            mtm_value = sum(
                portfolio.holdings.get(t, 0) * float(prices_today.get(t, 0) or 0)
                for t in portfolio.holdings
                if t in prices_today.index and np.isfinite(float(prices_today.get(t, 0) or 0))
            ) + portfolio.cash
            pre_nav = max(mtm_value, portfolio.nav) if np.isfinite(mtm_value) and mtm_value > 0 else portfolio.nav
            portfolio_mtm = self.simulator.value_portfolio(
                portfolio, prices_today, current_date.date()
            )
            exec_result = self.simulator.execute_rebalance(
                target_weights, portfolio_mtm, prices_today, current_date.date()
            )
            portfolio = exec_result.new_portfolio
            self._recent_turnovers.append(float(exec_result.total_turnover))
            if pre_nav > 0:
                self._recent_cost_ratios.append(float(exec_result.total_cost) / float(pre_nav))

            # Update sector weights
            sw: dict[str, float] = {}
            for t, w in portfolio.weights.items():
                if t != "CASH":
                    sec = sector_map.get(t, "Unknown")
                    sw[sec] = sw.get(sec, 0) + w
            portfolio = PortfolioState(
                date=portfolio.date,
                cash=portfolio.cash,
                holdings=portfolio.holdings,
                weights=portfolio.weights,
                nav=portfolio.nav,
                sector_weights=sw,
            )

            self.risk_engine.update(portfolio.nav, portfolio.date)
            nav_points.append((current_date, portfolio.nav))

            # ── K. Compute daily NAV for the period ───────────────────────────
            period_nav = self._compute_period_nav(
                portfolio, prices_today, current_date, next_date
            )
            nav_points.extend([(ts, v) for ts, v in period_nav if np.isfinite(v) and v > 0])

            # Period return
            end_nav = period_nav[-1][1] if period_nav else portfolio.nav
            if pre_nav > 0 and np.isfinite(end_nav) and np.isfinite(pre_nav):
                period_return = (end_nav - pre_nav) / pre_nav
            else:
                period_return = 0.0
            portfolio_returns.append(period_return)

            # ── L. Record rebalance ───────────────────────────────────────────
            rec = RebalanceRecord(
                rebalance_date=current_date.date(),
                pre_nav=pre_nav,
                post_nav=portfolio.nav,
                trades=exec_result.trades,
                target_weights=target_weights,
                sector_tilts=rl_decision["sector_tilts"],
                cash_target=cash_target,
                aggressiveness=rl_decision.get("aggressiveness", 1.0),
                posture=str(rl_decision.get("posture", "neutral")),
                selected_sector_count=len(selected_sectors),
                selected_stock_count=len(selected_stock_rows),
                turnover_cap=rl_decision.get("turnover_cap"),
                total_turnover=exec_result.total_turnover,
                total_cost=exec_result.total_cost,
                rl_action=rl_decision,
                emergency=risk_signal.emergency_rebalance,
            )
            self.rebalance_records.append(rec)
            selected_forward_returns = self._selected_forward_returns(
                selected_stock_rows,
                prices_today,
                next_date,
                period_nav,
            )
            universe_forward_returns = {
                ticker: ret
                for ticker, ret in (
                    self._forward_returns_for_tickers(
                        snapshot.tickers,
                        prices_today,
                        next_date,
                    ).items()
                )
                if np.isfinite(ret)
            }
            sector_forward_returns: dict[str, dict[str, float]] = {}
            for sector in snapshot.sectors:
                sector_tickers = [
                    ticker
                    for ticker in snapshot.tickers
                    if sector_map.get(ticker, "Unknown") == sector
                ]
                if not sector_tickers:
                    continue
                sector_forward_returns[sector] = self._forward_returns_for_tickers(
                    sector_tickers,
                    prices_today,
                    next_date,
                )

            selected_returns = list(selected_forward_returns.values())
            universe_returns = [
                ret for ret in universe_forward_returns.values() if np.isfinite(ret)
            ]
            top_k_avg = float(np.mean(selected_returns)) if selected_returns else None
            universe_avg = float(np.mean(universe_returns)) if universe_returns else None
            sector_medians: list[float] = []
            for row in selected_stock_rows:
                sector = str(row["sector"])
                sector_rets = [
                    ret
                    for ret in sector_forward_returns.get(sector, {}).values()
                    if np.isfinite(ret)
                ]
                if sector_rets:
                    sector_medians.append(float(np.median(sector_rets)))
            self.selection_diagnostics.append(
                {
                    "rebalance_date": current_date.date(),
                    "mode": self.mode,
                    "selected_sectors": selected_sectors,
                    "selected_stocks": [
                        {"ticker": str(row["ticker"]), "sector": str(row["sector"])}
                        for row in selected_stock_rows
                    ],
                    "candidate_stock_scores": {
                        str(ticker): float(score) for ticker, score in alpha_scores.items()
                    },
                    "candidate_stock_sectors": {
                        str(ticker): str(sector_map.get(ticker, "Unknown"))
                        for ticker in alpha_scores
                    },
                    "raw_stock_scores": {
                        str(row["ticker"]): float(row["score"]) for row in selected_stock_rows
                    },
                    "optimized_weights_before_rl": pre_risk_target_weights,
                    "final_weights_after_rl": target_weights,
                    "cash_pct": float(target_weights.get("CASH", 0.0)),
                    "turnover": float(exec_result.total_turnover),
                    "next_period_returns": selected_forward_returns,
                    "universe_forward_returns": universe_forward_returns,
                    "sector_forward_returns": sector_forward_returns,
                    "top_k_avg_forward_return": top_k_avg,
                    "top_k_minus_universe_forward_return": (
                        top_k_avg - universe_avg
                        if top_k_avg is not None and universe_avg is not None
                        else None
                    ),
                    "top_k_minus_sector_median_forward_return": (
                        top_k_avg - float(np.mean(sector_medians))
                        if top_k_avg is not None and sector_medians
                        else None
                    ),
                    "precision_at_k": (
                        float(np.mean([ret > 0 for ret in selected_returns]))
                        if selected_returns
                        else None
                    ),
                    "rank_ic": self._rank_ic(
                        {ticker: alpha_scores[ticker] for ticker in alpha_scores},
                        universe_forward_returns,
                    ),
                    "stability": self._selection_stability(selected_stock_rows),
                }
            )

            # ── N. Per-period detailed progress print ─────────────────────────
            total_periods = len(rebalance_dates) - 1
            pct_done = (i + 1) / total_periods * 100
            current_nav = end_nav if np.isfinite(end_nav) else portfolio.nav
            total_return = (current_nav - self.initial_capital) / self.initial_capital * 100 if np.isfinite(current_nav) else 0.0
            # Rolling Sharpe (last 12 periods)
            if len(portfolio_returns) >= 4:
                r_arr = np.array([r for r in portfolio_returns[-12:] if np.isfinite(r)])
                roll_sharpe = (r_arr.mean() / r_arr.std()) * (13 ** 0.5) if len(r_arr) >= 2 and r_arr.std() > 1e-9 else 0.0
            else:
                roll_sharpe = 0.0
            dd = self.risk_engine.current_drawdown() if hasattr(self.risk_engine, "current_drawdown") else 0.0
            cash_pct = portfolio.weights.get("CASH", 0.0) * 100
            free_cash_inr = portfolio.cash
            n_stocks = len([t for t in portfolio.holdings if portfolio.holdings[t] > 1e-6])
            turnover_pct = exec_result.total_turnover * 100
            cost_inr = exec_result.total_cost

            # RL tilt summary: sectors tilted above/below neutral
            tilts = rl_decision.get("sector_tilts", {})
            rl_overweights = sorted([(s, t) for s, t in tilts.items() if t > 1.1], key=lambda x: -x[1])[:3]
            rl_underweights = sorted([(s, t) for s, t in tilts.items() if t < 0.9], key=lambda x: x[1])[:3]
            if self.mode == "selection_only":
                rl_mode = "Selection"
            elif self.rl_model_enabled and self.rl_agent.is_trained:
                rl_mode = "RL"
            else:
                rl_mode = "Rule"
            aggressiveness = rl_decision.get("aggressiveness", 1.0)

            # Top holdings by weight
            stock_weights = {t: w for t, w in portfolio.weights.items() if t != "CASH" and w > 0}
            top_holdings = sorted(stock_weights.items(), key=lambda x: -x[1])[:5]

            # Sector distribution (all active sectors)
            sector_line = "  ".join(
                f"{s[:4]}:{v*100:.0f}%"
                for s, v in sorted(sw.items(), key=lambda x: -x[1])
                if v > 0.01
            ) if sw else "ALL CASH"

            # Period-over-period sector changes (vs previous rebalance)
            prev_rec = self.rebalance_records[-2] if len(self.rebalance_records) >= 2 else None
            if prev_rec:
                prev_sw = {s: w for s, w in prev_rec.target_weights.items()
                           if s != "CASH" and s in sector_map.values()}
                # recompute prev sector weights from prev target_weights + sector_map
                prev_sec_w: dict[str, float] = {}
                for tk, wt in prev_rec.target_weights.items():
                    if tk != "CASH":
                        sec = sector_map.get(tk, "Unknown")
                        prev_sec_w[sec] = prev_sec_w.get(sec, 0) + wt
                sec_changes = []
                for sec, cur_w in sw.items():
                    prev_w = prev_sec_w.get(sec, 0)
                    delta = cur_w - prev_w
                    if abs(delta) > 0.02:
                        sec_changes.append(f"{sec[:4]}{delta*100:+.0f}%")
                sec_change_str = " ".join(sec_changes) if sec_changes else "stable"
            else:
                sec_change_str = "initial"

            sep = "─" * 78
            print(f"\n{sep}", flush=True)
            print(
                f"  Period [{i+1:3d}/{total_periods}]  {str(current_date.date())} → {str(next_date.date())}  |  "
                f"{pct_done:.0f}% complete  |  Mode: {rl_mode}  |  Aggressiveness: {aggressiveness:.2f}",
                flush=True,
            )
            print(
                f"  NAV: ₹{current_nav:>10,.0f}  |  Period: {period_return*100:+.2f}%  |  "
                f"Total: {total_return:+.1f}%  |  DD: {dd*100:.1f}%  |  Roll.Sharpe: {roll_sharpe:.2f}",
                flush=True,
            )
            print(
                f"  Stocks: {n_stocks}  |  Cash: {cash_pct:.1f}%  (₹{free_cash_inr:,.0f})  |  "
                f"Turnover: {turnover_pct:.1f}%  |  TC: ₹{cost_inr:,.0f}",
                flush=True,
            )
            print(f"  Sectors  → {sector_line}", flush=True)
            print(f"  Δ Sector → {sec_change_str}", flush=True)
            if top_holdings:
                holdings_str = "  ".join(f"{t}:{w*100:.1f}%" for t, w in top_holdings)
                print(f"  Top5    → {holdings_str}", flush=True)
            if rl_overweights or rl_underweights:
                ow_str = " ".join(f"{s}×{t:.1f}" for s, t in rl_overweights) or "—"
                uw_str = " ".join(f"{s}×{t:.1f}" for s, t in rl_underweights) or "—"
                print(f"  RL↑     → {ow_str}  |  RL↓ → {uw_str}", flush=True)
            if exec_result.trades:
                buys = [tr for tr in exec_result.trades if tr.direction == "buy"]
                sells = [tr for tr in exec_result.trades if tr.direction == "sell"]
                print(
                    f"  Trades  → {len(buys)} buys  {len(sells)} sells  "
                    f"(largest buy: {max((tr.gross_value for tr in buys), default=0):,.0f}  "
                    f"largest sell: {max((tr.gross_value for tr in sells), default=0):,.0f})",
                    flush=True,
                )
            if risk_signal.emergency_rebalance:
                print(f"  ⚠ EMERGENCY REBALANCE — hard drawdown breach", flush=True)
            print(f"  [timing] Period total        → {time.perf_counter() - _t_period_start:.2f}s", flush=True)

            # ── M. Record RL experience ───────────────────────────────────────
            if self.use_rl:
                transition_info = {
                    "date": str(current_date.date()),
                    "mode": rl_mode.lower(),
                    "target_weights": {ticker: float(weight) for ticker, weight in target_weights.items()},
                    "pre_risk_target_weights": {
                        ticker: float(weight)
                        for ticker, weight in pre_risk_target_weights.items()
                    },
                    "realized_sector_weights": {sector: float(weight) for sector, weight in sw.items()},
                    "total_cost": float(exec_result.total_cost),
                    "turnover": float(exec_result.total_turnover),
                    "cash_target": float(cash_target),
                    "turnover_cap": (
                        float(rl_decision.get("turnover_cap"))
                        if rl_decision.get("turnover_cap") is not None
                        else None
                    ),
                    "posture": str(rl_decision.get("posture", "neutral")),
                    "optimizer_diagnostics": optimizer_diagnostics if self.mode != "selection_only" else {},
                    "optimizer_reason_code": (
                        str(optimizer_diagnostics.get("status", "selection_only"))
                        if self.mode != "selection_only"
                        else "selection_only"
                    ),
                    "optimizer_fallback_mode": (
                        str(optimizer_diagnostics.get("fallback_mode", "none"))
                        if self.mode != "selection_only"
                        else "none"
                    ),
                    "selected_sectors": [str(sector) for sector in selected_sectors],
                }
                outcome = {
                    "portfolio_return": float(period_return),
                    "max_drawdown_episode": float(risk_signal.drawdown),
                    "turnover": float(exec_result.total_turnover),
                    "concentration_hhi": float(risk_signal.hhi),
                    "liquidity_stress": float(risk_signal.liquidity_stress),
                    "realized_sector_weights": dict(sw),
                }
                next_state = self._build_next_rl_state(
                    next_date=next_date,
                    portfolio=portfolio,
                    nav_points=nav_points,
                )
                exp_step = {
                    "date": str(current_date.date()),
                    "state": transition_state,
                    "action": {
                        "sector_tilts": {
                            sector: float(weight)
                            for sector, weight in rl_decision.get("sector_tilts", {}).items()
                        },
                        "cash_target": float(rl_decision.get("cash_target", cash_target)),
                        "aggressiveness": float(rl_decision.get("aggressiveness", 1.0)),
                        "turnover_cap": (
                            float(rl_decision.get("turnover_cap"))
                            if rl_decision.get("turnover_cap") is not None
                            else None
                        ),
                        "posture": str(rl_decision.get("posture", "neutral")),
                    },
                    "reward": float(period_return),
                    "next_state": next_state,
                    "done": bool(i == len(rebalance_dates) - 2),
                    "info": transition_info,
                    # Legacy fields kept temporarily for backward compatibility.
                    "macro_state": transition_state["macro_state"],
                    "sector_state": transition_state["sector_state"],
                    "portfolio_state": transition_state["portfolio_state"],
                    "outcome": outcome,
                }
                self.rl_agent.record_step(exp_step)

            # ── N. Adaptive retrain trigger check ─────────────────────────────
            trigger_events = self.event_detector.update({
                "portfolio_return":      period_return,
                "max_drawdown_episode":  risk_signal.drawdown,
                "reward":                period_return,
                "macro_state":           macro_now.to_dict() if isinstance(macro_now, pd.Series) else macro_now,
                "risk_regime":           risk_regime,
                "sector_scores":         sector_scores,
            })
            if trigger_events:
                for ev in trigger_events:
                    self._trigger_log.append({
                        "date": str(current_date.date()),
                        "tier": ev.tier, "name": ev.name,
                        "severity": ev.severity, "reason": ev.reason,
                    })
                # Tier 1 triggers always force retrain; Tier 2/3 only if enough history.
                highest_tier = min(ev.tier for ev in trigger_events)
                current_idx = i
                min_history = int(self.cfg["rl"].get("min_history_rebalances", 3))
                if self.rl_model_enabled and current_idx >= min_history and (
                    highest_tier == 1 or (highest_tier <= 2 and self._should_retrain_rl(i))
                ):
                    logger.info(
                        "Event-triggered RL retrain at %s: %s",
                        current_date.date(),
                        "; ".join(f"[T{e.tier}] {e.name}" for e in trigger_events),
                    )
                    self._train_rl(current_idx=i)
                    self.event_detector.notify_retrained()

        # ── Build final NAV series ─────────────────────────────────────────────
        all_dates, all_navs = zip(*nav_points) if nav_points else ([], [])
        self.nav_series = pd.Series(
            list(all_navs),
            index=pd.DatetimeIndex(list(all_dates)),
            name="portfolio_nav",
        )
        self.nav_series = self.nav_series[~self.nav_series.index.duplicated(keep="last")]
        self.nav_series.sort_index(inplace=True)

        # ── Compute metrics ───────────────────────────────────────────────────
        metrics = self.simulator.compute_metrics(self.nav_series, bm_prices)
        metrics["total_rebalances"] = len(self.rebalance_records)
        metrics["avg_turnover"] = (
            np.mean([r.total_turnover for r in self.rebalance_records])
            if self.rebalance_records else 0.0
        )
        metrics["initial_capital"] = self.initial_capital
        metrics["n_rl_retrains"] = self._rl_retrain_count
        metrics["n_trigger_events"] = len(self._trigger_log)

        # Sector contributions
        metrics["sector_contributions"] = self._compute_sector_contributions()

        logger.info("=" * 70)
        logger.info("BACKTEST COMPLETE")
        logger.info("CAGR:         %.2f%%", metrics.get("cagr", 0) * 100)
        logger.info("Sharpe:       %.2f", metrics.get("sharpe", 0))
        logger.info("Max Drawdown: %.2f%%", metrics.get("max_drawdown", 0) * 100)
        logger.info("Final NAV:    INR %.0f", metrics.get("final_nav", 0))
        logger.info("=" * 70)

        return metrics

    # ── Model training ────────────────────────────────────────────────────────

    def _train_models(self, as_of: pd.Timestamp, idx: int = -1) -> None:
        t0_total = time.perf_counter()
        train_sector = self._should_retrain_sector_scorer(idx)
        train_stock  = self._should_retrain_stock_ranker(idx)
        logger.info("Training models as of %s  [sector=%s  stock=%s] ...",
                    as_of.date(), train_sector, train_stock)
        sector_lookback = self.cfg["sector_model"].get("train_lookback_years", 3)
        stock_lookback  = self.cfg["stock_model"].get("train_lookback_years", 2)
        train_start        = as_of - pd.DateOffset(years=sector_lookback)
        stock_train_start  = as_of - pd.DateOffset(years=stock_lookback)

        # ── Read pre-computed features from the store (no recomputation) ──────
        t0 = time.perf_counter()
        sector_feats = self.feature_store.load("sector", train_start, as_of)
        stock_feats  = self.feature_store.load("stock",  stock_train_start, as_of)
        logger.info("  [timing] feature_store.load  → %.2fs  (sector=%d rows, stock=%d rows)",
                    time.perf_counter() - t0, len(sector_feats), len(stock_feats))

        # Fallback: build on-the-fly if store is empty (first run before store exists)
        if sector_feats.empty or stock_feats.empty:
            train_prices = self.price_matrix.loc[
                (self.price_matrix.index >= train_start) &
                (self.price_matrix.index < as_of)
            ]
            snapshot = self.universe_mgr.get_universe(as_of.date(), price_matrix=train_prices)
            sector_map = self.universe_mgr.get_sector_map(snapshot)
            bm_ticker = self.cfg["backtest"].get("benchmark_ticker", "^NSEI")
            bm_prices = train_prices[bm_ticker] if bm_ticker in train_prices.columns else None
            if sector_feats.empty:
                sector_feats = self.sector_fb.build(train_prices, sector_map, None, bm_prices)
            if stock_feats.empty:
                stock_prices = self.price_matrix.loc[
                    (self.price_matrix.index >= stock_train_start) &
                    (self.price_matrix.index < as_of)
                ]
                stock_vol = self.volume_matrix.loc[
                    (self.volume_matrix.index >= stock_train_start) &
                    (self.volume_matrix.index < as_of)
                ]
                stock_feats = self.stock_fb.build(
                    stock_prices, stock_vol if not stock_vol.empty else None, sector_map, bm_prices
                )

        # Reconstruct sector_map and snapshot needed for labels and ranker
        train_prices = self.price_matrix.loc[
            (self.price_matrix.index >= train_start) &
            (self.price_matrix.index < as_of)
        ]
        train_volumes = self.volume_matrix.loc[
            (self.volume_matrix.index >= train_start) &
            (self.volume_matrix.index < as_of)
        ]
        membership = self.universe_mgr.membership_mask(
            train_prices,
            train_volumes if not train_volumes.empty else None,
        )
        sector_map = {
            sm.ticker: sm.sector
            for sm in self.universe_mgr._stock_meta
            if not sm.blacklisted and sm.ticker in membership.columns
        }
        snapshot = self.universe_mgr.get_universe(
            as_of.date(),
            price_matrix=train_prices,
            volume_matrix=train_volumes if not train_volumes.empty else None,
        )

        # sector return matrix for labels
        # NOTE: extend price window by fwd_window to allow label computation,
        # but sector_feats are truncated to label_cutoff to prevent lookahead.
        sector_label_horizon = self.sector_fwd_window_days
        label_cutoff = as_of - pd.offsets.BDay(sector_label_horizon + 2)
        extended_prices = self.price_matrix.loc[
            (self.price_matrix.index >= train_start) &
            (self.price_matrix.index < as_of)
        ]
        extended_membership = membership.reindex(extended_prices.index, fill_value=False)
        clean_prices = fill_price_gaps(extended_prices[list(sector_map)], limit=5)
        clean_returns = clean_prices.pct_change(fill_method=None)
        sec_returns = pd.DataFrame({
            sec: clean_returns[[t for t, s in sector_map.items() if s == sec]].where(
                extended_membership[[t for t, s in sector_map.items() if s == sec]]
            ).mean(axis=1)
            for sec in snapshot.sectors
        })

        # Truncate sector features to label_cutoff (prevents forward-label leakage)
        safe_sector_feats = sector_feats[sector_feats.index <= label_cutoff] if not sector_feats.empty else sector_feats

        # train sector scorer (monthly)
        if train_sector:
            t0 = time.perf_counter()
            self.sector_scorer.fit(
                safe_sector_feats,
                sec_returns,
                fwd_window=self.sector_fwd_window_days,
                macro_features=self.macro_features.loc[
                    (self.macro_features.index >= train_start) &
                    (self.macro_features.index < as_of)
                ],
            )
            logger.info("  [timing] sector_scorer.fit   → %.2fs", time.perf_counter() - t0)

        # train stock ranker (quarterly)
        if train_stock:
            t0 = time.perf_counter()
            self.stock_ranker.fit(stock_feats, train_prices, fwd_window=self.stock_fwd_window_days)
            logger.info("  [timing] stock_ranker.fit    → %.2fs", time.perf_counter() - t0)

        logger.info("  [timing] _train_models total → %.2fs", time.perf_counter() - t0_total)

    def _train_rl(self, current_idx: int) -> None:
        if not self.rl_model_enabled:
            return
        min_history = int(self.cfg["rl"].get("min_history_rebalances", 3))
        if current_idx < min_history:
            return

        total_ts = int(self.cfg["rl"].get("total_timesteps", 8000))
        logger.info(
            "Retraining RL agent: train_end_idx=%d, PPO total_timesteps=%d ...",
            current_idx - 1,
            total_ts,
        )
        executor = HistoricalPeriodExecutor(
            self,
            mode="full_rl",
            allow_model_retraining=False,
        )
        rl_env = self.rl_agent.build_causal_env(
            executor,
            start_idx=0,
            end_idx=current_idx - 1,
        )
        t0 = time.perf_counter()
        self.rl_agent.train(
            total_timesteps=total_ts,
            causal_env=rl_env,
        )
        if self.rl_agent.is_trained:
            self._rl_retrain_count += 1
        elapsed = time.perf_counter() - t0
        logger.info(
            "  [timing] rl_agent.train      → %.2fs  (%.0f ts/s)",
            elapsed,
            total_ts / max(elapsed, 1e-6),
        )

    # ── Scheduling helpers ────────────────────────────────────────────────────

    def _generate_rebalance_dates(self) -> list[pd.Timestamp]:
        warm_up = pd.DateOffset(years=self.min_train_years)
        first = self.start_date + warm_up
        freq = f"{self.rebalance_weeks * 7}D"
        dates = pd.date_range(start=first, end=self.end_date, freq=freq)
        # only trading days
        trading_days = self.price_matrix.index
        valid = [
            min(trading_days[trading_days >= d], key=lambda x: abs((x - d).days))
            for d in dates
            if d <= self.end_date and any(trading_days >= d)
        ]
        return sorted(set(valid))

    def _should_retrain_sector_scorer(self, idx: int) -> bool:
        every = self._retrain_every_rebalances("sector_model", fallback_weeks=4)
        return idx % every == 0

    def _should_retrain_stock_ranker(self, idx: int) -> bool:
        every = self._retrain_every_rebalances("stock_model", fallback_weeks=12)
        return idx % every == 0

    def _should_retrain_models(self, idx: int, date: pd.Timestamp) -> bool:
        return self._should_retrain_sector_scorer(idx) or self._should_retrain_stock_ranker(idx)

    def _should_retrain_rl(self, idx: int) -> bool:
        every = self._retrain_every_rebalances("rl", fallback_weeks=12)
        return idx > 0 and idx % every == 0

    def _retrain_every_rebalances(self, section: str, fallback_weeks: int) -> int:
        """
        Return the scheduled retrain cadence in rebalance units.

        `retrain_every_rebalances` is the canonical setting.
        Legacy `retrain_freq_weeks` is interpreted as calendar weeks and converted
        using the configured rebalance cadence so historical configs remain valid.
        """
        section_cfg = self.cfg.get(section, {})
        every = section_cfg.get("retrain_every_rebalances")
        if every is not None:
            return max(1, int(every))

        legacy_weeks = int(section_cfg.get("retrain_freq_weeks", fallback_weeks))
        return max(1, int(round(legacy_weeks / max(self.rebalance_weeks, 1))))

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _get_prices(self, date: pd.Timestamp) -> pd.Series:
        avail = self.price_matrix[self.price_matrix.index <= date]
        if avail.empty:
            return pd.Series(dtype=float)
        # ffill so holiday/sparse rows use the last known price for each ticker
        return avail.ffill(limit=5).iloc[-1].dropna()

    def _get_macro_features(self, date: pd.Timestamp) -> pd.Series:
        avail = self.macro_features[self.macro_features.index <= date]
        return avail.iloc[-1] if not avail.empty else pd.Series(dtype=float)

    def _get_sector_features_now(
        self,
        date: pd.Timestamp,
        snapshot,
        prices_today: pd.Series,
        bm_prices: pd.Series | None,
    ) -> pd.DataFrame:
        """Return latest sector feature row per sector as of `date` from the store."""
        snap = self.feature_store.snapshot("sector", date)
        if not snap.empty and "sector" in snap.columns:
            snap = snap[snap["sector"].isin(snapshot.sectors)]
            return snap

        # Fallback: build on-the-fly (store miss)
        sector_map = self.universe_mgr.get_sector_map(snapshot)
        lookback_start = date - pd.DateOffset(months=15)
        hist = self.price_matrix.loc[lookback_start:date]
        bm_hist = bm_prices.loc[lookback_start:date] if bm_prices is not None else None
        feats = self.sector_fb.build(hist, sector_map, None, bm_hist)
        if feats.empty:
            return feats
        result = []
        for sec in snapshot.sectors:
            sec_rows = feats[feats["sector"] == sec] if "sector" in feats.columns else feats
            if not sec_rows.empty:
                row = sec_rows.iloc[-1].to_dict()
                row["sector"] = sec
                result.append(row)
        return pd.DataFrame(result)

    def _get_stock_features_now(
        self,
        date: pd.Timestamp,
        snapshot,
        prices_today: pd.Series,
        bm_prices: pd.Series | None,
    ) -> pd.DataFrame:
        """Return latest stock feature row per ticker as of `date` from the store."""
        snap = self.feature_store.snapshot("stock", date)
        if not snap.empty:
            snap = snap[snap["ticker"].isin(snapshot.tickers)]
            return snap

        # Fallback: build on-the-fly (store miss)
        sector_map = self.universe_mgr.get_sector_map(snapshot)
        lookback_start = date - pd.DateOffset(months=15)
        hist = self.price_matrix.loc[lookback_start:date]
        vol_hist = self.volume_matrix.loc[lookback_start:date] if not self.volume_matrix.empty else None
        bm_hist = bm_prices.loc[lookback_start:date] if bm_prices is not None else None
        feats = self.stock_fb.build(hist, vol_hist, sector_map, bm_hist)
        if feats.empty:
            return feats
        snap = feats[feats["date"] <= date]
        snap = snap.sort_values("date").groupby("ticker").last().reset_index()
        return snap[snap["ticker"].isin(snapshot.tickers)]

    def _get_recent_portfolio_returns(
        self,
        nav_points: list[tuple],
        current_date: pd.Timestamp,
        lookback: int = 126,
    ) -> pd.Series:
        if len(nav_points) < 2:
            return pd.Series(dtype=float)
        dates, navs = zip(*nav_points)
        nav_s = pd.Series(list(navs), index=pd.DatetimeIndex(list(dates)))
        nav_s = nav_s[~nav_s.index.duplicated(keep="last")]
        nav_s = nav_s.loc[:current_date].tail(lookback + 1)
        return nav_s.pct_change().dropna()

    def _compute_period_nav(
        self,
        portfolio: PortfolioState,
        entry_prices: pd.Series,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[tuple]:
        """Compute daily NAV between rebalance dates using holdings."""
        period = self.price_matrix.loc[
            (self.price_matrix.index > start) &
            (self.price_matrix.index <= end)
        ]
        if period.empty:
            return []

        # Forward-fill then backward-fill so holidays/gaps use last known price
        period = period.ffill().bfill()

        holding_tickers = [t for t in portfolio.holdings if portfolio.holdings[t] > 1e-6]
        nav_points = []

        for ts, row in period.iterrows():
            value = portfolio.cash
            for t in holding_tickers:
                shares = portfolio.holdings[t]
                price = row.get(t, np.nan) if t in period.columns else np.nan
                if price is None or not np.isfinite(float(price)):
                    # last resort: entry price or prior weight-based value
                    price = float(entry_prices.get(t, 0) or 0)
                if price > 0:
                    value += shares * price
            nav_points.append((ts, max(value, 0)))

        return nav_points

    def _build_sector_state(self, sector_feats: pd.DataFrame) -> dict:
        state = {}
        for _, row in sector_feats.iterrows():
            sec = row.get("sector", "unknown")
            state[sec] = {
                "mom_1m": float(row.get("mom_1m", 0) or 0),
                "mom_3m": float(row.get("mom_3m", 0) or 0),
                "rel_str_1m": float(row.get("rel_str_1m", 0) or 0),
                "breadth_3m": float(row.get("breadth_3m", 0) or 0),
            }
        return state

    def _build_next_rl_state(
        self,
        next_date: pd.Timestamp,
        portfolio: PortfolioState,
        nav_points: list[tuple[pd.Timestamp, float]],
    ) -> dict:
        snapshot = self.universe_mgr.get_universe(
            next_date.date(),
            price_matrix=self.price_matrix,
            volume_matrix=self.volume_matrix,
        )
        prices_next = self._get_prices(next_date)
        bm_ticker = self.cfg["backtest"].get("benchmark_ticker", "^NSEI")
        bm_prices = self.price_matrix[bm_ticker] if bm_ticker in self.price_matrix.columns else None
        macro_next = self._get_macro_features(next_date)
        sector_feats_next = self._get_sector_features_now(
            next_date, snapshot, prices_next, bm_prices
        )
        recent_rets = self._get_recent_portfolio_returns(nav_points, next_date)
        bm_rets_recent = bm_prices.pct_change(fill_method=None) if bm_prices is not None else None
        risk_signal, risk_action = self.risk_engine.evaluate(
            portfolio,
            recent_rets,
            macro_features=(
                macro_next if isinstance(macro_next, pd.Series) else pd.Series(macro_next)
            ),
            volume_matrix=self.volume_matrix.loc[:next_date],
        )
        portfolio_state = compute_portfolio_features(
            portfolio,
            recent_rets,
            bm_rets_recent,
            control_context=build_control_context(
                sector_feats_next,
                risk_signal=risk_signal,
                risk_action=risk_action,
                recent_turnovers=self._recent_turnovers,
                recent_cost_ratios=self._recent_cost_ratios,
            ),
        )
        return {
            "macro_state": (
                macro_next.to_dict() if isinstance(macro_next, pd.Series) else dict(macro_next)
            ),
            "sector_state": self._build_sector_state(sector_feats_next),
            "portfolio_state": dict(portfolio_state),
        }

    def _compute_sector_contributions(self) -> dict[str, float]:
        contributions: dict[str, float] = {}
        for rec in self.rebalance_records:
            for sec, w in rec.sector_tilts.items():
                contributions[sec] = contributions.get(sec, 0) + (w - 1.0)
        return contributions

    @staticmethod
    def _default_decision(sectors: list[str]) -> dict:
        return {
            "sector_tilts": {sector: 1.0 for sector in sectors},
            "posture": "neutral",
            "cash_target": 0.05,
            "aggressiveness": 1.0,
            "turnover_cap": 0.40,
        }

    def _select_sectors(
        self,
        sectors: list[str],
        sector_scores: dict[str, float],
        rl_decision: dict,
    ) -> list[str]:
        ordered = sorted(
            ((sector, float(sector_scores.get(sector, 0.0))) for sector in sectors),
            key=lambda item: (-item[1], item[0]),
        )
        if self.mode == "full_rl":
            posture = str(rl_decision.get("posture", "neutral"))
            sector_top_n = posture_selection_profile(self.cfg, posture).get("sector_top_n")
            if sector_top_n is None:
                return [sector for sector, _ in ordered]
            top_n = min(len(ordered), int(sector_top_n))
            return [sector for sector, _ in ordered[:top_n]]
        top_n = min(len(ordered), 5)
        return [sector for sector, _ in ordered[:top_n]]

    def _build_equal_weight_targets(
        self,
        alpha_scores: dict[str, float],
        cash_target: float,
    ) -> dict[str, float]:
        """Deterministic equal-weight constructor used by selection_only mode."""
        if not alpha_scores:
            return {"CASH": 1.0}

        max_stock = float(self.cfg["optimizer"]["max_stock_weight"])
        investable = max(0.0, 1.0 - float(cash_target))
        ordered = sorted(alpha_scores, key=lambda ticker: (-alpha_scores[ticker], ticker))
        if investable <= 0 or not ordered:
            return {"CASH": 1.0}

        per_stock = min(investable / len(ordered), max_stock)
        weights = {ticker: per_stock for ticker in ordered if per_stock > 0}
        invested = per_stock * len(weights)
        weights["CASH"] = max(0.0, 1.0 - invested)
        return weights


    def _selected_forward_returns(
        self,
        selected_stock_rows: list[dict[str, object]],
        prices_today: pd.Series,
        next_date: pd.Timestamp,
        _period_nav: list[tuple[pd.Timestamp, float]] | None = None,
    ) -> dict[str, float]:
        tickers = [str(row["ticker"]) for row in selected_stock_rows]
        return self._forward_returns_for_tickers(tickers, prices_today, next_date)

    def _forward_returns_for_tickers(
        self,
        tickers: list[str],
        prices_today: pd.Series,
        next_date: pd.Timestamp,
    ) -> dict[str, float]:
        returns: dict[str, float] = {}
        if not tickers:
            return returns

        available = [ticker for ticker in tickers if ticker in self.price_matrix.columns]
        if not available:
            return returns

        next_prices = self.price_matrix.loc[:next_date, available].ffill().iloc[-1]
        for ticker in available:
            if ticker not in prices_today.index or ticker not in next_prices.index:
                continue
            start_price = float(prices_today.get(ticker, np.nan))
            end_price = float(next_prices.get(ticker, np.nan))
            if (
                np.isfinite(start_price)
                and np.isfinite(end_price)
                and start_price > 0
            ):
                returns[ticker] = (end_price - start_price) / start_price
        return returns

    @staticmethod
    def _rank_ic(scores: dict[str, float], forward_returns: dict[str, float]) -> float | None:
        common = [
            (float(scores[ticker]), float(forward_returns[ticker]))
            for ticker in sorted(set(scores) & set(forward_returns))
            if np.isfinite(scores[ticker]) and np.isfinite(forward_returns[ticker])
        ]
        if len(common) < 2:
            return None

        score_series = pd.Series([item[0] for item in common]).rank(method="average")
        return_series = pd.Series([item[1] for item in common]).rank(method="average")
        corr = score_series.corr(return_series, method="pearson")
        return float(corr) if pd.notna(corr) else None

    def _selection_stability(self, selected_stock_rows: list[dict[str, object]]) -> float | None:
        selected = {str(row["ticker"]) for row in selected_stock_rows}
        if not selected:
            return None

        previous = None
        for record in reversed(self.selection_diagnostics):
            previous_rows = record.get("selected_stocks", [])
            previous = {
                str(item["ticker"])
                for item in previous_rows
                if isinstance(item, dict) and item.get("ticker")
            }
            if previous:
                break

        if not previous:
            return None

        union = selected | previous
        if not union:
            return None
        return float(len(selected & previous) / len(union))

    # ── Save/load ─────────────────────────────────────────────────────────────

    def save_state(self) -> None:
        out_dir = Path(self.cfg["paths"]["artifact_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        self.nav_series.to_frame("nav").to_parquet(out_dir / "nav_series.parquet")
        with open(out_dir / "rebalance_records.pkl", "wb") as f:
            pickle.dump(self.rebalance_records, f)

        self.sector_scorer.save(
            Path(self.cfg["paths"]["model_dir"]) / "sector_scorer.pkl"
        )
        self.stock_ranker.save(
            Path(self.cfg["paths"]["model_dir"]) / "stock_ranker.pkl"
        )
        if self.use_rl:
            self.rl_agent.save(
                Path(self.cfg["paths"]["model_dir"]) / "rl_agent"
            )
        logger.info("Walk-forward state saved → %s", out_dir)
