"""
Portfolio Recommender — loads saved RL agent + models and produces
live allocation recommendations from the latest market data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

from src.data.contracts import PortfolioState
from src.data.ingestion import load_price_matrix, load_volume_matrix
from src.data.macro import MacroDataManager
from src.data.universe import UniverseManager
from src.features.feature_store import FeatureStore
from src.features.macro_features import MacroFeatureBuilder
from src.features.portfolio_features import compute_portfolio_features
from src.models.sector_scorer import SectorScorer
from src.models.stock_ranker import StockRanker
from src.optimizer.portfolio_optimizer import PortfolioOptimizer
from src.risk.risk_engine import RiskAction, RiskEngine
from src.rl.agent import RLSectorAgent
from src.rl.contract import CAUSAL_TRAINING_BACKEND
from src.rl.environment import SectorAllocationEnv
from src.rl.policy_utils import (
    apply_posture_policy,
    build_sector_state,
    posture_selection_profile,
    select_sectors,
)

logger = logging.getLogger(__name__)

_RISK_PROFILES = {
    "conservative":  {"aggressiveness": 0.6, "cash_floor": 0.15, "max_sector_w": 0.25},
    "moderate":      {"aggressiveness": 1.0, "cash_floor": 0.05, "max_sector_w": 0.35},
    "aggressive":    {"aggressiveness": 1.4, "cash_floor": 0.02, "max_sector_w": 0.45},
}


class PortfolioRecommender:
    """
    Loads saved models from artifacts/models/ and generates allocation
    recommendations using the latest available market data.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._model_dir = Path(cfg["paths"]["model_dir"])
        self._loaded = False
        self._price_matrix: pd.DataFrame | None = None
        self._volume_matrix: pd.DataFrame | None = None
        self._sector_scorer: SectorScorer | None = None
        self._stock_ranker: StockRanker | None = None
        self._rl_agent: RLSectorAgent | None = None
        self._universe_mgr: UniverseManager | None = None
        self._feature_store: FeatureStore | None = None
        self._macro_mgr: MacroDataManager | None = None
        self._optimizer: PortfolioOptimizer | None = None
        self._risk_engine: RiskEngine | None = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            self._price_matrix  = load_price_matrix(self.cfg)
            self._volume_matrix = load_volume_matrix(self.cfg)
            self._macro_mgr     = MacroDataManager(self.cfg)
            self._universe_mgr  = UniverseManager(self.cfg)
            store_dir = Path(self.cfg["paths"]["artifact_dir"]) / "feature_store"
            self._feature_store = FeatureStore(store_dir, self.cfg)
            self._optimizer     = PortfolioOptimizer(self.cfg)
            self._risk_engine   = RiskEngine(self.cfg)

            self._sector_scorer = SectorScorer(self.cfg)
            sc_path = self._model_dir / "sector_scorer.pkl"
            self._safe_load_model(self._sector_scorer, sc_path, "sector scorer")

            self._stock_ranker = StockRanker(self.cfg)
            sr_path = self._model_dir / "stock_ranker.pkl"
            self._safe_load_model(self._stock_ranker, sr_path, "stock ranker")

            self._rl_agent = RLSectorAgent(self.cfg)
            rl_path = self._model_dir / "rl_agent"
            self._safe_load_model(self._rl_agent, rl_path, "rl agent")

            self._loaded = True
            logger.info("PortfolioRecommender loaded (RL trained=%s)", self._rl_agent.is_trained)
        except Exception as e:
            logger.warning("PortfolioRecommender load failed: %s", e)

    def is_ready(self) -> bool:
        return self._loaded

    def policy_status(self) -> dict[str, object]:
        live_rl = self._serving_uses_rl()
        return {
            "rl_trained": bool(self._rl_agent is not None and self._rl_agent.is_trained),
            "rl_serving_enabled": live_rl,
            "default_serving_mode": "RL" if live_rl else "Baseline",
        }

    @staticmethod
    def _safe_load_model(model: object, path: Path, label: str) -> None:
        if not path.exists():
            logger.info("%s artifact not found at %s; using fallback inference path", label, path)
            return
        try:
            getattr(model, "load")(str(path))
        except Exception as exc:
            logger.warning("Failed to load %s from %s: %s; using fallback path", label, path, exc)

    @staticmethod
    def _coerce_snapshot_row(snapshot: pd.DataFrame | pd.Series | None) -> pd.Series:
        if snapshot is None:
            return pd.Series(dtype=float)
        if isinstance(snapshot, pd.Series):
            return snapshot
        if snapshot.empty:
            return pd.Series(dtype=float)
        row = snapshot.iloc[-1]
        if isinstance(row, pd.Series):
            return row
        return pd.Series(dtype=float)

    def _get_macro_features_now(self, as_of_ts: pd.Timestamp) -> pd.Series:
        if self._feature_store is not None:
            try:
                macro_snap = self._feature_store.snapshot("macro", as_of_ts)
                macro_now = self._coerce_snapshot_row(macro_snap)
                if not macro_now.empty:
                    return macro_now
            except Exception as exc:
                logger.warning("Feature store macro snapshot failed at %s: %s", as_of_ts.date(), exc)

        if self._macro_mgr is None:
            return pd.Series(dtype=float)

        try:
            raw_macro = self._macro_mgr.load()
            if raw_macro.empty:
                return pd.Series(dtype=float)
            macro_features = MacroFeatureBuilder(self.cfg).build(raw_macro)
            avail = macro_features[macro_features.index <= as_of_ts]
            return avail.iloc[-1] if not avail.empty else pd.Series(dtype=float)
        except Exception as exc:
            logger.warning("Macro feature fallback build failed at %s: %s", as_of_ts.date(), exc)
            return pd.Series(dtype=float)

    @staticmethod
    def _fallback_rank_stocks(
        stock_features_snap: pd.DataFrame,
        sector: str,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        sec_df = stock_features_snap[stock_features_snap["sector"] == sector].copy()
        if sec_df.empty:
            return pd.DataFrame(columns=["ticker", "score", "rank"])

        if "ret_3m" in sec_df.columns:
            scores = sec_df["ret_3m"].fillna(0).astype(float)
        elif "mom_3m" in sec_df.columns:
            scores = sec_df["mom_3m"].fillna(0).astype(float)
        else:
            scores = pd.Series(0.0, index=sec_df.index)

        result = pd.DataFrame({
            "ticker": sec_df["ticker"].values,
            "score": scores.values,
        })
        result = result.sort_values(
            ["score", "ticker"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result.head(top_k) if top_k else result

    def _rank_stocks(
        self,
        stock_features_snap: pd.DataFrame,
        sector: str,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        if self._stock_ranker is None:
            return self._fallback_rank_stocks(stock_features_snap, sector, top_k=top_k)
        try:
            return self._stock_ranker.rank_stocks(stock_features_snap, sector, top_k=top_k)
        except Exception as exc:
            logger.warning("Stock ranker failed for sector %s: %s; using fallback ranking", sector, exc)
            return self._fallback_rank_stocks(stock_features_snap, sector, top_k=top_k)

    def _serving_uses_rl(self) -> bool:
        rl_cfg = self.cfg.get("rl", {})
        if not rl_cfg.get("use_rl", True):
            return False
        if rl_cfg.get("training_backend", "disabled") != CAUSAL_TRAINING_BACKEND:
            return False
        if rl_cfg.get("serve_rl") is False:
            return False
        if rl_cfg.get("experimental", False) and not rl_cfg.get("serve_experimental", False):
            return False
        return self._rl_agent is not None and self._rl_agent.is_trained

    @staticmethod
    def _build_current_weights(
        capital_inr: float,
        current_holdings: dict[str, float] | None,
    ) -> tuple[dict[str, float], float]:
        if not current_holdings:
            return {"CASH": 1.0}, max(float(capital_inr), 1.0)

        gross_holdings = sum(max(float(v), 0.0) for v in current_holdings.values())
        nav = max(float(capital_inr), gross_holdings, 1.0)
        current_weights = {
            ticker: max(float(value), 0.0) / nav
            for ticker, value in current_holdings.items()
            if max(float(value), 0.0) > 0
        }
        cash_weight = max(0.0, 1.0 - sum(current_weights.values()))
        current_weights["CASH"] = cash_weight
        return current_weights, nav

    def _estimate_portfolio_returns(
        self,
        as_of_ts: pd.Timestamp,
        current_weights: dict[str, float],
    ) -> pd.Series:
        tickers = [
            ticker
            for ticker, weight in current_weights.items()
            if ticker != "CASH" and weight > 0 and ticker in self._price_matrix.columns
        ]
        if not tickers:
            return pd.Series(dtype=float)

        hist_prices = (
            self._price_matrix.loc[:as_of_ts, tickers]
            .tail(253)
            .ffill()
            .dropna(how="all")
        )
        if hist_prices.empty or len(hist_prices.index) < 2:
            return pd.Series(dtype=float)

        asset_returns = hist_prices.pct_change().fillna(0.0)
        weights = pd.Series({ticker: current_weights.get(ticker, 0.0) for ticker in tickers})
        return asset_returns.mul(weights, axis=1).sum(axis=1)

    def _estimate_benchmark_returns(self, as_of_ts: pd.Timestamp) -> pd.Series | None:
        benchmark_ticker = self.cfg.get("backtest", {}).get("benchmark_ticker")
        if not benchmark_ticker or benchmark_ticker not in self._price_matrix.columns:
            return None
        hist = (
            self._price_matrix.loc[:as_of_ts, benchmark_ticker]
            .tail(253)
            .ffill()
            .dropna()
        )
        if hist.empty or len(hist.index) < 2:
            return None
        return hist.pct_change().dropna()

    def _build_portfolio_context(
        self,
        *,
        as_of_ts: pd.Timestamp,
        capital_inr: float,
        current_holdings: dict[str, float] | None,
        sector_map: dict[str, str],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], object]:
        current_weights, nav = self._build_current_weights(capital_inr, current_holdings)

        sector_weights: dict[str, float] = {}
        for ticker, weight in current_weights.items():
            if ticker == "CASH" or weight <= 0:
                continue
            sector = sector_map.get(ticker)
            if sector is None:
                continue
            sector_weights[sector] = sector_weights.get(sector, 0.0) + float(weight)

        cash_value = nav * current_weights.get("CASH", 0.0)
        portfolio_state = PortfolioState(
            date=as_of_ts.date(),
            cash=cash_value,
            holdings={},
            weights=current_weights,
            nav=nav,
            sector_weights=sector_weights,
        )

        recent_returns = self._estimate_portfolio_returns(as_of_ts, current_weights)
        benchmark_returns = self._estimate_benchmark_returns(as_of_ts)
        port_feats = compute_portfolio_features(portfolio_state, recent_returns, benchmark_returns)

        if "risk" not in self.cfg:
            return current_weights, sector_weights, port_feats, RiskAction()

        risk_engine = RiskEngine(self.cfg)
        if recent_returns.empty:
            risk_engine.update(portfolio_state.nav, portfolio_state.date)
        else:
            nav_series = (1.0 + recent_returns).cumprod() * portfolio_state.nav
            for ts, nav_level in nav_series.items():
                risk_engine.update(float(nav_level), ts.date())

        macro_for_risk = pd.Series(dtype=float)
        if self._feature_store is not None:
            macro_for_risk = self._coerce_snapshot_row(self._feature_store.snapshot("macro", as_of_ts))
        risk_signal, risk_action = risk_engine.evaluate(
            portfolio_state,
            recent_returns,
            macro_features=macro_for_risk,
            volume_matrix=self._volume_matrix.loc[:as_of_ts] if self._volume_matrix is not None else None,
        )
        _ = risk_signal
        return current_weights, sector_weights, port_feats, risk_action

    def recommend(
        self,
        capital_inr: float = 500_000,
        risk_profile: str = "moderate",
        current_holdings: dict[str, float] | None = None,
        as_of: date | None = None,
    ) -> dict:
        """
        Generate a portfolio allocation recommendation.

        Args:
            capital_inr: total portfolio value in INR
            risk_profile: conservative / moderate / aggressive
            current_holdings: dict of ticker → current INR value (for turnover-aware rebalance)
            as_of: date to use for features (default: latest available)

        Returns dict with:
            allocation: {ticker: weight}
            trades: [{ticker, action, weight, value_inr}]
            sector_tilts: {sector: tilt}
            as_of_date: str
            model_mode: RL | Rule
        """
        if not self._loaded:
            self._try_load()
        if not self._loaded:
            raise RuntimeError("Models not loaded. Run run_backtest.py first.")

        profile = _RISK_PROFILES.get(risk_profile, _RISK_PROFILES["moderate"])
        as_of_ts = pd.Timestamp(as_of) if as_of else self._price_matrix.index[-1]

        # Build current market features
        sector_feats_now = self._feature_store.snapshot("sector", as_of_ts)
        stock_feats_now = self._feature_store.snapshot("stock", as_of_ts)
        macro_now = self._get_macro_features_now(as_of_ts)

        # Universe snapshot
        prices_now = self._price_matrix.loc[
            self._price_matrix.index <= as_of_ts
        ].iloc[-1].dropna()

        snapshot = self._universe_mgr.get_universe(as_of_ts.date(), price_matrix=self._price_matrix)
        sector_map = self._universe_mgr.get_sector_map(snapshot)
        cap_map = {s.ticker: s.cap for s in snapshot.stocks}
        current_weights, realized_sector_weights, port_feats, risk_action = self._build_portfolio_context(
            as_of_ts=as_of_ts,
            capital_inr=capital_inr,
            current_holdings=current_holdings,
            sector_map=sector_map,
        )

        # Sector scores
        sector_scores: dict[str, float] = {}
        if self._sector_scorer is not None and not sector_feats_now.empty:
            sector_scores = self._sector_scorer.predict(sector_feats_now, macro_now)

        use_live_rl = self._serving_uses_rl()
        sector_state = build_sector_state(sector_feats_now)

        # RL decision or neutral full-stack baseline
        if use_live_rl:
            try:
                rl_decision = self._rl_agent.decide(
                    macro_state=macro_now.to_dict(),
                    sector_state=sector_state,
                    portfolio_state=port_feats,
                    prev_realized_sector_weights=realized_sector_weights,
                )
            except TypeError:
                rl_decision = self._rl_agent.decide(
                    macro_state=macro_now.to_dict(),
                    sector_state=sector_state,
                    portfolio_state=port_feats,
                )
            model_mode = "RL"
        else:
            rl_decision = SectorAllocationEnv.neutral_action(self.cfg)
            model_mode = "Baseline"
        rl_decision = apply_posture_policy(self.cfg, rl_decision)

        cash_target = max(
            float(rl_decision.get("cash_target", 0.05)),
            float(profile["cash_floor"]),
            float(risk_action.cash_floor),
        )

        # Rank stocks
        posture = str(rl_decision.get("posture", "neutral"))
        selection_profile = posture_selection_profile(self.cfg, posture)
        top_k = int(selection_profile.get("stock_top_k_per_sector") or self.cfg["stock_model"]["top_k_per_sector"])
        alpha_scores: dict[str, float] = {}
        selected_sectors = select_sectors(
            snapshot.sectors,
            sector_scores,
            rl_decision,
            full_rl=True,
            cfg=self.cfg,
        )

        for sector in selected_sectors:
            ranking = self._rank_stocks(stock_feats_now, sector, top_k=top_k)
            for _, row in ranking.iterrows():
                alpha_scores[row["ticker"]] = float(row["score"])

        if not alpha_scores:
            for t in snapshot.tickers[:30]:
                if t in prices_now.index:
                    alpha_scores[t] = 0.5

        # Optimize
        cov = PortfolioOptimizer.estimate_covariance(
            self._price_matrix.loc[:as_of_ts], list(alpha_scores.keys())
        )
        target_weights = self._optimizer.optimize(
            alpha_scores=alpha_scores,
            cov_matrix=cov,
            sector_map=sector_map,
            current_weights=current_weights,
            sector_tilts=rl_decision["sector_tilts"],
            aggressiveness=float(rl_decision.get("aggressiveness", 1.0)),
            cash_target=cash_target,
            forced_exclude=risk_action.exclude_tickers,
            posture=posture,
        )
        if "risk" in self.cfg:
            risk_engine = self._risk_engine or RiskEngine(self.cfg)
            target_weights = risk_engine.check_pre_trade(
                target_weights,
                sector_map,
                cap_map,
                risk_action,
            )

        # Build trade list
        trades = []
        for ticker, new_w in target_weights.items():
            if ticker == "CASH":
                continue
            old_w = current_weights.get(ticker, 0.0)
            delta_w = new_w - old_w
            if abs(delta_w) < 0.003:
                continue
            trades.append({
                "ticker":    ticker,
                "action":    "buy" if delta_w > 0 else "sell",
                "weight":    round(new_w, 4),
                "delta_w":   round(delta_w, 4),
                "value_inr": round(new_w * capital_inr, 2),
            })

        return {
            "as_of_date":   str(as_of_ts.date()),
            "model_mode":   model_mode,
            "allocation":   {k: round(v, 4) for k, v in target_weights.items()},
            "trades":       sorted(trades, key=lambda x: -abs(x["delta_w"])),
            "sector_tilts": {k: round(v, 3) for k, v in rl_decision["sector_tilts"].items()},
            "cash_pct":     round(target_weights.get("CASH", 0) * 100, 1),
            "n_stocks":     len([t for t in target_weights if t != "CASH" and target_weights[t] > 0.001]),
            "risk_profile": risk_profile,
        }
