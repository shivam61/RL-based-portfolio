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

from src.data.ingestion import load_price_matrix, load_volume_matrix
from src.data.macro import MacroDataManager
from src.data.universe import UniverseManager
from src.features.feature_store import FeatureStore
from src.features.sector_features import SectorFeatureBuilder
from src.features.stock_features import StockFeatureBuilder
from src.models.sector_scorer import SectorScorer
from src.models.stock_ranker import StockRanker
from src.optimizer.portfolio_optimizer import PortfolioOptimizer
from src.risk.risk_engine import RiskEngine
from src.rl.agent import RLSectorAgent
from src.data.contracts import PortfolioState

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
            self._feature_store = FeatureStore(self.cfg)
            self._optimizer     = PortfolioOptimizer(self.cfg)
            self._risk_engine   = RiskEngine(self.cfg)

            self._sector_scorer = SectorScorer(self.cfg)
            sc_path = self._model_dir / "sector_scorer.pkl"
            if sc_path.exists():
                self._sector_scorer.load(str(sc_path))

            self._stock_ranker = StockRanker(self.cfg)
            sr_path = self._model_dir / "stock_ranker.pkl"
            if sr_path.exists():
                self._stock_ranker.load(str(sr_path))

            self._rl_agent = RLSectorAgent(self.cfg)
            rl_path = self._model_dir / "rl_agent"
            if rl_path.exists():
                self._rl_agent.load(str(rl_path))

            self._loaded = True
            logger.info("PortfolioRecommender loaded (RL trained=%s)", self._rl_agent.is_trained)
        except Exception as e:
            logger.warning("PortfolioRecommender load failed: %s", e)

    def is_ready(self) -> bool:
        return self._loaded

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
        sector_feats = self._feature_store.load("sector",
            as_of_ts - pd.DateOffset(days=30), as_of_ts)
        stock_feats  = self._feature_store.load("stock",
            as_of_ts - pd.DateOffset(days=90), as_of_ts)

        # Get latest macro snapshot
        macro_df = self._macro_mgr.load()
        macro_now = macro_df.loc[macro_df.index <= as_of_ts].iloc[-1] if not macro_df.empty else pd.Series()

        # Universe snapshot
        prices_now = self._price_matrix.loc[
            self._price_matrix.index <= as_of_ts
        ].iloc[-1].dropna()

        snapshot = self._universe_mgr.get_universe(as_of_ts.date(), price_matrix=self._price_matrix)
        sector_map = snapshot.sector_map if hasattr(snapshot, "sector_map") else {}
        cap_map    = snapshot.cap_map    if hasattr(snapshot, "cap_map")    else {}

        # Sector scores
        sector_scores: dict[str, float] = {}
        if self._sector_scorer and self._sector_scorer.is_fitted and not sector_feats.empty:
            sector_feats_now = sector_feats.loc[
                sector_feats.index == sector_feats.index.max()
            ] if not sector_feats.empty else sector_feats
            sector_scores = self._sector_scorer.predict(sector_feats_now)

        # RL decision or rule-based
        if self._rl_agent and self._rl_agent.is_trained:
            port_feats = {
                "cash_ratio": 1.0, "ret_1m": 0.0, "vol_1m": 0.01,
                "current_drawdown": 0.0, "max_drawdown": 0.0,
                "hhi": 0.0, "max_weight": 0.0, "sharpe_3m": 0.0,
                "active_ret_1m": 0.0, "n_stocks": 0,
            }
            sector_state = {}
            if not sector_feats.empty:
                for _, row in sector_feats.loc[sector_feats.index == sector_feats.index.max()].iterrows():
                    sec = row.get("sector", "unknown")
                    sector_state[sec] = {k: row.get(k, 0) for k in ["mom_1m", "mom_3m", "rel_str_1m", "breadth_3m"]}

            rl_decision = self._rl_agent.decide(
                macro_state=macro_now.to_dict() if isinstance(macro_now, pd.Series) else {},
                sector_state=sector_state,
                portfolio_state=port_feats,
            )
            model_mode = "RL"
        else:
            rl_decision = RLSectorAgent.rule_based_action(
                sector_scores,
                macro_now.to_dict() if isinstance(macro_now, pd.Series) else {},
            )
            model_mode = "Rule"

        # Apply risk profile overrides
        rl_decision["aggressiveness"] = profile["aggressiveness"]
        cash_target = max(rl_decision.get("cash_target", 0.05), profile["cash_floor"])

        # Rank stocks
        top_k = self.cfg["stock_model"]["top_k_per_sector"]
        alpha_scores: dict[str, float] = {}
        stock_feats_now = stock_feats.loc[
            stock_feats.index == stock_feats.index.max()
        ] if not stock_feats.empty else stock_feats

        for sector in snapshot.sectors:
            tilt = rl_decision["sector_tilts"].get(sector, 1.0)
            if tilt < 0.4:
                continue
            ranking = self._stock_ranker.rank_stocks(stock_feats_now, sector, top_k=top_k)
            for _, row in ranking.iterrows():
                alpha_scores[row["ticker"]] = float(row["score"]) * tilt

        if not alpha_scores:
            for t in snapshot.tickers[:30]:
                if t in prices_now.index:
                    alpha_scores[t] = 0.5

        # Build current weights from holdings
        current_weights: dict[str, float] = {"CASH": 1.0}
        if current_holdings:
            total = sum(current_holdings.values()) or capital_inr
            current_weights = {t: v / total for t, v in current_holdings.items()}
            current_weights["CASH"] = max(0, 1.0 - sum(v for k, v in current_weights.items() if k != "CASH"))

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
            aggressiveness=profile["aggressiveness"],
            cash_target=cash_target,
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
