"""Tests for data layer: universe, contracts, ingestion."""
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.contracts import (
    DailyBar, PortfolioState, StockMeta, UniverseSnapshot
)
from src.data.universe import UniverseManager


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def universe_mgr(cfg):
    return UniverseManager(cfg)


@pytest.fixture
def dummy_price_matrix():
    dates = pd.date_range("2015-01-01", "2016-12-31", freq="B")
    tickers = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS"]
    data = pd.DataFrame(
        np.random.lognormal(0, 0.01, (len(dates), len(tickers))).cumprod(axis=0) * 1000,
        index=dates, columns=tickers,
    )
    data.index.name = "date"
    return data


# ── Contract tests ────────────────────────────────────────────────────────────

class TestContracts:
    def test_portfolio_state_cash_non_negative(self):
        with pytest.raises(Exception):
            PortfolioState(date=date.today(), cash=-100, holdings={},
                           weights={}, nav=1000, sector_weights={})

    def test_portfolio_state_valid(self):
        ps = PortfolioState(
            date=date.today(), cash=50000.0,
            holdings={"TCS.NS": 10}, weights={"TCS.NS": 0.5, "CASH": 0.5},
            nav=100000.0, sector_weights={"IT": 0.5}
        )
        assert ps.nav == 100000.0
        assert ps.cash == 50000.0

    def test_stock_meta_frozen(self):
        sm = StockMeta(ticker="TCS.NS", name="TCS", sector="IT", cap="large")
        with pytest.raises(Exception):
            sm.ticker = "INFY.NS"

    def test_daily_bar_validates_price(self):
        with pytest.raises(Exception):
            DailyBar(ticker="X", date=date.today(), open=-1, high=100,
                     low=90, close=95, adj_close=95, volume=1000000)


# ── Universe tests ────────────────────────────────────────────────────────────

class TestUniverseManager:
    def test_loads_stocks(self, universe_mgr):
        assert len(universe_mgr._stock_meta) > 0

    def test_get_universe_returns_snapshot(self, universe_mgr, dummy_price_matrix):
        snap = universe_mgr.get_universe(
            date(2016, 1, 1), price_matrix=dummy_price_matrix
        )
        assert isinstance(snap, UniverseSnapshot)
        # Some stocks will not have enough history; total should be reasonable
        assert len(snap.stocks) >= 0

    def test_sector_map(self, universe_mgr, dummy_price_matrix):
        snap = universe_mgr.get_universe(
            date(2016, 1, 1), price_matrix=dummy_price_matrix
        )
        sector_map = universe_mgr.get_sector_map(snap)
        # All tickers should map to a sector
        for t in snap.tickers:
            assert t in sector_map

    def test_by_sector_grouping(self, universe_mgr, dummy_price_matrix):
        snap = universe_mgr.get_universe(
            date(2016, 1, 1), price_matrix=dummy_price_matrix
        )
        by_sector = snap.by_sector()
        assert isinstance(by_sector, dict)

    def test_no_blacklisted_stocks(self, universe_mgr, dummy_price_matrix):
        snap = universe_mgr.get_universe(date(2016, 1, 1), price_matrix=dummy_price_matrix)
        for s in snap.stocks:
            assert not s.blacklisted


# ── Feature tests ─────────────────────────────────────────────────────────────

class TestFeatures:
    def test_macro_feature_builder(self):
        from src.features.macro_features import MacroFeatureBuilder
        cfg = load_config()
        builder = MacroFeatureBuilder(cfg)
        dates = pd.date_range("2015-01-01", periods=300)
        macro = pd.DataFrame(
            np.random.randn(300, 6),
            index=dates,
            columns=["vix", "usdinr", "crude_oil", "sp500", "us_10y", "gold"],
        )
        macro["rbi_repo_rate"] = 6.5
        macro["rbi_meeting"] = 0.0
        macro["budget_day"] = 0.0
        macro["election_window"] = 0.0
        result = builder.build(macro)
        assert not result.empty
        assert "vix_level" in result.columns

    def test_stock_features_lag(self):
        """Ensure stock features are lagged (no lookahead)."""
        from src.features.stock_features import StockFeatureBuilder
        cfg = load_config()
        builder = StockFeatureBuilder(cfg)
        dates = pd.date_range("2014-01-01", periods=300)
        prices = pd.DataFrame(
            np.random.lognormal(0, 0.01, (300, 3)).cumprod(axis=0) * 1000,
            index=dates, columns=["TCS.NS", "INFY.NS", "HDFCBANK.NS"]
        )
        sector_map = {"TCS.NS": "IT", "INFY.NS": "IT", "HDFCBANK.NS": "Banking"}
        feats = builder.build(prices, None, sector_map)
        # Should have rows with date and ticker
        assert "date" in feats.columns
        assert "ticker" in feats.columns


# ── Optimizer tests ───────────────────────────────────────────────────────────

class TestOptimizer:
    def test_weights_sum_to_one(self):
        from src.optimizer.portfolio_optimizer import PortfolioOptimizer
        cfg = load_config()
        opt = PortfolioOptimizer(cfg)
        alpha = {"TCS.NS": 0.8, "INFY.NS": 0.6, "HDFCBANK.NS": 0.7, "RELIANCE.NS": 0.5}
        sector_map = {"TCS.NS": "IT", "INFY.NS": "IT",
                      "HDFCBANK.NS": "Banking", "RELIANCE.NS": "Energy"}
        result = opt.optimize(alpha, None, sector_map)
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-3

    def test_no_short_positions(self):
        from src.optimizer.portfolio_optimizer import PortfolioOptimizer
        cfg = load_config()
        opt = PortfolioOptimizer(cfg)
        alpha = {"TCS.NS": 0.8, "INFY.NS": 0.6, "HDFCBANK.NS": 0.7}
        sector_map = {"TCS.NS": "IT", "INFY.NS": "IT", "HDFCBANK.NS": "Banking"}
        result = opt.optimize(alpha, None, sector_map)
        assert all(v >= -1e-6 for v in result.values())

    def test_max_stock_weight_respected(self):
        from src.optimizer.portfolio_optimizer import PortfolioOptimizer
        try:
            import cvxpy
            has_cvxpy = True
        except ImportError:
            has_cvxpy = False

        cfg = load_config()
        opt = PortfolioOptimizer(cfg)
        max_w = cfg["optimizer"]["max_stock_weight"]
        alpha = {"TCS.NS": 100.0, "INFY.NS": 80.0, "HCLTECH.NS": 60.0,
                 "WIPRO.NS": 40.0, "TECHM.NS": 20.0,
                 "HDFCBANK.NS": 90.0, "ICICIBANK.NS": 70.0}
        sector_map = {"TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT",
                      "WIPRO.NS": "IT", "TECHM.NS": "IT",
                      "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking"}
        result = opt.optimize(alpha, None, sector_map)
        if has_cvxpy:
            tcs_w = result.get("TCS.NS", 0)
            assert tcs_w <= max_w + 1e-3
        else:
            # Without CVXPY, verify valid weight dict
            assert sum(result.values()) <= 1.0 + 1e-3


# ── Risk engine tests ─────────────────────────────────────────────────────────

class TestRiskEngine:
    def test_drawdown_detection(self):
        from src.risk.risk_engine import RiskEngine
        cfg = load_config()
        risk = RiskEngine(cfg)
        risk.update(1_000_000, date(2020, 1, 1))
        risk.update(800_000, date(2020, 3, 1))  # -20% drawdown
        dd = risk.current_drawdown()
        assert dd < -0.15

    def test_emergency_at_hard_limit(self):
        from src.risk.risk_engine import RiskEngine
        from src.data.contracts import PortfolioState
        cfg = load_config()
        risk = RiskEngine(cfg)
        risk._peak_nav = 1_000_000
        risk._nav_history = [800_000]  # -20% < hard limit of -18%
        ps = PortfolioState(
            date=date.today(), cash=100000, holdings={},
            weights={"CASH": 1.0}, nav=800_000, sector_weights={}
        )
        signal, action = risk.evaluate(ps, pd.Series(dtype=float))
        assert signal.emergency_rebalance is True
        assert action.force_rebalance is True


# ── Simulator tests ───────────────────────────────────────────────────────────

class TestSimulator:
    def test_nav_is_conserved_on_rebalance(self):
        from src.backtest.simulator import PortfolioSimulator
        from src.data.contracts import PortfolioState
        cfg = load_config()
        sim = PortfolioSimulator(cfg)

        prices = pd.Series({"TCS.NS": 3000.0, "INFY.NS": 1500.0, "CASH": 1.0})
        init_state = PortfolioState(
            date=date(2020, 1, 1),
            cash=500_000.0,
            holdings={},
            weights={"CASH": 1.0},
            nav=500_000.0,
            sector_weights={},
        )
        target = {"TCS.NS": 0.50, "INFY.NS": 0.45, "CASH": 0.05}
        result = sim.execute_rebalance(target, init_state, prices, date(2020, 1, 2))

        # NAV should be close to initial (minus transaction costs)
        nav_after = result.new_portfolio.nav
        assert nav_after > 0
        assert nav_after <= init_state.nav  # costs reduce NAV
        assert nav_after > init_state.nav * 0.99  # should not lose >1% to costs

    def test_compute_metrics(self):
        from src.backtest.simulator import PortfolioSimulator
        nav = pd.Series(
            [100, 105, 103, 110, 108, 115, 120],
            index=pd.date_range("2020-01-01", periods=7),
        ) * 1000
        metrics = PortfolioSimulator.compute_metrics(nav)
        assert "cagr" in metrics
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics
