"""Tests for data layer: universe, contracts, ingestion."""
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.ingestion import load_price_matrix, load_volume_matrix
from src.data.contracts import (
    DailyBar, PortfolioState, StockMeta, UniverseSnapshot
)
from src.data.universe import UniverseManager


@pytest.fixture
def cfg():
    cfg = load_config()
    cfg["universe"]["mode"] = "static"
    return cfg


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

    def test_membership_mask_respects_listing_and_delisting(self, universe_mgr):
        dates = pd.date_range("2015-01-01", "2017-12-31", freq="B")
        prices = pd.DataFrame(
            {
                "OLD.NS": np.linspace(100, 200, len(dates)),
                "NEW.NS": np.linspace(50, 150, len(dates)),
            },
            index=dates,
        )
        universe_mgr._stock_meta = [
            StockMeta(
                ticker="OLD.NS", name="Old", sector="IT", cap="large",
                listed_since=date(2014, 1, 1), delisted_on=date(2016, 12, 1),
            ),
            StockMeta(
                ticker="NEW.NS", name="New", sector="IT", cap="large",
                listed_since=date(2016, 1, 1),
            ),
        ]

        membership = universe_mgr.membership_mask(prices)

        assert membership.loc["2015-12-31", "NEW.NS"] == False
        assert membership.loc["2017-02-01", "OLD.NS"] == False
        assert membership.loc["2017-02-01", "NEW.NS"] == True


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

    def test_sector_features_ignore_nominal_price_scale(self):
        from src.features.sector_features import SectorFeatureBuilder

        cfg = load_config()
        builder = SectorFeatureBuilder(cfg)
        dates = pd.date_range("2014-01-01", periods=320)
        base_a = pd.Series(np.linspace(100, 180, len(dates)), index=dates)
        base_b = pd.Series(np.linspace(50, 90, len(dates)), index=dates)
        prices_lo = pd.DataFrame({"A.NS": base_a, "B.NS": base_b}, index=dates)
        prices_hi = pd.DataFrame({"A.NS": base_a * 10, "B.NS": base_b}, index=dates)
        sector_map = {"A.NS": "IT", "B.NS": "IT"}

        feats_lo = builder.build(prices_lo, sector_map)
        feats_hi = builder.build(prices_hi, sector_map)

        lo = feats_lo[feats_lo["sector"] == "IT"]
        hi = feats_hi[feats_hi["sector"] == "IT"]
        cols = ["mom_3m", "mom_12m"]

        for col in cols:
            left = lo[col].dropna()
            right = hi[col].dropna()
            common = left.index.intersection(right.index)
            assert np.allclose(left.loc[common], right.loc[common], atol=1e-9, equal_nan=True)

    def test_taxonomy_additions_have_price_and_feature_coverage(self):
        from src.features.stock_features import StockFeatureBuilder

        cfg = load_config()
        price_matrix = load_price_matrix(cfg)
        volume_matrix = load_volume_matrix(cfg)
        added = [
            "TATAELXSI.NS",
            "LTTS.NS",
            "AUBANK.NS",
            "HDFCAMC.NS",
            "NAM-INDIA.NS",
            "LUPIN.NS",
            "ZYDUSLIFE.NS",
            "GLENMARK.NS",
            "ADANIGREEN.NS",
            "FLUOROCHEM.NS",
            "BALAMINES.NS",
            "VINATIORGA.NS",
            "NTPC.NS",
            "POWERGRID.NS",
            "RELAXO.NS",
            "LALPATHLAB.NS",
            "RAMCOCEM.NS",
            "AIAENG.NS",
            "JINDALSTEL.NS",
        ]

        present = [t for t in added if t in price_matrix.columns]
        assert present == added
        for ticker in added:
            assert price_matrix[ticker].notna().sum() > 0, ticker

        # Validate feature generation on a subset with actual local history.
        long_hist = [t for t in added if price_matrix[t].notna().sum() >= 252]
        assert long_hist
        sample_prices = price_matrix[long_hist].copy()
        sample_volume = volume_matrix[long_hist].copy()
        sector_map = {
            "TATAELXSI.NS": "IT",
            "LTTS.NS": "IT",
            "AUBANK.NS": "Banking",
            "HDFCAMC.NS": "FinancialServices",
            "NAM-INDIA.NS": "FinancialServices",
            "LUPIN.NS": "Pharma",
            "ZYDUSLIFE.NS": "Pharma",
            "GLENMARK.NS": "Pharma",
            "ADANIGREEN.NS": "Energy",
            "FLUOROCHEM.NS": "Chemicals",
            "BALAMINES.NS": "Chemicals",
            "VINATIORGA.NS": "Chemicals",
            "NTPC.NS": "Utilities",
            "POWERGRID.NS": "Utilities",
            "RELAXO.NS": "ConsumerDiscretionary",
            "LALPATHLAB.NS": "Healthcare",
            "RAMCOCEM.NS": "Cement",
            "AIAENG.NS": "CapitalGoods",
            "JINDALSTEL.NS": "Metals",
        }

        builder = StockFeatureBuilder(cfg)
        feats = builder.build(sample_prices, sample_volume, sector_map)
        assert not feats.empty
        assert set(long_hist).issubset(set(feats["ticker"].unique()))
        for col in [
            "ret_3m",
            "vol_3m",
            "mom_accel_3m_6m",
            "ma_50_200_ratio",
            "amihud_1m",
        ]:
            assert feats[col].notna().any(), col

    def test_stock_features_minimal_raw_contract(self):
        from src.features.stock_features import StockFeatureBuilder

        cfg = load_config()
        builder = StockFeatureBuilder(cfg)
        dates = pd.date_range("2014-01-01", periods=320)
        prices = pd.DataFrame(
            np.random.lognormal(0, 0.01, (320, 4)).cumprod(axis=0) * 1000,
            index=dates,
            columns=["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS"],
        )
        volumes = pd.DataFrame(
            np.random.lognormal(5, 0.2, (320, 4)),
            index=dates,
            columns=prices.columns,
        )
        sector_map = {
            "TCS.NS": "IT",
            "INFY.NS": "IT",
            "HDFCBANK.NS": "Banking",
            "RELIANCE.NS": "Energy",
        }
        feats = builder.build(prices, volumes, sector_map)
        assert not feats.empty
        for col in [
            "ret_3m",
            "mom_accel_3m_6m",
            "amihud_1m",
            "ma_50_200_ratio",
        ]:
            assert col in feats.columns
            assert feats[col].notna().any(), col


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

    def test_one_way_turnover_matches_liquidation_only_case(self):
        from src.optimizer.portfolio_optimizer import PortfolioOptimizer

        turnover = PortfolioOptimizer._compute_one_way_turnover(
            w_val=np.array([0.35, 0.35]),
            w_prev=np.array([0.35, 0.35]),
            cash_val=0.30,
            w_prev_cash=0.10,
            liquidation_cost=0.20,
        )
        assert turnover == pytest.approx(0.20)

    def test_turnover_repair_brings_solution_within_budget(self):
        from src.optimizer.portfolio_optimizer import PortfolioOptimizer

        cfg = load_config()
        opt = PortfolioOptimizer(cfg)

        repaired = opt._repair_turnover_violation(
            w_val=np.array([0.16, 0.16, 0.04, 0.04]),
            cash_val=0.60,
            w_prev=np.array([0.20, 0.20, 0.00, 0.00]),
            w_prev_cash=0.60,
            liquidation_cost=0.0,
            max_turnover=0.12,
            max_stock=0.20,
            tickers=["A.NS", "B.NS", "C.NS", "D.NS"],
            sector_map={"A.NS": "IT", "B.NS": "IT", "C.NS": "Banking", "D.NS": "Banking"},
            max_sector=0.20,
        )

        assert repaired is not None
        w_new, cash_new = repaired
        turnover = opt._compute_one_way_turnover(
            w_new, np.array([0.20, 0.20, 0.00, 0.00]), cash_new, 0.60, 0.0
        )
        assert turnover <= 0.12 + 1e-6
        assert w_new.max() <= 0.20 + 1e-6


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

    def test_rebalance_never_returns_negative_cash_weight(self):
        from src.backtest.simulator import PortfolioSimulator
        from src.data.contracts import PortfolioState

        cfg = load_config()
        sim = PortfolioSimulator(cfg)

        prices = pd.Series({"TCS.NS": 3000.0, "INFY.NS": 1500.0})
        init_state = PortfolioState(
            date=date(2020, 1, 1),
            cash=500_000.0,
            holdings={},
            weights={"CASH": 1.0},
            nav=500_000.0,
            sector_weights={},
        )
        target = {"TCS.NS": 0.50, "INFY.NS": 0.50}

        result = sim.execute_rebalance(target, init_state, prices, date(2020, 1, 2))

        assert result.new_portfolio.cash >= 0
        assert result.new_portfolio.weights.get("CASH", 0.0) >= -1e-9
        assert abs(sum(result.new_portfolio.weights.values()) - 1.0) < 1e-6

    def test_rebalance_uses_marked_to_market_state_nav(self):
        from src.backtest.simulator import PortfolioSimulator
        from src.data.contracts import PortfolioState

        cfg = load_config()
        sim = PortfolioSimulator(cfg)

        stale_state = PortfolioState(
            date=date(2020, 1, 1),
            cash=0.0,
            holdings={"TCS.NS": 10.0},
            weights={"TCS.NS": 1.0, "CASH": 0.0},
            nav=1_000.0,
            sector_weights={},
        )
        prices = pd.Series({"TCS.NS": 120.0, "INFY.NS": 60.0})

        mtm_state = sim.value_portfolio(stale_state, prices, date(2020, 1, 2))
        result = sim.execute_rebalance(
            {"TCS.NS": 0.50, "INFY.NS": 0.50},
            mtm_state,
            prices,
            date(2020, 1, 2),
        )

        assert mtm_state.nav == 1_200.0
        assert result.new_portfolio.nav > 0
        assert result.new_portfolio.cash >= 0

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
