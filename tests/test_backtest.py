"""Integration tests for the walk-forward backtest engine."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config


def _make_synthetic_data(n_tickers=10, n_days=600):
    """Generate synthetic price/volume/macro data for testing."""
    dates = pd.date_range("2013-01-01", periods=n_days, freq="B")
    np.random.seed(42)

    tickers = [
        "TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS",
        "SUNPHARMA.NS", "TATASTEEL.NS", "MARUTI.NS", "ITC.NS",
        "BHARTIARTL.NS", "ULTRACEMCO.NS",
    ][:n_tickers]
    benchmark = "^NSEI"

    # Geometric Brownian Motion
    prices = {}
    for t in tickers + [benchmark]:
        drift = np.random.uniform(0.0002, 0.0006)
        vol = np.random.uniform(0.01, 0.025)
        rets = np.random.normal(drift, vol, n_days)
        prices[t] = np.cumprod(1 + rets) * np.random.uniform(500, 3000)

    price_matrix = pd.DataFrame(prices, index=dates)
    price_matrix.index.name = "date"

    volume_matrix = pd.DataFrame(
        np.random.lognormal(3, 0.5, (n_days, n_tickers)),
        index=dates, columns=tickers
    )
    volume_matrix.index.name = "date"

    macro_df = pd.DataFrame({
        "vix": np.random.uniform(12, 40, n_days),
        "usdinr": np.random.uniform(60, 85, n_days),
        "crude_oil": np.random.uniform(30, 100, n_days),
        "sp500": np.cumprod(1 + np.random.normal(0.0003, 0.01, n_days)) * 2000,
        "gold": np.cumprod(1 + np.random.normal(0.0001, 0.005, n_days)) * 1200,
        "us_10y": np.random.uniform(1, 5, n_days),
        "rbi_repo_rate": 6.5,
        "rbi_meeting": 0.0,
        "budget_day": 0.0,
        "election_window": 0.0,
    }, index=dates)

    return price_matrix, volume_matrix, macro_df


class TestWalkForwardIntegration:
    """Lightweight integration test using synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        return _make_synthetic_data(n_tickers=10, n_days=800)

    def test_walkforward_runs_without_error(self, synthetic_data):
        """Core integration test: engine runs end-to-end."""
        cfg = load_config()
        cfg["backtest"]["start_date"] = "2013-01-01"
        cfg["backtest"]["end_date"] = "2015-12-31"
        cfg["backtest"]["min_train_years"] = 1
        cfg["rl"]["use_rl"] = False   # skip RL for speed
        cfg["sector_model"]["n_estimators"] = 10
        cfg["stock_model"]["n_estimators"] = 10

        from src.features.macro_features import MacroFeatureBuilder
        price_matrix, volume_matrix, macro_df = synthetic_data

        # Build macro features
        macro_fb = MacroFeatureBuilder(cfg)
        macro_features = macro_fb.build(macro_df)

        from src.backtest.walk_forward import WalkForwardEngine
        engine = WalkForwardEngine(
            price_matrix=price_matrix,
            volume_matrix=volume_matrix,
            macro_df=macro_df,
            cfg=cfg,
            use_rl=False,
        )

        metrics = engine.run()
        assert "cagr" in metrics
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics
        assert metrics["total_rebalances"] > 0
        assert not engine.nav_series.empty

    def test_nav_always_positive(self, synthetic_data):
        cfg = load_config()
        cfg["backtest"]["start_date"] = "2013-01-01"
        cfg["backtest"]["end_date"] = "2014-12-31"
        cfg["backtest"]["min_train_years"] = 1
        cfg["rl"]["use_rl"] = False
        cfg["sector_model"]["n_estimators"] = 5
        cfg["stock_model"]["n_estimators"] = 5

        price_matrix, volume_matrix, macro_df = synthetic_data

        from src.backtest.walk_forward import WalkForwardEngine
        engine = WalkForwardEngine(
            price_matrix=price_matrix,
            volume_matrix=volume_matrix,
            macro_df=macro_df,
            cfg=cfg,
            use_rl=False,
        )
        engine.run()

        nav = engine.nav_series.dropna()
        assert (nav >= 0).all(), "NAV should never go negative"

    def test_rebalance_records_structure(self, synthetic_data):
        cfg = load_config()
        cfg["backtest"]["start_date"] = "2013-01-01"
        cfg["backtest"]["end_date"] = "2014-06-30"
        cfg["backtest"]["min_train_years"] = 1
        cfg["rl"]["use_rl"] = False
        cfg["sector_model"]["n_estimators"] = 5
        cfg["stock_model"]["n_estimators"] = 5

        price_matrix, volume_matrix, macro_df = synthetic_data

        from src.backtest.walk_forward import WalkForwardEngine
        engine = WalkForwardEngine(
            price_matrix=price_matrix,
            volume_matrix=volume_matrix,
            macro_df=macro_df,
            cfg=cfg,
            use_rl=False,
        )
        engine.run()

        for rec in engine.rebalance_records:
            assert rec.pre_nav > 0
            assert rec.post_nav > 0
            total_w = sum(rec.target_weights.values())
            assert abs(total_w - 1.0) < 0.01, f"Weights don't sum to 1: {total_w}"

    def test_stock_fwd_window_override_is_respected(self, synthetic_data):
        cfg = load_config()
        cfg["backtest"]["start_date"] = "2013-01-01"
        cfg["backtest"]["end_date"] = "2014-06-30"
        cfg["backtest"]["min_train_years"] = 1
        cfg["rl"]["use_rl"] = False
        cfg["sector_model"]["n_estimators"] = 5
        cfg["stock_model"]["n_estimators"] = 5
        cfg["sector_model"]["fwd_window_days"] = 28
        cfg["stock_model"]["fwd_window_days"] = 56

        price_matrix, volume_matrix, macro_df = synthetic_data

        from src.backtest.walk_forward import WalkForwardEngine
        engine = WalkForwardEngine(
            price_matrix=price_matrix,
            volume_matrix=volume_matrix,
            macro_df=macro_df,
            cfg=cfg,
            use_rl=False,
        )
        assert engine.sector_fwd_window_days == 28
        assert engine.stock_fwd_window_days == 56

    def test_retrain_frequency_weeks_are_converted_to_rebalances(self, synthetic_data):
        cfg = load_config()
        cfg["backtest"]["start_date"] = "2013-01-01"
        cfg["backtest"]["end_date"] = "2014-06-30"
        cfg["backtest"]["min_train_years"] = 1
        cfg["backtest"]["rebalance_freq_weeks"] = 4
        cfg["rl"]["use_rl"] = False
        cfg["sector_model"]["n_estimators"] = 5
        cfg["stock_model"]["n_estimators"] = 5
        cfg["sector_model"].pop("retrain_every_rebalances", None)
        cfg["stock_model"].pop("retrain_every_rebalances", None)
        cfg["rl"].pop("retrain_every_rebalances", None)
        cfg["sector_model"]["retrain_freq_weeks"] = 12
        cfg["stock_model"]["retrain_freq_weeks"] = 12
        cfg["rl"]["retrain_freq_weeks"] = 12

        price_matrix, volume_matrix, macro_df = synthetic_data

        from src.backtest.walk_forward import WalkForwardEngine
        engine = WalkForwardEngine(
            price_matrix=price_matrix,
            volume_matrix=volume_matrix,
            macro_df=macro_df,
            cfg=cfg,
            use_rl=False,
        )

        assert engine._retrain_every_rebalances("sector_model", fallback_weeks=4) == 3
        assert engine._retrain_every_rebalances("stock_model", fallback_weeks=12) == 3
        assert engine._retrain_every_rebalances("rl", fallback_weeks=12) == 3
