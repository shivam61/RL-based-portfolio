from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.contracts import PortfolioState
from src.features.portfolio_features import compute_portfolio_features


def test_compute_portfolio_features_emits_control_features_without_nans():
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    recent_returns = pd.Series(
        np.linspace(-0.01, 0.012, len(dates)),
        index=dates,
        dtype=float,
    )
    benchmark_returns = pd.Series(
        np.linspace(-0.005, 0.008, len(dates)),
        index=dates,
        dtype=float,
    )
    state = PortfolioState(
        date=date(2024, 4, 30),
        cash=125000.0,
        holdings={"AAA.NS": 10.0, "BBB.NS": 5.0},
        weights={"AAA.NS": 0.35, "BBB.NS": 0.40, "CASH": 0.25},
        nav=500000.0,
        sector_weights={"IT": 0.35, "Banking": 0.40},
    )

    features = compute_portfolio_features(
        state,
        recent_returns,
        benchmark_returns,
        control_context={
            "market_breadth_3m": 0.42,
            "recent_turnover_3p": 0.31,
            "recent_cost_ratio_3p": 0.002,
            "risk_cash_floor": 0.15,
            "emergency_rebalance": 1.0,
            "current_stress_signal": 0.45,
            "previous_stress_signal": 0.20,
            "target_posture_score": 1.0,
            "previous_posture_score": 0.0,
            "previous_target_posture_score": 1.0,
            "target_posture_streak": 3.0,
            "previous_posture_mismatch": 0.5,
        },
    )

    expected = {
        "drawdown_slope_1m",
        "vol_shock_1m_3m",
        "breadth_deterioration",
        "recent_turnover_3p",
        "recent_cost_ratio_3p",
        "risk_cash_floor",
        "emergency_flag",
        "current_stress_signal",
        "previous_stress_signal",
        "target_posture_score",
        "previous_posture_score",
        "previous_target_posture_score",
        "target_posture_streak",
        "previous_posture_mismatch",
    }
    assert expected <= set(features)
    assert all(np.isfinite(float(features[key])) for key in expected)
    assert features["breadth_deterioration"] == pytest.approx(0.58)
    assert features["recent_turnover_3p"] == pytest.approx(0.31)
    assert features["risk_cash_floor"] == pytest.approx(0.15)
    assert features["emergency_flag"] == pytest.approx(1.0)
    assert features["current_stress_signal"] == pytest.approx(0.45)
    assert features["target_posture_score"] == pytest.approx(1.0)
    assert features["target_posture_streak"] == pytest.approx(3.0)


def test_compute_portfolio_features_defaults_and_clips_control_inputs():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    recent_returns = pd.Series(np.linspace(-0.02, 0.01, len(dates)), index=dates, dtype=float)
    state = PortfolioState(
        date=date(2024, 2, 12),
        cash=50000.0,
        holdings={"AAA.NS": 10.0},
        weights={"AAA.NS": 0.9, "CASH": 0.1},
        nav=100000.0,
        sector_weights={"IT": 0.9},
    )

    features = compute_portfolio_features(
        state,
        recent_returns,
        benchmark_returns=None,
        control_context={
            "market_breadth_3m": 1.8,
            "recent_turnover_3p": -0.2,
            "recent_cost_ratio_3p": -0.001,
            "risk_cash_floor": -0.3,
            "emergency_rebalance": 0.0,
            "current_stress_signal": 2.0,
            "previous_stress_signal": -1.0,
            "target_posture_score": 2.0,
            "previous_posture_score": -2.0,
            "previous_target_posture_score": 3.0,
            "target_posture_streak": 12.0,
            "previous_posture_mismatch": 2.0,
        },
    )

    assert features["breadth_deterioration"] == pytest.approx(0.0)
    assert features["recent_turnover_3p"] == pytest.approx(0.0)
    assert features["recent_cost_ratio_3p"] == pytest.approx(0.0)
    assert features["risk_cash_floor"] == pytest.approx(0.0)
    assert features["emergency_flag"] == pytest.approx(0.0)
    assert features["current_stress_signal"] == pytest.approx(1.0)
    assert features["previous_stress_signal"] == pytest.approx(0.0)
    assert features["target_posture_score"] == pytest.approx(1.0)
    assert features["previous_posture_score"] == pytest.approx(-1.0)
    assert features["previous_target_posture_score"] == pytest.approx(1.0)
    assert features["target_posture_streak"] == pytest.approx(6.0)
    assert features["previous_posture_mismatch"] == pytest.approx(1.0)
