"""Unit tests for stock-ranker horizon blending."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models.stock_ranker import StockRanker


def _make_blend_fixture():
    dates = pd.date_range("2018-01-01", periods=260, freq="B")
    tickers = [f"T{i}.NS" for i in range(8)]

    rng = np.random.default_rng(42)
    prices = {}
    for i, ticker in enumerate(tickers):
        drift = 0.0003 + i * 0.00005
        vol = 0.012 + i * 0.001
        rets = rng.normal(drift, vol, len(dates))
        prices[ticker] = np.cumprod(1 + rets) * (100 + i * 10)
    price_matrix = pd.DataFrame(prices, index=dates)
    price_matrix.index.name = "date"

    rows: list[dict[str, object]] = []
    for dt in dates:
        for i, ticker in enumerate(tickers):
            base = float(price_matrix.loc[dt, ticker])
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "sector": "IT",
                    "ret_3m": base / 100.0,
                    "mom_12m_skip1m": base / 120.0,
                    "mom_accel_3m_6m": base / 140.0,
                    "vol_3m": abs(np.sin(i + dt.dayofyear / 90.0)) + 0.1,
                    "amihud_1m": abs(np.cos(i + dt.dayofyear / 60.0)) + 0.05,
                    "ma_50_200_ratio": base / 150.0,
                }
            )
    stock_features = pd.DataFrame(rows)
    return stock_features, price_matrix


def test_stock_ranker_can_blend_multiple_horizons():
    cfg = load_config()
    cfg["stock_model"]["n_estimators"] = 5
    cfg["stock_model"]["blend_horizons_days"] = [28, 56, 84]
    cfg["stock_model"]["blend_validation_fraction"] = 0.2
    cfg["stock_model"]["blend_grid_step"] = 0.25

    stock_features, price_matrix = _make_blend_fixture()

    ranker = StockRanker(cfg)
    ranker.fit(stock_features, price_matrix, fwd_window=cfg["stock_model"]["blend_horizons_days"])

    assert ranker.is_fitted
    assert ranker.horizons == [28, 56, 84]
    assert abs(sum(ranker.blend_weights.values()) - 1.0) < 1e-6
    assert set(ranker.models.keys()) == {"IT"}

    snap = stock_features[stock_features["date"] == stock_features["date"].max()].copy()
    ranked = ranker.rank_stocks(snap, "IT", top_k=3)
    assert not ranked.empty
    assert list(ranked.columns) == ["ticker", "score", "rank"]
    assert len(ranked) == 3
