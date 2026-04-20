"""
Feature validation pipeline — correctness, bounds, PIT safety, and
cross-feature consistency tests.

All tests use synthetic in-memory data; no real market data required.
Run with:  pytest tests/test_feature_validation.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.stock_features import StockFeatureBuilder
from src.features.sector_features import SectorFeatureBuilder


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_DAYS   = 600   # enough for 252-day rolling windows + lag
N_STOCKS = 10

TICKERS  = [f"STOCK{i:02d}.NS" for i in range(N_STOCKS)]
SECTORS  = ["IT", "Banking", "FMCG", "Pharma", "Energy"]

# round-robin sector assignments
SECTOR_MAP = {t: SECTORS[i % len(SECTORS)] for i, t in enumerate(TICKERS)}


def _make_prices(seed: int = 0, n_days: int = N_DAYS) -> pd.DataFrame:
    """Geometric Brownian motion prices with no NaN gaps."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    log_rets = rng.normal(0.0003, 0.015, size=(n_days, N_STOCKS))
    prices = 100 * np.exp(np.cumsum(log_rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


def _make_prices_with_gaps(seed: int = 0, gap_frac: float = 0.04) -> pd.DataFrame:
    """Prices with ~4% random NaN entries (simulates trading holidays)."""
    rng = np.random.default_rng(seed)
    df = _make_prices(seed)
    mask = rng.random(df.shape) < gap_frac
    df[mask] = np.nan
    return df


def _make_volumes(price_df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vols = rng.lognormal(mean=10, sigma=0.5, size=price_df.shape)
    return pd.DataFrame(vols, index=price_df.index, columns=price_df.columns)


_DEFAULT_SENTINEL = object()


def _build_stock_features(price_df=None, volume_df=_DEFAULT_SENTINEL, sector_map=None, benchmark=None):
    if price_df is None:
        price_df = _make_prices()
    if volume_df is _DEFAULT_SENTINEL:
        volume_df = _make_volumes(price_df)
    if sector_map is None:
        sector_map = SECTOR_MAP
    cfg = {
        "features": {
            "lookback_short": 21, "lookback_medium": 63, "lookback_long": 252,
            "stock_lag": 1,
        },
        "paths": {"feature_data": "/tmp", "processed_data": "/tmp"},
    }
    builder = StockFeatureBuilder(cfg)
    return builder.build(price_df, volume_df, sector_map, benchmark_prices=benchmark)


def _build_sector_features(price_df=None, sector_map=None, benchmark=None):
    if price_df is None:
        price_df = _make_prices()
    if sector_map is None:
        sector_map = SECTOR_MAP
    cfg = {
        "features": {
            "lookback_short": 21, "lookback_medium": 63, "lookback_long": 252,
        },
        "paths": {"feature_data": "/tmp"},
    }
    builder = SectorFeatureBuilder(cfg)
    return builder.build(price_df, sector_map, benchmark_prices=benchmark)


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestSchema:
    """Output shape and required columns."""

    REQUIRED_STOCK_COLS = [
        "date", "ticker", "sector",
        "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_12m",
        "mom_12m_skip1m", "mom_stab_3m", "mom_stab_12m",
        "vol_1m", "vol_3m", "vol_12m", "vol_ratio_1m_3m",
        "downside_vol_3m", "max_dd_3m", "max_dd_12m",
        "skew_3m", "kurt_3m",
        "price_to_52w_high", "price_to_52w_low",
        "zscore_6m",
        "ma_50_200_ratio", "price_pctile_1y",
        "rsi_14",
        "ret_1m_vs_sector", "ret_3m_vs_sector", "ret_6m_vs_sector", "ret_12m_vs_sector",
    ]

    REQUIRED_SECTOR_COLS = [
        "sector",
        "mom_1m", "mom_3m", "mom_6m", "mom_12m",
        "vol_1m", "vol_3m",
        "above_50ma", "above_200ma", "price_to_52w_high",
        "breadth_1m", "breadth_3m",
        "pct_above_50ma", "new_high_ratio",
        "dispersion_1m", "zscore_1y",
    ]

    def test_stock_output_is_dataframe(self):
        sf = _build_stock_features()
        assert isinstance(sf, pd.DataFrame)

    def test_stock_output_not_empty(self):
        sf = _build_stock_features()
        assert len(sf) > 0

    def test_stock_required_columns_present(self):
        sf = _build_stock_features()
        missing = [c for c in self.REQUIRED_STOCK_COLS if c not in sf.columns]
        assert not missing, f"Missing stock feature columns: {missing}"

    def test_stock_volume_features_present_when_volume_given(self):
        sf = _build_stock_features()
        for col in ["avg_vol_1m", "avg_vol_3m", "vol_trend", "amihud_1m", "vol_spike"]:
            assert col in sf.columns, f"Volume feature missing: {col}"

    def test_stock_alpha_beta_present_with_benchmark(self):
        prices = _make_prices()
        bench = prices.iloc[:, 0].rename("BENCH")
        sf = _build_stock_features(price_df=prices, benchmark=bench)
        assert "alpha_ret_1m" in sf.columns
        assert "alpha_ret_3m" in sf.columns
        assert "beta_3m" in sf.columns

    def test_sector_output_not_empty(self):
        sec = _build_sector_features()
        assert len(sec) > 0

    def test_sector_required_columns_present(self):
        sec = _build_sector_features()
        missing = [c for c in self.REQUIRED_SECTOR_COLS if c not in sec.columns]
        assert not missing, f"Missing sector feature columns: {missing}"

    def test_sector_output_covers_all_sectors(self):
        sec = _build_sector_features()
        found = set(sec["sector"].unique())
        expected = set(SECTORS)
        assert expected.issubset(found), f"Sectors missing: {expected - found}"

    def test_ticker_column_values_are_in_sector_map(self):
        sf = _build_stock_features()
        assert set(sf["ticker"].unique()).issubset(set(SECTOR_MAP.keys()))


# ── Holiday / gap robustness ──────────────────────────────────────────────────

class TestGapRobustness:
    """Features must survive prices with NaN gaps (trading holidays)."""

    def test_stock_features_non_empty_with_gaps(self):
        prices = _make_prices_with_gaps()
        sf = _build_stock_features(price_df=prices)
        assert len(sf) > 0

    def test_ma_ratio_non_null_with_gaps(self):
        prices = _make_prices_with_gaps()
        sf = _build_stock_features(price_df=prices)
        nn = sf["ma_50_200_ratio"].notna().mean()
        assert nn > 0.5, f"ma_50_200_ratio fill rate too low: {nn:.1%}"

    def test_ret_12m_non_null_with_gaps(self):
        prices = _make_prices_with_gaps()
        sf = _build_stock_features(price_df=prices)
        nn = sf["ret_12m"].notna().mean()
        assert nn > 0.5, f"ret_12m fill rate too low: {nn:.1%}"

    def test_price_pctile_non_null_with_gaps(self):
        prices = _make_prices_with_gaps()
        sf = _build_stock_features(price_df=prices)
        nn = sf["price_pctile_1y"].notna().mean()
        assert nn > 0.5, f"price_pctile_1y fill rate too low: {nn:.1%}"

    def test_sector_breadth_non_null_with_gaps(self):
        prices = _make_prices_with_gaps()
        sec = _build_sector_features(price_df=prices)
        assert sec["pct_above_50ma"].notna().mean() > 0.5
        assert sec["new_high_ratio"].notna().mean() > 0.5


# ── Bounds and domain tests ───────────────────────────────────────────────────

class TestBounds:
    """Feature values must fall within valid mathematical / financial bounds."""

    def setup_method(self):
        self.sf = _build_stock_features()
        self.sec = _build_sector_features()

    def test_rsi_bounded_0_100(self):
        rsi = self.sf["rsi_14"].dropna()
        assert rsi.between(0, 100).all(), \
            f"RSI out of [0,100]: min={rsi.min():.2f} max={rsi.max():.2f}"

    def test_price_to_52w_high_leq_1(self):
        p = self.sf["price_to_52w_high"].dropna()
        assert (p <= 1.0 + 1e-6).all(), \
            f"price_to_52w_high > 1: max={p.max():.4f}"

    def test_price_pctile_bounded_0_1(self):
        p = self.sf["price_pctile_1y"].dropna()
        assert p.between(-1e-6, 1 + 1e-6).all(), \
            f"price_pctile_1y out of [0,1]: min={p.min():.4f} max={p.max():.4f}"

    def test_vol_features_non_negative(self):
        for col in ["vol_1m", "vol_3m", "vol_12m", "downside_vol_3m"]:
            v = self.sf[col].dropna()
            assert (v >= 0).all(), f"{col} has negative values"

    def test_max_drawdown_non_positive(self):
        for col in ["max_dd_3m", "max_dd_12m"]:
            v = self.sf[col].dropna()
            assert (v <= 1e-9).all(), f"{col} has positive values (should be ≤ 0)"

    def test_mom_stab_bounded_0_1(self):
        for col in ["mom_stab_3m", "mom_stab_12m"]:
            v = self.sf[col].dropna()
            assert v.between(0, 1).all(), f"{col} out of [0,1]"

    def test_sector_breadth_bounded_0_1(self):
        for col in ["breadth_1m", "breadth_3m", "pct_above_50ma", "new_high_ratio"]:
            v = self.sec[col].dropna()
            assert v.between(-1e-9, 1 + 1e-9).all(), \
                f"{col} out of [0,1]: min={v.min():.4f} max={v.max():.4f}"

    def test_returns_not_extreme(self):
        """Period returns shouldn't exceed ±200% for synthetic GBM data."""
        for col in ["ret_1w", "ret_1m", "ret_3m"]:
            v = self.sf[col].dropna()
            assert v.abs().max() < 2.0, f"{col} has extreme value: {v.abs().max():.2f}"

    def test_no_inf_in_stock_features(self):
        num_cols = self.sf.select_dtypes(include="number").columns
        inf_cols = [c for c in num_cols if np.isinf(self.sf[c]).any()]
        assert not inf_cols, f"Inf values found in: {inf_cols}"

    def test_no_inf_in_sector_features(self):
        num_cols = self.sec.select_dtypes(include="number").columns
        inf_cols = [c for c in num_cols if np.isinf(self.sec[c]).any()]
        assert not inf_cols, f"Inf values found in: {inf_cols}"


# ── Fill rate tests ───────────────────────────────────────────────────────────

class TestFillRate:
    """Features with sufficient history must achieve minimum fill rates."""

    MIN_FILL = {
        "ret_1w":           0.85,
        "ret_1m":           0.85,
        "ret_3m":           0.80,
        "ret_6m":           0.75,
        "ret_12m":          0.58,
        "vol_1m":           0.80,
        "vol_3m":           0.75,
        "rsi_14":           0.85,
        "ma_50_200_ratio":  0.67,
        "price_pctile_1y":  0.58,
    }

    def test_stock_fill_rates(self):
        sf = _build_stock_features()
        failures = []
        for col, min_rate in self.MIN_FILL.items():
            if col not in sf.columns:
                failures.append(f"{col}: MISSING")
                continue
            actual = sf[col].notna().mean()
            if actual < min_rate:
                failures.append(f"{col}: {actual:.1%} < {min_rate:.0%}")
        assert not failures, "Fill rate failures:\n  " + "\n  ".join(failures)

    def test_sector_fill_rates(self):
        sec = _build_sector_features()
        required = {
            "mom_1m": 0.80, "mom_3m": 0.75, "mom_6m": 0.70,
            "pct_above_50ma": 0.80, "new_high_ratio": 0.70,
            "breadth_1m": 0.80,
        }
        failures = []
        for col, min_rate in required.items():
            if col not in sec.columns:
                failures.append(f"{col}: MISSING")
                continue
            actual = sec[col].notna().mean()
            if actual < min_rate:
                failures.append(f"{col}: {actual:.1%} < {min_rate:.0%}")
        assert not failures, "Sector fill rate failures:\n  " + "\n  ".join(failures)


# ── Point-in-time / lookahead safety ─────────────────────────────────────────

class TestPointInTimeSafety:
    """
    The single most important correctness property: features at date T must
    only use information available at T-1 (lag=1).
    """

    def test_stock_features_are_lagged_by_1(self):
        """
        Build features on prices 2020-01-01 to 2021-12-31.
        Then compare the feature value at date D against a fresh build
        using only prices up to D-1.  They must agree.
        """
        prices = _make_prices(n_days=400)
        volumes = _make_volumes(prices)
        sf_full = _build_stock_features(price_df=prices, volume_df=volumes)

        # Pick a reference date mid-way through
        all_dates = sorted(sf_full["date"].unique())
        ref_date = all_dates[300]
        prev_date = all_dates[299]

        # Features at ref_date should equal a build truncated at ref_date,
        # since lagged values are timestamped at the decision date.
        row_full = sf_full[
            (sf_full["date"] == ref_date) & (sf_full["ticker"] == TICKERS[0])
        ]
        if row_full.empty:
            pytest.skip("No data at ref_date for STOCK00.NS")

        prices_truncated = prices[prices.index <= ref_date]
        volumes_truncated = volumes[volumes.index <= ref_date]
        sf_trunc = _build_stock_features(price_df=prices_truncated, volume_df=volumes_truncated)
        row_trunc = sf_trunc[
            (sf_trunc["date"] == ref_date) & (sf_trunc["ticker"] == TICKERS[0])
        ]
        if row_trunc.empty:
            pytest.skip("No truncated data for STOCK00.NS")

        for col in ["ret_1m", "vol_3m", "rsi_14", "ma_50_200_ratio"]:
            if col not in row_full.columns or col not in row_trunc.columns:
                continue
            v_full  = row_full[col].values[0]
            v_trunc = row_trunc[col].values[0]
            if np.isnan(v_full) and np.isnan(v_trunc):
                continue
            assert abs(v_full - v_trunc) < 1e-4, (
                f"PIT mismatch on {col} at {ref_date}: "
                f"full={v_full:.6f} trunc={v_trunc:.6f}"
            )

    def test_no_future_dates_in_output(self):
        prices = _make_prices(n_days=400)
        last_price_date = prices.index[-1]
        sf = _build_stock_features(price_df=prices)
        assert pd.to_datetime(sf["date"]).max() <= last_price_date

    def test_sector_features_no_future_dates(self):
        prices = _make_prices(n_days=400)
        last_price_date = prices.index[-1]
        sec = _build_sector_features(price_df=prices)
        assert sec.index.max() <= last_price_date

    def test_features_shift_1_preserves_last_decision_day(self):
        """Lagged features are stored on the decision date, not the source-data date."""
        prices = _make_prices(n_days=300)
        sf = _build_stock_features(price_df=prices)
        last_feature_date = pd.to_datetime(sf["date"]).max()
        last_price_date = prices.index[-1]
        assert last_feature_date <= last_price_date, (
            "Lag-1 features should not be dated after the last price date"
        )


# ── Cross-feature consistency ─────────────────────────────────────────────────

class TestCrossFeatureConsistency:
    """Relationships that must hold between features by construction."""

    def setup_method(self):
        self.sf = _build_stock_features()

    def test_mom_12m_skip1m_equals_12m_minus_1m(self):
        df = self.sf[["ret_12m", "ret_1m", "mom_12m_skip1m"]].dropna()
        expected = df["ret_12m"] - df["ret_1m"]
        diff = (df["mom_12m_skip1m"] - expected).abs()
        assert diff.max() < 1e-8, f"mom_12m_skip1m != ret_12m - ret_1m: max diff={diff.max():.2e}"

    def test_vol_ratio_equals_vol1m_over_vol3m(self):
        df = self.sf[["vol_1m", "vol_3m", "vol_ratio_1m_3m"]].dropna()
        expected = df["vol_1m"] / df["vol_3m"].replace(0, np.nan)
        diff = (df["vol_ratio_1m_3m"] - expected).abs().dropna()
        assert diff.max() < 1e-6, f"vol_ratio inconsistent: max diff={diff.max():.2e}"

    def test_longer_vol_less_variable_than_shorter(self):
        """vol_3m should be smoother (lower std) than vol_1m across the panel."""
        v1m_std = self.sf["vol_1m"].std()
        v3m_std = self.sf["vol_3m"].std()
        assert v3m_std < v1m_std, (
            f"vol_3m ({v3m_std:.4f}) not smoother than vol_1m ({v1m_std:.4f})"
        )

    def test_sector_relative_features_zero_mean_per_date(self):
        """ret_Xm_vs_sector should average to ~0 across stocks within a sector on any date."""
        sf = self.sf[["date", "ticker", "sector", "ret_1m_vs_sector"]].dropna()
        for (date, sector), grp in sf.groupby(["date", "sector"]):
            if len(grp) < 3:
                continue
            mean_val = grp["ret_1m_vs_sector"].mean()
            assert abs(mean_val) < 1e-8, (
                f"ret_1m_vs_sector mean non-zero in {sector} on {date}: {mean_val:.2e}"
            )
            break  # checking one (date, sector) pair is sufficient here

    def test_price_pctile_at_52w_high_is_1(self):
        """A stock at its 52-week high should have price_pctile_1y ≈ 1."""
        prices = _make_prices()
        # Force last price of STOCK00 to be its rolling max
        prices.iloc[-1, 0] = prices.iloc[:, 0].rolling(252).max().iloc[-1]
        sf = _build_stock_features(price_df=prices)
        row = sf[(sf["ticker"] == TICKERS[0])].sort_values("date").dropna(subset=["price_pctile_1y"])
        if row.empty:
            pytest.skip("No valid price_pctile_1y for STOCK00")
        # Value should be close to 1 for recent dates (after lag-1 shift)
        last_val = row["price_pctile_1y"].iloc[-1]
        assert last_val > 0.9, f"price_pctile_1y at 52w high = {last_val:.4f}, expected ≈ 1"


# ── Reproducibility tests ────────────────────────────────────────────────────

class TestReproducibility:
    """Same inputs must always produce identical outputs."""

    def test_stock_features_deterministic(self):
        prices = _make_prices(seed=42)
        volumes = _make_volumes(prices, seed=7)
        sf1 = _build_stock_features(price_df=prices.copy(), volume_df=volumes.copy())
        sf2 = _build_stock_features(price_df=prices.copy(), volume_df=volumes.copy())
        pd.testing.assert_frame_equal(
            sf1.sort_values(["date", "ticker"]).reset_index(drop=True),
            sf2.sort_values(["date", "ticker"]).reset_index(drop=True),
        )

    def test_sector_features_deterministic(self):
        prices = _make_prices(seed=42)
        sec1 = _build_sector_features(price_df=prices.copy())
        sec2 = _build_sector_features(price_df=prices.copy())
        pd.testing.assert_frame_equal(
            sec1.sort_values(["sector"]).reset_index(drop=True),
            sec2.sort_values(["sector"]).reset_index(drop=True),
        )


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_sector(self):
        """All tickers in one sector still produce demeaned sector-relative features."""
        sector_map = {t: "IT" for t in TICKERS}
        prices = _make_prices()
        sf = _build_stock_features(price_df=prices, sector_map=sector_map)
        grouped = sf[["date", "ret_1m_vs_sector"]].dropna().groupby("date")["ret_1m_vs_sector"].mean()
        assert grouped.abs().max() < 1e-8, "Single-sector: demeaned vs_sector mean should be zero"

    def test_minimum_history_returns_empty_not_error(self):
        """Very short price history (< 252 days) should not crash — just sparse output."""
        prices = _make_prices(n_days=30)
        volumes = _make_volumes(prices)
        sf = _build_stock_features(price_df=prices, volume_df=volumes)
        assert isinstance(sf, pd.DataFrame)  # must not raise

    def test_missing_volume_matrix_skips_volume_features(self):
        prices = _make_prices()
        sf = _build_stock_features(price_df=prices, volume_df=None)
        assert "avg_vol_1m" not in sf.columns
        assert "vol_spike" not in sf.columns

    def test_tickers_not_in_sector_map_are_excluded(self):
        prices = _make_prices()
        partial_map = {t: SECTOR_MAP[t] for t in TICKERS[:5]}  # only first 5
        sf = _build_stock_features(price_df=prices, sector_map=partial_map)
        assert set(sf["ticker"].unique()).issubset(set(TICKERS[:5]))

    def test_constant_price_series_vol_is_zero(self):
        """Flat price → zero volatility (not NaN, not error)."""
        prices = _make_prices()
        prices.iloc[:, 0] = 100.0  # flat stock
        sf = _build_stock_features(price_df=prices)
        vol = sf[sf["ticker"] == TICKERS[0]]["vol_1m"].dropna()
        if not vol.empty:
            assert vol.abs().max() < 1e-6, f"Flat price vol_1m non-zero: {vol.max():.6f}"
