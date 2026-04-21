"""
Tests for FeatureStore — correctness, PIT safety, incremental append, dtype optimisation.
All tests use purely synthetic in-memory data; no real market data required.
"""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_store import FeatureStore, _read_shard


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_store(tmp_path):
    """Return a fresh FeatureStore backed by a temp directory."""
    cfg = {
        "backtest": {"benchmark_ticker": "^NSEI"},
        "paths": {"artifact_dir": str(tmp_path)},
    }
    return FeatureStore(tmp_path / "feature_store", cfg)


def _make_macro_df(start="2020-01-01", end="2020-12-31") -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.standard_normal((len(idx), 5)),
        index=idx,
        columns=["vix_level", "usdinr_ret_1m", "crude_ret_1m", "sp500_ret_1m", "risk_on_score"],
    )


def _make_sector_df(start="2020-01-01", end="2020-12-31") -> pd.DataFrame:
    sectors = ["IT", "Banking", "FMCG"]
    idx = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(1)
    rows = []
    for d in idx:
        for sec in sectors:
            row = {"sector": sec, "mom_1m": rng.standard_normal(), "mom_3m": rng.standard_normal()}
            rows.append(row)
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex([d for d in idx for _ in sectors])
    return df


def _make_stock_df(start="2020-01-01", end="2020-12-31") -> pd.DataFrame:
    tickers = ["TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    dates = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(2)
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t, "ret_1m": rng.standard_normal(),
                         "vol_1m": abs(rng.standard_normal()), "sector": "IT"})
    return pd.DataFrame(rows)


# ── Metadata tests ─────────────────────────────────────────────────────────────

class TestMetadata:
    def test_fresh_store_has_no_last_date(self, tmp_store):
        assert tmp_store.last_computed_date("macro") is None
        assert tmp_store.last_computed_date("sector") is None
        assert tmp_store.last_computed_date("stock") is None

    def test_is_fresh_returns_false_on_empty_store(self, tmp_store):
        assert not tmp_store.is_fresh("macro", pd.Timestamp("2020-06-01"))

    def test_meta_persists_across_instances(self, tmp_path):
        cfg = {"backtest": {"benchmark_ticker": "^NSEI"}, "paths": {"artifact_dir": str(tmp_path)}}
        store1 = FeatureStore(tmp_path / "fs", cfg)
        macro_df = _make_macro_df()
        store1._append_macro(macro_df, macro_df.index[0], macro_df.index[-1])

        store2 = FeatureStore(tmp_path / "fs", cfg)
        assert store2.last_computed_date("macro") is not None

    def test_universe_hash_changes_logic_hash_for_stock_and_sector(self, tmp_store):
        class DummyUniverse:
            def __init__(self, tickers):
                self._uni_cfg = {
                    "stocks": [
                        {"ticker": ticker, "sector": "IT", "cap": "large"}
                        for ticker in tickers
                    ]
                }

        uni_a = DummyUniverse(["AAA.NS", "BBB.NS"])
        uni_b = DummyUniverse(["AAA.NS", "CCC.NS"])

        stock_a = tmp_store._current_logic_hash("stock", universe_mgr=uni_a)
        stock_b = tmp_store._current_logic_hash("stock", universe_mgr=uni_b)
        sector_a = tmp_store._current_logic_hash("sector", universe_mgr=uni_a)
        sector_b = tmp_store._current_logic_hash("sector", universe_mgr=uni_b)

        assert stock_a != stock_b
        assert sector_a != sector_b


# ── Macro store tests ─────────────────────────────────────────────────────────

class TestMacroStore:
    def test_roundtrip(self, tmp_store):
        macro_df = _make_macro_df()
        tmp_store._append_macro(macro_df, macro_df.index[0], macro_df.index[-1])
        loaded = tmp_store.load("macro", macro_df.index[0], macro_df.index[-1])
        assert not loaded.empty
        assert len(loaded) == len(macro_df)

    def test_dtype_optimisation(self, tmp_store):
        macro_df = _make_macro_df()
        tmp_store._append_macro(macro_df, macro_df.index[0], macro_df.index[-1])
        loaded = tmp_store.load("macro", macro_df.index[0], macro_df.index[-1])
        for col in loaded.select_dtypes(include="number").columns:
            assert loaded[col].dtype == np.float32, f"{col} should be float32"

    def test_no_future_rows_in_load(self, tmp_store):
        macro_df = _make_macro_df("2020-01-01", "2020-12-31")
        tmp_store._append_macro(macro_df, macro_df.index[0], macro_df.index[-1])
        as_of = pd.Timestamp("2020-06-30")
        loaded = tmp_store.load("macro", macro_df.index[0], as_of)
        assert loaded.index.max() <= as_of

    def test_snapshot_returns_single_row(self, tmp_store):
        macro_df = _make_macro_df()
        tmp_store._append_macro(macro_df, macro_df.index[0], macro_df.index[-1])
        snap = tmp_store.snapshot("macro", pd.Timestamp("2020-06-15"))
        assert len(snap) == 1


# ── Stock store tests ─────────────────────────────────────────────────────────

class TestStockStore:
    def test_roundtrip(self, tmp_store):
        # Just verify the stock path helper works
        path = tmp_store._stock_path(2020, 1)
        assert "year=2020" in str(path) and "month=01" in str(path)

    def test_append_and_load(self, tmp_store):
        stock_df = _make_stock_df()
        start, end = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")
        # Directly test merge_and_write with synthetic data
        for (yr, mo), grp in stock_df.groupby(
            [pd.DatetimeIndex(stock_df["date"]).year, pd.DatetimeIndex(stock_df["date"]).month]
        ):
            path = tmp_store._stock_path(yr, mo)
            tmp_store._merge_and_write(grp, path, dedup_cols=["date", "ticker"])

        loaded = tmp_store.load("stock", start, end)
        assert len(loaded) == len(stock_df)
        assert set(loaded["ticker"].unique()) == {"TCS.NS", "INFY.NS", "HDFCBANK.NS"}

    def test_snapshot_returns_one_row_per_ticker(self, tmp_store):
        stock_df = _make_stock_df()
        for (yr, mo), grp in stock_df.groupby(
            [pd.DatetimeIndex(stock_df["date"]).year, pd.DatetimeIndex(stock_df["date"]).month]
        ):
            tmp_store._merge_and_write(grp, tmp_store._stock_path(yr, mo), ["date", "ticker"])

        as_of = pd.Timestamp("2020-06-30")
        snap = tmp_store.snapshot("stock", as_of)
        assert len(snap) == 3                           # one per ticker
        assert snap["date"].max() <= as_of

    def test_pit_no_future_tickers(self, tmp_store):
        stock_df = _make_stock_df()
        for (yr, mo), grp in stock_df.groupby(
            [pd.DatetimeIndex(stock_df["date"]).year, pd.DatetimeIndex(stock_df["date"]).month]
        ):
            tmp_store._merge_and_write(grp, tmp_store._stock_path(yr, mo), ["date", "ticker"])

        as_of = pd.Timestamp("2020-03-31")
        snap = tmp_store.snapshot("stock", as_of)
        assert snap["date"].max() <= as_of, "snapshot contains future data — PIT violation"


# ── Point-in-time correctness tests ──────────────────────────────────────────

class TestPointInTime:
    """
    These tests directly exercise assert_no_lookahead() and verify that
    load() / snapshot() never return rows beyond the requested as_of date.
    """

    def test_assert_no_lookahead_passes_on_clean_df(self, tmp_store):
        as_of = pd.Timestamp("2020-06-30")
        idx = pd.date_range("2020-01-01", "2020-06-30", freq="B")
        df = pd.DataFrame({"x": 1.0}, index=idx)
        tmp_store.assert_no_lookahead(df, as_of, "test")   # must not raise

    def test_assert_no_lookahead_raises_on_future_row(self, tmp_store):
        as_of = pd.Timestamp("2020-06-30")
        idx = pd.date_range("2020-01-01", "2020-07-15", freq="B")
        df = pd.DataFrame({"x": 1.0}, index=idx)
        with pytest.raises(AssertionError, match="PIT violation"):
            tmp_store.assert_no_lookahead(df, as_of, "test")

    def test_load_macro_pit(self, tmp_store):
        macro_df = _make_macro_df("2020-01-01", "2020-12-31")
        tmp_store._append_macro(macro_df, macro_df.index[0], macro_df.index[-1])
        as_of = pd.Timestamp("2020-06-15")
        loaded = tmp_store.load("macro", macro_df.index[0], as_of)
        tmp_store.assert_no_lookahead(loaded, as_of, "macro load")

    def test_snapshot_stock_pit(self, tmp_store):
        stock_df = _make_stock_df("2020-01-01", "2020-12-31")
        for (yr, mo), grp in stock_df.groupby(
            [pd.DatetimeIndex(stock_df["date"]).year, pd.DatetimeIndex(stock_df["date"]).month]
        ):
            tmp_store._merge_and_write(grp, tmp_store._stock_path(yr, mo), ["date", "ticker"])
        as_of = pd.Timestamp("2020-08-31")
        snap = tmp_store.snapshot("stock", as_of)
        # PIT check via date column
        future = snap[pd.to_datetime(snap["date"]) > as_of]
        assert future.empty, f"PIT violation: {len(future)} future rows in snapshot"

    def test_cross_year_load_stays_within_bounds(self, tmp_store):
        macro_2020 = _make_macro_df("2020-01-01", "2020-12-31")
        macro_2021 = _make_macro_df("2021-01-01", "2021-12-31")
        full = pd.concat([macro_2020, macro_2021])
        tmp_store._append_macro(full, full.index[0], full.index[-1])

        as_of = pd.Timestamp("2021-06-30")
        start = pd.Timestamp("2020-01-01")
        loaded = tmp_store.load("macro", start, as_of)
        assert loaded.index.max() <= as_of
        assert loaded.index.min() >= start


# ── Incremental append tests ──────────────────────────────────────────────────

class TestIncrementalAppend:
    def test_second_append_does_not_duplicate(self, tmp_store):
        macro_h1 = _make_macro_df("2020-01-01", "2020-06-30")
        macro_h2 = _make_macro_df("2020-07-01", "2020-12-31")

        tmp_store._append_macro(macro_h1, macro_h1.index[0], macro_h1.index[-1])
        tmp_store._append_macro(macro_h2, macro_h2.index[0], macro_h2.index[-1])

        full = tmp_store.load("macro", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
        # No duplicate indices
        assert full.index.is_unique

    def test_overlapping_append_keeps_latest(self, tmp_store):
        macro_first = _make_macro_df("2020-01-01", "2020-06-30")
        tmp_store._append_macro(macro_first, macro_first.index[0], macro_first.index[-1])

        # Overlap: append from May with different values
        macro_overlap = _make_macro_df("2020-05-01", "2020-09-30")
        macro_overlap["vix_level"] = 999.0   # sentinel
        tmp_store._append_macro(macro_overlap, macro_overlap.index[0], macro_overlap.index[-1])

        loaded = tmp_store.load("macro", pd.Timestamp("2020-05-01"), pd.Timestamp("2020-06-30"))
        # Latest values (999.0) should win — check a sample of rows
        assert float(loaded["vix_level"].mean()) == pytest.approx(999.0, abs=1.0)

    def test_is_fresh_after_append(self, tmp_store):
        macro_df = _make_macro_df()
        tmp_store._append_macro(macro_df, macro_df.index[0], macro_df.index[-1])
        assert tmp_store.is_fresh("macro", pd.Timestamp("2020-06-01"))
        assert not tmp_store.is_fresh("macro", pd.Timestamp("2021-01-01"))


# ── Partition layout tests ────────────────────────────────────────────────────

class TestPartitions:
    def test_macro_partitioned_by_year(self, tmp_store):
        full = pd.concat([_make_macro_df("2019-01-01", "2019-12-31"),
                          _make_macro_df("2020-01-01", "2020-12-31")])
        tmp_store._append_macro(full, full.index[0], full.index[-1])
        assert (tmp_store.base_dir / "macro" / "year=2019" / "data.parquet").exists()
        assert (tmp_store.base_dir / "macro" / "year=2020" / "data.parquet").exists()

    def test_stock_partitioned_by_year_month(self, tmp_store):
        stock_df = _make_stock_df("2020-01-01", "2020-03-31")
        for (yr, mo), grp in stock_df.groupby(
            [pd.DatetimeIndex(stock_df["date"]).year, pd.DatetimeIndex(stock_df["date"]).month]
        ):
            tmp_store._merge_and_write(grp, tmp_store._stock_path(yr, mo), ["date", "ticker"])
        assert (tmp_store.base_dir / "stock" / "year=2020" / "month=01" / "data.parquet").exists()
        assert (tmp_store.base_dir / "stock" / "year=2020" / "month=02" / "data.parquet").exists()

    def test_load_reads_only_needed_partitions(self, tmp_store):
        full = pd.concat([_make_macro_df("2019-01-01", "2019-12-31"),
                          _make_macro_df("2020-01-01", "2020-12-31"),
                          _make_macro_df("2021-01-01", "2021-12-31")])
        tmp_store._append_macro(full, full.index[0], full.index[-1])

        # Verify that requesting only 2020 returns only 2020 rows
        loaded = tmp_store.load("macro", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
        assert loaded.index.min().year == 2020
        assert loaded.index.max().year == 2020

        # Verify correct partition files exist on disk
        assert (tmp_store.base_dir / "macro" / "year=2019" / "data.parquet").exists()
        assert (tmp_store.base_dir / "macro" / "year=2020" / "data.parquet").exists()
        assert (tmp_store.base_dir / "macro" / "year=2021" / "data.parquet").exists()
