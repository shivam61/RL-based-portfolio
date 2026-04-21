"""Tests for historical union sector universe builder and eligibility."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.universe import UniverseManager
from src.universe.eligibility import get_sector_candidates
from src.universe.historical_sector_universe import HistoricalSectorUniverseBuilder


def _synthetic_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2017-01-02", "2026-01-30", freq="B")
    rng = np.random.default_rng(7)
    tickers = ["TCS.NS", "INFY.NS", "NEWIT.NS", "ILLQ.NS"]

    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    vols = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in tickers:
        rets = rng.normal(0.0005, 0.015, len(dates))
        prices[t] = 100 * np.cumprod(1 + rets)

    # NEWIT only appears from 2021 onward.
    prices.loc[prices.index < pd.Timestamp("2021-01-04"), "NEWIT.NS"] = np.nan
    vols["TCS.NS"] = 50.0
    vols["INFY.NS"] = 40.0
    vols["NEWIT.NS"] = 20.0
    vols["ILLQ.NS"] = 0.2
    vols.loc[prices.index < pd.Timestamp("2021-01-04"), "NEWIT.NS"] = np.nan
    return prices, vols


def test_builder_and_union_outputs(tmp_path):
    cfg = load_config()
    cfg["universe"]["historical_union"]["output_dir"] = str(tmp_path / "historical")
    override_path = tmp_path / "sector_map.csv"
    pd.DataFrame(
        [
            {"ticker": "NEWIT.NS", "sector": "IT", "name": "New IT", "cap": "mid"},
            {"ticker": "ILLQ.NS", "sector": "IT", "name": "Illiquid IT", "cap": "small"},
        ]
    ).to_csv(override_path, index=False)
    cfg["universe"]["historical_union"]["candidate_sector_map_file"] = str(override_path)
    cfg["universe"]["historical_union"]["relevance"]["min_median_traded_value_cr"] = 5.0
    cfg["universe"]["historical_union"]["relevance"]["market_cap_proxy_rank_max"] = 2

    prices, vols = _synthetic_matrices()
    builder = HistoricalSectorUniverseBuilder(cfg)
    master, union_df, diagnostics = builder.build(prices, vols, as_of=pd.Timestamp("2026-01-30"))
    artifacts = builder.persist(master, union_df, diagnostics)

    assert artifacts.sector_master_path.exists()
    assert artifacts.union_path.exists()
    assert "NEWIT.NS" in set(union_df["ticker"])
    assert "added_on" in union_df.columns
    new_added = union_df.loc[union_df["ticker"] == "NEWIT.NS", "added_on"].iloc[0]
    assert pd.notna(new_added)
    assert "ILLQ.NS" not in set(union_df["ticker"])


def test_time_aware_sector_candidates_and_universe_manager_mode(tmp_path):
    cfg = load_config()
    cfg["universe"]["historical_union"]["output_dir"] = str(tmp_path / "historical")
    override_path = tmp_path / "sector_map.csv"
    pd.DataFrame(
        [{"ticker": "NEWIT.NS", "sector": "IT", "name": "New IT", "cap": "mid"}]
    ).to_csv(override_path, index=False)
    cfg["universe"]["historical_union"]["candidate_sector_map_file"] = str(override_path)
    cfg["universe"]["historical_union"]["min_price_history_days"] = 252
    cfg["universe"]["historical_union"]["min_median_traded_value_cr"] = 2.0
    cfg["universe"]["historical_union"]["relevance"]["min_median_traded_value_cr"] = 4.0
    cfg["universe"]["historical_union"]["relevance"]["market_cap_proxy_rank_max"] = 3

    prices, vols = _synthetic_matrices()
    builder = HistoricalSectorUniverseBuilder(cfg)
    master, union_df, diagnostics = builder.build(prices, vols, as_of=pd.Timestamp("2026-01-30"))
    builder.persist(master, union_df, diagnostics)

    pre_candidates = get_sector_candidates(
        sector="IT",
        as_of_date=pd.Timestamp("2020-01-31"),
        union_df=union_df,
        price_matrix=prices,
        volume_matrix=vols,
        cfg=cfg,
    )
    post_candidates = get_sector_candidates(
        sector="IT",
        as_of_date=pd.Timestamp("2022-01-31"),
        union_df=union_df,
        price_matrix=prices,
        volume_matrix=vols,
        cfg=cfg,
    )
    assert "NEWIT.NS" not in pre_candidates
    assert "NEWIT.NS" in post_candidates

    cfg["universe"]["mode"] = "historical_union_10y"
    uni_mgr = UniverseManager(cfg)
    snap_2020 = uni_mgr.get_universe(
        date(2020, 1, 31),
        price_matrix=prices,
        volume_matrix=vols,
    )
    snap_2022 = uni_mgr.get_universe(
        date(2022, 1, 31),
        price_matrix=prices,
        volume_matrix=vols,
    )
    assert "NEWIT.NS" not in set(snap_2020.tickers)
    assert "NEWIT.NS" in set(snap_2022.tickers)
