"""
Time-aware eligibility filters for historical union universes.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def _as_timestamp(dt_like) -> pd.Timestamp:
    return pd.Timestamp(dt_like).normalize()


def _filter_active_window(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    use_active_window_filter: bool,
) -> pd.DataFrame:
    if not use_active_window_filter:
        return df
    out = df.copy()
    if "active_from" in out.columns:
        out = out[
            out["active_from"].isna() | (pd.to_datetime(out["active_from"]) <= as_of)
        ]
    if "active_to" in out.columns:
        out = out[
            out["active_to"].isna() | (pd.to_datetime(out["active_to"]) > as_of)
        ]
    return out


def _filter_added_on(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if "added_on" not in df.columns:
        return df
    return df[df["added_on"].isna() | (pd.to_datetime(df["added_on"]) <= as_of)]


def _filter_price_history(
    df: pd.DataFrame,
    price_matrix: pd.DataFrame,
    as_of: pd.Timestamp,
    min_price_history_days: int,
) -> pd.DataFrame:
    hist = price_matrix.loc[price_matrix.index <= as_of]
    keep = []
    for ticker in df["ticker"].tolist():
        if ticker not in hist.columns:
            continue
        if hist[ticker].dropna().shape[0] >= min_price_history_days:
            keep.append(ticker)
    return df[df["ticker"].isin(keep)]


def _filter_liquidity(
    df: pd.DataFrame,
    volume_matrix: pd.DataFrame | None,
    as_of: pd.Timestamp,
    min_median_traded_value_cr: float,
    liquidity_lookback_days: int,
) -> pd.DataFrame:
    if volume_matrix is None or volume_matrix.empty:
        return df
    hist = volume_matrix.loc[volume_matrix.index <= as_of].tail(liquidity_lookback_days)
    keep = []
    for ticker in df["ticker"].tolist():
        if ticker not in hist.columns:
            # No volume data: keep conservatively, matching existing behavior.
            keep.append(ticker)
            continue
        med_val = hist[ticker].median(skipna=True)
        if pd.isna(med_val) or float(med_val) >= min_median_traded_value_cr:
            keep.append(ticker)
    return df[df["ticker"].isin(keep)]


def apply_time_aware_eligibility(
    union_df: pd.DataFrame,
    as_of_date,
    price_matrix: pd.DataFrame,
    volume_matrix: pd.DataFrame | None,
    min_price_history_days: int,
    min_median_traded_value_cr: float,
    liquidity_lookback_days: int,
    use_active_window_filter: bool,
) -> pd.DataFrame:
    """
    Return union rows eligible at `as_of_date` with no future leakage.
    """
    if union_df.empty:
        return union_df.copy()
    as_of = _as_timestamp(as_of_date)
    out = union_df.copy()
    for col in ["active_from", "active_to", "added_on"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    out = _filter_active_window(out, as_of, use_active_window_filter)
    out = _filter_added_on(out, as_of)
    out = _filter_price_history(out, price_matrix, as_of, min_price_history_days)
    out = _filter_liquidity(
        out,
        volume_matrix,
        as_of,
        min_median_traded_value_cr,
        liquidity_lookback_days,
    )
    return out.sort_values(["sector", "ticker"]).reset_index(drop=True)


def get_sector_candidates(
    sector: str,
    as_of_date,
    union_df: pd.DataFrame,
    price_matrix: pd.DataFrame,
    volume_matrix: pd.DataFrame | None,
    cfg: dict,
) -> list[str]:
    """
    Public API: candidates in a sector as of a rebalance date.
    """
    hu_cfg = cfg.get("universe", {}).get("historical_union", {})
    eligible = apply_time_aware_eligibility(
        union_df=union_df,
        as_of_date=as_of_date,
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        min_price_history_days=int(hu_cfg.get("min_price_history_days", 252)),
        min_median_traded_value_cr=float(hu_cfg.get("min_median_traded_value_cr", 2.0)),
        liquidity_lookback_days=int(hu_cfg.get("liquidity_lookback_days", 63)),
        use_active_window_filter=bool(hu_cfg.get("use_active_window_filter", True)),
    )
    return sorted(eligible.loc[eligible["sector"] == sector, "ticker"].unique().tolist())
