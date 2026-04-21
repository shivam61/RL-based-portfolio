"""
Sector master construction helpers.

This module builds a reusable ticker→sector master table with optional
active windows and provenance fields. It is intentionally conservative:
if exact dates are unknown, columns remain nullable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import load_universe_config

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SectorMasterBuildResult:
    master: pd.DataFrame
    missing_sector_tickers: list[str]
    duplicate_tickers: list[str]


def _resolve_repo_path(path_like: str | None) -> Path | None:
    if not path_like:
        return None
    p = Path(path_like)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _load_sector_override_map(path_like: str | None) -> pd.DataFrame:
    path = _resolve_repo_path(path_like)
    if path is None or not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported sector map file type: {path}")

    if df.empty:
        return df

    required = {"ticker", "sector"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Sector override map missing required columns: {sorted(missing)}"
        )
    return df


def _normalize_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def build_sector_master(
    price_matrix: pd.DataFrame,
    cfg: dict,
) -> SectorMasterBuildResult:
    """
    Build sector master from:
    1) config/universe.yaml stocks (base)
    2) optional historical override file (can add old tickers + sectors)
    """
    uni_cfg = load_universe_config()
    base_rows = []
    for row in uni_cfg.get("stocks", []):
        base_rows.append(
            {
                "ticker": row.get("ticker"),
                "name": row.get("name"),
                "sector": row.get("sector"),
                "cap": row.get("cap"),
                "active_from": row.get("listed_since"),
                "active_to": row.get("delisted_on"),
                "source": "universe_yaml",
                "derivation_note": "Configured static universe metadata",
                "inclusion_reason": "base_universe",
            }
        )
    base_df = pd.DataFrame(base_rows)
    base_df = _normalize_dates(base_df, ["active_from", "active_to"])

    hu_cfg = cfg.get("universe", {}).get("historical_union", {})
    override_df = _load_sector_override_map(hu_cfg.get("candidate_sector_map_file"))
    if not override_df.empty:
        override_df = _normalize_dates(override_df, ["active_from", "active_to"])
        if "source" not in override_df.columns:
            override_df["source"] = "historical_sector_map_file"
        if "derivation_note" not in override_df.columns:
            override_df["derivation_note"] = "External historical mapping override"
        if "inclusion_reason" not in override_df.columns:
            override_df["inclusion_reason"] = "external_mapping"
        if "cap" not in override_df.columns:
            override_df["cap"] = "mid"
        if "name" not in override_df.columns:
            override_df["name"] = override_df["ticker"]

    combined = pd.concat([base_df, override_df], ignore_index=True, sort=False)
    if combined.empty:
        return SectorMasterBuildResult(
            master=combined, missing_sector_tickers=[], duplicate_tickers=[]
        )

    combined["ticker"] = combined["ticker"].astype(str).str.strip()
    combined["sector"] = combined["sector"].astype(str).str.strip()
    combined = combined[combined["ticker"].str.len() > 0]

    # Keep latest override if duplicates exist.
    dup_tickers = (
        combined["ticker"].value_counts().loc[lambda s: s > 1].index.tolist()
    )
    combined = combined.drop_duplicates(subset=["ticker"], keep="last")

    equity_cols = [
        c
        for c in price_matrix.columns
        if isinstance(c, str) and (c.endswith(".NS") or c.endswith(".BO"))
    ]
    mapped = set(combined["ticker"])
    missing_sector = sorted([t for t in equity_cols if t not in mapped])

    # Infer active windows from available price history where missing.
    prices = price_matrix.sort_index()
    for ticker in combined["ticker"]:
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        if series.empty:
            continue
        first_seen = pd.Timestamp(series.index[0]).normalize()
        last_seen = pd.Timestamp(series.index[-1]).normalize()

        row_mask = combined["ticker"] == ticker
        if "active_from" in combined.columns:
            if pd.isna(combined.loc[row_mask, "active_from"]).all():
                combined.loc[row_mask, "active_from"] = first_seen
        else:
            combined["active_from"] = pd.NaT
            combined.loc[row_mask, "active_from"] = first_seen

        if "active_to" not in combined.columns:
            combined["active_to"] = pd.NaT
        # Consider stock still active if seen in the recent data tail.
        if pd.isna(combined.loc[row_mask, "active_to"]).all():
            if last_seen < prices.index.max() - pd.Timedelta(days=120):
                combined.loc[row_mask, "active_to"] = last_seen

    combined["added_on"] = pd.NaT  # populated by relevance stage
    combined["source"] = combined["source"].fillna("unknown")
    combined["derivation_note"] = combined["derivation_note"].fillna("")
    combined["inclusion_reason"] = combined["inclusion_reason"].fillna("")
    combined["in_union_10y"] = False

    ordered_cols = [
        "ticker",
        "name",
        "sector",
        "cap",
        "active_from",
        "active_to",
        "added_on",
        "source",
        "derivation_note",
        "inclusion_reason",
        "in_union_10y",
    ]
    for col in ordered_cols:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[ordered_cols].sort_values(["sector", "ticker"]).reset_index(
        drop=True
    )

    return SectorMasterBuildResult(
        master=combined,
        missing_sector_tickers=missing_sector,
        duplicate_tickers=sorted(dup_tickers),
    )
