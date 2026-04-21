"""
Build and query survivorship-aware sector union universes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.universe.sector_master import SectorMasterBuildResult, build_sector_master

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HistoricalSectorUniverseArtifacts:
    sector_master_path: Path
    union_path: Path
    diagnostics_path: Path


def _resolve_repo_path(path_like: str | None) -> Optional[Path]:
    if not path_like:
        return None
    p = Path(path_like)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _resolve_output_dir(cfg: dict) -> Path:
    hu_cfg = cfg.get("universe", {}).get("historical_union", {})
    out = _resolve_repo_path(hu_cfg.get("output_dir"))
    if out is not None:
        return out
    return Path(cfg["paths"]["processed_data"]) / "universe" / "historical_union_10y"


def _load_broad_index_membership(path_like: str | None) -> pd.DataFrame:
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
        raise ValueError(f"Unsupported index membership file type: {path}")

    if df.empty:
        return df
    if "ticker" not in df.columns:
        raise ValueError("Broad index membership file must include `ticker` column")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


class HistoricalSectorUniverseBuilder:
    """Construct 10-year sector union from price/volume history."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hu_cfg = cfg.get("universe", {}).get("historical_union", {})

    def build(
        self,
        price_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        if price_matrix.empty:
            raise ValueError("price_matrix is empty")

        as_of_ts = pd.Timestamp(as_of) if as_of is not None else price_matrix.index.max()
        lookback_years = int(self.hu_cfg.get("lookback_years", 10))
        window_start = as_of_ts - pd.DateOffset(years=lookback_years)

        master_res: SectorMasterBuildResult = build_sector_master(price_matrix, self.cfg)
        master = master_res.master.copy()
        if master.empty:
            raise ValueError("Sector master is empty; no ticker-sector mappings available")

        mapped = [t for t in master["ticker"].tolist() if t in price_matrix.columns]
        if not mapped:
            raise ValueError("No mapped tickers present in price_matrix columns")

        prices_win = price_matrix.loc[(price_matrix.index >= window_start) & (price_matrix.index <= as_of_ts), mapped]
        vols_win = volume_matrix.reindex(index=prices_win.index, columns=mapped) if volume_matrix is not None and not volume_matrix.empty else pd.DataFrame(index=prices_win.index, columns=mapped)

        liq_lookback = int(self.hu_cfg.get("liquidity_lookback_days", 63))
        relevance_cfg = self.hu_cfg.get("relevance", {})
        min_history_for_relevance = int(relevance_cfg.get("min_history_days_for_relevance", 126))
        min_median_traded_value_cr = float(relevance_cfg.get("min_median_traded_value_cr", 8.0))
        market_cap_proxy_rank_max = int(relevance_cfg.get("market_cap_proxy_rank_max", 300))
        include_index = bool(relevance_cfg.get("include_if_in_broad_index", True))
        index_df = _load_broad_index_membership(relevance_cfg.get("broad_index_membership_file"))

        rolling_liq = vols_win.rolling(liq_lookback, min_periods=max(20, liq_lookback // 3)).median()
        cap_proxy_rank = rolling_liq.rank(axis=1, ascending=False, method="average")

        union_rows: list[dict] = []
        insufficient_history = 0
        missing_sector = master_res.missing_sector_tickers

        for _, row in master.iterrows():
            ticker = row["ticker"]
            if ticker not in prices_win.columns:
                continue

            px = prices_win[ticker].dropna()
            if len(px) < min_history_for_relevance:
                insufficient_history += 1
                continue

            rank_series = cap_proxy_rank[ticker].dropna() if ticker in cap_proxy_rank.columns else pd.Series(dtype=float)
            liq_series = rolling_liq[ticker].dropna() if ticker in rolling_liq.columns else pd.Series(dtype=float)

            rank_hit = bool((rank_series <= market_cap_proxy_rank_max).any())
            liq_hit = bool((liq_series >= min_median_traded_value_cr).any())

            index_hit = False
            index_hit_date: pd.Timestamp | None = None
            if include_index and not index_df.empty:
                if "date" in index_df.columns:
                    hits = index_df[
                        (index_df["ticker"] == ticker)
                        & (index_df["date"] >= window_start)
                        & (index_df["date"] <= as_of_ts)
                    ]
                    if not hits.empty:
                        index_hit = True
                        index_hit_date = pd.Timestamp(hits["date"].min())
                else:
                    index_hit = bool((index_df["ticker"] == ticker).any())
                    if index_hit:
                        index_hit_date = window_start

            relevant = rank_hit or liq_hit or index_hit
            if not relevant:
                continue

            dates = []
            if rank_hit:
                dates.append(rank_series[rank_series <= market_cap_proxy_rank_max].index.min())
            if liq_hit:
                dates.append(liq_series[liq_series >= min_median_traded_value_cr].index.min())
            if index_hit and index_hit_date is not None:
                dates.append(index_hit_date)
            added_on = min(d for d in dates if d is not None) if dates else pd.NaT

            reasons = []
            if rank_hit:
                reasons.append("market_cap_proxy_rank")
            if liq_hit:
                reasons.append("liquidity")
            if index_hit:
                reasons.append("broad_index_presence")

            union_rows.append(
                {
                    "ticker": ticker,
                    "name": row.get("name", ticker),
                    "sector": row["sector"],
                    "cap": row.get("cap", "mid"),
                    "active_from": row.get("active_from"),
                    "active_to": row.get("active_to"),
                    "added_on": added_on,
                    "source": row.get("source", "unknown"),
                    "derivation_note": row.get("derivation_note", ""),
                    "inclusion_reason": "|".join(reasons),
                    "rank_hit": rank_hit,
                    "liquidity_hit": liq_hit,
                    "broad_index_hit": index_hit,
                }
            )

        union_df = pd.DataFrame(union_rows).sort_values(["sector", "ticker"]).reset_index(drop=True)
        if union_df.empty:
            logger.warning("Historical union build returned empty universe")

        master = master.copy()
        if not union_df.empty:
            add_map = union_df.set_index("ticker")["added_on"]
            reason_map = union_df.set_index("ticker")["inclusion_reason"]
            union_set = set(union_df["ticker"])
            master["in_union_10y"] = master["ticker"].isin(union_set)
            master["added_on"] = master["ticker"].map(add_map)
            master.loc[master["ticker"].isin(union_set), "inclusion_reason"] = (
                master.loc[master["ticker"].isin(union_set), "ticker"].map(reason_map).fillna("relevance_filter")
            )
            master.loc[master["ticker"].isin(union_set), "derivation_note"] = (
                master.loc[master["ticker"].isin(union_set), "derivation_note"].fillna("")
                + " | Added by 10y relevance union filter"
            ).str.strip(" |")

        diagnostics = self._build_diagnostics_markdown(
            master=master,
            union_df=union_df,
            as_of=as_of_ts,
            window_start=window_start,
            missing_sector=missing_sector,
            duplicate_tickers=master_res.duplicate_tickers,
            insufficient_history_count=insufficient_history,
        )
        return master, union_df, diagnostics

    def persist(
        self,
        master: pd.DataFrame,
        union_df: pd.DataFrame,
        diagnostics_md: str,
    ) -> HistoricalSectorUniverseArtifacts:
        out_dir = _resolve_output_dir(self.cfg)
        out_dir.mkdir(parents=True, exist_ok=True)

        sector_master_path = out_dir / "sector_historical_master.parquet"
        union_path = out_dir / "sector_union_universe_10y.parquet"
        diagnostics_path = out_dir / "historical_universe_diagnostics.md"

        master.to_parquet(sector_master_path, index=False)
        union_df.to_parquet(union_path, index=False)
        # Convenience files for inspection.
        master.to_csv(out_dir / "sector_historical_master.csv", index=False)
        union_df.to_csv(out_dir / "sector_union_universe_10y.csv", index=False)
        with open(out_dir / "sector_union_universe_10y.json", "w") as f:
            json.dump(
                union_df.fillna(pd.NA).replace({pd.NA: None}).to_dict(orient="records"),
                f,
                indent=2,
                default=str,
            )
        with open(diagnostics_path, "w") as f:
            f.write(diagnostics_md)

        return HistoricalSectorUniverseArtifacts(
            sector_master_path=sector_master_path,
            union_path=union_path,
            diagnostics_path=diagnostics_path,
        )

    def _build_diagnostics_markdown(
        self,
        master: pd.DataFrame,
        union_df: pd.DataFrame,
        as_of: pd.Timestamp,
        window_start: pd.Timestamp,
        missing_sector: list[str],
        duplicate_tickers: list[str],
        insufficient_history_count: int,
    ) -> str:
        static_set = set(master.loc[master["source"] == "universe_yaml", "ticker"])
        union_set = set(union_df["ticker"]) if not union_df.empty else set()
        additions = sorted(union_set - static_set)
        removals = sorted(static_set - union_set)

        hu_cfg = self.hu_cfg
        min_per_sector = int(hu_cfg.get("sanity", {}).get("min_stocks_per_sector", 2))
        max_per_sector = int(hu_cfg.get("sanity", {}).get("max_stocks_per_sector", 250))
        by_sector = (
            union_df.groupby("sector")["ticker"].nunique().sort_values(ascending=False)
            if not union_df.empty else pd.Series(dtype=int)
        )
        flagged = [
            f"{sec}={int(n)}"
            for sec, n in by_sector.items()
            if int(n) < min_per_sector or int(n) > max_per_sector
        ]

        lines = [
            "# Historical 10-Year Sector Union Diagnostics",
            "",
            f"- Build window: {window_start.date()} to {as_of.date()}",
            f"- Union tickers: {len(union_set)}",
            f"- Static baseline tickers: {len(static_set)}",
            f"- Additions vs static: {len(additions)}",
            f"- Removals vs static: {len(removals)}",
            f"- Missing sector assignments (candidate tickers): {len(missing_sector)}",
            f"- Duplicate ticker mappings resolved: {len(duplicate_tickers)}",
            f"- Tickers dropped for insufficient history: {insufficient_history_count}",
            "",
            "## Stocks per sector",
            "",
        ]
        if by_sector.empty:
            lines.append("- (empty)")
        else:
            for sec, n in by_sector.items():
                lines.append(f"- {sec}: {int(n)}")

        lines.extend(
            [
                "",
                "## Overlap with static universe",
                "",
                f"- Overlap: {len(static_set & union_set)}",
                f"- Additions: {', '.join(additions[:40]) if additions else '(none)'}",
                f"- Removals: {', '.join(removals[:40]) if removals else '(none)'}",
                "",
                "## Data quality checks",
                "",
                f"- Missing sector assignments sample: {', '.join(missing_sector[:20]) if missing_sector else '(none)'}",
                f"- Duplicate mapping sample: {', '.join(duplicate_tickers[:20]) if duplicate_tickers else '(none)'}",
                f"- Suspicious sector sizes: {', '.join(flagged) if flagged else '(none)'}",
                "",
                "## Assumptions and limitations",
                "",
                "- Market-cap filter uses traded-value rank proxy when true market-cap history is unavailable.",
                "- Sector history defaults to current mapping unless override file supplies historical mappings.",
                "- Active windows are inferred from price availability when explicit listing/delisting dates are absent.",
            ]
        )
        return "\n".join(lines) + "\n"


class HistoricalSectorUniverseStore:
    """
    Runtime loader for persisted historical union artifacts.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hu_cfg = cfg.get("universe", {}).get("historical_union", {})
        out_dir = _resolve_output_dir(cfg)
        self.sector_master_path = out_dir / "sector_historical_master.parquet"
        self.union_path = out_dir / "sector_union_universe_10y.parquet"
        self._sector_master = self._safe_read(self.sector_master_path)
        self._union_df = self._safe_read(self.union_path)
        for df in [self._sector_master, self._union_df]:
            if not df.empty:
                for col in ["active_from", "active_to", "added_on"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce")

    @staticmethod
    def _safe_read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    @property
    def is_available(self) -> bool:
        return not self._union_df.empty

    @property
    def union_df(self) -> pd.DataFrame:
        return self._union_df.copy()

    @property
    def sector_master(self) -> pd.DataFrame:
        return self._sector_master.copy()
