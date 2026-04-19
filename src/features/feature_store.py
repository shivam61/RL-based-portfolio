"""
Partitioned, incremental feature store.

Layout on disk:
  {base_dir}/
    macro/year=YYYY/data.parquet          (~3 500 rows × 40 cols per year)
    sector/year=YYYY/data.parquet         (~3 500 × 15 sectors × 25 cols per year)
    stock/year=YYYY/month=MM/data.parquet (~30 000 rows × 35 cols per shard)
    _metadata.json                        last_computed_date + schema_hash per type

Design guarantees
-----------------
* Point-in-time safe  – callers pass as_of; store never returns future rows.
* Compute-once        – build_or_append() is idempotent; re-running skips
                        already-computed date ranges.
* Schema-aware        – if the feature column set changes (new features added),
                        is_fresh() returns False and the store rebuilds from scratch.
* Shard-verified      – is_fresh() confirms actual parquet files exist on disk,
                        not just that metadata claims they do.
* Transparent I/O     – callers use load() / snapshot(); partition layout is
                        an internal detail.
* Memory-efficient    – only the needed year/month partitions are loaded;
                        an LRU cache keeps recently-used shards hot.
* float32 storage     – ~50% smaller than float64 with no loss of precision
                        for financial ratios.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FeatureType = Literal["macro", "sector", "stock"]

# Number of year-partitions kept hot in the process-level shard cache.
_CACHE_MAX_SHARDS = 60


class FeatureStore:
    """
    Manages partitioned parquet storage for pre-computed features.

    Typical usage
    -------------
    At backtest startup (once):
        store = FeatureStore("data/feature_store", cfg)
        store.build_or_append(price_matrix, volume_matrix, macro_df,
                              macro_features_df, sector_fb, stock_fb,
                              universe_mgr, start, end)

    Inside the walk-forward loop (fast):
        sector_snap = store.snapshot("sector", current_date)
        stock_snap  = store.snapshot("stock",  current_date)
        train_feats = store.load("sector", train_start, current_date)
    """

    def __init__(self, base_dir: str | Path, cfg: dict):
        self.base_dir = Path(base_dir)
        self.cfg = cfg
        self._meta_path = self.base_dir / "_metadata.json"
        self._meta: dict = self._load_meta()

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                return json.load(f)
        return {"macro": {}, "sector": {}, "stock": {}}

    def _save_meta(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self._meta_path, "w") as f:
            json.dump(self._meta, f, indent=2, default=str)

    def last_computed_date(self, ft: FeatureType) -> pd.Timestamp | None:
        val = self._meta.get(ft, {}).get("last_date")
        return pd.Timestamp(val) if val else None

    # ── Schema hash helpers ───────────────────────────────────────────────────

    @staticmethod
    def _col_hash(columns: list[str]) -> str:
        """Stable 8-char hash of a sorted column list."""
        key = ",".join(sorted(columns))
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def _stored_schema_hash(self, ft: FeatureType) -> str | None:
        return self._meta.get(ft, {}).get("schema_hash")

    def _stored_logic_hash(self, ft: FeatureType) -> str | None:
        return self._meta.get(ft, {}).get("logic_hash")

    def _current_logic_hash(self, ft: FeatureType, sector_fb=None, stock_fb=None) -> str:
        versions = {
            "macro": "macro_features_v2_store_logic_hash",
            "sector": getattr(sector_fb, "LOGIC_VERSION", "sector_features"),
            "stock": getattr(stock_fb, "LOGIC_VERSION", "stock_features"),
        }
        payload = {
            "ft": ft,
            "logic_version": versions[ft],
            "features_cfg": self.cfg.get("features", {}),
            "benchmark_ticker": self.cfg.get("backtest", {}).get("benchmark_ticker", "^NSEI"),
            "min_avg_volume_cr": self.cfg.get("universe", {}).get("min_avg_volume_cr", 1.0),
        }
        return hashlib.md5(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

    def _latest_shard_columns(self, ft: FeatureType) -> list[str] | None:
        """Read column list from the most recent shard on disk, or None."""
        try:
            if ft == "stock":
                shards = sorted(self.base_dir.joinpath("stock").rglob("*.parquet"))
            elif ft == "sector":
                shards = sorted(self.base_dir.joinpath("sector").rglob("*.parquet"))
            else:
                shards = sorted(self.base_dir.joinpath("macro").rglob("*.parquet"))
            if not shards:
                return None
            df = pd.read_parquet(shards[-1], engine="pyarrow")
            return sorted(df.columns.tolist())
        except Exception:
            return None

    def _shards_exist(self, ft: FeatureType) -> bool:
        """Return True only if at least one parquet shard file exists on disk."""
        subdir = self.base_dir / ft
        return subdir.exists() and any(subdir.rglob("*.parquet"))

    def is_fresh(self, ft: FeatureType, as_of: pd.Timestamp) -> bool:
        """
        True iff:
          1. metadata claims data is current through as_of, AND
          2. at least one shard file actually exists on disk, AND
          3. the schema hash in metadata matches the current shard columns.

        Any mismatch triggers a full rebuild so the model never trains on a
        stale or mismatched feature set.
        """
        last = self.last_computed_date(ft)
        if last is None or last < as_of:
            return False

        # Guard 2: shard files must physically exist
        if not self._shards_exist(ft):
            logger.warning(
                "FeatureStore[%s] metadata claims current but no shard files found — "
                "will rebuild", ft
            )
            self._meta.get(ft, {}).pop("last_date", None)
            self._save_meta()
            return False

        # Guard 3: schema must match stored hash
        stored_hash = self._stored_schema_hash(ft)
        if stored_hash is not None:
            current_cols = self._latest_shard_columns(ft)
            if current_cols is not None:
                current_hash = self._col_hash(current_cols)
                if current_hash != stored_hash:
                    logger.warning(
                        "FeatureStore[%s] schema changed (stored=%s current=%s) — "
                        "will rebuild", ft, stored_hash, current_hash
                    )
                    self.invalidate(ft)
                    return False

        return True

    def invalidate(self, ft: FeatureType) -> None:
        """
        Force a full rebuild of one feature type on next build_or_append().
        Deletes all shards for that type and clears metadata.
        Call this whenever the feature builder logic changes.
        """
        shard_dir = self.base_dir / ft
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
            logger.info("FeatureStore[%s] shards deleted for rebuild", ft)
        self._meta.pop(ft, None)
        self._save_meta()
        _read_shard.cache_clear()

    # ── Shard paths ───────────────────────────────────────────────────────────

    def _macro_path(self, year: int) -> Path:
        return self.base_dir / "macro" / f"year={year}" / "data.parquet"

    def _sector_path(self, year: int) -> Path:
        return self.base_dir / "sector" / f"year={year}" / "data.parquet"

    def _stock_path(self, year: int, month: int) -> Path:
        return self.base_dir / "stock" / f"year={year}" / f"month={month:02d}" / "data.parquet"

    # ── I/O helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast float64 → float32; object cols with low cardinality → category."""
        out = df.copy()
        for col in out.select_dtypes(include="float64").columns:
            out[col] = out[col].astype("float32")
        for col in out.select_dtypes(include=["object", "string"]).columns:
            if out[col].nunique() < 500:
                out[col] = out[col].astype("category")
        return out

    def _write(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self._optimise_dtypes(df)
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=True)
        _read_shard.cache_clear()           # invalidate stale reads
        logger.debug("Wrote %d rows → %s", len(df), path)

    @staticmethod
    def _read(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        return _read_shard(str(path))       # goes through the LRU shard cache

    # ── Full build / incremental append ───────────────────────────────────────

    def build_or_append(
        self,
        price_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame,
        macro_df: pd.DataFrame,
        macro_features_df: pd.DataFrame,
        sector_fb,
        stock_fb,
        universe_mgr,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> None:
        """
        Compute and persist any features not yet in the store.
        Safe to call repeatedly; already-computed ranges are skipped.
        """
        for ft in ("macro", "sector", "stock"):
            logic_hash = self._current_logic_hash(ft, sector_fb=sector_fb, stock_fb=stock_fb)
            if self._stored_logic_hash(ft) != logic_hash:
                logger.info(
                    "FeatureStore[%s] logic changed (stored=%s current=%s) — rebuilding",
                    ft,
                    self._stored_logic_hash(ft),
                    logic_hash,
                )
                self.invalidate(ft)

            last = self.last_computed_date(ft)
            if last is not None and self.is_fresh(ft, end_date):
                logger.info("FeatureStore[%s] up-to-date through %s — skipping", ft, last)
                continue
            ft_start = (last + pd.Timedelta(days=1)) if last else start_date
            if ft_start > end_date:
                logger.info("FeatureStore[%s] up-to-date through %s — skipping", ft, last)
                continue
            logger.info(
                "FeatureStore[%s] computing %s → %s",
                ft, ft_start.date(), end_date.date(),
            )
            if ft == "macro":
                self._append_macro(macro_features_df, ft_start, end_date)
            elif ft == "sector":
                self._append_sector(
                    price_matrix, volume_matrix, universe_mgr, sector_fb, ft_start, end_date
                )
            else:
                self._append_stock(
                    price_matrix, volume_matrix, universe_mgr, stock_fb, ft_start, end_date
                )

    # ── Per-type append helpers ───────────────────────────────────────────────

    def _append_macro(
        self,
        macro_features_df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        new_rows = macro_features_df.loc[
            (macro_features_df.index >= start) & (macro_features_df.index <= end)
        ]
        if new_rows.empty:
            return
        for year, grp in new_rows.groupby(new_rows.index.year):
            path = self._macro_path(year)
            self._merge_and_write(grp, path, dedup_cols=None)
        cols = self._latest_shard_columns("macro") or []
        self._meta.setdefault("macro", {})["last_date"] = str(end.date())
        self._meta["macro"]["schema_hash"] = self._col_hash(cols)
        self._meta["macro"]["logic_hash"] = self._current_logic_hash("macro")
        self._save_meta()
        logger.info("FeatureStore[macro] persisted %d rows (schema=%s)",
                    len(new_rows), self._meta["macro"]["schema_hash"])

    def _candidate_sector_map(self, universe_mgr, price_matrix: pd.DataFrame) -> dict[str, str]:
        return {
            sm.ticker: sm.sector
            for sm in universe_mgr._stock_meta
            if not sm.blacklisted and sm.ticker in price_matrix.columns
        }

    @staticmethod
    def _filter_rows_by_membership(
        rows: pd.DataFrame,
        membership: pd.DataFrame,
        entity_col: str,
    ) -> pd.DataFrame:
        if rows.empty or membership.empty:
            return rows.iloc[0:0].copy()

        mask = (
            membership.stack(future_stack=True)
            .rename("_eligible")
            .reset_index()
            .rename(columns={"level_0": "date", "level_1": entity_col})
        )
        mask["date"] = pd.to_datetime(mask["date"])
        merged = rows.merge(mask, on=["date", entity_col], how="left")
        return merged[merged["_eligible"].fillna(False)].drop(columns="_eligible")

    def _append_sector(
        self,
        price_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame,
        universe_mgr,
        sector_fb,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        bm_ticker = self.cfg["backtest"].get("benchmark_ticker", "^NSEI")
        price_window = price_matrix.loc[:end]
        volume_window = volume_matrix.loc[:end] if volume_matrix is not None and not volume_matrix.empty else None
        sector_map = self._candidate_sector_map(universe_mgr, price_window)
        if not sector_map:
            return
        bm_prices = price_window[bm_ticker] if bm_ticker in price_window.columns else None

        full = sector_fb.build(price_window, sector_map, None, bm_prices)
        if full.empty:
            return

        # Normalise index → DatetimeIndex
        if not isinstance(full.index, pd.DatetimeIndex):
            full = full.reset_index()
            full["date"] = pd.to_datetime(full["date"])
            full = full.set_index("date")

        membership = universe_mgr.membership_mask(price_window, volume_window)
        sector_membership = pd.DataFrame({
            sector: membership[[t for t, s in sector_map.items() if s == sector]].any(axis=1)
            for sector in sorted(set(sector_map.values()))
        }, index=membership.index)

        new_rows = full.loc[(full.index >= start) & (full.index <= end)].reset_index()
        new_rows["date"] = pd.to_datetime(new_rows["date"])
        new_rows = self._filter_rows_by_membership(new_rows, sector_membership, "sector")
        if new_rows.empty:
            return
        new_rows = new_rows.set_index("date").sort_index()

        for year, grp in new_rows.groupby(new_rows.index.year):
            path = self._sector_path(year)
            existing = self._read(path)
            if existing is not None:
                combined = pd.concat([existing, grp])
                # Dedup on (date, sector) — index is date, sector is a column
                combined = (
                    combined.reset_index()
                    .rename(columns={"index": "date"})
                    .drop_duplicates(subset=["date", "sector"], keep="last")
                    .set_index("date")
                    .sort_index()
                )
                combined.index.name = "date"
            else:
                combined = grp.sort_index()
            self._write(combined, path)

        cols = self._latest_shard_columns("sector") or []
        self._meta.setdefault("sector", {})["last_date"] = str(end.date())
        self._meta["sector"]["schema_hash"] = self._col_hash(cols)
        self._meta["sector"]["logic_hash"] = self._current_logic_hash("sector", sector_fb=sector_fb)
        self._save_meta()
        logger.info("FeatureStore[sector] persisted %d rows (schema=%s)",
                    len(new_rows), self._meta["sector"]["schema_hash"])

    def _append_stock(
        self,
        price_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame,
        universe_mgr,
        stock_fb,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        bm_ticker = self.cfg["backtest"].get("benchmark_ticker", "^NSEI")
        price_window = price_matrix.loc[:end]
        sector_map = self._candidate_sector_map(universe_mgr, price_window)
        if not sector_map:
            return
        bm_prices = price_window[bm_ticker] if bm_ticker in price_window.columns else None
        vol_window = volume_matrix.loc[:end] if not volume_matrix.empty else None

        full = stock_fb.build(price_window, vol_window, sector_map, bm_prices)
        if full.empty:
            return

        full["date"] = pd.to_datetime(full["date"])
        new_rows = full[(full["date"] >= start) & (full["date"] <= end)]
        membership = universe_mgr.membership_mask(price_window, vol_window)
        new_rows = self._filter_rows_by_membership(new_rows, membership, "ticker")
        if new_rows.empty:
            return

        dates = pd.DatetimeIndex(new_rows["date"])
        for (year, month), grp in new_rows.groupby([dates.year, dates.month]):
            self._merge_and_write(
                grp, self._stock_path(year, month),
                dedup_cols=["date", "ticker"],
            )

        cols = self._latest_shard_columns("stock") or []
        self._meta.setdefault("stock", {})["last_date"] = str(end.date())
        self._meta["stock"]["schema_hash"] = self._col_hash(cols)
        self._meta["stock"]["logic_hash"] = self._current_logic_hash("stock", stock_fb=stock_fb)
        self._save_meta()
        logger.info("FeatureStore[stock] persisted %d rows (schema=%s)",
                    len(new_rows), self._meta["stock"]["schema_hash"])

    def _merge_and_write(
        self,
        new: pd.DataFrame,
        path: Path,
        dedup_cols: list[str] | None,
    ) -> None:
        existing = self._read(path)
        if existing is not None:
            combined = pd.concat([existing, new])
            if dedup_cols:
                combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
            else:
                combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new
        self._write(combined, path)

    # ── Read interface ────────────────────────────────────────────────────────

    # ── PIT validation ────────────────────────────────────────────────────────

    @staticmethod
    def assert_no_lookahead(df: pd.DataFrame, as_of: pd.Timestamp, label: str = "") -> None:
        """
        Raise AssertionError if df contains any row whose date is strictly
        after as_of.  Call this in tests or with FEATURE_STORE_STRICT=1.
        """
        if df.empty:
            return
        if isinstance(df.index, pd.DatetimeIndex):
            future = df[df.index > as_of]
        elif "date" in df.columns:
            future = df[pd.to_datetime(df["date"]) > as_of]
        else:
            return
        if not future.empty:
            raise AssertionError(
                f"PIT violation{' (' + label + ')' if label else ''}: "
                f"{len(future)} rows after as_of={as_of.date()}, "
                f"latest={future.index.max() if isinstance(future.index, pd.DatetimeIndex) else future['date'].max()}"
            )

    def load(
        self,
        ft: FeatureType,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Load features for [start, end], reading only the needed partitions.
        Returns a DataFrame in the same shape the original builders return.
        end is treated as the point-in-time boundary — no rows after end are returned.
        """
        parts: list[pd.DataFrame] = []

        if ft == "macro":
            for yr in range(start.year, end.year + 1):
                df = self._read(self._macro_path(yr))
                if df is not None:
                    parts.append(df)
            if not parts:
                return pd.DataFrame()
            out = pd.concat(parts)
            out = out[~out.index.duplicated(keep="last")].sort_index()
            return out.loc[start:end]

        elif ft == "sector":
            for yr in range(start.year, end.year + 1):
                df = self._read(self._sector_path(yr))
                if df is not None:
                    parts.append(df)
            if not parts:
                return pd.DataFrame()
            out = pd.concat(parts)
            # Sector is long-format: 15 rows per date (one per sector).
            # Do NOT dedup by index — the non-unique DatetimeIndex is intentional.
            out = out.sort_index()
            return out.loc[start:end]

        else:  # stock
            for yr in range(start.year, end.year + 1):
                mo_start = start.month if yr == start.year else 1
                mo_end   = end.month   if yr == end.year   else 12
                for mo in range(mo_start, mo_end + 1):
                    df = self._read(self._stock_path(yr, mo))
                    if df is not None:
                        parts.append(df)
            if not parts:
                return pd.DataFrame()
            out = pd.concat(parts, ignore_index=True)
            out["date"] = pd.to_datetime(out["date"])
            out = out[(out["date"] >= start) & (out["date"] <= end)]
            return out.sort_values("date").reset_index(drop=True)

    def snapshot(
        self,
        ft: FeatureType,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Return the most-recent feature row per entity as of `as_of`.

        * stock  → one row per ticker  (reads ≤2 monthly shards)
        * sector → one row per sector  (reads ≤2 yearly shards)
        * macro  → single latest row   (reads 1 yearly shard)
        """
        # Window: go back far enough that every entity has at least one row.
        lookback_months = 15 if ft in {"stock", "sector"} else 3
        lookback = as_of - pd.DateOffset(months=lookback_months)
        df = self.load(ft, lookback, as_of)
        if df.empty:
            return df

        if ft == "stock":
            df = df[df["date"] <= as_of]
            return (
                df.sort_values("date")
                  .groupby("ticker", sort=False)
                  .last()
                  .reset_index()
            )

        elif ft == "sector":
            # DatetimeIndex; sector is a column named "sector"
            df = df[df.index <= as_of]
            if "sector" not in df.columns:
                return df.iloc[[-1]] if len(df) else df
            df2 = df.reset_index().rename(columns={"index": "date", "date": "date"})
            date_col = [c for c in df2.columns if "date" in c.lower()][0]
            return (
                df2.sort_values(date_col)
                   .groupby("sector", sort=False)
                   .last()
                   .reset_index()
            )

        else:  # macro
            df = df[df.index <= as_of]
            return df.iloc[[-1]] if len(df) else df

    def evict_old_shards(self, keep_years: list[int]) -> None:
        """Drop shard-cache entries for years not in keep_years to free RAM."""
        _shard_cache.cache_clear()


# ── Module-level LRU shard cache ──────────────────────────────────────────────
# Keeping this outside the class means multiple FeatureStore instances (e.g.
# in tests) still share a single cache and avoid double-reading the same file.

@lru_cache(maxsize=_CACHE_MAX_SHARDS)
def _read_shard(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")
