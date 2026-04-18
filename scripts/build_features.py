"""
Feature store builder.

First run (full backfill, ~5–15 min):
    python scripts/build_features.py

Daily incremental update (seconds):
    python scripts/build_features.py --incremental

Only recomputes dates not already in the store.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.macro import MacroDataManager
from src.data.universe import UniverseManager
from src.features.feature_store import FeatureStore
from src.features.macro_features import MacroFeatureBuilder
from src.features.sector_features import SectorFeatureBuilder
from src.features.stock_features import StockFeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--incremental", is_flag=True, default=False,
              help="Only append dates after the last computed date.")
@click.option("--start", default=None, help="Override start date (YYYY-MM-DD).")
@click.option("--end", default=None, help="Override end date (YYYY-MM-DD).")
@click.option("--feature-types", default="macro,sector,stock",
              help="Comma-separated list of feature types to build.")
def main(incremental: bool, start: str | None, end: str | None, feature_types: str) -> None:
    cfg = load_config()
    bt  = cfg["backtest"]

    # ── Load raw data ─────────────────────────────────────────────────────────
    processed = Path(cfg["paths"]["processed_data"])
    price_path  = processed / "price_matrix.parquet"
    volume_path = processed / "volume_matrix.parquet"
    macro_path  = processed / "macro_data.parquet"

    if not price_path.exists():
        logger.error("Price matrix not found at %s — run download_data.py first", price_path)
        sys.exit(1)

    logger.info("Loading price matrix ...")
    price_matrix  = pd.read_parquet(price_path)
    volume_matrix = pd.read_parquet(volume_path) if volume_path.exists() else pd.DataFrame()
    macro_df      = pd.read_parquet(macro_path)  if macro_path.exists()  else pd.DataFrame()

    # ── Compute date range ────────────────────────────────────────────────────
    start_ts = pd.Timestamp(start) if start else pd.Timestamp(bt["start_date"])
    end_ts   = pd.Timestamp(end)   if end   else (
        pd.Timestamp(bt["end_date"]) if bt["end_date"] != "latest"
        else price_matrix.index.max()
    )

    # ── Build macro features ──────────────────────────────────────────────────
    logger.info("Building macro features ...")
    macro_fb       = MacroFeatureBuilder(cfg)
    macro_features = macro_fb.build(macro_df)

    # ── Init managers / builders ──────────────────────────────────────────────
    universe_mgr = UniverseManager(cfg)
    sector_fb    = SectorFeatureBuilder(cfg)
    stock_fb     = StockFeatureBuilder(cfg)

    # ── Feature store ─────────────────────────────────────────────────────────
    store_dir = Path(cfg["paths"]["artifact_dir"]) / "feature_store"
    store     = FeatureStore(store_dir, cfg)

    if incremental:
        for ft in feature_types.split(","):
            ft = ft.strip()
            last = store.last_computed_date(ft)
            logger.info(
                "Incremental mode: %s last_date=%s → appending up to %s",
                ft, last, end_ts.date(),
            )

    logger.info("Building feature store %s → %s ...", start_ts.date(), end_ts.date())
    store.build_or_append(
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        macro_df=macro_df,
        macro_features_df=macro_features,
        sector_fb=sector_fb,
        stock_fb=stock_fb,
        universe_mgr=universe_mgr,
        start_date=start_ts,
        end_date=end_ts,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Feature store complete — %s", store_dir)
    for ft in ("macro", "sector", "stock"):
        last = store.last_computed_date(ft)
        logger.info("  %-8s last_date = %s", ft, last)

    # Disk usage
    total_mb = sum(f.stat().st_size for f in store_dir.rglob("*.parquet")) / 1e6
    logger.info("  Disk usage: %.1f MB", total_mb)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
