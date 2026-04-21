#!/usr/bin/env python3
"""Export clean per-sector stock-ranker feature importance artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.models.stock_ranker import StockRanker


def main() -> None:
    cfg = load_config()
    model_path = Path(cfg["paths"]["model_dir"]) / "stock_ranker.pkl"
    out_dir = Path(cfg["paths"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ranker = StockRanker(cfg)
    ranker.load(model_path)

    rows: list[dict] = []
    for sector in sorted(ranker.models.keys()):
        importance = ranker.feature_importance(sector)
        if importance is None or importance.empty:
            continue

        total = float(importance.sum())
        for rank, (feature, value) in enumerate(importance.items(), start=1):
            rows.append(
                {
                    "sector": sector,
                    "feature": str(feature),
                    "importance": float(value),
                    "importance_share": float(value / total) if total > 0 else 0.0,
                    "rank": int(rank),
                }
            )

    if not rows:
        raise SystemExit("No stock-ranker importance rows found")

    df = pd.DataFrame(rows).sort_values(["sector", "rank"], kind="mergesort")
    csv_path = out_dir / "stock_ranker_feature_importance.csv"
    json_path = out_dir / "stock_ranker_feature_importance.json"
    df.to_csv(csv_path, index=False)

    summary = {
        "n_sectors": int(df["sector"].nunique()),
        "n_features": int(len(df)),
        "sectors": {},
    }
    for sector in sorted(df["sector"].unique()):
        sec_df = df.loc[df["sector"] == sector, ["feature", "importance", "importance_share", "rank"]]
        summary["sectors"][sector] = [
            {
                "feature": str(row["feature"]),
                "importance": float(row["importance"]),
                "importance_share": float(row["importance_share"]),
                "rank": int(row["rank"]),
            }
            for _, row in sec_df.iterrows()
        ]

    json_path.write_text(json.dumps(summary, indent=2))
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
