"""JSON-file backed portfolio store. One file per portfolio ID."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class PortfolioStore:
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, portfolio_id: str) -> Path:
        return self.store_dir / f"{portfolio_id}.json"

    def save(self, portfolio_id: str, data: dict) -> None:
        with open(self._path(portfolio_id), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, portfolio_id: str) -> Optional[dict]:
        p = self._path(portfolio_id)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def list_all(self) -> list[dict]:
        result = []
        for f in sorted(self.store_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    d = json.load(fh)
                result.append({
                    "id":           d.get("id"),
                    "label":        d.get("label"),
                    "capital_inr":  d.get("capital_inr"),
                    "risk_profile": d.get("risk_profile"),
                    "created_at":   d.get("created_at"),
                    "last_updated": d.get("last_updated"),
                    "as_of_date":   d.get("as_of_date"),
                })
            except Exception:
                pass
        return result
