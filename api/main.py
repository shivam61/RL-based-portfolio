"""
Portfolio Management API

Endpoints:
  POST /portfolios              → Create new portfolio, returns ID + initial allocation
  GET  /portfolios/{id}         → Get current portfolio state
  POST /portfolios/{id}/rebalance → Suggest rebalance trades based on latest market state
  GET  /suggest                 → Fresh allocation recommendation (no tracking)

Usage:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Thread isolation to prevent OpenMP/PyTorch conflicts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
import json
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import load_config
from src.api.recommender import PortfolioRecommender
from src.api.portfolio_store import PortfolioStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RL Portfolio API",
    description="Indian equity portfolio recommendations powered by RL + ML",
    version="1.0.0",
)

_cfg = load_config()
_recommender = PortfolioRecommender(_cfg)
_store = PortfolioStore(Path(_cfg["paths"]["artifact_dir"]) / "portfolios")


# ── Request / Response models ──────────────────────────────────────────────────

class CreatePortfolioRequest(BaseModel):
    capital_inr: float = 500_000
    risk_profile: str = "moderate"   # conservative / moderate / aggressive
    label: str = ""                   # optional human label

class RebalanceRequest(BaseModel):
    current_holdings: dict[str, float] = {}  # ticker → current value in INR (optional)


def _run_recommendation(**kwargs):
    try:
        return _recommender.recommend(**kwargs)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _recommender.is_ready(),
        **_recommender.policy_status(),
    }


@app.get("/suggest")
def suggest_allocation(capital_inr: float = 500_000, risk_profile: str = "moderate"):
    """
    Suggest a fresh portfolio allocation without tracking state.
    Returns target weights and trades to execute from cash.
    """
    rec = _run_recommendation(capital_inr=capital_inr, risk_profile=risk_profile)
    return rec


@app.post("/portfolios")
def create_portfolio(req: CreatePortfolioRequest):
    """
    Create a new tracked portfolio. Returns a unique portfolio ID.
    Use this ID for subsequent rebalance calls.
    """
    portfolio_id = str(uuid.uuid4())[:8]
    rec = _run_recommendation(
        capital_inr=req.capital_inr,
        risk_profile=req.risk_profile,
    )

    portfolio = {
        "id":            portfolio_id,
        "label":         req.label or f"Portfolio-{portfolio_id}",
        "created_at":    datetime.utcnow().isoformat(),
        "last_updated":  datetime.utcnow().isoformat(),
        "capital_inr":   req.capital_inr,
        "risk_profile":  req.risk_profile,
        "allocation":    rec["allocation"],
        "trades":        rec["trades"],
        "sector_tilts":  rec["sector_tilts"],
        "as_of_date":    rec["as_of_date"],
        "history":       [{"date": rec["as_of_date"], "event": "created", "allocation": rec["allocation"]}],
    }
    _store.save(portfolio_id, portfolio)
    logger.info("Created portfolio %s with capital ₹%.0f", portfolio_id, req.capital_inr)
    return portfolio


@app.get("/portfolios/{portfolio_id}")
def get_portfolio(portfolio_id: str):
    """Get current state of a tracked portfolio."""
    p = _store.load(portfolio_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_id}' not found")
    return p


@app.get("/portfolios")
def list_portfolios():
    """List all tracked portfolio IDs and labels."""
    return _store.list_all()


@app.post("/portfolios/{portfolio_id}/rebalance")
def rebalance_portfolio(portfolio_id: str, req: RebalanceRequest):
    """
    Suggest rebalance trades for an existing portfolio.
    Optionally pass current_holdings (ticker → INR value) for accurate diff.
    Returns: target_allocation, trades (buys/sells), expected turnover.
    """
    p = _store.load(portfolio_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"Portfolio '{portfolio_id}' not found")

    current_alloc = req.current_holdings or {
        t: w * p["capital_inr"]
        for t, w in p["allocation"].items()
    }

    rec = _run_recommendation(
        capital_inr=p["capital_inr"],
        risk_profile=p["risk_profile"],
        current_holdings=current_alloc,
    )

    # Diff: what changed vs last known allocation
    prev = p["allocation"]
    diff = {}
    for t, new_w in rec["allocation"].items():
        old_w = prev.get(t, 0.0)
        if abs(new_w - old_w) > 0.005:
            diff[t] = {"from": round(old_w, 4), "to": round(new_w, 4), "delta": round(new_w - old_w, 4)}

    # Update stored portfolio
    p["allocation"]   = rec["allocation"]
    p["trades"]       = rec["trades"]
    p["sector_tilts"] = rec["sector_tilts"]
    p["as_of_date"]   = rec["as_of_date"]
    p["last_updated"] = datetime.utcnow().isoformat()
    p["history"].append({
        "date":       rec["as_of_date"],
        "event":      "rebalance",
        "allocation": rec["allocation"],
    })
    _store.save(portfolio_id, p)

    return {
        "portfolio_id":  portfolio_id,
        "as_of_date":    rec["as_of_date"],
        "target":        rec["allocation"],
        "trades":        rec["trades"],
        "changes":       diff,
        "sector_tilts":  rec["sector_tilts"],
        "turnover_est":  round(sum(abs(v["delta"]) for v in diff.values()) / 2, 4),
    }
