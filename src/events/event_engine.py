"""
Event and geopolitical signal engine.

Converts structured event records into sector-level impact scores
with temporal decay. All events are pre-catalogued for the backtest
period; extension hooks allow adding real-time news signals.
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.contracts import MarketEvent

logger = logging.getLogger(__name__)


# ── Historical event catalog ──────────────────────────────────────────────────
# Format: (event_type, date_str, geography, severity, affected_sectors, sentiment)
_EVENT_CATALOG: list[dict] = [
    # ── Global macro shocks ───────────────────────────────────────────────────
    {"type": "oil_shock", "date": "2014-11-01", "geo": "global",
     "severity": 0.7, "sectors": ["Energy", "Automobiles", "Chemicals"],
     "sentiment": 0.5, "description": "OPEC oil price collapse begins"},
    {"type": "currency_shock", "date": "2015-08-11", "geo": "china",
     "severity": 0.6, "sectors": ["IT", "Metals", "Energy"],
     "sentiment": -0.6, "description": "China yuan devaluation"},
    {"type": "commodity_shock", "date": "2016-02-01", "geo": "global",
     "severity": 0.5, "sectors": ["Metals", "Energy", "CapitalGoods"],
     "sentiment": -0.5, "description": "Commodity rout; crude at 12Y low"},
    {"type": "geopolitical", "date": "2016-11-09", "geo": "usa",
     "severity": 0.4, "sectors": ["IT", "Pharma", "FMCG"],
     "sentiment": -0.3, "description": "US election: Trump win"},
    {"type": "policy", "date": "2016-11-08", "geo": "india",
     "severity": 0.8, "sectors": ["Banking", "FMCG", "RealEstate"],
     "sentiment": -0.7, "description": "India demonetisation"},
    {"type": "macro", "date": "2018-10-01", "geo": "global",
     "severity": 0.5, "sectors": ["IT", "Banking", "FinancialServices"],
     "sentiment": -0.5, "description": "IL&FS crisis; NBFC credit crunch"},
    {"type": "macro", "date": "2018-09-01", "geo": "india",
     "severity": 0.6, "sectors": ["Energy", "Automobiles", "FMCG"],
     "sentiment": -0.6, "description": "Crude surge + rupee at 74"},
    {"type": "macro", "date": "2019-08-05", "geo": "global",
     "severity": 0.5, "sectors": ["IT", "Pharma"],
     "sentiment": -0.4, "description": "US-China trade war escalation"},
    {"type": "pandemic", "date": "2020-03-01", "geo": "global",
     "severity": 1.0, "sectors": ["all"],
     "sentiment": -1.0, "description": "COVID-19 pandemic market crash"},
    {"type": "macro", "date": "2020-03-27", "geo": "india",
     "severity": 0.5, "sectors": ["Banking", "FinancialServices", "RealEstate"],
     "sentiment": 0.7, "description": "RBI emergency rate cut 75bps"},
    {"type": "geopolitical", "date": "2020-06-15", "geo": "india",
     "severity": 0.6, "sectors": ["CapitalGoods", "Metals", "Energy"],
     "sentiment": -0.5, "description": "India-China Galwan Valley clash"},
    {"type": "commodity_shock", "date": "2022-02-24", "geo": "global",
     "severity": 0.8, "sectors": ["Energy", "Metals", "Chemicals", "Cement"],
     "sentiment": -0.7, "description": "Russia-Ukraine war begins"},
    {"type": "macro", "date": "2022-05-04", "geo": "usa",
     "severity": 0.7, "sectors": ["IT", "Banking", "FinancialServices"],
     "sentiment": -0.6, "description": "Fed rate hike cycle begins"},
    {"type": "macro", "date": "2022-10-01", "geo": "india",
     "severity": 0.4, "sectors": ["Banking", "CapitalGoods", "RealEstate"],
     "sentiment": 0.5, "description": "India capex cycle; RBI rate hikes"},
    {"type": "geopolitical", "date": "2023-10-07", "geo": "middle_east",
     "severity": 0.5, "sectors": ["Energy", "CapitalGoods"],
     "sentiment": -0.4, "description": "Israel-Hamas war escalation"},
    {"type": "election", "date": "2024-05-23", "geo": "india",
     "severity": 0.6, "sectors": ["CapitalGoods", "RealEstate", "Banking"],
     "sentiment": 0.6, "description": "India general election results"},
    {"type": "macro", "date": "2025-04-02", "geo": "global",
     "severity": 0.7, "sectors": ["IT", "Pharma", "Chemicals"],
     "sentiment": -0.6, "description": "US tariff announcements"},
    {"type": "macro", "date": "2025-04-09", "geo": "india",
     "severity": 0.4, "sectors": ["Banking", "FinancialServices", "FMCG"],
     "sentiment": 0.6, "description": "RBI rate cut 25bps"},
]

ALL_SECTORS = [
    "IT", "Banking", "FinancialServices", "FMCG", "Automobiles",
    "Pharma", "Energy", "Metals", "Telecom", "Cement",
    "CapitalGoods", "ConsumerDiscretionary", "Healthcare", "RealEstate",
    "Chemicals",
]


class EventEngine:
    """Converts events into time-decayed sector impact scores."""

    def __init__(self, decay_halflife_days: int = 21):
        self.decay_halflife = decay_halflife_days
        self._events = self._load_catalog()

    def _load_catalog(self) -> list[MarketEvent]:
        events = []
        for i, raw in enumerate(_EVENT_CATALOG):
            sectors = raw["sectors"]
            if sectors == ["all"]:
                sectors = ALL_SECTORS

            impact = {sec: raw["sentiment"] * raw["severity"] for sec in sectors}
            events.append(MarketEvent(
                event_id=f"evt_{i:04d}",
                event_type=raw["type"],
                date=date.fromisoformat(raw["date"]),
                geography=raw["geo"],
                affected_sectors=sectors,
                affected_stocks=[],
                severity=raw["severity"],
                duration_days=30,
                confidence=0.8,
                sentiment=raw["sentiment"],
                first_order_impact=impact,
                decay_halflife_days=self.decay_halflife,
            ))
        return events

    def get_sector_impact(
        self, as_of: date, lookback_days: int = 90
    ) -> dict[str, float]:
        """
        Return decayed event impact score per sector as of `as_of`.
        Positive = tailwind, negative = headwind.
        """
        cutoff = as_of - timedelta(days=lookback_days)
        impacts: dict[str, float] = {sec: 0.0 for sec in ALL_SECTORS}

        for evt in self._events:
            if evt.date > as_of or evt.date < cutoff:
                continue
            days_elapsed = (as_of - evt.date).days
            decay = math.exp(-days_elapsed * math.log(2) / evt.decay_halflife_days)

            for sector, raw_impact in evt.first_order_impact.items():
                if sector in impacts:
                    impacts[sector] += raw_impact * decay

        return impacts

    def get_recent_events(
        self, as_of: date, lookback_days: int = 30
    ) -> list[MarketEvent]:
        cutoff = as_of - timedelta(days=lookback_days)
        return [e for e in self._events if cutoff <= e.date <= as_of]

    def add_event(self, event: MarketEvent) -> None:
        """Add a new event (e.g., from real-time news feed)."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.date)

    def build_event_feature_series(
        self,
        date_index: pd.DatetimeIndex,
        sector: str,
    ) -> pd.Series:
        """Build daily event impact time series for a sector."""
        values = []
        for ts in date_index:
            score = self.get_sector_impact(ts.date())
            values.append(score.get(sector, 0.0))
        return pd.Series(values, index=date_index, name=f"event_impact_{sector}")
