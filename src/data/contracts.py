"""Pydantic data contracts — strict schemas for all inter-module data."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Universe ──────────────────────────────────────────────────────────────────

class StockMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    name: str
    sector: str
    cap: Literal["large", "mid", "small"]
    listed_since: Optional[date] = None
    delisted_on: Optional[date] = None
    blacklisted: bool = False


class UniverseSnapshot(BaseModel):
    """Point-in-time universe valid for a given date."""
    as_of: date
    stocks: list[StockMeta]

    @property
    def tickers(self) -> list[str]:
        return [s.ticker for s in self.stocks]

    @property
    def sectors(self) -> list[str]:
        return sorted({s.sector for s in self.stocks})

    def by_sector(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.stocks:
            out.setdefault(s.sector, []).append(s.ticker)
        return out


# ── OHLCV ─────────────────────────────────────────────────────────────────────

class DailyBar(BaseModel):
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float

    @field_validator("close", "adj_close", "open", "high", "low")
    @classmethod
    def positive_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Price must be positive, got {v}")
        return v


# ── Macro ─────────────────────────────────────────────────────────────────────

class MacroSnapshot(BaseModel):
    date: date
    usdinr: Optional[float] = None
    crude_oil: Optional[float] = None
    gold: Optional[float] = None
    vix: Optional[float] = None
    dxy: Optional[float] = None
    sp500: Optional[float] = None
    us_10y_yield: Optional[float] = None
    us_2y_yield: Optional[float] = None
    nifty50: Optional[float] = None


# ── Portfolio state ───────────────────────────────────────────────────────────

class PortfolioState(BaseModel):
    date: date
    cash: float = Field(ge=0)
    holdings: dict[str, float] = Field(default_factory=dict)  # ticker → shares
    weights: dict[str, float] = Field(default_factory=dict)   # ticker → weight
    nav: float = Field(ge=0)                                   # total portfolio value
    sector_weights: dict[str, float] = Field(default_factory=dict)

    @field_validator("cash")
    @classmethod
    def cash_non_negative(cls, v: float) -> float:
        if v < -1e-6:
            raise ValueError(f"Cash cannot be negative: {v}")
        return max(v, 0.0)


# ── Trade ─────────────────────────────────────────────────────────────────────

class Trade(BaseModel):
    ticker: str
    date: date
    direction: Literal["buy", "sell"]
    shares: float
    price: float                   # execution price (incl. slippage)
    gross_value: float
    transaction_cost: float
    net_value: float               # gross ± cost


# ── Rebalance record ──────────────────────────────────────────────────────────

class RebalanceRecord(BaseModel):
    rebalance_date: date
    pre_nav: float
    post_nav: float
    trades: list[Trade]
    target_weights: dict[str, float]
    sector_tilts: dict[str, float]
    cash_target: float
    aggressiveness: float
    total_turnover: float
    total_cost: float
    rl_action: Optional[dict] = None
    emergency: bool = False


# ── Performance metrics ───────────────────────────────────────────────────────

class PeriodReturn(BaseModel):
    start_date: date
    end_date: date
    portfolio_return: float
    benchmark_return: float
    active_return: float
    nav_start: float
    nav_end: float


class BacktestMetrics(BaseModel):
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    cagr: float
    ann_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_start: date
    max_drawdown_end: date
    hit_rate: float               # fraction of periods with positive return
    avg_turnover: float
    total_rebalances: int
    benchmark_cagr: float
    information_ratio: float
    year_returns: dict[int, float]
    sector_contributions: dict[str, float]


# ── Feature vectors ───────────────────────────────────────────────────────────

class MacroFeatureVector(BaseModel):
    date: date
    features: dict[str, float]


class SectorFeatureVector(BaseModel):
    date: date
    sector: str
    features: dict[str, float]


class StockFeatureVector(BaseModel):
    date: date
    ticker: str
    sector: str
    features: dict[str, float]


# ── Event ─────────────────────────────────────────────────────────────────────

class MarketEvent(BaseModel):
    event_id: str
    event_type: str                # war | election | rbi | budget | commodity_shock | ...
    date: date
    geography: str
    affected_sectors: list[str]
    affected_stocks: list[str]
    severity: float = Field(ge=0, le=1)
    duration_days: int
    confidence: float = Field(ge=0, le=1)
    sentiment: float = Field(ge=-1, le=1)
    first_order_impact: dict[str, float]   # sector → impact score
    decay_halflife_days: int = 21
