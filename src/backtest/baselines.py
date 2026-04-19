"""
Baseline strategies for benchmarking.

1. Nifty buy-and-hold         — track Nifty 50 price series
2. Equal-weight rebalance     — 1/N across all universe stocks
3. Sector-momentum            — overweight top-3 sectors each period
4. Non-RL optimizer           — full pipeline but rule-based (no RL overlay)
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.backtest.simulator import PortfolioSimulator
from src.data.contracts import PortfolioState
from src.data.universe import UniverseManager
from src.optimizer.portfolio_optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)


# ── 1. Buy-and-hold benchmark ─────────────────────────────────────────────────

def nifty_buy_and_hold(
    price_matrix: pd.DataFrame,
    initial_capital: float,
    start_date: str,
    end_date: str,
    benchmark_ticker: str = "^NSEI",
) -> pd.Series:
    """NAV series for full-period Nifty 50 buy-and-hold."""
    if benchmark_ticker not in price_matrix.columns:
        logger.warning("Benchmark ticker %s not in price matrix", benchmark_ticker)
        return pd.Series(dtype=float)

    prices = price_matrix[benchmark_ticker].loc[start_date:end_date].dropna()
    if prices.empty:
        return pd.Series(dtype=float)

    nav = prices / prices.iloc[0] * initial_capital
    nav.name = "nifty_buyhold"
    return nav


# ── 2. Equal-weight rebalance ─────────────────────────────────────────────────

def equal_weight_backtest(
    price_matrix: pd.DataFrame,
    cfg: dict,
    rebalance_dates: list[pd.Timestamp],
    initial_capital: float,
) -> pd.Series:
    """Equal-weight rebalance every period."""
    simulator = PortfolioSimulator(cfg)
    uni_mgr = UniverseManager(cfg)

    portfolio = PortfolioState(
        date=rebalance_dates[0].date(),
        cash=float(initial_capital),
        holdings={},
        weights={"CASH": 1.0},
        nav=float(initial_capital),
        sector_weights={},
    )

    nav_points: list[tuple] = [(rebalance_dates[0], portfolio.nav)]

    for i in range(len(rebalance_dates) - 1):
        current = rebalance_dates[i]
        nxt = rebalance_dates[i + 1]

        prices = price_matrix.loc[:current].iloc[-1].dropna()
        snapshot = uni_mgr.get_universe(current.date(), price_matrix=price_matrix)

        tickers = [t for t in snapshot.tickers if t in prices.index]
        if not tickers:
            continue

        # Equal weight
        w = 1.0 / len(tickers)
        target = {t: w for t in tickers}
        target["CASH"] = 0.02

        # Scale to sum to 1
        total = sum(target.values())
        target = {k: v / total for k, v in target.items()}

        portfolio = simulator.value_portfolio(portfolio, prices, current.date())
        result = simulator.execute_rebalance(target, portfolio, prices, current.date())
        portfolio = result.new_portfolio

        # Daily NAV
        period_data = price_matrix.loc[current:nxt]
        nav_pts = _interpolate_nav(portfolio, period_data, current)
        nav_points.extend(nav_pts)

    dates, navs = zip(*nav_points)
    s = pd.Series(list(navs), index=pd.DatetimeIndex(list(dates)), name="equal_weight")
    return s[~s.index.duplicated(keep="last")].sort_index()


# ── 3. Sector momentum ────────────────────────────────────────────────────────

def sector_momentum_backtest(
    price_matrix: pd.DataFrame,
    cfg: dict,
    rebalance_dates: list[pd.Timestamp],
    initial_capital: float,
    n_top_sectors: int = 5,
    lookback: int = 63,
) -> pd.Series:
    """Overweight top-N sectors by 3-month momentum, equal weight within."""
    simulator = PortfolioSimulator(cfg)
    uni_mgr = UniverseManager(cfg)
    optimizer = PortfolioOptimizer(cfg)

    portfolio = PortfolioState(
        date=rebalance_dates[0].date(),
        cash=float(initial_capital),
        holdings={},
        weights={"CASH": 1.0},
        nav=float(initial_capital),
        sector_weights={},
    )

    nav_points = [(rebalance_dates[0], portfolio.nav)]

    for i in range(len(rebalance_dates) - 1):
        current = rebalance_dates[i]
        nxt = rebalance_dates[i + 1]

        prices = price_matrix.loc[:current].iloc[-1].dropna()
        snapshot = uni_mgr.get_universe(current.date(), price_matrix=price_matrix)
        sector_map = uni_mgr.get_sector_map(snapshot)

        # Sector 3m returns
        hist = price_matrix.loc[:current].iloc[-(lookback + 1):]
        sector_rets: dict[str, float] = {}
        for sec in snapshot.sectors:
            sec_tickers = [t for t in snapshot.tickers if sector_map.get(t) == sec and t in hist.columns]
            if not sec_tickers:
                continue
            sec_prices = hist[sec_tickers].mean(axis=1).dropna()
            if len(sec_prices) < 2:
                continue
            ret = sec_prices.iloc[-1] / sec_prices.iloc[0] - 1
            sector_rets[sec] = float(ret)

        top_sectors = sorted(sector_rets, key=sector_rets.get, reverse=True)[:n_top_sectors]

        # Equal weight within top sectors
        target: dict[str, float] = {}
        eligible = [t for t in snapshot.tickers if sector_map.get(t) in top_sectors and t in prices.index]
        if not eligible:
            eligible = [t for t in snapshot.tickers if t in prices.index][:20]

        for t in eligible:
            target[t] = 1.0 / len(eligible)
        target["CASH"] = 0.03

        total = sum(target.values())
        target = {k: v / total for k, v in target.items()}

        portfolio = simulator.value_portfolio(portfolio, prices, current.date())
        result = simulator.execute_rebalance(target, portfolio, prices, current.date())
        portfolio = result.new_portfolio

        period_data = price_matrix.loc[current:nxt]
        nav_pts = _interpolate_nav(portfolio, period_data, current)
        nav_points.extend(nav_pts)

    dates, navs = zip(*nav_points)
    s = pd.Series(list(navs), index=pd.DatetimeIndex(list(dates)), name="sector_momentum")
    return s[~s.index.duplicated(keep="last")].sort_index()


# ── Helper ────────────────────────────────────────────────────────────────────

def _interpolate_nav(
    portfolio: PortfolioState,
    price_period: pd.DataFrame,
    entry_date: pd.Timestamp,
) -> list[tuple]:
    tickers = [t for t in portfolio.holdings if t in price_period.columns]
    nav_pts = []
    for ts, row in price_period.iterrows():
        if ts <= entry_date:
            continue
        value = sum(
            portfolio.holdings.get(t, 0) * float(row.get(t, 0) or 0)
            for t in tickers
        ) + portfolio.cash
        nav_pts.append((ts, max(value, 0)))
    return nav_pts


# ── Summary ───────────────────────────────────────────────────────────────────

def compare_strategies(
    strategy_navs: dict[str, pd.Series],
    metrics_fn,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Build comparison table of performance metrics across strategies."""
    rows = []
    for name, nav in strategy_navs.items():
        m = metrics_fn(nav, benchmark)
        m["strategy"] = name
        rows.append(m)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("strategy")
    key_cols = ["cagr", "ann_volatility", "sharpe", "sortino", "calmar",
                "max_drawdown", "hit_rate", "benchmark_cagr", "information_ratio"]
    available = [c for c in key_cols if c in df.columns]
    return df[available].round(4)
