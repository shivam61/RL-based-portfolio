"""
Core backtesting simulator.

Handles:
- Trade execution with transaction costs and slippage
- Portfolio valuation at each date
- NAV time series construction
- P&L attribution per trade
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.contracts import PortfolioState, RebalanceRecord, Trade

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    trades: list[Trade]
    new_portfolio: PortfolioState
    total_cost: float
    total_turnover: float


class PortfolioSimulator:
    """
    Simulates portfolio execution given target weights.

    Assumptions:
    - Prices are adj_close from price_matrix
    - Transaction costs: fixed BPS round-trip per trade
    - Slippage: fixed BPS per trade (modeled as adverse price move)
    - Execution: same-day close (T+0 for Indian market simplicity,
      configurable to T+1)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        bt_cfg = cfg["backtest"]
        self.tc_bps = bt_cfg.get("transaction_cost_bps", 25)  # one-way BPS
        self.slippage_bps = bt_cfg.get("slippage_bps", 10)    # one-way BPS
        self.total_cost_bps = self.tc_bps + self.slippage_bps

    # ── Trade execution ───────────────────────────────────────────────────────

    def execute_rebalance(
        self,
        target_weights: dict[str, float],   # ticker → weight (incl CASH)
        current_state: PortfolioState,
        prices: pd.Series,                   # ticker → current price
        exec_date: date,
    ) -> ExecutionResult:
        """
        Execute rebalance from current state to target weights.
        Returns updated portfolio state and trade list.
        """
        nav = current_state.nav
        trades = []
        total_cost = 0.0

        # Compute target holdings
        target_holdings: dict[str, float] = {}
        for ticker, w in target_weights.items():
            if ticker == "CASH":
                continue
            if ticker not in prices.index or pd.isna(prices[ticker]) or prices[ticker] <= 0:
                continue
            target_value = nav * w
            raw_shares = target_value / prices[ticker]
            target_holdings[ticker] = raw_shares

        # Determine sells and buys
        current_holdings = dict(current_state.holdings)
        all_tickers = set(list(current_holdings.keys()) + list(target_holdings.keys()))

        for ticker in all_tickers:
            if ticker == "CASH":
                continue
            current_shares = current_holdings.get(ticker, 0.0)
            target_shares = target_holdings.get(ticker, 0.0)
            delta_shares = target_shares - current_shares

            if abs(delta_shares) < 1e-6:
                continue

            if ticker not in prices.index or pd.isna(prices[ticker]):
                continue

            base_price = prices[ticker]
            direction = "buy" if delta_shares > 0 else "sell"

            # apply slippage (adverse for both buy and sell)
            slippage_factor = self.slippage_bps / 10000
            if direction == "buy":
                exec_price = base_price * (1 + slippage_factor)
            else:
                exec_price = base_price * (1 - slippage_factor)

            gross_value = abs(delta_shares) * exec_price
            tc = gross_value * (self.tc_bps / 10000)
            net_value = gross_value + tc if direction == "buy" else gross_value - tc
            total_cost += tc

            trades.append(Trade(
                ticker=ticker,
                date=exec_date,
                direction=direction,
                shares=abs(delta_shares),
                price=exec_price,
                gross_value=gross_value,
                transaction_cost=tc,
                net_value=net_value,
            ))

        # Compute new holdings and cash
        new_holdings = dict(current_holdings)
        cash = current_state.cash

        for trade in trades:
            if trade.direction == "buy":
                new_holdings[trade.ticker] = new_holdings.get(trade.ticker, 0) + trade.shares
                cash -= trade.net_value
            else:
                new_holdings[trade.ticker] = new_holdings.get(trade.ticker, 0) - trade.shares
                cash += trade.net_value

        # Remove near-zero holdings
        new_holdings = {t: s for t, s in new_holdings.items() if s > 1e-6}

        # Compute new NAV
        holdings_value = sum(
            shares * prices.get(t, 0)
            for t, shares in new_holdings.items()
            if not pd.isna(prices.get(t, np.nan))
        )
        new_nav = max(holdings_value + cash, 1.0)  # floor at 1 rupee to avoid degenerate state

        # Compute new weights
        new_weights: dict[str, float] = {}
        for t, shares in new_holdings.items():
            p = prices.get(t, 0)
            if p > 0 and new_nav > 0:
                new_weights[t] = shares * p / new_nav

        if new_nav > 0:
            new_weights["CASH"] = cash / new_nav

        # Sector weights
        sector_map = {t: "unknown" for t in new_holdings}  # will be filled by caller
        new_state = PortfolioState(
            date=exec_date,
            cash=max(cash, 0),
            holdings=new_holdings,
            weights=new_weights,
            nav=new_nav,
            sector_weights={},
        )

        # Compute turnover
        all_t = set(list(current_state.weights.keys()) + list(new_weights.keys()))
        turnover = sum(
            abs(new_weights.get(t, 0) - current_state.weights.get(t, 0))
            for t in all_t
        ) / 2.0  # one-way turnover

        return ExecutionResult(
            trades=trades,
            new_portfolio=new_state,
            total_cost=total_cost,
            total_turnover=turnover,
        )

    # ── Portfolio valuation ───────────────────────────────────────────────────

    def value_portfolio(
        self,
        state: PortfolioState,
        prices: pd.Series,
        as_of: date,
    ) -> PortfolioState:
        """Mark portfolio to market at given date."""
        holdings_value = 0.0
        new_weights: dict[str, float] = {}

        for ticker, shares in state.holdings.items():
            price = prices.get(ticker)
            if price is not None and not pd.isna(price) and price > 0:
                value = shares * price
                holdings_value += value
            elif shares > 0:
                # use last known price from current state
                old_w = state.weights.get(ticker, 0)
                holdings_value += old_w * state.nav  # approximation

        new_nav = holdings_value + state.cash
        if new_nav <= 0:
            new_nav = state.nav  # fallback to prevent zero NAV

        for ticker, shares in state.holdings.items():
            price = prices.get(ticker)
            if price is not None and not pd.isna(price) and price > 0:
                new_weights[ticker] = shares * price / new_nav

        if new_nav > 0:
            new_weights["CASH"] = state.cash / new_nav

        return PortfolioState(
            date=as_of,
            cash=state.cash,
            holdings=state.holdings,
            weights=new_weights,
            nav=new_nav,
            sector_weights=state.sector_weights,
        )

    # ── NAV series ────────────────────────────────────────────────────────────

    @staticmethod
    def compute_nav_series(
        rebalance_records: list[RebalanceRecord],
        price_matrix: pd.DataFrame,
        initial_state: PortfolioState,
    ) -> pd.Series:
        """
        Interpolate daily NAV between rebalance dates using price changes.
        """
        if not rebalance_records:
            return pd.Series(dtype=float)

        nav_points: list[tuple] = [(initial_state.date, initial_state.nav)]
        holdings = dict(initial_state.holdings)
        cash = initial_state.cash

        for i, rec in enumerate(rebalance_records):
            start = nav_points[-1][0]
            end = rec.rebalance_date

            date_range = price_matrix.loc[
                (price_matrix.index >= pd.Timestamp(start)) &
                (price_matrix.index <= pd.Timestamp(end))
            ]
            tickers = [t for t in holdings if t in price_matrix.columns]

            prev_nav = nav_points[-1][1]
            prev_prices: dict[str, float] = {}

            for ts, row in date_range.iterrows():
                day_value = sum(
                    holdings.get(t, 0) * float(row.get(t, prev_prices.get(t, 0)))
                    for t in tickers
                ) + cash

                for t in tickers:
                    v = row.get(t)
                    if v is not None and not pd.isna(v):
                        prev_prices[t] = float(v)

                nav_points.append((ts.date(), max(day_value, 0)))

            # After rebalance, update holdings/cash from record
            # (approximate — detailed tracking happens in walk_forward)

        dates, navs = zip(*nav_points)
        return pd.Series(navs, index=pd.DatetimeIndex(dates), name="nav")

    # ── Performance statistics ────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(nav_series: pd.Series, benchmark: pd.Series | None = None) -> dict:
        """Compute standard portfolio performance metrics from NAV series."""
        if nav_series.empty or len(nav_series) < 5:
            return {}

        nav = nav_series.dropna()
        daily_returns = nav.pct_change().dropna()

        if len(daily_returns) < 2:
            return {}

        n_years = (nav.index[-1] - nav.index[0]).days / 365.25
        if n_years <= 0:
            return {}

        total_return = nav.iloc[-1] / nav.iloc[0] - 1
        cagr = (1 + total_return) ** (1 / n_years) - 1
        ann_vol = daily_returns.std() * np.sqrt(252)
        sharpe = (cagr - 0.06) / ann_vol if ann_vol > 0 else 0.0  # 6% risk-free proxy

        # Sortino
        downside = daily_returns[daily_returns < 0]
        sortino_denom = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
        sortino = (cagr - 0.06) / sortino_denom

        # Max drawdown
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        max_dd = float(drawdown.min())
        dd_end = drawdown.idxmin()
        dd_peak = nav[:dd_end].idxmax() if not nav[:dd_end].empty else nav.index[0]

        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
        hit_rate = float((daily_returns > 0).mean())

        metrics = {
            "cagr": float(cagr),
            "total_return": float(total_return),
            "ann_volatility": float(ann_vol),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_drawdown": float(max_dd),
            "max_drawdown_start": dd_peak.date() if hasattr(dd_peak, "date") else dd_peak,
            "max_drawdown_end": dd_end.date() if hasattr(dd_end, "date") else dd_end,
            "hit_rate": hit_rate,
            "final_nav": float(nav.iloc[-1]),
            "initial_nav": float(nav.iloc[0]),
            "n_years": float(n_years),
        }

        # Year-wise returns
        year_returns = {}
        for year in daily_returns.index.year.unique():
            yr_ret = daily_returns[daily_returns.index.year == year]
            year_returns[int(year)] = float((1 + yr_ret).prod() - 1)
        metrics["year_returns"] = year_returns

        # Benchmark metrics
        if benchmark is not None and not benchmark.empty:
            bm_common = benchmark.reindex(nav.index).ffill().dropna()
            bm_rets = bm_common.pct_change().dropna()
            common = daily_returns.reindex(bm_rets.index).dropna()
            bm_aligned = bm_rets.reindex(common.index).dropna()

            if len(common) > 10:
                bm_total = bm_common.iloc[-1] / bm_common.iloc[0] - 1
                bm_n_years = (bm_common.index[-1] - bm_common.index[0]).days / 365.25
                bm_cagr = (1 + bm_total) ** (1 / bm_n_years) - 1 if bm_n_years > 0 else 0
                active_rets = common.values - bm_aligned.values
                ir = (np.mean(active_rets) * 252) / (np.std(active_rets) * np.sqrt(252) + 1e-9)
                metrics["benchmark_cagr"] = float(bm_cagr)
                metrics["information_ratio"] = float(ir)

        return metrics
