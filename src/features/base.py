"""Shared feature-engineering utilities (returns, rolling stats, etc.)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def safe_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods).replace([np.inf, -np.inf], np.nan)


def rolling_return(prices: pd.Series, window: int) -> pd.Series:
    return prices / prices.shift(window) - 1


def rolling_vol(returns: pd.Series, window: int, ann: int = 252) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(ann)


def rolling_downside_vol(
    returns: pd.Series, window: int, threshold: float = 0.0, ann: int = 252
) -> pd.Series:
    downside = returns.clip(upper=threshold) - threshold
    return downside.rolling(window).std() * np.sqrt(ann)


def rolling_sharpe(returns: pd.Series, window: int, rf: float = 0.0) -> pd.Series:
    mu = returns.rolling(window).mean() * 252
    sig = returns.rolling(window).std() * np.sqrt(252)
    return (mu - rf) / sig.replace(0, np.nan)


def rolling_max_drawdown(prices: pd.Series, window: int) -> pd.Series:
    def _mdd(x: np.ndarray) -> float:
        peak = np.maximum.accumulate(x)
        dd = (x - peak) / np.where(peak == 0, 1, peak)
        return float(dd.min())

    return prices.rolling(window).apply(_mdd, raw=True)


def rolling_skew(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).skew()


def rolling_kurt(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).kurt()


def rank_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Rank columns cross-sectionally at each row. Output in [0, 1]."""
    return df.rank(axis=1, pct=True)


def zscore_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score columns cross-sectionally at each row."""
    mu = df.mean(axis=1)
    sig = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sig, axis=0)


def lag_series(df: pd.DataFrame | pd.Series, lag: int = 1) -> pd.DataFrame | pd.Series:
    return df.shift(lag)


def ewma(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def momentum_stability(returns: pd.Series, window: int = 63) -> pd.Series:
    """Fraction of positive days in rolling window (consistency measure)."""
    return (returns > 0).rolling(window).mean()


def compute_beta(
    stock_returns: pd.Series, market_returns: pd.Series, window: int = 252
) -> pd.Series:
    cov = stock_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var.replace(0, np.nan)
