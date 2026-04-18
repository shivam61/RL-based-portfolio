"""
FII/DII flow proxy builder.

Real NSE FII/DII data (daily net buy/sell in ₹ crore) is only available via
NSE's website for recent periods. For a historical backtest (2013–present),
we build a synthetic proxy that captures the same regime signal using the
market footprint FII activity leaves behind:

  FII buying  → INR appreciates (USDINR ↓) + Nifty outperforms SP500
  FII selling → INR depreciates (USDINR ↑) + Nifty underperforms SP500

Additionally, the India VIX premium over US VIX signals domestic-specific
fear (typically elevated during FII-driven sell-offs).

If real FII data becomes available (e.g. downloaded from NSE monthly reports
as CSVs and placed at data/raw/fii_dii_historical.csv), the builder will
use the real data and fall back to proxy only for dates not covered.

Real CSV format expected (NSE monthly report columns):
  Date, Category, Buy Value (Cr), Sell Value (Cr), Net Value (Cr)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REAL_DATA_PATH = Path("data/raw/fii_dii_historical.csv")


def build_fii_features(macro_df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Build FII/DII flow features from macro_df (output of MacroDataManager.build()).

    Returns a DataFrame with DatetimeIndex aligned to macro_df, with columns:
      fii_proxy_flow       — daily composite FII flow signal (z-scored)
      fii_flow_5d          — 5-day rolling sum of proxy
      fii_flow_20d         — 20-day rolling sum of proxy
      fii_flow_zscore      — (fii_flow_20d - mean_1y) / std_1y  (regime signal)
      fii_sell_regime      — 1 if z-score < -1.5 (FII heavy selling), else 0
      fii_buy_regime       — 1 if z-score > 1.5  (FII heavy buying),  else 0
      india_vs_global      — Nifty 5d return minus SP500 5d return (FII preference)
      inr_strength_5d      — inverted USDINR 5d return (INR ↑ = FII inflow)
      vix_premium          — India VIX minus US VIX (domestic fear above global)
      fii_net_real         — real FII net buy in ₹ crore (if CSV available, else NaN)
      fii_net_5d_real      — 5d rolling sum of real FII net (if available)
    """
    feats: dict[str, pd.Series] = {}
    idx = macro_df.index

    # ── Synthetic FII flow proxy ──────────────────────────────────────────────
    # Component 1: INR strength (inverted USDINR return)
    if "usdinr" in macro_df.columns:
        usdinr_ret_5d = macro_df["usdinr"].pct_change(5)
        inr_strength = -usdinr_ret_5d  # negative return = INR appreciated = FII inflow
        feats["inr_strength_5d"] = inr_strength
    else:
        inr_strength = pd.Series(0.0, index=idx)

    # Component 2: Nifty outperformance vs SP500 (FII preference for India)
    if "nifty50" in macro_df.columns and "sp500" in macro_df.columns:
        nifty_5d = macro_df["nifty50"].pct_change(5)
        sp500_5d = macro_df["sp500"].pct_change(5)
        india_vs_global = nifty_5d - sp500_5d
        feats["india_vs_global"] = india_vs_global
    else:
        india_vs_global = pd.Series(0.0, index=idx)

    # Component 3: India VIX premium over US VIX
    if "india_vix" in macro_df.columns and "vix" in macro_df.columns:
        vix_premium = macro_df["india_vix"] - macro_df["vix"]
        feats["vix_premium"] = vix_premium
    elif "india_vix" in macro_df.columns:
        vix_premium = macro_df["india_vix"] - 18.0  # long-run US VIX avg
        feats["vix_premium"] = vix_premium
    else:
        vix_premium = pd.Series(0.0, index=idx)

    # Composite daily proxy: weight INR strength more (most direct FII signal)
    daily_proxy = (
        0.50 * _zscore(inr_strength, window=252)
        + 0.35 * _zscore(india_vs_global, window=252)
        - 0.15 * _zscore(vix_premium, window=252)   # high VIX premium → FII selling
    ).fillna(0.0)

    feats["fii_proxy_flow"] = daily_proxy
    feats["fii_flow_5d"] = daily_proxy.rolling(5).sum()
    feats["fii_flow_20d"] = daily_proxy.rolling(20).sum()

    flow_20d = feats["fii_flow_20d"]
    feats["fii_flow_zscore"] = _zscore(flow_20d, window=252)
    feats["fii_sell_regime"] = (feats["fii_flow_zscore"] < -1.5).astype(float)
    feats["fii_buy_regime"]  = (feats["fii_flow_zscore"] >  1.5).astype(float)

    # ── Real FII data (if available) ─────────────────────────────────────────
    feats["fii_net_real"]    = pd.Series(np.nan, index=idx)
    feats["fii_net_5d_real"] = pd.Series(np.nan, index=idx)

    if _REAL_DATA_PATH.exists():
        try:
            real = _load_real_fii(_REAL_DATA_PATH, idx)
            feats["fii_net_real"]    = real["fii_net"].reindex(idx)
            feats["fii_net_5d_real"] = feats["fii_net_real"].rolling(5).sum()
            logger.info("Real FII data loaded: %d rows from %s", real["fii_net"].count(), _REAL_DATA_PATH)
        except Exception as e:
            logger.warning("Could not load real FII data: %s", e)

    result = pd.DataFrame(feats, index=idx)
    result.index.name = "date"
    result = result.shift(lag)  # enforce lag to prevent lookahead
    return result


def _zscore(s: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score with minimum 63 periods."""
    mean = s.rolling(window, min_periods=63).mean()
    std  = s.rolling(window, min_periods=63).std()
    return (s - mean) / (std + 1e-8)


def _load_real_fii(path: Path, target_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load real FII/DII CSV. Expected columns after parsing:
      Date (dd-MMM-yyyy or dd-MM-yyyy), Category (FII/DII), Net Value (Cr)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # normalise date column
    date_col = [c for c in df.columns if "date" in c][0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])

    # net value column
    net_col = [c for c in df.columns if "net" in c][0]
    df[net_col] = pd.to_numeric(df[net_col].astype(str).str.replace(",", ""), errors="coerce")

    # filter FII rows only; aggregate if both FII and DII present
    cat_col = [c for c in df.columns if "cat" in c]
    if cat_col:
        fii = df[df[cat_col[0]].str.upper().str.contains("FII", na=False)]
    else:
        fii = df

    fii = fii.set_index(date_col)[net_col].rename("fii_net")
    fii = fii.reindex(target_idx).ffill(limit=3)
    return fii.to_frame()
