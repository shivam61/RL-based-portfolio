"""
Macro and global signal features.

All features are lagged by at least 1 day before being used in
any model to enforce point-in-time correctness.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.fii_proxy import build_fii_features
from src.features.base import ewma, rolling_vol

logger = logging.getLogger(__name__)


class MacroFeatureBuilder:
    """Builds macro feature DataFrame from raw macro data."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self._lag = self.cfg["features"]["macro_lag"]

    def build(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Input : macro_df from MacroDataManager (date × raw signals)
        Output: feature DataFrame (date × macro_features), lagged
        """
        feats: dict[str, pd.Series] = {}

        # ── Volatility / risk regime ──────────────────────────────────────────
        if "vix" in macro_df.columns:
            vix = macro_df["vix"]
            feats["vix_level"] = vix
            feats["vix_chg_1m"] = vix.pct_change(21)
            feats["vix_pctile_1y"] = vix.rolling(252).rank(pct=True)
            feats["high_vix_regime"] = (vix > 25).astype(float)

        # ── USD / FX ──────────────────────────────────────────────────────────
        if "usdinr" in macro_df.columns:
            fx = macro_df["usdinr"]
            feats["usdinr_level"] = fx
            feats["usdinr_ret_1m"] = fx.pct_change(21)
            feats["usdinr_ret_3m"] = fx.pct_change(63)
            feats["usdinr_above_200ma"] = (fx > fx.rolling(200).mean()).astype(float)
            feats["fx_stress"] = (
                fx.pct_change(21) > fx.pct_change(21).rolling(252).quantile(0.80)
            ).astype(float)

        # ── Crude oil ─────────────────────────────────────────────────────────
        if "crude_oil" in macro_df.columns:
            oil = macro_df["crude_oil"]
            feats["crude_ret_1m"] = oil.pct_change(21)
            feats["crude_ret_3m"] = oil.pct_change(63)
            feats["crude_vol_1m"] = rolling_vol(oil.pct_change(), 21, 252)
            feats["oil_shock_up"] = (
                oil.pct_change(21) > oil.pct_change(21).rolling(252).quantile(0.80)
            ).astype(float)
            feats["oil_shock_dn"] = (
                oil.pct_change(21) < oil.pct_change(21).rolling(252).quantile(0.20)
            ).astype(float)

        # ── Gold ──────────────────────────────────────────────────────────────
        if "gold" in macro_df.columns:
            gold = macro_df["gold"]
            feats["gold_ret_1m"] = gold.pct_change(21)
            feats["gold_ret_3m"] = gold.pct_change(63)
            feats["gold_above_200ma"] = (gold > gold.rolling(200).mean()).astype(float)

        # ── DXY ───────────────────────────────────────────────────────────────
        if "dxy" in macro_df.columns:
            dxy = macro_df["dxy"]
            feats["dxy_ret_1m"] = dxy.pct_change(21)
            feats["strong_dollar"] = (dxy > dxy.rolling(252).quantile(0.75)).astype(float)

        # ── S&P 500 ───────────────────────────────────────────────────────────
        if "sp500" in macro_df.columns:
            sp = macro_df["sp500"]
            feats["sp500_ret_1m"] = sp.pct_change(21)
            feats["sp500_ret_3m"] = sp.pct_change(63)
            feats["sp500_above_200ma"] = (sp > sp.rolling(200).mean()).astype(float)
            feats["global_risk_on"] = (
                (sp > sp.rolling(50).mean()) & (sp.pct_change(21) > 0)
            ).astype(float)

        # ── US yield curve ────────────────────────────────────────────────────
        if "us_10y" in macro_df.columns:
            y10 = macro_df["us_10y"]
            feats["us_10y_level"] = y10
            feats["us_10y_chg_1m"] = y10.diff(21)
            feats["rising_rates"] = (y10.diff(21) > 0).astype(float)

        if "us_2y" in macro_df.columns and "us_10y" in macro_df.columns:
            feats["yield_curve"] = macro_df["us_10y"] - macro_df["us_2y"]
            feats["inverted_yield"] = (feats["yield_curve"] < 0).astype(float)

        # ── RBI / India macro ─────────────────────────────────────────────────
        if "rbi_repo_rate" in macro_df.columns:
            rbi = macro_df["rbi_repo_rate"]
            feats["rbi_rate"] = rbi
            feats["rbi_rate_chg_6m"] = rbi.diff(126)
            feats["rate_cutting_cycle"] = (rbi.diff(126) < 0).astype(float)

        if "rbi_meeting" in macro_df.columns:
            feats["rbi_meeting"] = macro_df["rbi_meeting"]
        if "budget_day" in macro_df.columns:
            feats["budget_day"] = macro_df["budget_day"]
        if "election_window" in macro_df.columns:
            feats["election_window"] = macro_df["election_window"]

        # ── Nifty 50 breadth ──────────────────────────────────────────────────
        if "nifty50" in macro_df.columns:
            nifty = macro_df["nifty50"]
            feats["nifty_ret_1m"] = nifty.pct_change(21)
            feats["nifty_ret_3m"] = nifty.pct_change(63)
            feats["nifty_above_200ma"] = (nifty > nifty.rolling(200).mean()).astype(float)
            feats["nifty_vol_1m"] = rolling_vol(nifty.pct_change(), 21, 252)

        # ── NSE sector indices ────────────────────────────────────────────────
        if "nifty_bank" in macro_df.columns:
            nb = macro_df["nifty_bank"]
            feats["niftybank_ret_1m"] = nb.pct_change(21)
            feats["niftybank_ret_3m"] = nb.pct_change(63)
            if "nifty50" in macro_df.columns:
                feats["bank_vs_nifty_rel_1m"] = (
                    nb.pct_change(21) - macro_df["nifty50"].pct_change(21)
                )

        if "nifty_it" in macro_df.columns:
            ni = macro_df["nifty_it"]
            feats["niftyit_ret_1m"] = ni.pct_change(21)
            feats["niftyit_ret_3m"] = ni.pct_change(63)
            if "nifty50" in macro_df.columns:
                feats["it_vs_nifty_rel_1m"] = (
                    ni.pct_change(21) - macro_df["nifty50"].pct_change(21)
                )

        # ── India VIX ─────────────────────────────────────────────────────────
        if "india_vix" in macro_df.columns:
            feats["india_vix_ret_1m"] = macro_df["india_vix"].pct_change(21)
            feats["india_vix_pctile_1y"] = (
                macro_df["india_vix"]
                .rolling(252)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            )

        # ── Composite signals ─────────────────────────────────────────────────
        # Risk-on / risk-off score (0 = full risk-off, 1 = full risk-on)
        risk_on_components = []
        if "global_risk_on" in feats:
            risk_on_components.append(feats["global_risk_on"])
        if "gold_above_200ma" in feats:
            risk_on_components.append(1 - feats["gold_above_200ma"])  # gold up = risk-off
        if "high_vix_regime" in feats:
            risk_on_components.append(1 - feats["high_vix_regime"])
        if "strong_dollar" in feats:
            risk_on_components.append(1 - feats["strong_dollar"])

        if risk_on_components:
            feats["risk_on_score"] = pd.concat(risk_on_components, axis=1).mean(axis=1)

        # Macro stress composite
        stress_components = []
        if "oil_shock_up" in feats:
            stress_components.append(feats["oil_shock_up"])
        if "fx_stress" in feats:
            stress_components.append(feats["fx_stress"])
        if "high_vix_regime" in feats:
            stress_components.append(feats["high_vix_regime"])
        if "election_window" in feats:
            stress_components.append(feats["election_window"])

        if stress_components:
            feats["macro_stress_score"] = pd.concat(stress_components, axis=1).mean(axis=1)

        result = pd.DataFrame(feats)
        result.index.name = "date"

        # ── FII/DII flow proxy features ───────────────────────────────────────
        fii_feats = build_fii_features(macro_df, lag=self._lag)
        fii_feats = fii_feats.reindex(result.index)
        result = pd.concat([result, fii_feats], axis=1)

        # incorporate FII sell regime into macro stress score
        if "fii_sell_regime" in result.columns and "macro_stress_score" in result.columns:
            result["macro_stress_score"] = (
                result["macro_stress_score"] * 0.7 + result["fii_sell_regime"] * 0.3
            )

        # ── Enforce lag (no lookahead) ────────────────────────────────────────
        # Note: lag already applied inside build_fii_features; only shift non-FII cols
        non_fii = [c for c in result.columns if c not in fii_feats.columns]
        result[non_fii] = result[non_fii].shift(self._lag)

        out_path = Path(self.cfg["paths"]["feature_data"]) / "macro_features.parquet"
        result.to_parquet(out_path, engine="pyarrow")
        logger.info("Macro features: %s → %s", result.shape, out_path)
        return result

    def get_feature_names(self) -> list[str]:
        dummy_idx = pd.date_range("2015-01-01", periods=300)
        dummy_macro = pd.DataFrame(
            np.random.randn(300, 10),
            index=dummy_idx,
            columns=["vix", "usdinr", "crude_oil", "gold", "dxy",
                     "sp500", "us_10y", "us_2y", "rbi_repo_rate", "nifty50"],
        )
        dummy_macro["rbi_meeting"] = 0.0
        dummy_macro["budget_day"] = 0.0
        dummy_macro["election_window"] = 0.0
        result = self.build(dummy_macro)
        return list(result.columns)
