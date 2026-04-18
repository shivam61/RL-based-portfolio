"""
Performance attribution and diagnostics.

Decomposes portfolio return into:
  - sector allocation effect
  - stock selection effect
  - interaction effect
  - macro regime attribution
  - drawdown episode analysis
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.data.contracts import RebalanceRecord

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    total_return: float
    sector_allocation_effect: dict[str, float]
    stock_selection_effect: dict[str, float]
    interaction_effect: dict[str, float]
    macro_regime_returns: dict[str, float]
    drawdown_episodes: list[dict]
    year_returns: dict[int, float]
    feature_importance: dict[str, float]


class AttributionEngine:
    """Brinson-Hood-Beebower attribution + drawdown episode analysis."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def compute(
        self,
        nav_series: pd.Series,
        rebalance_records: list[RebalanceRecord],
        price_matrix: pd.DataFrame,
        sector_map: dict[str, str],
        benchmark_nav: pd.Series | None = None,
        macro_features: pd.DataFrame | None = None,
        sector_scorer=None,
        stock_ranker=None,
    ) -> AttributionResult:
        """
        Full attribution analysis.
        """
        sector_alloc, stock_sel, interact = self._brinson_attribution(
            rebalance_records, price_matrix, sector_map, benchmark_nav
        )
        regime_returns = self._regime_attribution(nav_series, macro_features)
        dd_episodes = self._drawdown_episodes(nav_series, rebalance_records, macro_features)
        year_rets = self._year_returns(nav_series)
        feat_imp = self._feature_importance(sector_scorer, stock_ranker)

        total = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1) if len(nav_series) > 1 else 0.0

        return AttributionResult(
            total_return=total,
            sector_allocation_effect=sector_alloc,
            stock_selection_effect=stock_sel,
            interaction_effect=interact,
            macro_regime_returns=regime_returns,
            drawdown_episodes=dd_episodes,
            year_returns=year_rets,
            feature_importance=feat_imp,
        )

    # ── Brinson attribution ────────────────────────────────────────────────────

    def _brinson_attribution(
        self,
        records: list[RebalanceRecord],
        price_matrix: pd.DataFrame,
        sector_map: dict[str, str],
        benchmark_nav: pd.Series | None,
    ) -> tuple[dict, dict, dict]:
        sector_alloc: dict[str, float] = {}
        stock_sel: dict[str, float] = {}
        interact: dict[str, float] = {}

        for i, rec in enumerate(records[:-1]):
            nxt_date = records[i + 1].rebalance_date if i + 1 < len(records) else None
            if nxt_date is None:
                continue

            # Realized return per ticker for this period
            tickers = [t for t in rec.target_weights if t != "CASH"]
            period = price_matrix.loc[
                pd.Timestamp(rec.rebalance_date):pd.Timestamp(nxt_date)
            ]
            if period.empty or len(period) < 2:
                continue

            for t in tickers:
                if t not in period.columns:
                    continue
                p_series = period[t].dropna()
                if len(p_series) < 2:
                    continue
                t_ret = float(p_series.iloc[-1] / p_series.iloc[0] - 1)
                sector = sector_map.get(t, "Unknown")
                w = rec.target_weights.get(t, 0.0)

                # Approximate BM sector weight = equal across all sectors
                bm_sector_w = 1.0 / max(len(set(sector_map.values())), 1)
                bm_ret = 0.0  # simplified; use benchmark ticker return if available

                # Allocation effect: (wp - wb) * rb
                alloc = (w - bm_sector_w) * bm_ret
                # Selection effect: wb * (rp - rb)
                sel = bm_sector_w * (t_ret - bm_ret)
                # Interaction: (wp - wb) * (rp - rb)
                inter = (w - bm_sector_w) * (t_ret - bm_ret)

                sector_alloc[sector] = sector_alloc.get(sector, 0) + alloc
                stock_sel[sector] = stock_sel.get(sector, 0) + sel
                interact[sector] = interact.get(sector, 0) + inter

        return sector_alloc, stock_sel, interact

    # ── Regime attribution ────────────────────────────────────────────────────

    def _regime_attribution(
        self,
        nav: pd.Series,
        macro: pd.DataFrame | None,
    ) -> dict[str, float]:
        if macro is None or nav.empty:
            return {}

        returns = nav.pct_change().dropna()
        results: dict[str, float] = {}

        # VIX regime
        if "vix_level" in macro.columns:
            vix = macro["vix_level"].reindex(returns.index).ffill()
            high_vix_rets = returns[vix > 25]
            low_vix_rets = returns[vix <= 25]
            results["high_vix_return"] = float((1 + high_vix_rets).prod() - 1)
            results["low_vix_return"] = float((1 + low_vix_rets).prod() - 1)

        # Rate regime
        if "rate_cutting_cycle" in macro.columns:
            cutting = macro["rate_cutting_cycle"].reindex(returns.index).ffill()
            results["rate_cutting_return"] = float((1 + returns[cutting == 1]).prod() - 1)
            results["rate_hiking_return"] = float((1 + returns[cutting == 0]).prod() - 1)

        # Election
        if "election_window" in macro.columns:
            elec = macro["election_window"].reindex(returns.index).ffill()
            results["election_window_return"] = float((1 + returns[elec == 1]).prod() - 1)

        return results

    # ── Drawdown episodes ─────────────────────────────────────────────────────

    def _drawdown_episodes(
        self,
        nav: pd.Series,
        records: list[RebalanceRecord],
        macro: pd.DataFrame | None,
    ) -> list[dict]:
        if nav.empty:
            return []

        cummax = nav.cummax()
        dd = (nav - cummax) / cummax
        threshold = -0.05

        episodes = []
        in_dd = False
        start_date = None
        peak_nav = None

        for ts, dd_val in dd.items():
            if dd_val <= threshold and not in_dd:
                in_dd = True
                start_date = ts
                peak_nav = float(cummax.loc[ts])
            elif in_dd and dd_val >= -0.01:
                end_date = ts
                trough_val = float(nav.loc[start_date:end_date].min())
                max_dd = float((trough_val - peak_nav) / peak_nav)
                duration = (end_date - start_date).days

                # Attribution: which sectors were overweight during episode?
                ep_records = [
                    r for r in records
                    if start_date.date() <= r.rebalance_date <= end_date.date()
                ]
                top_sectors = {}
                for r in ep_records:
                    for sec, tilt in r.sector_tilts.items():
                        top_sectors[sec] = top_sectors.get(sec, 0) + tilt

                cause = "unknown"
                if macro is not None and "election_window" in macro.columns:
                    ep_macro = macro.loc[start_date:end_date]
                    if ep_macro["election_window"].mean() > 0.3:
                        cause = "election_uncertainty"
                if macro is not None and "macro_stress_score" in macro.columns:
                    ep_macro = macro.loc[start_date:end_date]
                    if ep_macro.get("macro_stress_score", pd.Series()).mean() > 0.5:
                        cause = "macro_shock"

                episodes.append({
                    "start": str(start_date.date()),
                    "end": str(end_date.date()),
                    "max_drawdown": max_dd,
                    "duration_days": duration,
                    "top_overweight_sectors": sorted(
                        top_sectors.items(), key=lambda x: x[1], reverse=True
                    )[:3],
                    "probable_cause": cause,
                })
                in_dd = False

        return episodes

    # ── Year returns ──────────────────────────────────────────────────────────

    def _year_returns(self, nav: pd.Series) -> dict[int, float]:
        if nav.empty:
            return {}
        rets = nav.pct_change().dropna()
        year_rets = {}
        for yr in rets.index.year.unique():
            yr_rets = rets[rets.index.year == yr]
            year_rets[int(yr)] = float((1 + yr_rets).prod() - 1)
        return year_rets

    # ── Feature importance ────────────────────────────────────────────────────

    def _feature_importance(
        self, sector_scorer=None, stock_ranker=None
    ) -> dict[str, float]:
        importance: dict[str, float] = {}

        if sector_scorer is not None and sector_scorer.is_fitted:
            imp = sector_scorer.feature_importance()
            if not imp.empty:
                for feat, val in imp.head(10).items():
                    importance[f"sector__{feat}"] = float(val)

        if stock_ranker is not None and stock_ranker.is_fitted:
            for sector in list(stock_ranker.models.keys())[:3]:
                imp = stock_ranker.feature_importance(sector)
                if not imp.empty:
                    for feat, val in imp.head(5).items():
                        importance[f"stock_{sector}__{feat}"] = float(val)

        return importance

    # ── SHAP analysis ─────────────────────────────────────────────────────────

    @staticmethod
    def run_shap(model, X: pd.DataFrame, n_samples: int = 200) -> pd.Series:
        """Compute SHAP feature importance using shap library."""
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            sample = X.sample(min(n_samples, len(X)))
            shap_values = explainer.shap_values(sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            return pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=X.columns,
            ).sort_values(ascending=False)
        except Exception as e:
            logger.warning("SHAP failed: %s", e)
            return pd.Series(dtype=float)
