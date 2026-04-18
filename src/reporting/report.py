"""
Report generation: text summary, CSV outputs, and matplotlib charts.
All outputs are dashboard-ready (parquet + PNG).
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.attribution.attribution import AttributionResult
from src.data.contracts import RebalanceRecord

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available; charts will be skipped")


class ReportGenerator:
    """Generate full backtest reports: text, CSV, charts, JSON."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.report_dir = Path(cfg["paths"]["report_dir"])
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        metrics: dict,
        nav_series: pd.Series,
        rebalance_records: list[RebalanceRecord],
        attribution: Optional[AttributionResult] = None,
        strategy_navs: dict[str, pd.Series] | None = None,
        current_portfolio: dict | None = None,
        benchmark_nav: pd.Series | None = None,
    ) -> Path:
        """Generate all report artifacts and return report directory."""
        logger.info("Generating report → %s", self.report_dir)

        self._save_metrics_json(metrics)
        self._save_nav_parquet(nav_series, strategy_navs)
        self._save_rebalance_log(rebalance_records)
        if current_portfolio:
            self._save_current_portfolio(current_portfolio)
        if attribution:
            self._save_attribution(attribution)

        if HAS_MPL:
            self._plot_nav(nav_series, benchmark_nav, strategy_navs)
            self._plot_drawdown(nav_series)
            self._plot_year_returns(metrics.get("year_returns", {}))
            self._plot_sector_contributions(
                metrics.get("sector_contributions", {}),
                attribution,
            )
            self._plot_rolling_returns(nav_series, benchmark_nav)

        self._print_summary(metrics, attribution, current_portfolio)
        return self.report_dir

    # ── Text / JSON ───────────────────────────────────────────────────────────

    def _save_metrics_json(self, metrics: dict) -> None:
        cleaned = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                cleaned[k] = float(v) if isinstance(v, float) else v
            elif isinstance(v, dict):
                cleaned[k] = {str(kk): float(vv) if isinstance(vv, float) else vv
                              for kk, vv in v.items()}
            elif hasattr(v, "isoformat"):
                cleaned[k] = v.isoformat()
            else:
                cleaned[k] = str(v)

        path = self.report_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(cleaned, f, indent=2)
        logger.info("Metrics saved → %s", path)

    def _save_nav_parquet(
        self,
        nav: pd.Series,
        strategy_navs: dict[str, pd.Series] | None,
    ) -> None:
        frames = {"portfolio": nav}
        if strategy_navs:
            frames.update(strategy_navs)
        df = pd.DataFrame(frames)
        df.to_parquet(self.report_dir / "nav_series.parquet", engine="pyarrow")

    def _save_rebalance_log(self, records: list[RebalanceRecord]) -> None:
        rows = []
        for r in records:
            rows.append({
                "date": str(r.rebalance_date),
                "pre_nav": r.pre_nav,
                "post_nav": r.post_nav,
                "turnover": r.total_turnover,
                "cost": r.total_cost,
                "cash_target": r.cash_target,
                "aggressiveness": r.aggressiveness,
                "n_trades": len(r.trades),
                "emergency": r.emergency,
                **{f"tilt_{k}": v for k, v in r.sector_tilts.items()},
            })
        df = pd.DataFrame(rows)
        df.to_parquet(self.report_dir / "rebalance_log.parquet", engine="pyarrow")
        df.to_csv(self.report_dir / "rebalance_log.csv", index=False)
        logger.info("Rebalance log saved: %d records", len(records))

    def _save_current_portfolio(self, portfolio: dict) -> None:
        path = self.report_dir / "current_portfolio.json"
        with open(path, "w") as f:
            json.dump(portfolio, f, indent=2)
        logger.info("Current portfolio saved → %s", path)

    def _save_attribution(self, attr: AttributionResult) -> None:
        data = {
            "total_return": attr.total_return,
            "sector_allocation_effect": attr.sector_allocation_effect,
            "stock_selection_effect": attr.stock_selection_effect,
            "interaction_effect": attr.interaction_effect,
            "macro_regime_returns": attr.macro_regime_returns,
            "year_returns": {str(k): v for k, v in attr.year_returns.items()},
            "drawdown_episodes": attr.drawdown_episodes,
            "feature_importance": attr.feature_importance,
        }
        path = self.report_dir / "attribution.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Charts ────────────────────────────────────────────────────────────────

    def _plot_nav(
        self,
        portfolio_nav: pd.Series,
        benchmark_nav: pd.Series | None,
        strategy_navs: dict[str, pd.Series] | None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(14, 6))
        initial = float(portfolio_nav.iloc[0])

        ax.plot(portfolio_nav.index, portfolio_nav / initial, label="RL Portfolio",
                linewidth=2, color="#1f77b4")

        if benchmark_nav is not None and not benchmark_nav.empty:
            bm_aligned = benchmark_nav.reindex(portfolio_nav.index).ffill()
            bm_norm = bm_aligned / float(bm_aligned.iloc[0])
            ax.plot(bm_norm.index, bm_norm, label="Nifty 50", linestyle="--",
                    linewidth=1.5, color="#ff7f0e")

        if strategy_navs:
            colors = ["#2ca02c", "#d62728", "#9467bd"]
            for (name, nav), color in zip(strategy_navs.items(), colors):
                if nav.empty:
                    continue
                norm = nav / float(nav.iloc[0])
                ax.plot(norm.index, norm, label=name, linestyle=":",
                        linewidth=1.2, color=color)

        ax.set_title("Portfolio NAV (normalized to 1.0)", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Value")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        fig.tight_layout()
        fig.savefig(self.report_dir / "nav_chart.png", dpi=150)
        plt.close(fig)
        logger.info("NAV chart saved")

    def _plot_drawdown(self, nav: pd.Series) -> None:
        cummax = nav.cummax()
        dd = (nav - cummax) / cummax

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
        ax.plot(dd.index, dd.values, color="red", linewidth=0.8)
        ax.set_title("Portfolio Drawdown", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.report_dir / "drawdown_chart.png", dpi=150)
        plt.close(fig)

    def _plot_year_returns(self, year_returns: dict) -> None:
        if not year_returns:
            return
        years = sorted(year_returns.keys())
        rets = [year_returns[y] * 100 for y in years]
        colors = ["#2ca02c" if r >= 0 else "#d62728" for r in rets]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar([str(y) for y in years], rets, color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Annual Returns (%)", fontsize=14)
        ax.set_ylabel("Return (%)")
        ax.grid(True, alpha=0.3, axis="y")
        for i, (y, r) in enumerate(zip(years, rets)):
            ax.text(i, r + (1 if r >= 0 else -2), f"{r:.1f}%",
                    ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(self.report_dir / "year_returns.png", dpi=150)
        plt.close(fig)

    def _plot_sector_contributions(
        self, sector_contributions: dict, attribution: Optional[AttributionResult]
    ) -> None:
        data = sector_contributions or {}
        if attribution and attribution.stock_selection_effect:
            data = attribution.stock_selection_effect or data

        if not data:
            return

        sectors = list(data.keys())
        values = [data[s] * 100 for s in sectors]
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(sectors, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Sector Attribution (Selection Effect, %)", fontsize=14)
        ax.set_xlabel("Contribution (%)")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(self.report_dir / "sector_attribution.png", dpi=150)
        plt.close(fig)

    def _plot_rolling_returns(
        self, nav: pd.Series, benchmark: pd.Series | None
    ) -> None:
        rets = nav.pct_change().dropna()

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Rolling 1Y return
        rolling_1y = (1 + rets).rolling(252).apply(lambda x: x.prod() - 1, raw=True)
        axes[0].plot(rolling_1y.index, rolling_1y * 100, color="#1f77b4", linewidth=1.2)
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].set_title("Rolling 1-Year Return (%)")
        axes[0].grid(True, alpha=0.3)

        # Rolling Sharpe
        rolling_sharpe = (
            rets.rolling(252).mean() * 252
            / (rets.rolling(252).std() * np.sqrt(252)).replace(0, np.nan)
        )
        axes[1].plot(rolling_sharpe.index, rolling_sharpe, color="#2ca02c", linewidth=1.2)
        axes[1].axhline(1.0, color="orange", linewidth=0.8, linestyle="--", label="Sharpe=1")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_title("Rolling 1-Year Sharpe Ratio")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.report_dir / "rolling_returns.png", dpi=150)
        plt.close(fig)

    # ── Console summary ───────────────────────────────────────────────────────

    def _print_summary(
        self, metrics: dict, attribution: Optional[AttributionResult], current_portfolio: dict | None
    ) -> None:
        print("\n" + "=" * 70)
        print("  BACKTEST PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"  Period:           {metrics.get('start_date', '')} → {metrics.get('end_date', '')}")
        print(f"  Initial Capital:  INR {metrics.get('initial_capital', 0):,.0f}")
        print(f"  Final NAV:        INR {metrics.get('final_nav', 0):,.0f}")
        print(f"  CAGR:             {metrics.get('cagr', 0):.2%}")
        print(f"  Total Return:     {metrics.get('total_return', 0):.2%}")
        print(f"  Annualized Vol:   {metrics.get('ann_volatility', 0):.2%}")
        print(f"  Sharpe Ratio:     {metrics.get('sharpe', 0):.2f}")
        print(f"  Sortino Ratio:    {metrics.get('sortino', 0):.2f}")
        print(f"  Calmar Ratio:     {metrics.get('calmar', 0):.2f}")
        print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Hit Rate:         {metrics.get('hit_rate', 0):.2%}")
        print(f"  Avg Turnover:     {metrics.get('avg_turnover', 0):.2%}")
        print(f"  Total Rebalances: {metrics.get('total_rebalances', 0)}")
        if "benchmark_cagr" in metrics:
            print(f"  Benchmark CAGR:   {metrics['benchmark_cagr']:.2%}")
            print(f"  Information Ratio:{metrics.get('information_ratio', 0):.2f}")

        print("\n  YEAR-WISE RETURNS:")
        for yr, ret in sorted(metrics.get("year_returns", {}).items()):
            bar = "█" * int(abs(ret) * 100 / 3)
            sign = "+" if ret >= 0 else "-"
            print(f"  {yr}: {sign}{abs(ret):.1%}  {bar}")

        if current_portfolio:
            print("\n  CURRENT PORTFOLIO RECOMMENDATION:")
            top = sorted(
                [(t, w) for t, w in current_portfolio.items() if t != "CASH"],
                key=lambda x: x[1], reverse=True
            )[:10]
            for t, w in top:
                print(f"    {t:20s}  {w:.1%}")
            print(f"    {'CASH':20s}  {current_portfolio.get('CASH', 0):.1%}")

        print("=" * 70)
        print(f"  Full report saved → {self.report_dir}")
        print("=" * 70 + "\n")
