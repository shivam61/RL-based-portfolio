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
from src.reporting.selection_diagnostics import prepare_selection_diagnostics

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
        selection_diagnostics: object | None = None,
        stock_ranker: object | None = None,
    ) -> Path:
        """Generate all report artifacts and return report directory."""
        logger.info("Generating report → %s", self.report_dir)
        prepared_selection = prepare_selection_diagnostics(selection_diagnostics)

        self._save_metrics_json(metrics)
        self._save_nav_parquet(nav_series, strategy_navs)
        self._save_rebalance_log(rebalance_records)
        self._save_selection_diagnostics(prepared_selection)
        self._save_stock_ranker_importance(stock_ranker)
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

        self._print_summary(metrics, attribution, current_portfolio, prepared_selection)
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
        prev_weights: dict[str, float] = {}
        cum_nav = records[0].pre_nav if records else 1.0

        for i, r in enumerate(records):
            cur_weights = {t: w for t, w in r.target_weights.items() if t != "CASH"}

            stocks_added   = sorted(set(cur_weights) - set(prev_weights))
            stocks_removed = sorted(set(prev_weights) - set(cur_weights))

            top5 = sorted(cur_weights.items(), key=lambda x: -x[1])[:5]
            top5_str = "|".join(f"{t}:{w:.1%}" for t, w in top5)

            period_ret = (r.post_nav - r.pre_nav) / r.pre_nav if r.pre_nav > 0 else 0.0
            cum_ret    = (r.post_nav - cum_nav) / cum_nav if cum_nav > 0 else 0.0

            # RL tilt summary
            over  = sorted([(s, t) for s, t in r.sector_tilts.items() if t > 1.1], key=lambda x: -x[1])[:3]
            under = sorted([(s, t) for s, t in r.sector_tilts.items() if t < 0.9], key=lambda x: x[1])[:3]

            buys  = [t for t in r.trades if t.direction == "buy"]
            sells = [t for t in r.trades if t.direction == "sell"]

            rows.append({
                "date":              str(r.rebalance_date),
                "mode":              "RL" if r.rl_action.get("sector_tilts") else "Rule",
                "pre_nav":           round(r.pre_nav, 2),
                "post_nav":          round(r.post_nav, 2),
                "period_return_pct": round(period_ret * 100, 3),
                "cum_return_pct":    round(cum_ret * 100, 3),
                "n_stocks":          len(cur_weights),
                "cash_pct":          round(r.target_weights.get("CASH", 0) * 100, 2),
                "turnover_pct":      round(r.total_turnover * 100, 2),
                "cost_inr":          round(r.total_cost, 2),
                "n_buys":            len(buys),
                "n_sells":           len(sells),
                "stocks_added":      ",".join(stocks_added),
                "stocks_removed":    ",".join(stocks_removed),
                "top5_holdings":     top5_str,
                "rl_overweight":     "|".join(f"{s}×{t:.1f}" for s, t in over),
                "rl_underweight":    "|".join(f"{s}×{t:.1f}" for s, t in under),
                "aggressiveness":    round(r.aggressiveness, 2),
                "emergency":         r.emergency,
                **{f"tilt_{k}": round(v, 3) for k, v in r.sector_tilts.items()},
            })
            prev_weights = cur_weights

        df = pd.DataFrame(rows)
        df.to_parquet(self.report_dir / "rebalance_log.parquet", engine="pyarrow")
        df.to_csv(self.report_dir / "rebalance_log.csv", index=False)
        logger.info("Rebalance log saved: %d records", len(records))
        self._print_decision_log(df)

    def _print_decision_log(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 90)
        print("  DETAILED REBALANCE DECISION LOG")
        print("=" * 90)
        print(f"  {'Date':<12} {'Mode':<5} {'NAV (₹)':<12} {'Period%':>8} {'Stocks':>7} "
              f"{'Cash%':>6} {'TO%':>5} {'Cost₹':>7}  Top Holdings / Changes")
        print("-" * 90)
        for _, row in df.iterrows():
            nav_str  = f"{row['post_nav']:>11,.0f}"
            top5     = row['top5_holdings'] or "ALL CASH"
            added    = row['stocks_added']
            removed  = row['stocks_removed']
            changes  = []
            if added:   changes.append(f"+[{added}]")
            if removed: changes.append(f"-[{removed}]")
            change_str = "  ".join(changes) if changes else ""
            rl_info  = ""
            if row['rl_overweight']:  rl_info += f"  ↑{row['rl_overweight']}"
            if row['rl_underweight']: rl_info += f"  ↓{row['rl_underweight']}"
            emerg = " ⚠ EMERGENCY" if row['emergency'] else ""
            print(f"  {row['date']:<12} {row['mode']:<5} {nav_str} "
                  f"{row['period_return_pct']:>+8.2f}% "
                  f"{int(row['n_stocks']):>6}  "
                  f"{row['cash_pct']:>5.1f}% "
                  f"{row['turnover_pct']:>5.1f}% "
                  f"{row['cost_inr']:>7,.0f}  {top5}")
            if change_str: print(f"    {'':12}  Changes: {change_str}")
            if rl_info:    print(f"    {'':12}  RL:      {rl_info.strip()}{emerg}")
        print("=" * 90 + "\n")

    def _save_current_portfolio(self, portfolio: dict) -> None:
        path = self.report_dir / "current_portfolio.json"
        with open(path, "w") as f:
            json.dump(portfolio, f, indent=2)
        logger.info("Current portfolio saved → %s", path)

    def _save_selection_diagnostics(self, prepared: dict | None) -> None:
        if not prepared:
            return

        summary = prepared.get("summary")
        if summary:
            path = self.report_dir / "selection_diagnostics.json"
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Selection diagnostics saved → %s", path)

        per_rebalance = prepared.get("per_rebalance")
        if per_rebalance is not None:
            frame = (
                per_rebalance.copy()
                if isinstance(per_rebalance, pd.DataFrame)
                else pd.DataFrame(per_rebalance)
            )
            if not frame.empty:
                frame.to_parquet(
                    self.report_dir / "selection_rebalance_log.parquet", engine="pyarrow"
                )
                frame.to_csv(self.report_dir / "selection_rebalance_log.csv", index=False)
                logger.info("Selection diagnostics rebalance log saved: %d records", len(frame))

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

    def _save_stock_ranker_importance(self, stock_ranker: object | None) -> None:
        if stock_ranker is None:
            return
        if not getattr(stock_ranker, "is_fitted", False):
            return

        models = getattr(stock_ranker, "models", {})
        if not models:
            return

        rows: list[dict] = []
        for sector in sorted(models.keys()):
            try:
                importance = stock_ranker.feature_importance(sector)
            except Exception as exc:
                logger.warning(
                    "Skipping stock-ranker importance for %s: %s", sector, exc
                )
                continue

            if importance is None or importance.empty:
                continue

            total = float(importance.sum())
            for rank, (feature, value) in enumerate(importance.items(), start=1):
                rows.append({
                    "sector": sector,
                    "feature": feature,
                    "importance": float(value),
                    "importance_share": float(value / total) if total > 0 else 0.0,
                    "rank": rank,
                })

        if not rows:
            return

        df = pd.DataFrame(rows)
        df = df.sort_values(["sector", "rank"], ascending=[True, True], kind="mergesort")
        csv_path = self.report_dir / "stock_ranker_feature_importance.csv"
        json_path = self.report_dir / "stock_ranker_feature_importance.json"
        df.to_csv(csv_path, index=False)

        summary = {
            "n_sectors": int(df["sector"].nunique()),
            "n_features": int(len(df)),
            "sectors": {},
        }
        horizons = list(getattr(stock_ranker, "horizons", []) or [])
        blend_weights = getattr(stock_ranker, "blend_weights", {}) or {}
        if horizons:
            summary["horizons"] = [int(h) for h in horizons]
        if blend_weights:
            summary["blend_weights"] = {
                str(int(h)): float(w) for h, w in blend_weights.items()
            }
        for sector in sorted(df["sector"].unique()):
            sec_df = df.loc[df["sector"] == sector, ["feature", "importance", "importance_share", "rank"]]
            summary["sectors"][sector] = [
                {
                    "feature": str(row["feature"]),
                    "importance": float(row["importance"]),
                    "importance_share": float(row["importance_share"]),
                    "rank": int(row["rank"]),
                }
                for _, row in sec_df.iterrows()
            ]
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Stock-ranker feature importance saved → %s", csv_path)

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
        self,
        metrics: dict,
        attribution: Optional[AttributionResult],
        current_portfolio: dict | None,
        selection_diagnostics: dict | None,
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
            weights = current_portfolio.get("weights", current_portfolio)
            if not isinstance(weights, dict):
                weights = {}
            top = sorted(
                [(t, w) for t, w in weights.items()
                 if t != "CASH" and isinstance(w, (int, float))],
                key=lambda x: x[1], reverse=True
            )[:10]
            for t, w in top:
                print(f"    {t:20s}  {w:.1%}")
            cash = current_portfolio.get("cash", weights.get("CASH", 0))
            print(f"    {'CASH':20s}  {float(cash):.1%}")

        summary = (selection_diagnostics or {}).get("summary", {})
        if summary:
            print("\n  STOCK SELECTION DIAGNOSTICS:")
            labels = [
                ("avg_selected_count", "Avg selected names", "{:.1f}"),
                ("top_k_avg_forward_return", "Top-k avg forward return", "{:.2%}"),
                ("top_k_minus_universe", "Top-k vs universe", "{:+.2%}"),
                ("top_k_minus_sector_median", "Top-k vs sector median", "{:+.2%}"),
                ("precision_at_k", "Precision@k", "{:.2%}"),
                ("rank_ic", "Rank IC", "{:.3f}"),
                ("stability", "Selection stability", "{:.2%}"),
            ]
            for key, label, fmt in labels:
                value = summary.get(key)
                if value is None:
                    continue
                print(f"    {label:24s}  {fmt.format(value)}")

        print("=" * 70)
        print(f"  Full report saved → {self.report_dir}")
        print("=" * 70 + "\n")
