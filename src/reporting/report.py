"""
Report generation: text summary, CSV outputs, and matplotlib charts.
All outputs are dashboard-ready (parquet + PNG).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
        self._run_context: dict[str, Any] = {}
        self._generated_artifacts: list[Path] = []
        self._preexisting_report_files: set[Path] = set()

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
        self._preexisting_report_files = {
            path.resolve()
            for path in self.report_dir.glob("*")
            if path.is_file()
        }
        self._run_context = self._build_run_context(metrics, nav_series)
        self._generated_artifacts = []

        self._generated_artifacts.append(self._save_metrics_json(metrics))
        self._generated_artifacts.extend(self._save_nav_parquet(nav_series, strategy_navs))
        self._generated_artifacts.extend(self._save_rebalance_log(rebalance_records))
        self._generated_artifacts.extend(self._save_selection_diagnostics(prepared_selection))
        self._generated_artifacts.extend(self._save_stock_ranker_importance(stock_ranker))
        if current_portfolio:
            self._generated_artifacts.append(self._save_current_portfolio(current_portfolio))
        if attribution:
            self._generated_artifacts.append(self._save_attribution(attribution))

        if HAS_MPL:
            self._generated_artifacts.append(
                self._plot_nav(nav_series, benchmark_nav, strategy_navs)
            )
            self._generated_artifacts.append(self._plot_drawdown(nav_series))
            year_returns_chart = self._plot_year_returns(metrics.get("year_returns", {}))
            if year_returns_chart is not None:
                self._generated_artifacts.append(year_returns_chart)
            sector_chart = self._plot_sector_contributions(
                metrics.get("sector_contributions", {}),
                attribution,
            )
            if sector_chart is not None:
                self._generated_artifacts.append(sector_chart)
            self._generated_artifacts.append(
                self._plot_rolling_returns(nav_series, benchmark_nav)
            )

        self._write_run_manifest()

        self._print_summary(metrics, attribution, current_portfolio, prepared_selection)
        return self.report_dir

    # ── Text / JSON ───────────────────────────────────────────────────────────

    def _save_metrics_json(self, metrics: dict) -> Path:
        cleaned = {}
        for k, v in metrics.items():
            cleaned[str(k)] = self._jsonify(v)

        cleaned["run_mode"] = self._run_context["run_mode"]
        cleaned["backtest_start_date"] = self._run_context["backtest_start_date"]
        cleaned["backtest_end_date"] = self._run_context["backtest_end_date"]
        cleaned["report_generated_at_utc"] = self._run_context["report_generated_at_utc"]
        cleaned = self._with_artifact_metadata(cleaned, artifact_type="metrics")

        path = self.report_dir / "metrics.json"
        self._write_json(path, cleaned)
        logger.info("Metrics saved → %s", path)
        return path

    def _save_nav_parquet(
        self,
        nav: pd.Series,
        strategy_navs: dict[str, pd.Series] | None,
    ) -> list[Path]:
        frames = {"portfolio": nav}
        if strategy_navs:
            frames.update(strategy_navs)
        df = self._attach_frame_metadata(pd.DataFrame(frames))
        path = self.report_dir / "nav_series.parquet"
        df.to_parquet(path, engine="pyarrow")
        return [path]

    def _save_rebalance_log(self, records: list[RebalanceRecord]) -> list[Path]:
        base_columns = [
            "date",
            "mode",
            "pre_nav",
            "post_nav",
            "period_return_pct",
            "cum_return_pct",
            "n_stocks",
            "selected_sector_count",
            "selected_stock_count",
            "turnover_cap_pct",
            "cash_pct",
            "turnover_pct",
            "cost_inr",
            "n_buys",
            "n_sells",
            "stocks_added",
            "stocks_removed",
            "top5_holdings",
            "rl_overweight",
            "rl_underweight",
            "aggressiveness",
            "emergency",
        ]
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
                "selected_sector_count": int(getattr(r, "selected_sector_count", 0)),
                "selected_stock_count": int(getattr(r, "selected_stock_count", 0)),
                "turnover_cap_pct": (
                    round(float(getattr(r, "turnover_cap")) * 100, 2)
                    if getattr(r, "turnover_cap", None) is not None
                    else np.nan
                ),
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
        if df.empty:
            df = pd.DataFrame(columns=base_columns)
        else:
            for column in base_columns:
                if column not in df.columns:
                    df[column] = np.nan
        df = self._attach_frame_metadata(df)
        parquet_path = self.report_dir / "rebalance_log.parquet"
        csv_path = self.report_dir / "rebalance_log.csv"
        df.to_parquet(parquet_path, engine="pyarrow")
        df.to_csv(csv_path, index=False)
        logger.info("Rebalance log saved: %d records", len(records))
        self._print_decision_log(df)
        return [parquet_path, csv_path]

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

    def _save_current_portfolio(self, portfolio: dict) -> Path:
        path = self.report_dir / "current_portfolio.json"
        payload = self._with_artifact_metadata(
            self._jsonify(portfolio),
            artifact_type="current_portfolio",
        )
        self._write_json(path, payload)
        logger.info("Current portfolio saved → %s", path)
        return path

    def _save_selection_diagnostics(self, prepared: dict | None) -> list[Path]:
        outputs: list[Path] = []
        if not prepared:
            return outputs

        summary = prepared.get("summary")
        if summary:
            path = self.report_dir / "selection_diagnostics.json"
            payload = self._with_artifact_metadata(
                self._jsonify(summary),
                artifact_type="selection_diagnostics",
            )
            self._write_json(path, payload)
            logger.info("Selection diagnostics saved → %s", path)
            outputs.append(path)

        per_rebalance = prepared.get("per_rebalance")
        if per_rebalance is not None:
            frame = (
                per_rebalance.copy()
                if isinstance(per_rebalance, pd.DataFrame)
                else pd.DataFrame(per_rebalance)
            )
            if not frame.empty:
                frame = self._attach_frame_metadata(frame)
                parquet_path = self.report_dir / "selection_rebalance_log.parquet"
                csv_path = self.report_dir / "selection_rebalance_log.csv"
                frame.to_parquet(
                    parquet_path, engine="pyarrow"
                )
                frame.to_csv(csv_path, index=False)
                logger.info("Selection diagnostics rebalance log saved: %d records", len(frame))
                outputs.extend([parquet_path, csv_path])
        return outputs

    def _save_attribution(self, attr: AttributionResult) -> Path:
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
        self._write_json(path, self._with_artifact_metadata(data, artifact_type="attribution"))
        return path

    def _save_stock_ranker_importance(self, stock_ranker: object | None) -> list[Path]:
        outputs: list[Path] = []
        if stock_ranker is None:
            return outputs
        if not getattr(stock_ranker, "is_fitted", False):
            return outputs

        models = getattr(stock_ranker, "models", {})
        if not models:
            return outputs

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
            return outputs

        df = self._attach_frame_metadata(pd.DataFrame(rows))
        df = df.sort_values(["sector", "rank"], ascending=[True, True], kind="mergesort")
        csv_path = self.report_dir / "stock_ranker_feature_importance.csv"
        json_path = self.report_dir / "stock_ranker_feature_importance.json"
        df.to_csv(csv_path, index=False)

        summary = {
            "n_sectors": int(df["sector"].nunique()),
            "n_features": int(len(df)),
            "sectors": {},
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
        self._write_json(
            json_path,
            self._with_artifact_metadata(
                summary,
                artifact_type="stock_ranker_feature_importance",
            ),
        )
        logger.info("Stock-ranker feature importance saved → %s", csv_path)
        outputs.extend([csv_path, json_path])
        return outputs

    # ── Charts ────────────────────────────────────────────────────────────────

    def _plot_nav(
        self,
        portfolio_nav: pd.Series,
        benchmark_nav: pd.Series | None,
        strategy_navs: dict[str, pd.Series] | None,
    ) -> Path:
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
        path = self.report_dir / "nav_chart.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("NAV chart saved")
        return path

    def _plot_drawdown(self, nav: pd.Series) -> Path:
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
        path = self.report_dir / "drawdown_chart.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_year_returns(self, year_returns: dict) -> Path | None:
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
        path = self.report_dir / "year_returns.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_sector_contributions(
        self, sector_contributions: dict, attribution: Optional[AttributionResult]
    ) -> Path | None:
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
        path = self.report_dir / "sector_attribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_rolling_returns(
        self, nav: pd.Series, benchmark: pd.Series | None
    ) -> Path:
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
        path = self.report_dir / "rolling_returns.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _build_run_context(self, metrics: dict, nav_series: pd.Series) -> dict[str, Any]:
        generated_at = self._now_utc()
        start_date = self._resolve_backtest_date(metrics.get("start_date"), nav_series, first=True)
        end_date = self._resolve_backtest_date(metrics.get("end_date"), nav_series, first=False)
        run_mode = self._resolve_run_mode(metrics)
        report_timestamp = self._isoformat_utc(generated_at)
        return {
            "schema_version": 1,
            "run_id": self._build_run_id(run_mode, start_date, end_date, generated_at),
            "run_mode": run_mode,
            "backtest_start_date": start_date,
            "backtest_end_date": end_date,
            "report_generated_at_utc": report_timestamp,
            "manifest_filename": "run_manifest.json",
            "report_dir": str(self.report_dir.resolve()),
            "report_started_epoch": generated_at.timestamp(),
        }

    def _resolve_run_mode(self, metrics: dict) -> str:
        mode = metrics.get("run_mode") or metrics.get("mode") or self.cfg.get("mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
        if self.cfg.get("rl", {}).get("use_rl") is True:
            return "full_rl"
        if self.cfg.get("rl", {}).get("use_rl") is False:
            return "optimizer_only"
        return "unknown"

    def _resolve_backtest_date(
        self,
        value: Any,
        nav_series: pd.Series,
        *,
        first: bool,
    ) -> str:
        if value is not None:
            resolved = self._jsonify(value)
            if isinstance(resolved, str) and resolved:
                return resolved
        if nav_series.empty:
            return ""
        idx = nav_series.index.min() if first else nav_series.index.max()
        if hasattr(idx, "date"):
            return idx.date().isoformat()
        return str(idx)

    def _build_run_id(
        self,
        run_mode: str,
        start_date: str,
        end_date: str,
        generated_at: datetime,
    ) -> str:
        return "_".join(
            [
                self._slug(run_mode),
                self._slug(start_date),
                self._slug(end_date),
                generated_at.strftime("%Y%m%dT%H%M%SZ"),
            ]
        )

    def _with_artifact_metadata(self, payload: dict[str, Any], *, artifact_type: str) -> dict[str, Any]:
        enriched = dict(payload)
        enriched["_report_metadata"] = self._artifact_metadata(artifact_type)
        return enriched

    def _artifact_metadata(self, artifact_type: str) -> dict[str, Any]:
        return {
            "schema_version": self._run_context["schema_version"],
            "artifact_type": artifact_type,
            "run_id": self._run_context["run_id"],
            "run_mode": self._run_context["run_mode"],
            "backtest_start_date": self._run_context["backtest_start_date"],
            "backtest_end_date": self._run_context["backtest_end_date"],
            "report_generated_at_utc": self._run_context["report_generated_at_utc"],
            "freshness_verified_at_utc": self._isoformat_utc(self._now_utc()),
            "manifest_file": self._run_context["manifest_filename"],
        }

    def _attach_frame_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        metadata = {
            "run_id": self._run_context["run_id"],
            "run_mode": self._run_context["run_mode"],
            "backtest_start_date": self._run_context["backtest_start_date"],
            "backtest_end_date": self._run_context["backtest_end_date"],
            "report_generated_at_utc": self._run_context["report_generated_at_utc"],
        }
        for key, value in metadata.items():
            enriched[key] = value
        ordered = list(metadata.keys()) + [col for col in enriched.columns if col not in metadata]
        return enriched.loc[:, ordered]

    def _write_run_manifest(self) -> Path:
        model_dir_raw = self.cfg.get("paths", {}).get("model_dir")
        model_dir = Path(model_dir_raw) if model_dir_raw else None
        manifest = {
            "schema_version": self._run_context["schema_version"],
            "run": {
                "run_id": self._run_context["run_id"],
                "mode": self._run_context["run_mode"],
                "backtest_start_date": self._run_context["backtest_start_date"],
                "backtest_end_date": self._run_context["backtest_end_date"],
                "report_generated_at_utc": self._run_context["report_generated_at_utc"],
            },
            "config": {
                "sha256": self._sha256_text(
                    json.dumps(self._jsonify(self.cfg), sort_keys=True)
                ),
                "snapshot": self._jsonify(self.cfg),
            },
            "models": {
                "model_dir": str(model_dir.resolve()) if model_dir else "",
                "artifacts": self._collect_model_artifacts(model_dir),
            },
            "reports": {
                "report_dir": str(self.report_dir.resolve()),
                "artifacts": self._collect_report_artifacts(),
            },
        }
        path = self.report_dir / self._run_context["manifest_filename"]
        self._write_json(path, manifest)
        logger.info("Run manifest saved → %s", path)
        return path

    def _collect_model_artifacts(self, model_dir: Path | None) -> list[dict[str, Any]]:
        if model_dir is None or not model_dir.exists():
            return []
        return [
            self._describe_path(path, artifact_group="model", generated_in_run=False)
            for path in sorted(model_dir.iterdir())
        ]

    def _collect_report_artifacts(self) -> list[dict[str, Any]]:
        generated = {path.resolve() for path in self._generated_artifacts}
        artifacts: list[dict[str, Any]] = []
        for path in sorted(self.report_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name == self._run_context["manifest_filename"]:
                continue
            resolved = path.resolve()
            generated_in_run = resolved in generated
            artifacts.append(
                self._describe_path(
                    path,
                    artifact_group="report",
                    generated_in_run=generated_in_run,
                    stale_relative_to_run=(
                        not generated_in_run
                        and resolved in self._preexisting_report_files
                        and path.stat().st_mtime < self._run_context["report_started_epoch"]
                    ),
                )
            )
        return artifacts

    def _describe_path(
        self,
        path: Path,
        *,
        artifact_group: str,
        generated_in_run: bool,
        stale_relative_to_run: bool = False,
    ) -> dict[str, Any]:
        stat = path.stat()
        payload = {
            "group": artifact_group,
            "name": path.name,
            "path": str(path.resolve()),
            "artifact_type": self._artifact_type_for_path(path.name),
            "exists": True,
            "is_dir": path.is_dir(),
            "size_bytes": None if path.is_dir() else int(stat.st_size),
            "modified_at_utc": self._isoformat_utc(datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)),
            "generated_in_current_run": generated_in_run,
            "stale_relative_to_current_run": stale_relative_to_run,
        }
        if path.is_file():
            payload["sha256"] = self._sha256_file(path)
        else:
            payload["entry_count"] = len(list(path.iterdir()))
        return payload

    def _artifact_type_for_path(self, name: str) -> str:
        mapping = {
            "metrics.json": "metrics",
            "nav_series.parquet": "nav_series",
            "rebalance_log.csv": "rebalance_log",
            "rebalance_log.parquet": "rebalance_log",
            "selection_diagnostics.json": "selection_diagnostics",
            "selection_rebalance_log.csv": "selection_rebalance_log",
            "selection_rebalance_log.parquet": "selection_rebalance_log",
            "stock_ranker_feature_importance.csv": "stock_ranker_feature_importance",
            "stock_ranker_feature_importance.json": "stock_ranker_feature_importance",
            "current_portfolio.json": "current_portfolio",
            "attribution.json": "attribution",
            "rl_holdout_comparison.json": "rl_holdout_comparison",
            "rl_full_backtest_comparison.json": "rl_full_backtest_comparison",
            "rl_full_neutral_comparison.json": "rl_full_neutral_comparison",
            "rl_control_evaluation.json": "rl_control_evaluation",
            "nav_chart.png": "nav_chart",
            "drawdown_chart.png": "drawdown_chart",
            "year_returns.png": "year_returns_chart",
            "sector_attribution.png": "sector_attribution_chart",
            "rolling_returns.png": "rolling_returns_chart",
            "run_manifest.json": "run_manifest",
        }
        return mapping.get(name, "unknown")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with open(path, "w") as f:
            json.dump(self._jsonify(payload), f, indent=2, sort_keys=True)

    def _jsonify(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonify(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (datetime, pd.Timestamp)):
            ts = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if pd.isna(value):
            return None
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _sha256_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _slug(value: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
        return "_".join(filter(None, cleaned.split("_"))) or "unknown"

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc).replace(microsecond=0)

    @staticmethod
    def _isoformat_utc(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

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
