#!/usr/bin/env python3
"""
CLI: Run the full walk-forward backtest.

Usage:
    python scripts/run_backtest.py [--mode full_rl] [--config path/to/base.yaml]

This script:
1. Loads cached data (run download_data.py first)
2. Runs walk-forward simulation from 2013→2026
3. Computes baseline strategies
4. Runs attribution engine
5. Generates full report
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Prevent OpenMP/loky workers from conflicting with PyTorch thread pool
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd

from src.attribution.attribution import AttributionEngine
from src.backtest.baselines import (
    compare_strategies,
    equal_weight_backtest,
    nifty_buy_and_hold,
    sector_momentum_backtest,
)
from src.backtest.simulator import PortfolioSimulator
from src.backtest.walk_forward import WalkForwardEngine
from src.config import load_config, setup_logging
from src.data.ingestion import load_price_matrix, load_volume_matrix
from src.data.macro import MacroDataManager
from src.reporting.report import ReportGenerator


@click.command()
@click.option("--no-rl", "disable_rl", is_flag=True, default=False, help="Disable RL overlay (maps to optimizer_only when --mode is omitted)")
@click.option(
    "--mode",
    type=click.Choice(["selection_only", "optimizer_only", "full_rl"], case_sensitive=False),
    default="full_rl",
    show_default=True,
    help="Backtest mode: stock selection only, selection+optimizer, or full RL pipeline",
)
@click.option("--config", default=None, help="Path to custom config file")
@click.option("--start", default=None, help="Override backtest start date")
@click.option("--end", default=None, help="Override backtest end date")
@click.option(
    "--stock-fwd-window-days",
    type=int,
    default=None,
    help="Override the stock-ranker label horizon in trading days (28≈4W, 56≈8W, 84≈12W)",
)
@click.option("--baselines/--no-baselines", default=True, help="Run baseline strategies")
@click.option("--report/--no-report", default=True, help="Generate report")
def main(disable_rl, mode, config, start, end, stock_fwd_window_days, baselines, report):
    """Run the full walk-forward backtest and generate report."""
    cfg = load_config(config)
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    if disable_rl and mode == "full_rl":
        mode = "optimizer_only"
    elif disable_rl and mode != "optimizer_only":
        raise click.UsageError("--no-rl only supports optimizer_only mode")

    if start:
        cfg["backtest"]["start_date"] = start
    if end:
        cfg["backtest"]["end_date"] = end
    if stock_fwd_window_days is not None:
        cfg["stock_model"]["fwd_window_days"] = int(stock_fwd_window_days)

    logger.info("=" * 70)
    logger.info("WALK-FORWARD BACKTEST")
    logger.info("Period: %s → %s", cfg["backtest"]["start_date"], cfg["backtest"]["end_date"])
    logger.info("Mode: %s", mode)
    logger.info("Stock label horizon: %s trading days", cfg["stock_model"].get("fwd_window_days", 28))
    logger.info("RL overlay: %s", "ENABLED" if mode == "full_rl" else "DISABLED")
    logger.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading price matrix ...")
    try:
        price_matrix = load_price_matrix(cfg)
        volume_matrix = load_volume_matrix(cfg)
    except FileNotFoundError as e:
        logger.error("%s\nRun: python scripts/download_data.py first", e)
        sys.exit(1)

    logger.info("Loading macro data ...")
    macro_mgr = MacroDataManager(cfg)
    macro_df = macro_mgr.load()

    # Trim to backtest window
    bt_start = pd.Timestamp(cfg["backtest"]["start_date"])
    bt_end_str = cfg["backtest"]["end_date"]
    bt_end = pd.Timestamp(bt_end_str) if bt_end_str != "latest" else pd.Timestamp.today()

    price_matrix = price_matrix.loc[:bt_end]
    macro_df = macro_df.loc[:bt_end]

    # Benchmark prices
    bm_ticker = cfg["backtest"].get("benchmark_ticker", "^NSEI")
    bm_nav = None
    if bm_ticker in price_matrix.columns:
        bm_prices = price_matrix[bm_ticker].loc[bt_start:bt_end].dropna()
        bm_nav = bm_prices / bm_prices.iloc[0] * cfg["backtest"]["initial_capital"]

    # ── 2. Main backtest ──────────────────────────────────────────────────────
    engine = WalkForwardEngine(
        price_matrix=price_matrix,
        volume_matrix=volume_matrix,
        macro_df=macro_df,
        cfg=cfg,
        mode=mode,
    )

    metrics = engine.run()
    engine.save_state()
    metrics["mode"] = mode
    metrics["stock_fwd_window_days"] = int(cfg["stock_model"].get("fwd_window_days", 28))

    # Add dates to metrics
    metrics["start_date"] = str(bt_start.date())
    metrics["end_date"] = str(bt_end.date())

    # ── 3. Baseline strategies ────────────────────────────────────────────────
    strategy_navs: dict[str, pd.Series] = {}

    if baselines:
        logger.info("Running baseline strategies ...")
        rebal_dates = engine._generate_rebalance_dates()

        # Nifty B&H
        nifty_nav = nifty_buy_and_hold(
            price_matrix, cfg["backtest"]["initial_capital"],
            str(rebal_dates[0].date()), str(bt_end.date()), bm_ticker
        )
        if not nifty_nav.empty:
            strategy_navs["Nifty B&H"] = nifty_nav

        # Equal weight
        try:
            ew_nav = equal_weight_backtest(
                price_matrix, cfg, rebal_dates, cfg["backtest"]["initial_capital"]
            )
            strategy_navs["Equal Weight"] = ew_nav
        except Exception as e:
            logger.warning("Equal-weight baseline failed: %s", e)

        # Sector momentum
        try:
            sm_nav = sector_momentum_backtest(
                price_matrix, cfg, rebal_dates, cfg["backtest"]["initial_capital"]
            )
            strategy_navs["Sector Momentum"] = sm_nav
        except Exception as e:
            logger.warning("Sector momentum baseline failed: %s", e)

        # Comparison table
        if strategy_navs:
            sim = PortfolioSimulator(cfg)
            comp = compare_strategies(
                {"RL Portfolio": engine.nav_series, **strategy_navs},
                sim.compute_metrics,
                bm_nav,
            )
            print("\n  STRATEGY COMPARISON:")
            print(comp.to_string())
            comp.to_csv(
                Path(cfg["paths"]["report_dir"]) / "strategy_comparison.csv"
            )

    # ── 4. Attribution ────────────────────────────────────────────────────────
    attribution = None
    try:
        from src.data.universe import UniverseManager
        uni_mgr = UniverseManager(cfg)
        snap = uni_mgr.get_universe(bt_end.date(), price_matrix=price_matrix)
        sector_map = uni_mgr.get_sector_map(snap)

        attr_engine = AttributionEngine(cfg)
        attribution = attr_engine.compute(
            nav_series=engine.nav_series,
            rebalance_records=engine.rebalance_records,
            price_matrix=price_matrix,
            sector_map=sector_map,
            benchmark_nav=bm_nav,
            macro_features=engine.macro_features,
            sector_scorer=engine.sector_scorer,
            stock_ranker=engine.stock_ranker,
        )
    except Exception as e:
        logger.warning("Attribution failed: %s", e)

    # ── 5. Current portfolio ──────────────────────────────────────────────────
    current_portfolio = None
    if engine.rebalance_records:
        last_rec = engine.rebalance_records[-1]
        current_portfolio = {
            "as_of": str(last_rec.rebalance_date),
            "weights": last_rec.target_weights,
            "sector_tilts": last_rec.sector_tilts,
            "cash": last_rec.cash_target,
            "note": "Latest walk-forward portfolio recommendation",
        }

    # ── 6. Report ─────────────────────────────────────────────────────────────
    if report:
        reporter = ReportGenerator(cfg)
        report_dir = reporter.generate_full_report(
            metrics=metrics,
            nav_series=engine.nav_series,
            rebalance_records=engine.rebalance_records,
            attribution=attribution,
            strategy_navs=strategy_navs if strategy_navs else None,
            current_portfolio=current_portfolio,
            benchmark_nav=bm_nav,
            selection_diagnostics=engine.selection_diagnostics,
            stock_ranker=engine.stock_ranker,
        )
        logger.info("Report generated → %s", report_dir)

    logger.info("Backtest complete.")
    return metrics


if __name__ == "__main__":
    main()
