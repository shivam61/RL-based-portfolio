#!/usr/bin/env python3
"""
Ablation study: RL retrain frequency vs adaptive event triggers.

Runs 6 configurations back-to-back and prints a comparison table.
Each config uses the same data / random seed; only retrain_freq_weeks
and rl_triggers.enabled vary.

Configs:
  A  4-week  retrain  (triggers OFF)
  B  8-week  retrain  (triggers OFF)
  C  12-week retrain  (triggers OFF)  ← current default
  D  26-week retrain  (triggers OFF)
  E  26-week retrain  (triggers ON)
  F   8-week retrain  (triggers ON)

Usage:
    python scripts/ablation_retrain_freq.py [--configs A,B,C] [--no-report]

Output:
    Prints comparison table to stdout.
    Saves CSV to artifacts/reports/ablation_retrain_freq.csv
"""
from __future__ import annotations

import copy
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd

from src.backtest.walk_forward import WalkForwardEngine
from src.config import load_config, setup_logging
from src.data.ingestion import load_price_matrix, load_volume_matrix
from src.data.macro import MacroDataManager

logger = logging.getLogger(__name__)

# ── Ablation config definitions ───────────────────────────────────────────────

ABLATION_CONFIGS: dict[str, dict] = {
    "A": {
        "label": "4-week retrain (no triggers)",
        "rl.retrain_freq_weeks": 4,
        "rl_triggers.enabled": False,
    },
    "B": {
        "label": "8-week retrain (no triggers)",
        "rl.retrain_freq_weeks": 8,
        "rl_triggers.enabled": False,
    },
    "C": {
        "label": "12-week retrain (no triggers) ← baseline",
        "rl.retrain_freq_weeks": 12,
        "rl_triggers.enabled": False,
    },
    "D": {
        "label": "26-week retrain (no triggers)",
        "rl.retrain_freq_weeks": 26,
        "rl_triggers.enabled": False,
    },
    "E": {
        "label": "26-week retrain + event triggers",
        "rl.retrain_freq_weeks": 26,
        "rl_triggers.enabled": True,
    },
    "F": {
        "label": "8-week retrain + event triggers",
        "rl.retrain_freq_weeks": 8,
        "rl_triggers.enabled": True,
    },
}


def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply dotted-key overrides to a nested config dict."""
    cfg = copy.deepcopy(cfg)
    for dotkey, val in overrides.items():
        if dotkey == "label":
            continue
        keys = dotkey.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = val
    return cfg


def _run_one(cfg: dict, price_matrix, volume_matrix, macro_df) -> dict:
    """Run one backtest and return the metrics dict."""
    bt_end_str = cfg["backtest"]["end_date"]
    bt_end = (
        pd.Timestamp.today()
        if bt_end_str == "latest"
        else pd.Timestamp(bt_end_str)
    )

    engine = WalkForwardEngine(
        price_matrix=price_matrix.loc[:bt_end],
        volume_matrix=volume_matrix.loc[:bt_end],
        macro_df=macro_df.loc[:bt_end],
        cfg=cfg,
        use_rl=True,
    )
    metrics = engine.run()

    # Count event-triggered retrains
    n_event_retrains = sum(
        1 for e in engine._trigger_log if e.get("tier") is not None
    )
    metrics["n_event_retrains"] = n_event_retrains
    metrics["n_trigger_events"] = len(engine._trigger_log)

    return metrics


def _fmt(val, fmt=".2f"):
    if val is None:
        return "—"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


@click.command()
@click.option(
    "--configs",
    default="A,B,C,D,E,F",
    help="Comma-separated list of config IDs to run (e.g. A,C,E)",
)
@click.option("--no-report", "skip_report", is_flag=True, default=False)
def main(configs: str, skip_report: bool):
    """Ablation: retrain frequency × event triggers."""
    base_cfg = load_config()
    setup_logging(base_cfg)

    selected = [c.strip().upper() for c in configs.split(",")]
    unknown = [c for c in selected if c not in ABLATION_CONFIGS]
    if unknown:
        logger.error("Unknown config IDs: %s. Valid: %s", unknown, list(ABLATION_CONFIGS))
        sys.exit(1)

    logger.info("Loading shared data (price, volume, macro) …")
    try:
        price_matrix = load_price_matrix(base_cfg)
        volume_matrix = load_volume_matrix(base_cfg)
    except FileNotFoundError as e:
        logger.error("%s\nRun: python scripts/download_data.py first", e)
        sys.exit(1)

    macro_mgr = MacroDataManager(base_cfg)
    macro_df = macro_mgr.load()

    rows = []
    for cfg_id in selected:
        spec = ABLATION_CONFIGS[cfg_id]
        logger.info("=" * 65)
        logger.info("Config %s: %s", cfg_id, spec["label"])
        logger.info("=" * 65)

        cfg = _apply_overrides(base_cfg, spec)
        try:
            m = _run_one(cfg, price_matrix, volume_matrix, macro_df)
            rows.append(
                {
                    "Config": cfg_id,
                    "Description": spec["label"],
                    "CAGR %": _fmt(m.get("cagr", 0) * 100),
                    "Sharpe": _fmt(m.get("sharpe")),
                    "MaxDD %": _fmt(m.get("max_drawdown", 0) * 100),
                    "Final NAV ₹L": _fmt(
                        m.get("final_nav", 0) / 1e5, ".1f"
                    ),
                    "Retrains (sched)": _fmt(
                        m.get("n_rl_retrains", 0), ".0f"
                    ),
                    "Events fired": _fmt(
                        m.get("n_trigger_events", 0), ".0f"
                    ),
                    "Event retrains": _fmt(
                        m.get("n_event_retrains", 0), ".0f"
                    ),
                }
            )
        except Exception as exc:
            logger.exception("Config %s failed: %s", cfg_id, exc)
            rows.append(
                {
                    "Config": cfg_id,
                    "Description": spec["label"],
                    "CAGR %": "ERROR",
                    "Sharpe": "ERROR",
                    "MaxDD %": "ERROR",
                    "Final NAV ₹L": str(exc)[:40],
                    "Retrains (sched)": "—",
                    "Events fired": "—",
                    "Event retrains": "—",
                }
            )

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("  ABLATION RESULTS — RL Retrain Frequency × Event Triggers")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    if not skip_report:
        out = Path(base_cfg["paths"]["report_dir"]) / "ablation_retrain_freq.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Ablation CSV saved → %s", out)


if __name__ == "__main__":
    main()
