#!/usr/bin/env python3
"""Train and evaluate the posture utility regression model.

Usage:
    python scripts/train_posture_model.py [--parquet PATH] [--out-dir DIR]

Steps:
    1. Load posture dataset parquet + build 23-feature matrix
    2. Pre-training sanity checks
    3. Fit PostureUtilityModel (3 LGBMRegressors on normalized utilities)
    4. LOO cross-validation evaluation
    5. Post-training sanity checks
    6. Save model + evaluation report
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.posture_model import (
    FEATURE_NAMES,
    POSTURES,
    PostureUtilityModel,
    build_features,
    build_regression_targets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).parent.parent
DEFAULT_PARQUET = REPO / "artifacts/reports/posture_dataset_return_only_h2_2016.parquet"
DEFAULT_MACRO_DIR = REPO / "artifacts/feature_store/macro"
DEFAULT_PRICE_PATH = REPO / "data/processed/price_matrix.parquet"
DEFAULT_OUT_DIR = REPO / "artifacts/models/posture_model"
DEFAULT_EVAL_PATH = REPO / "artifacts/reports/posture_regression_eval.json"


def _print_section(title: str) -> None:
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def _pre_training_checks(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    sample_info: pd.DataFrame,
    epsilon: float,
) -> bool:
    """Run pre-training sanity checks. Returns True if all pass."""
    ok = True
    _print_section("PRE-TRAINING SANITY CHECKS")

    # 1. Class distribution after ε filter
    margins = sample_info["utility_margin"].values
    non_indiff = margins >= epsilon
    filtered_info = sample_info[non_indiff]
    class_counts = filtered_info["target_posture"].value_counts()
    n_total = len(filtered_info)
    logger.info("Class distribution after ε=%.4f filter (n=%d):", epsilon, n_total)
    for cls, cnt in class_counts.items():
        pct = 100 * cnt / n_total
        flag = " ← SKEWED" if pct > 70 else ""
        logger.info("  %-12s  %2d / %d  (%.0f%%)%s", cls, cnt, n_total, pct, flag)
    if any((cnt / n_total) > 0.70 for cnt in class_counts.values):
        logger.warning("CHECK 1 FAIL: one class > 70%% — ε filter may not have rebalanced")
        ok = False
    else:
        logger.info("CHECK 1 PASS: no class > 70%%")

    # 2. Mean margin post-filter vs pre-filter
    mean_pre = float(margins.mean())
    mean_post = float(margins[non_indiff].mean()) if non_indiff.sum() > 0 else 0.0
    logger.info("Mean margin: pre-filter=%.4f, post-filter=%.4f", mean_pre, mean_post)
    if mean_post <= mean_pre * 1.05:
        logger.warning("CHECK 2 WARN: post-filter margin not meaningfully higher than pre — ε may be too small")
    else:
        logger.info("CHECK 2 PASS: post-filter margin higher than pre-filter")

    # 3. Minimum sample count post-filter
    n_after = int(non_indiff.sum())
    logger.info("Samples after ε filter: %d / %d", n_after, len(margins))
    if n_after < 10:
        logger.error("CHECK 3 FAIL: < 10 non-indifferent samples — lower ε threshold")
        ok = False
    else:
        logger.info("CHECK 3 PASS: %d samples ≥ 10 minimum", n_after)

    # 4. Feature NaN check
    nan_counts = features[FEATURE_NAMES].isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    if not nan_features.empty:
        logger.warning("CHECK 4 WARN: NaN in features (filled with 0): %s", nan_features.to_dict())
    else:
        logger.info("CHECK 4 PASS: no NaN in feature matrix")

    return ok


def _post_training_checks(
    model: PostureUtilityModel,
    loo_result: dict,
    sample_info: pd.DataFrame,
) -> None:
    """Run post-training sanity checks."""
    _print_section("POST-TRAINING SANITY CHECKS")

    # 1. Top-3 features should include at least one regime/breadth feature
    top3 = [name for name, _ in loo_result["top_features"][:3]]
    regime_breadth = {
        "vol_percentile_1y", "current_drawdown", "sector_dispersion",
        "vix_percentile_1y", "max_drawdown_3m", "cross_sectional_vol",
        "spread_decile", "nifty_vol_1m",
    }
    has_regime = any(f in regime_breadth for f in top3)
    logger.info("Top-3 features: %s", top3)
    if not has_regime:
        logger.warning(
            "CHECK 5 WARN: top-3 features don't include regime/breadth features — "
            "model may be fitting noise"
        )
    else:
        logger.info("CHECK 5 PASS: at least one regime/breadth feature in top-3")

    # 2. Residual pattern by stress_bucket
    preds = loo_result["loo_predictions"]
    truths = loo_result["true_postures"]
    margins = sample_info["utility_margin"].values
    non_indiff = margins >= model.epsilon

    if "stress_bucket" in sample_info.columns:
        common_idx = sample_info.index
        buckets = sample_info["stress_bucket"].values
        for bucket in ["low", "medium", "high"]:
            mask = (buckets == bucket) & non_indiff
            if mask.sum() == 0:
                continue
            correct = sum(
                preds[i] == truths[i]
                for i in range(len(preds))
                if mask[i]
            )
            n = int(mask.sum())
            acc = correct / n
            logger.info("Accuracy in %-8s stress bucket: %d/%d = %.1f%%", bucket, correct, n, 100 * acc)

    # 3. Spearman rank check: predicted vs actual utility order
    try:
        from scipy.stats import spearmanr
        ranks_ok = []
        for i in range(len(preds)):
            if not non_indiff[i]:
                continue
            pred_rank = {p: j for j, p in enumerate(sorted(POSTURES, key=lambda p: p == preds[i], reverse=True))}
            true_rank = {p: j for j, p in enumerate(sorted(POSTURES, key=lambda p: p == truths[i], reverse=True))}
            corr, _ = spearmanr([pred_rank[p] for p in POSTURES], [true_rank[p] for p in POSTURES])
            if not np.isnan(corr):
                ranks_ok.append(corr)
        if ranks_ok:
            mean_corr = float(np.mean(ranks_ok))
            logger.info("Mean Spearman rank corr (predicted vs actual order): %.3f", mean_corr)
            if mean_corr < 0:
                logger.warning("CHECK 6 WARN: negative Spearman correlation — model inverted signal")
            elif mean_corr >= 0.3:
                logger.info("CHECK 6 PASS: positive Spearman correlation ≥ 0.3")
            else:
                logger.info("CHECK 6 INFO: low positive Spearman correlation (%.3f < 0.3)", mean_corr)
    except ImportError:
        logger.info("scipy not available — skipping Spearman check")


def _compute_utility_capture(
    parquet_path: Path,
    loo_preds: list[str],
    sample_info: pd.DataFrame,
    epsilon: float,
) -> tuple[float | None, float | None]:
    """Compute utility_capture = chosen_utility / oracle_utility."""
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])

    # Build raw utility table: date → {posture: raw_utility}
    raw_utils: dict[str, dict[str, float]] = {}
    for date, group in df.groupby("date", sort=True):
        raw_utils[str(date)] = {
            str(row["posture"]): float(row["utility_return_only"])
            for _, row in group.iterrows()
        }

    dates = [str(d) for d in sample_info.index]
    margins = sample_info["utility_margin"].values
    non_indiff = margins >= epsilon

    captures_all, captures_non_indiff = [], []
    for i, date in enumerate(dates):
        utils = raw_utils.get(date, {})
        if not utils or len(utils) < 3:
            continue
        oracle = max(utils.values())
        if oracle <= 0:
            continue
        chosen_posture = loo_preds[i]
        chosen_util = utils.get(chosen_posture, 0.0)
        cap = chosen_util / oracle
        captures_all.append(cap)
        if non_indiff[i]:
            captures_non_indiff.append(cap)

    uc_all = float(np.mean(captures_all)) if captures_all else None
    uc_non_indiff = float(np.mean(captures_non_indiff)) if captures_non_indiff else None
    return uc_non_indiff, uc_all


def main(
    parquet_path: Path = DEFAULT_PARQUET,
    macro_dir: Path = DEFAULT_MACRO_DIR,
    price_path: Path = DEFAULT_PRICE_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    eval_path: Path = DEFAULT_EVAL_PATH,
    epsilon_percentile: int = 25,
) -> dict:
    _print_section("POSTURE UTILITY REGRESSION — TRAINING & EVALUATION")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    logger.info("Loading features from parquet: %s", parquet_path)
    features = build_features(str(parquet_path), str(macro_dir), str(price_path))
    logger.info("Feature matrix shape: %s", features.shape)
    logger.info("Sample dates: %s ... %s", features.index[0].date(), features.index[-1].date())

    targets, sample_info = build_regression_targets(str(parquet_path))
    logger.info("Targets shape: %s", targets.shape)
    logger.info("Class distribution (all samples):\n%s",
                sample_info["target_posture"].value_counts().to_string())

    # ── 2. Pre-training sanity (dry-run epsilon first) ────────────────────────
    margins = sample_info["utility_margin"].values
    epsilon_preview = float(np.percentile(margins, epsilon_percentile))
    n_non_indiff_preview = int((margins >= epsilon_preview).sum())
    if n_non_indiff_preview < 10:
        epsilon_percentile = 15
        logger.warning("p25 leaves < 10 samples — using p15 instead")

    logger.info("Margin distribution: min=%.4f, p25=%.4f, median=%.4f, p75=%.4f, max=%.4f",
                float(np.min(margins)), float(np.percentile(margins, 25)),
                float(np.median(margins)), float(np.percentile(margins, 75)),
                float(np.max(margins)))

    # Histogram (ASCII)
    hist, edges = np.histogram(margins, bins=10)
    logger.info("Margin histogram:")
    for i, (lo, hi, cnt) in enumerate(zip(edges, edges[1:], hist)):
        bar = "#" * min(cnt * 2, 40)
        logger.info("  [%.4f–%.4f] %s %d", lo, hi, bar, cnt)

    epsilon_for_checks = float(np.percentile(margins, epsilon_percentile))
    pre_ok = _pre_training_checks(features, targets, sample_info, epsilon_for_checks)
    if not pre_ok:
        logger.warning("Some pre-training checks failed — proceeding anyway")

    # ── 3. Fit model ─────────────────────────────────────────────────────────
    _print_section("FITTING PostureUtilityModel")
    model = PostureUtilityModel(epsilon_percentile=epsilon_percentile)
    model.fit(features, targets, sample_info)

    # ── 4. LOO evaluation ─────────────────────────────────────────────────────
    _print_section("LOO CROSS-VALIDATION")
    loo_result = model.evaluate_loo(features, targets, sample_info)

    # Compute utility capture with raw utilities
    uc_non_indiff, uc_all = _compute_utility_capture(
        parquet_path, loo_result["loo_predictions"], sample_info, model.epsilon
    )
    loo_result["utility_capture_non_indifferent"] = round(uc_non_indiff, 4) if uc_non_indiff is not None else None
    loo_result["utility_capture_all"] = round(uc_all, 4) if uc_all is not None else None

    # Compute regret
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    regrets = []
    for i, (date, row_info) in enumerate(sample_info.iterrows()):
        if row_info["utility_margin"] < model.epsilon:
            continue
        group = df[df["date"] == date]
        utils = {str(r["posture"]): float(r["utility_return_only"]) for _, r in group.iterrows()}
        oracle = max(utils.values())
        chosen = utils.get(loo_result["loo_predictions"][i], 0.0)
        regrets.append(oracle - chosen)
    loo_result["mean_regret"] = round(float(np.mean(regrets)), 6) if regrets else None

    # ── 5. Print results ──────────────────────────────────────────────────────
    _print_section("RESULTS")
    acc = loo_result["loo_accuracy"]
    n = loo_result["loo_accuracy_n"]
    logger.info("LOO Accuracy (non-indifferent, n=%d): %.1f%%", n, 100 * acc)
    logger.info("Baseline always_neutral:   37.5%%")
    logger.info("Baseline always_risk_off:  50.0%%  ← binding gate")
    logger.info("Beats binding baseline: %s", "YES ✓" if acc > 0.5 else "NO ✗")
    if uc_non_indiff is not None:
        logger.info("Utility capture (non-indifferent): %.1f%%  (gate: ≥ 90%%)", 100 * uc_non_indiff)
        logger.info("Utility capture gate: %s", "PASS ✓" if uc_non_indiff >= 0.90 else "FAIL ✗")
    if uc_all is not None:
        logger.info("Utility capture (all samples): %.1f%%", 100 * uc_all)
    logger.info("Mean regret: %.4f", loo_result.get("mean_regret") or 0)
    logger.info("")

    gate_pass = acc > 0.50 and (uc_non_indiff is None or uc_non_indiff >= 0.90)
    logger.info("FINAL VERDICT: %s", "SIGNAL EXISTS — proceed to next step" if gate_pass else "NO SIGNAL — close posture research track")

    logger.info("")
    logger.info("Feature importance (top 10 combined across 3 models):")
    for fname, fval in loo_result["top_features"]:
        logger.info("  %-36s  %d", fname, fval)

    # ── 6. Post-training checks ───────────────────────────────────────────────
    _post_training_checks(model, loo_result, sample_info)

    # ── 7. Save model + eval ──────────────────────────────────────────────────
    model.save(out_dir)
    logger.info("Model saved → %s", out_dir)

    eval_out = {
        "parquet_path": str(parquet_path),
        "n_samples_total": loo_result["n_samples_total"],
        "n_non_indifferent": loo_result["n_non_indifferent"],
        "epsilon": loo_result["epsilon"],
        "epsilon_percentile": epsilon_percentile,
        "loo_accuracy": loo_result["loo_accuracy"],
        "loo_accuracy_n": loo_result["loo_accuracy_n"],
        "baseline_always_neutral": 0.375,
        "baseline_always_risk_off": 0.500,
        "beats_binding_baseline": loo_result["beats_binding_baseline"],
        "utility_capture_non_indifferent": loo_result["utility_capture_non_indifferent"],
        "utility_capture_all": loo_result["utility_capture_all"],
        "utility_capture_gate_90pct": (
            uc_non_indiff is not None and uc_non_indiff >= 0.90
        ),
        "mean_regret": loo_result.get("mean_regret"),
        "gate_pass": gate_pass,
        "top_features": loo_result["top_features"],
        "loo_predictions": loo_result["loo_predictions"],
        "true_postures": loo_result["true_postures"],
        "feature_importance_full": model.feature_importance_summary(),
    }

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(eval_out, indent=2, default=str))
    logger.info("Evaluation saved → %s", eval_path)

    return eval_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train posture utility regression model")
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--macro-dir", default=str(DEFAULT_MACRO_DIR))
    parser.add_argument("--price-path", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--eval-out", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--epsilon-percentile", type=int, default=25)
    args = parser.parse_args()

    main(
        parquet_path=Path(args.parquet),
        macro_dir=Path(args.macro_dir),
        price_path=Path(args.price_path),
        out_dir=Path(args.out_dir),
        eval_path=Path(args.eval_out),
        epsilon_percentile=args.epsilon_percentile,
    )
