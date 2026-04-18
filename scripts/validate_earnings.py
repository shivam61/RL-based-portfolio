#!/usr/bin/env python3
"""
4-layer earnings data validation suite.

Layer 1 — Structural integrity   (panel shape, duplicates, index sanity)
Layer 2 — Per-ticker coverage     (first/last date, % missing, longest streak)
Layer 3 — Temporal / leakage      (feature onset vs quarter-end date)
Layer 4 — Economic sense          (growth magnitudes, sign consistency)

Usage:
    python scripts/validate_earnings.py [--verbose] [--sample-n 10]

Exit code 0 = all checks pass (warnings do not fail).
Exit code 1 = at least one RED-FLAG check failed.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import pandas as pd
import yfinance as yf

from src.config import load_config, setup_logging

logger = logging.getLogger(__name__)

RED   = "\033[91m"
YELLOW= "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

_FAILURES: list[str] = []
_WARNINGS: list[str] = []


def fail(msg: str) -> None:
    _FAILURES.append(msg)
    logger.error("%s[FAIL]%s %s", RED, RESET, msg)


def warn(msg: str) -> None:
    _WARNINGS.append(msg)
    logger.warning("%s[WARN]%s %s", YELLOW, RESET, msg)


def ok(msg: str) -> None:
    logger.info("%s[OK]%s   %s", GREEN, RESET, msg)


# ── Layer 1: Structural integrity ────────────────────────────────────────────

def layer1_structural(panel: pd.DataFrame) -> None:
    logger.info("=" * 65)
    logger.info("LAYER 1 — Structural integrity")
    logger.info("=" * 65)

    # Shape
    ok(f"Panel shape: {panel.shape}  (rows=days, cols=feature×ticker)")

    # Index type
    if not isinstance(panel.index, pd.DatetimeIndex):
        fail("Panel index is not DatetimeIndex")
    else:
        ok("Index is DatetimeIndex")

    # Sorted
    if not panel.index.is_monotonic_increasing:
        fail("Index is NOT strictly sorted ascending")
    else:
        ok("Index sorted ascending")

    # Duplicate dates
    dup_dates = panel.index[panel.index.duplicated()]
    if len(dup_dates):
        fail(f"Duplicate dates in index: {dup_dates[:5].tolist()}")
    else:
        ok("No duplicate dates")

    # Duplicate columns
    if panel.columns.duplicated().any():
        dup_cols = panel.columns[panel.columns.duplicated()].tolist()
        fail(f"Duplicate columns: {dup_cols[:5]}")
    else:
        ok("No duplicate columns")

    # MultiIndex column check
    if isinstance(panel.columns, pd.MultiIndex):
        features = panel.columns.get_level_values(0).unique().tolist()
        tickers  = panel.columns.get_level_values(1).unique().tolist()
        ok(f"MultiIndex columns: {len(features)} features × {len(tickers)} tickers")
        logger.info("  Features: %s", features)
    else:
        fail("Columns are not a MultiIndex(feature, ticker) — unexpected format")
        return

    # Naming consistency: all tickers end with .NS or .BO
    bad_tickers = [t for t in tickers if not (t.endswith(".NS") or t.endswith(".BO"))]
    if bad_tickers:
        warn(f"Non-equity tickers in panel: {bad_tickers} — should have been filtered")
    else:
        ok("All tickers are equity (.NS/.BO)")

    # Weekend / holiday rows
    biz_days = pd.bdate_range(panel.index.min(), panel.index.max())
    non_biz = panel.index.difference(biz_days)
    if len(non_biz) > 5:
        warn(f"{len(non_biz)} non-business-day rows in panel (expected some for holidays)")
    else:
        ok(f"Business-day coverage looks normal ({len(non_biz)} non-biz rows)")

    # Overall missingness
    total_cells = panel.shape[0] * panel.shape[1]
    null_pct = panel.isna().sum().sum() / total_cells * 100
    if null_pct > 85:
        warn(f"Very high overall missingness: {null_pct:.1f}% NaN "
             f"(expected — earnings data only covers ~4–5 years)")
    else:
        ok(f"Overall missingness: {null_pct:.1f}% NaN")


# ── Layer 2: Per-ticker coverage ─────────────────────────────────────────────

def _longest_null_streak(s: pd.Series) -> int:
    mask = s.isna().astype(int)
    if mask.sum() == 0:
        return 0
    # run-length encoding
    streaks = (mask != mask.shift()).cumsum()
    return int(mask.groupby(streaks).sum().max())


def layer2_coverage(
    panel: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    logger.info("=" * 65)
    logger.info("LAYER 2 — Per-ticker coverage")
    logger.info("=" * 65)

    if not isinstance(panel.columns, pd.MultiIndex):
        warn("Skipping layer 2 — not a MultiIndex panel")
        return pd.DataFrame()

    features = panel.columns.get_level_values(0).unique().tolist()
    tickers  = panel.columns.get_level_values(1).unique().tolist()

    rows = []
    for ticker in tickers:
        for feat in features:
            if (feat, ticker) not in panel.columns:
                continue
            s = panel[(feat, ticker)]
            non_null = s.dropna()
            if non_null.empty:
                first_date = last_date = None
                pct_missing = 100.0
                longest_streak = len(s)
                n_obs = 0
            else:
                first_date = non_null.index.min().date()
                last_date  = non_null.index.max().date()
                pct_missing = s.isna().mean() * 100
                longest_streak = _longest_null_streak(s)
                n_obs = len(non_null)

            rows.append({
                "ticker": ticker,
                "feature": feat,
                "first_date": first_date,
                "last_date":  last_date,
                "n_obs": n_obs,
                "pct_missing": round(pct_missing, 1),
                "longest_null_streak": longest_streak,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        warn("No coverage data computed")
        return df

    # Red flags
    # 1. Tickers where ALL features are 100% missing
    all_missing = (
        df.groupby("ticker")["pct_missing"].min() == 100.0
    )
    dead_tickers = all_missing[all_missing].index.tolist()
    if dead_tickers:
        warn(f"Tickers with 0 earnings data at all: {dead_tickers}")
    else:
        ok("All tickers have at least some earnings data for at least one feature")

    # 2. "Forward-filled forever" — detect via value-change frequency.
    # A forward-filled series changes value only when a new quarter arrives.
    # If a series hasn't changed in >18 months, the source data is stale.
    stale_cutoff = pd.Timestamp.today() - pd.DateOffset(months=18)
    stale_tickers = []
    if isinstance(panel.columns, pd.MultiIndex):
        feat = "rev_growth_yoy"
        if feat in panel.columns.get_level_values(0):
            for ticker in panel.columns.get_level_values(1).unique():
                if (feat, ticker) not in panel.columns:
                    continue
                s = panel[(feat, ticker)].dropna()
                if s.empty:
                    continue
                # Last date where the value actually changed
                changes = s[s != s.shift()]
                if changes.empty:
                    continue
                last_change = changes.index[-1]
                if last_change < stale_cutoff:
                    stale_tickers.append(
                        f"{ticker} (last update: {last_change.date()})"
                    )
    if stale_tickers:
        warn(
            f"Tickers with no earnings update in 18+ months "
            f"(forward-filled stale): {stale_tickers}"
        )
    else:
        ok("All tickers had an earnings update within the last 18 months")

    # Summary stats
    logger.info(
        "Coverage summary: median pct_missing=%.1f%%  median n_obs=%.0f",
        df["pct_missing"].median(),
        df["n_obs"].median(),
    )

    if verbose:
        print("\nPer-ticker × feature coverage:")
        print(df.sort_values(["ticker", "feature"]).to_string(index=False))

    return df


# ── Layer 3: Temporal / leakage checks ───────────────────────────────────────

def _get_listing_date(ticker: str) -> pd.Timestamp | None:
    """Best-effort listing date from yfinance info."""
    try:
        info = yf.Ticker(ticker).info
        ipo = info.get("ipoExpectedDate") or info.get("firstTradeDateEpochUtc")
        if ipo:
            return pd.Timestamp(ipo, unit="s") if isinstance(ipo, (int, float)) else pd.Timestamp(ipo)
    except Exception:
        pass
    return None


def layer3_temporal(
    panel: pd.DataFrame,
    raw_dir: Path,
    sample_tickers: list[str],
    n_events: int = 10,
) -> None:
    logger.info("=" * 65)
    logger.info("LAYER 3 — Temporal / leakage (sample: %d tickers × %d events)",
                len(sample_tickers), n_events)
    logger.info("=" * 65)

    leakage_found = False

    for ticker in sample_tickers:
        # Screener raw files use NSE symbol without .NS
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        path = raw_dir / f"{symbol}.parquet"
        if not path.exists():
            warn(f"{ticker}: no raw Screener file, skipping temporal check")
            continue

        try:
            raw = pd.read_parquet(path)
        except Exception as e:
            warn(f"{ticker}: cannot read raw file ({e})")
            continue

        raw.index = pd.to_datetime(raw.index)
        raw.sort_index(inplace=True)

        if "available_from" not in raw.columns:
            warn(f"{ticker}: raw file missing available_from column")
            continue

        if not isinstance(panel.columns, pd.MultiIndex):
            continue

        feat = "rev_growth_yoy"
        if (feat, ticker) not in panel.columns:
            continue

        panel_series = panel[(feat, ticker)]

        # Take last n_events quarter-end dates that have available_from
        check_rows = raw.dropna(subset=["available_from"]).tail(n_events)

        # Correct leakage test: the feature value must NOT CHANGE in the window
        # [q_date, available_from). Any change means new quarter data appeared early.
        # (Non-null values in that window are expected — they are the previous
        # quarter's forward-filled value, which is correct and not a leak.)
        for q_date, row in check_rows.iterrows():
            avail_from = pd.Timestamp(row["available_from"])

            prior = panel_series[panel_series.index < q_date].dropna()
            if prior.empty:
                continue
            prev_val = prior.iloc[-1]

            window = panel_series[
                (panel_series.index >= q_date) & (panel_series.index < avail_from)
            ].dropna()
            if window.empty:
                continue

            changed = window[abs(window - prev_val) > 1e-9]
            if not changed.empty:
                logger.warning(
                    "%s[LEAK]%s  %s | q_end=%s  available_from=%s  "
                    "value changed on %s (%.4f → %.4f, %d days early)",
                    RED, RESET, ticker,
                    q_date.date(), avail_from.date(),
                    changed.index[0].date(), prev_val, changed.iloc[0],
                    (avail_from - changed.index[0]).days,
                )
                leakage_found = True

    if not leakage_found:
        ok(f"No lookahead leakage detected across {len(sample_tickers)} tickers × {n_events} events")
    else:
        fail("Lookahead leakage detected — feature appears before available_from date")

    # Check for backward-fill artifacts: values changing on weekends
    for ticker in sample_tickers[:3]:
        feat = "rev_growth_yoy"
        if not isinstance(panel.columns, pd.MultiIndex):
            break
        if (feat, ticker) not in panel.columns:
            continue
        s = panel[(feat, ticker)].dropna()
        changes = s[s != s.shift()]
        if changes.empty:
            continue
        weekend_changes = changes[changes.index.dayofweek >= 5]
        if not weekend_changes.empty:
            warn(f"{ticker}: {feat} changes on weekends: {weekend_changes.index[:3].tolist()}")
        else:
            ok(f"{ticker}: no weekend value changes")


# ── Layer 4: Economic sense ───────────────────────────────────────────────────

def layer4_economic(panel: pd.DataFrame) -> None:
    logger.info("=" * 65)
    logger.info("LAYER 4 — Economic sense / feature logic")
    logger.info("=" * 65)

    if not isinstance(panel.columns, pd.MultiIndex):
        warn("Skipping layer 4 — not a MultiIndex panel")
        return

    issues = 0

    for feat in panel.columns.get_level_values(0).unique():
        feat_data = panel[feat].stack()  # long series of all (date, ticker) values
        feat_data = feat_data.dropna()

        if feat_data.empty:
            warn(f"{feat}: completely empty after dropna")
            continue

        p1, p99 = feat_data.quantile(0.01), feat_data.quantile(0.99)
        median = feat_data.median()
        logger.info("  %-30s  median=%7.3f  p1=%7.3f  p99=%7.3f  n=%d",
                    feat, median, p1, p99, len(feat_data))

        # Growth features should be centred around 0, not extreme
        if "growth" in feat:
            if abs(median) > 5.0:
                warn(f"{feat}: median growth {median:.2f} is extreme — possible unit mismatch")
                issues += 1
            if p99 > 50 or p1 < -50:
                warn(f"{feat}: growth range [{p1:.1f}, {p99:.1f}] includes >5000% — check for near-zero base")
                issues += 1
            else:
                ok(f"{feat}: growth range looks reasonable [{p1:.1f}, {p99:.1f}]")

        # Margins should be 0–1 (or 0–100 for %)
        if "margin" in feat and "chg" not in feat:
            if p99 > 5 or p1 < -2:
                warn(f"{feat}: margin values outside [-2, 5] — check for % vs decimal")
                issues += 1
            else:
                ok(f"{feat}: margin range looks reasonable [{p1:.3f}, {p99:.3f}]")

        # Margin change should be small delta
        if "margin_chg" in feat:
            if abs(p99) > 2:
                warn(f"{feat}: margin change p99={p99:.2f} — may be noisy base period")
            else:
                ok(f"{feat}: margin change range looks reasonable")

    # Cross-ticker consistency: rev_growth and earnings_growth should be correlated
    if "rev_growth_yoy" in panel.columns.get_level_values(0) and \
       "earnings_growth_yoy" in panel.columns.get_level_values(0):
        common_tickers = (
            panel["rev_growth_yoy"].columns.intersection(
                panel["earnings_growth_yoy"].columns
            )
        )
        if len(common_tickers) > 5:
            rev = panel["rev_growth_yoy"][common_tickers].stack().dropna()
            ni  = panel["earnings_growth_yoy"][common_tickers].stack().dropna()
            aligned = rev.align(ni, join="inner")
            if len(aligned[0]) > 100:
                corr = aligned[0].corr(aligned[1])
                if corr < 0:
                    warn(f"rev_growth_yoy and earnings_growth_yoy negatively correlated ({corr:.2f}) — unexpected")
                    issues += 1
                else:
                    ok(f"rev_growth_yoy × earnings_growth_yoy correlation: {corr:.2f} (positive, expected)")

    if issues == 0:
        ok("All economic sense checks passed")
    else:
        warn(f"{issues} economic-sense warnings raised (see above)")


# ── Main ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--verbose", is_flag=True, default=False, help="Print full per-ticker table")
@click.option("--sample-n", default=10, help="Number of tickers to sample for temporal checks")
@click.option("--config", default=None)
def main(verbose: bool, sample_n: int, config: str | None) -> None:
    """Run 4-layer earnings data validation."""
    cfg = load_config(config)
    setup_logging(cfg)

    panel_path = Path(cfg["paths"]["processed_data"]) / "screener_panel.parquet"
    raw_dir    = Path(cfg["paths"]["raw_data"]) / "screener"

    if not panel_path.exists():
        logger.error("Earnings panel not found at %s — run download_data.py --earnings-only first", panel_path)
        sys.exit(1)

    logger.info("Loading earnings panel from %s ...", panel_path)
    panel = pd.read_parquet(panel_path)
    logger.info("Panel loaded: %s", panel.shape)

    # Layer 1
    layer1_structural(panel)

    # Layer 2
    coverage_df = layer2_coverage(panel, verbose=verbose)
    if not coverage_df.empty:
        out = Path(cfg["paths"]["report_dir"]) / "earnings_coverage.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        coverage_df.to_csv(out, index=False)
        logger.info("Coverage report saved → %s", out)

    # Layer 3 — sample tickers
    if isinstance(panel.columns, pd.MultiIndex):
        all_tickers = panel.columns.get_level_values(1).unique().tolist()
        rng = np.random.default_rng(42)
        sample_tickers = list(rng.choice(all_tickers, size=min(sample_n, len(all_tickers)), replace=False))
        layer3_temporal(panel, raw_dir, sample_tickers)
    else:
        warn("Skipping layer 3 — panel columns are not MultiIndex")

    # Layer 4
    layer4_economic(panel)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 65)
    logger.info("%s[FAIL]%s  %d red-flag failures", RED, RESET, len(_FAILURES))
    logger.info("%s[WARN]%s  %d warnings", YELLOW, RESET, len(_WARNINGS))

    if _FAILURES:
        logger.error("Failures:")
        for f in _FAILURES:
            logger.error("  • %s", f)

    if _WARNINGS:
        logger.warning("Warnings:")
        for w in _WARNINGS:
            logger.warning("  • %s", w)

    if not _FAILURES:
        logger.info("%s[PASS]%s  All hard checks passed.", GREEN, RESET)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
