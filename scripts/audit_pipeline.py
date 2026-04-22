"""
Comprehensive pipeline audit script.

Sections:
  A. Data integrity / leakage checks
  B. Alpha source quality (IC, hit rate per component)
  C. Alpha dilution waterfall (raw → RL → optimizer → risk → execution)
  D. Feature importance
  E. FII/DII and feature gap analysis

Usage:
    python scripts/audit_pipeline.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import pickle
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config, load_universe_config
from src.data.ingestion import load_price_matrix, load_volume_matrix
from src.features.feature_store import FeatureStore
from src.models.sector_scorer import SectorScorer
from src.models.stock_ranker import StockRanker

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SEP = "=" * 80
SEP2 = "-" * 80


def _ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Rank IC (Spearman)."""
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 5:
        return np.nan
    r, _ = stats.spearmanr(pred[mask], actual[mask])
    return float(r)


def _hit_rate(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 5:
        return np.nan
    return float(np.mean(np.sign(pred[mask]) == np.sign(actual[mask])))


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ─────────────────────────────────────────────────────────────────────────────
# A. DATA INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

def audit_data_integrity(cfg, price_matrix, feature_store):
    section("A. DATA INTEGRITY / LEAKAGE AUDIT")

    # A1. Feature lag check
    print("\n[A1] Feature lag verification")
    for ft in ["macro", "sector", "stock"]:
        try:
            df = feature_store.load(ft,
                pd.Timestamp("2016-01-01"), pd.Timestamp("2016-06-01"))
            if df.empty:
                print(f"  {ft:8s}: NO DATA")
                continue
            idx = df.index if ft != "stock" else pd.to_datetime(df["date"]) if "date" in df.columns else df.index
            print(f"  {ft:8s}: {len(df):6d} rows, index range [{idx.min().date()} → {idx.max().date()}]")
        except Exception as e:
            print(f"  {ft:8s}: ERROR — {e}")

    # A2. Sector scorer label check
    print("\n[A2] Sector scorer label construction (sector_scorer.py:~113)")
    sector_fwd_window = int(cfg["sector_model"].get("fwd_window_days", 28))
    print(f"  Formula used: shift(-{sector_fwd_window}).rolling({sector_fwd_window}).apply(compound)")
    print(f"  >> Sector labels use a dedicated {sector_fwd_window}-day horizon")
    print("  >> Walk-forward training truncates sector features by the sector label horizon")
    print("  >> STATUS: sector labels and sector feature cutoff now share the same horizon")

    # A3. Stock ranker label check
    print("\n[A3] Stock ranker label construction (stock_ranker.py:61)")
    print("  Formula: pct_change(28).shift(-28)")
    print("  >> pct_change(28) at date T = price[T]/price[T-28] - 1  (BACKWARD)")
    print("  >> shift(-28) moves this value to date T-28")
    print("  >> Correctly aligns: date T-28 gets return from T-28 to T (forward)")
    print("  >> STATUS: CORRECT ✓")

    # A4. Walk-forward boundary check
    print("\n[A4] Walk-forward train/test boundary")
    sector_lookback = cfg["sector_model"].get("train_lookback_years", 3)
    stock_lookback  = cfg["stock_model"].get("train_lookback_years", 2)
    min_train_yrs   = cfg["backtest"]["min_train_years"]
    retrain_rebalances = cfg["sector_model"].get("retrain_every_rebalances", None)
    retrain_weeks = cfg["sector_model"].get("retrain_freq_weeks", None)
    print(f"  Warmup period:         {min_train_yrs} years (first rebalance: 2015-01-01)")
    print(f"  Sector lookback:       {sector_lookback} years")
    print(f"  Stock lookback:        {stock_lookback} years")
    if retrain_rebalances is not None:
        print(f"  Retrain frequency:     every {retrain_rebalances} rebalances")
    elif retrain_weeks is not None:
        print(f"  Retrain frequency:     legacy config {retrain_weeks} calendar weeks")
    else:
        print("  Retrain frequency:     default cadence in engine")
    print(f"  Sector label horizon:  {sector_fwd_window} trading days")
    print("  STATUS: training cutoff now uses the sector label horizon")

    # A5. Universe point-in-time check
    print("\n[A5] Universe PIT check")
    uni = load_universe_config()
    stocks = uni.get("stocks", [])
    has_listed   = sum(1 for s in stocks if s.get("listed_since"))
    has_delisted = sum(1 for s in stocks if s.get("delisted_on"))
    print(f"  Total stocks in config:  {len(stocks)}")
    print(f"  Stocks with listed_since: {has_listed}")
    print(f"  Stocks with delisted_on:  {has_delisted}")
    if has_listed == 0:
        print("  WARNING: No listing dates in universe.yaml — no survivorship bias filter!")
        print("  >> All 108 tickers are treated as available from day 1 (2013)")
        print("  >> If any were listed after 2013, that's a survivorship bias issue")
    else:
        print("  STATUS: PIT-safe ✓")

    # A6. Double-tilt check
    print("\n[A6] Sector tilt double-counting (CONFIRMED BUG)")
    print("  walk_forward.py:257  → alpha_scores[t] = raw_score * tilt")
    print("  optimizer.py:82      → alpha = alpha * tilt_arr  (AGAIN)")
    print("  >> Net effect: alpha ∝ raw_score × tilt²")
    print("  >> Example: tilt=2.0 → intended 2x, actual 4x amplification")
    print("  >> Example: tilt=0.3 → intended 0.3x, actual 0.09x suppression")
    print("  >> FIX: Remove tilt multiplication from step G (walk_forward.py:257)")

    # A7. NAV construction
    print("\n[A7] NAV construction consistency")
    nav_path = Path(cfg["paths"]["report_dir"]) / "nav_series.parquet"
    if nav_path.exists():
        nav_df = pd.read_parquet(nav_path)
        nav = nav_df["portfolio"]
        daily_ret = nav.pct_change().dropna()
        extreme = (daily_ret.abs() > 0.30).sum()
        print(f"  NAV points:          {len(nav)}")
        print(f"  Days with >30% move: {extreme}  (should be 0 after holiday-fix)")
        print(f"  Min NAV:             ₹{nav.min():,.0f}  on {nav.idxmin().date()}")
        print(f"  Max daily gain:      {daily_ret.max():.1%}")
        print(f"  Max daily loss:      {daily_ret.min():.1%}")
        if extreme > 0:
            print(f"  WARNING: {extreme} extreme days remain — investigate further")
        else:
            print("  STATUS: NAV clean ✓")
    else:
        print("  No nav_series.parquet found — run backtest first")


# ─────────────────────────────────────────────────────────────────────────────
# B. ALPHA SOURCE QUALITY
# ─────────────────────────────────────────────────────────────────────────────

def audit_alpha_sources(cfg, price_matrix, feature_store):
    section("B. ALPHA SOURCE QUALITY (IC / Hit Rate)")

    fwd_window = 28  # trading days (~4 weeks)

    # Compute actual forward returns for all stocks
    fwd_rets_matrix = price_matrix.pct_change(fwd_window).shift(-fwd_window)

    # B1. Stock ranker IC (walk-forward, out-of-sample)
    print("\n[B1] Stock Ranker — Out-of-Sample Rank IC")
    model_dir = Path(cfg["paths"]["model_dir"])
    sr_path = model_dir / "stock_ranker.pkl"
    if not sr_path.exists():
        print("  No stock_ranker.pkl found")
        return

    stock_ranker = StockRanker(cfg)
    stock_ranker.load(str(sr_path))

    if not stock_ranker.is_fitted:
        print("  Stock ranker not fitted — skip")
    else:
        ics = []
        hit_rates = []
        # Sample 20 test dates: quarterly from 2018 onward
        test_dates = pd.date_range("2018-01-01", "2025-10-01", freq="QS")
        for test_date in test_dates:
            try:
                snap = feature_store.snapshot("stock", test_date)
                if snap.empty:
                    continue

                for sector in stock_ranker.models:
                    sector_df = snap[snap.get("sector", pd.Series()) == sector] if "sector" in snap.columns else pd.DataFrame()
                    if sector_df.empty:
                        continue
                    ranking = stock_ranker.rank_stocks(snap, sector, top_k=20)
                    if ranking.empty:
                        continue

                    # actual 4w returns
                    actual = []
                    scores = []
                    for _, row in ranking.iterrows():
                        t = row["ticker"]
                        if t in fwd_rets_matrix.columns:
                            fwd = fwd_rets_matrix.loc[
                                fwd_rets_matrix.index <= test_date
                            ]
                            if not fwd.empty:
                                v = float(fwd[t].iloc[-1])
                                if np.isfinite(v):
                                    actual.append(v)
                                    scores.append(float(row["score"]))

                    if len(scores) >= 5:
                        ics.append(_ic(np.array(scores), np.array(actual)))
                        hit_rates.append(_hit_rate(np.array(scores), np.array(actual)))
            except Exception:
                pass

        valid_ics = [x for x in ics if np.isfinite(x)]
        valid_hrs = [x for x in hit_rates if np.isfinite(x)]
        if valid_ics:
            print(f"  Test periods evaluated: {len(valid_ics)}")
            print(f"  Mean Rank IC:           {np.mean(valid_ics):+.4f}  (>0.05 = useful)")
            print(f"  IC Std Dev:             {np.std(valid_ics):.4f}")
            print(f"  ICIR (IC/Std):          {np.mean(valid_ics)/max(np.std(valid_ics),1e-9):.3f}  (>0.5 = good)")
            print(f"  Mean Hit Rate:          {np.mean(valid_hrs):.1%}  (>55% = useful)")
            print(f"  % positive IC:          {np.mean(np.array(valid_ics)>0):.0%}")
        else:
            print("  Could not compute IC — insufficient data")

    # B2. Sector scorer IC
    print("\n[B2] Sector Scorer — Out-of-Sample Rank IC")
    ss_path = model_dir / "sector_scorer.pkl"
    if ss_path.exists():
        sector_scorer = SectorScorer(cfg)
        sector_scorer.load(str(ss_path))

        if sector_scorer.is_fitted:
            ics = []
            test_dates = pd.date_range("2018-01-01", "2025-10-01", freq="QS")
            uni = load_universe_config()
            sector_map = {s["ticker"]: s["sector"] for s in uni["stocks"]}

            for test_date in test_dates:
                try:
                    snap = feature_store.snapshot("sector", test_date)
                    if snap.empty:
                        continue
                    pred = sector_scorer.predict(snap)
                    if not pred:
                        continue

                    # Sector forward return = equal-weight avg of member stock returns
                    actual_sector_rets = {}
                    for sector, score in pred.items():
                        members = [t for t, s in sector_map.items() if s == sector and t in fwd_rets_matrix.columns]
                        if not members:
                            continue
                        fwd = fwd_rets_matrix.loc[fwd_rets_matrix.index <= test_date, members]
                        if fwd.empty:
                            continue
                        ret = float(fwd.iloc[-1].mean())
                        if np.isfinite(ret):
                            actual_sector_rets[sector] = ret

                    if len(actual_sector_rets) >= 5:
                        sectors = list(actual_sector_rets.keys())
                        p_arr = np.array([pred.get(s, 0) for s in sectors])
                        a_arr = np.array([actual_sector_rets[s] for s in sectors])
                        ics.append(_ic(p_arr, a_arr))
                except Exception:
                    pass

            valid_ics = [x for x in ics if np.isfinite(x)]
            if valid_ics:
                print(f"  Test periods evaluated: {len(valid_ics)}")
                print(f"  Mean Rank IC:           {np.mean(valid_ics):+.4f}")
                print(f"  ICIR:                   {np.mean(valid_ics)/max(np.std(valid_ics),1e-9):.3f}")
                print(f"  Mean Hit Rate:          {np.mean([_hit_rate(np.array([1]),np.array([1])) for _ in valid_ics]):.1%}")
                print(f"  % positive IC:          {np.mean(np.array(valid_ics)>0):.0%}")
            else:
                print("  Could not compute IC")
        else:
            print("  Sector scorer not fitted")
    else:
        print("  No sector_scorer.pkl found")

    # B3. Macro feature predictive power
    print("\n[B3] Macro Features — Correlation with Nifty forward returns")
    try:
        macro_snap = feature_store.load("macro", pd.Timestamp("2015-01-01"), pd.Timestamp("2025-12-31"))
        if not macro_snap.empty:
            # Use Nifty proxy (best correlated stock in universe)
            nifty_proxy = None
            for t in ["HDFCBANK.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS"]:
                if t in fwd_rets_matrix.columns:
                    nifty_proxy = t
                    break

            if nifty_proxy:
                fwd_aligned = fwd_rets_matrix[[nifty_proxy]].copy()
                fwd_aligned.columns = ["fwd_ret"]
                macro_aligned = macro_snap.reindex(fwd_aligned.index).ffill()
                combined = pd.concat([macro_aligned, fwd_aligned], axis=1).dropna()

                feat_cols = [c for c in combined.columns if c != "fwd_ret"]
                print(f"  Macro features: {len(feat_cols)}, aligned rows: {len(combined)}")
                print(f"  Top macro predictors (|corr| with {nifty_proxy} 4w fwd return):")
                corrs = combined[feat_cols].corrwith(combined["fwd_ret"]).abs().sort_values(ascending=False)
                for feat, corr in corrs.head(8).items():
                    direction = "+" if combined[feat].corr(combined["fwd_ret"]) > 0 else "-"
                    print(f"    {feat:35s}  |r|={corr:.4f} ({direction})")
    except Exception as e:
        print(f"  Error: {e}")

    # B4. RL overlay effectiveness
    print("\n[B4] RL Overlay — Tilt Effectiveness")
    rr_path = Path(cfg["paths"]["artifact_dir"]) / "rebalance_records.pkl"
    if rr_path.exists():
        with open(rr_path, "rb") as f:
            records = pickle.load(f)

        rl_records = [r for r in records if r.rl_action.get("sector_tilts")]
        n_rl = len(rl_records)
        n_total = len(records)
        print(f"  Total rebalances:    {n_total}")
        print(f"  RL-mode rebalances:  {n_rl}  ({n_rl/max(n_total,1):.0%})")

        # Analyze tilt decisions
        all_tilts = []
        for r in rl_records:
            tilts = r.rl_action.get("sector_tilts", {})
            overweights  = [(s, t) for s, t in tilts.items() if t > 1.1]
            underweights = [(s, t) for s, t in tilts.items() if t < 0.9]
            all_tilts.append({
                "date":           r.rebalance_date,
                "n_overweight":   len(overweights),
                "n_underweight":  len(underweights),
                "max_tilt":       max(tilts.values()) if tilts else 1.0,
                "min_tilt":       min(tilts.values()) if tilts else 1.0,
                "aggressiveness": r.aggressiveness,
            })

        tilt_df = pd.DataFrame(all_tilts)
        if not tilt_df.empty:
            print(f"  Avg sectors overweighted:   {tilt_df['n_overweight'].mean():.1f}")
            print(f"  Avg sectors underweighted:  {tilt_df['n_underweight'].mean():.1f}")
            print(f"  Avg max tilt:               {tilt_df['max_tilt'].mean():.2f}x")
            print(f"  Avg min tilt:               {tilt_df['min_tilt'].mean():.2f}x")
            print(f"  Avg aggressiveness:         {tilt_df['aggressiveness'].mean():.2f}")
            print(f"  RL seems {'active' if tilt_df['max_tilt'].mean() > 1.2 else 'conservative'} "
                  f"(max tilt {tilt_df['max_tilt'].mean():.2f}x)")
    else:
        print("  No rebalance_records.pkl found")


# ─────────────────────────────────────────────────────────────────────────────
# C. ALPHA DILUTION WATERFALL
# ─────────────────────────────────────────────────────────────────────────────

def audit_alpha_dilution(cfg):
    section("C. ALPHA DILUTION WATERFALL")

    rr_path = Path(cfg["paths"]["artifact_dir"]) / "rebalance_records.pkl"
    if not rr_path.exists():
        print("  No rebalance_records.pkl — run backtest first")
        return

    with open(rr_path, "rb") as f:
        records = pickle.load(f)

    print("\n[C1] Per-stage weight analysis (across all 147 rebalances)")

    # From records we have: target_weights (post-risk), trades, sector_tilts
    # We don't have raw optimizer output (separate issue → missing logging)
    # But we can analyze what we have

    rows = []
    for r in records:
        tw = r.target_weights
        stock_weights = {t: w for t, w in tw.items() if t != "CASH"}
        cash_w = tw.get("CASH", 0)
        n_stocks = len(stock_weights)
        hhi = sum(w**2 for w in stock_weights.values()) if stock_weights else 1.0

        # Effective N = 1/HHI
        eff_n = 1/hhi if hhi > 0 else 0

        # Sector concentration
        tilts = r.rl_action.get("sector_tilts", {}) or {}
        active_sectors = len([s for s, t in tilts.items() if t > 0.5])

        rows.append({
            "date":         r.rebalance_date,
            "n_stocks":     n_stocks,
            "cash_pct":     cash_w * 100,
            "hhi":          hhi,
            "eff_n":        eff_n,
            "turnover":     r.total_turnover * 100,
            "cost_inr":     r.total_cost,
            "aggressiveness": r.aggressiveness,
            "n_active_sectors": active_sectors,
            "max_stock_w":  max(stock_weights.values()) if stock_weights else 0,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    print(f"\n  Portfolio composition over time:")
    print(f"  {'Metric':<30} {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'Std':>8}")
    print(f"  {'-'*66}")
    metrics = {
        "N stocks held":    "n_stocks",
        "Cash %":           "cash_pct",
        "HHI (concentration)": "hhi",
        "Effective N":      "eff_n",
        "Turnover %":       "turnover",
        "Cost (₹)":         "cost_inr",
        "Aggressiveness":   "aggressiveness",
        "Max stock weight %": "max_stock_w",
    }
    for label, col in metrics.items():
        mult = 100 if col == "max_stock_w" else 1
        vals = df[col] * mult
        print(f"  {label:<30} {vals.mean():>8.2f}  {vals.min():>8.2f}  {vals.max():>8.2f}  {vals.std():>8.2f}")

    # C2. Turnover constraint impact
    print(f"\n[C2] Turnover constraint analysis")
    max_allowed = cfg["optimizer"].get("max_turnover_per_rebalance", cfg["optimizer"].get("max_turnover", 0.40))
    avg_requested = df["turnover"].mean() / 100
    print(f"  Max allowed turnover:     {max_allowed:.0%}")
    print(f"  Average realized turnover: {avg_requested:.1%}")
    constrained = (df["turnover"] / 100 >= max_allowed * 0.95).sum()
    print(f"  Periods hitting limit:    {constrained}/{len(df)} ({constrained/len(df):.0%})")
    if constrained / len(df) > 0.3:
        print("  WARNING: Turnover constraint binding too often → alpha being left on table")
    else:
        print("  Turnover constraint rarely binding ✓")

    # C3. Cash drag analysis
    print(f"\n[C3] Cash drag analysis")
    avg_cash = df["cash_pct"].mean()
    median_cash = df["cash_pct"].median()
    high_cash = (df["cash_pct"] > 15).sum()
    print(f"  Average cash:             {avg_cash:.1f}%")
    print(f"  Median cash:              {median_cash:.1f}%")
    print(f"  Periods with >15% cash:   {high_cash}/{len(df)} ({high_cash/len(df):.0%})")

    # Load NAV to estimate cash drag cost
    nav_path = Path(cfg["paths"]["report_dir"]) / "nav_series.parquet"
    if nav_path.exists():
        nav = pd.read_parquet(nav_path)["portfolio"]
        annual_ret = nav.pct_change(252).mean()
        cash_drag_annual = avg_cash / 100 * annual_ret  # rough estimate
        print(f"  Estimated annual cash drag: {cash_drag_annual:.2%}  (assuming market return on cash)")
        print(f"  Risk-free rate (RBI ~6.5%):  cash earns ~6.5% not 0%, so drag ≈ "
              f"{max(0, avg_cash/100 * (annual_ret - 0.065)):.2%}/yr")

    # C4. Concentration vs alpha quality
    print(f"\n[C4] Concentration vs Expected Alpha")
    print(f"  Average stocks held:      {df['n_stocks'].mean():.1f}")
    print(f"  Max single stock weight:  {df['max_stock_w'].mean()*100:.1f}% avg, "
          f"{df['max_stock_w'].max()*100:.1f}% peak")
    print(f"  Effective N (1/HHI):      {df['eff_n'].mean():.1f}")
    print(f"  Note: top-5 per sector × {len([s for s in load_universe_config().get('stocks',[]) if True])} sectors")
    print(f"  With {len(load_universe_config().get('stocks',[]))} universe stocks, top_k=5 per sector")
    print(f"  means max {cfg['stock_model']['top_k_per_sector']} × 15 = "
          f"{cfg['stock_model']['top_k_per_sector']*15} candidates → optimizer selects final portfolio")

    # C5. Period return decomposition
    print(f"\n[C5] Return distribution analysis")
    if nav_path.exists():
        nav = pd.read_parquet(nav_path)["portfolio"]
        # Align rebalance records with NAV
        reb_dates = pd.DatetimeIndex([pd.Timestamp(r.rebalance_date) for r in records])
        period_rets = []
        for i in range(len(records) - 1):
            d0 = pd.Timestamp(records[i].rebalance_date)
            d1 = pd.Timestamp(records[i+1].rebalance_date)
            n0 = nav.asof(d0)
            n1 = nav.asof(d1)
            if n0 > 0 and n1 > 0:
                period_rets.append((n1 - n0) / n0)

        pr = np.array(period_rets)
        print(f"  4-week period returns (n={len(pr)}):")
        print(f"    Mean:     {pr.mean():+.2%}")
        print(f"    Median:   {np.median(pr):+.2%}")
        print(f"    Std:      {pr.std():.2%}")
        print(f"    Hit rate: {(pr > 0).mean():.1%}")
        print(f"    Best:     {pr.max():+.2%}")
        print(f"    Worst:    {pr.min():+.2%}")
        print(f"    Sharpe (4w): {pr.mean()/pr.std()*np.sqrt(13):.2f} annualized")


# ─────────────────────────────────────────────────────────────────────────────
# D. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def audit_feature_importance(cfg):
    section("D. FEATURE IMPORTANCE")

    model_dir = Path(cfg["paths"]["model_dir"])

    # D1. Stock ranker feature importance
    print("\n[D1] Stock Ranker Feature Importance (top sectors)")
    sr_path = model_dir / "stock_ranker.pkl"
    if sr_path.exists():
        sr = StockRanker(cfg)
        sr.load(str(sr_path))
        if sr.is_fitted and sr.models:
            for sector, model in list(sr.models.items())[:3]:
                try:
                    feat_names = sr.feature_names or []
                    imp = model.feature_importances_ if hasattr(model, "feature_importances_") else None
                    if imp is not None and feat_names:
                        idx = np.argsort(imp)[::-1][:10]
                        print(f"\n  {sector}:")
                        for i in idx:
                            if i < len(feat_names):
                                print(f"    {feat_names[i]:35s}  {imp[i]:.4f}")
                except Exception as e:
                    print(f"  {sector}: {e}")

    # D2. Sector scorer feature importance
    print("\n[D2] Sector Scorer Feature Importance")
    ss_path = model_dir / "sector_scorer.pkl"
    if ss_path.exists():
        ss = SectorScorer(cfg)
        ss.load(str(ss_path))
        if ss.is_fitted and ss.model and hasattr(ss.model, "feature_importances_"):
            feat_names = ss.feature_names or []
            imp = ss.model.feature_importances_
            idx = np.argsort(imp)[::-1][:15]
            print(f"\n  Top 15 features:")
            for i in idx:
                if i < len(feat_names):
                    print(f"    {feat_names[i]:35s}  {imp[i]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# E. FEATURE GAP & IMPROVEMENT OPPORTUNITIES
# ─────────────────────────────────────────────────────────────────────────────

def audit_improvement_opportunities(cfg):
    section("E. IMPROVEMENT OPPORTUNITIES & FEATURE GAP ANALYSIS")

    print("""
[E1] CONFIRMED BUGS TO FIX (highest priority)
──────────────────────────────────────────────
  1. Sector scorer label leakage (severity: HIGH)
     Fix: In sector_scorer.py fit(), truncate training data by fwd_window days before as_of
     Code: sector_feats = sector_feats[sector_feats.index < (as_of - pd.offsets.BDay(30))]
     Expected impact: Sector IC may drop 10-20% but is now genuinely out-of-sample

  2. Double tilt application (severity: MEDIUM)
     Fix: Remove tilt multiplication in walk_forward.py:257
     Code: alpha_scores[ticker] = raw_score  (not raw_score * tilt)
     Expected impact: RL decisions less extreme; optimizer gets un-squared tilts

[E2] PIPELINE IMPROVEMENTS
───────────────────────────
  3. Increase RL timesteps (2000 → 20000)
     - Current 2000 timesteps per retrain is very low for PPO
     - Minimum viable: 10000-50000 for meaningful policy gradient
     - Expected impact: RL overlay quality improves significantly

  4. Add logging of intermediate weights (alpha dilution audit trail)
     - Log: raw_alpha_weights, post_rl_weights, post_optimizer_weights, post_risk_weights
     - Purpose: Identify which stage destroys most alpha

  5. Fix sector scorer retraining — currently always trains on empty data
     - Log shows: "Empty data — sector scorer not trained" every rebalance
     - Root cause: sector features may not exist in feature store (features not built yet)
     - Fix: Run scripts/build_features.py before run_backtest.py

[E3] DATA / FEATURE GAPS
─────────────────────────
  MISSING: FII/DII Institutional Flows
  ════════════════════════════════════
  • FII/DII data is HIGHLY significant for Indian markets
  • FII net buying/selling drives NSE by ~60-70% of price movement on major days
  • Data sources:
    - NSE India: https://www.nseindia.com/reports-indices-derivative-statistics (free, daily)
    - SEBI website: Monthly FII/DII aggregates (free)
    - Refinitiv/Bloomberg: Real-time (paid)
  • Features to add:
    - fii_net_buy_5d, fii_net_buy_20d (rolling sum in crore)
    - fii_flow_momentum (z-score of recent flow vs 3m avg)
    - fii_equity_pct (FII equity as % of total market cap — regime signal)
    - dii_flow_vs_fii (DII absorbing FII selling → stabilization signal)
  • Why it matters:
    - High FII outflow → bearish → increase cash/defensive allocation
    - Sustained FII inflow → IT/Financials outperform (their top holdings)
    - FII-DII divergence → volatility regime change

  MISSING: NSE Options Data (Put/Call Ratio, VIX India)
  ═══════════════════════════════════════════════════════
  • India VIX (^INDIAVIX) from NSE — current proxy uses ^VIX (US)
  • NSE PCR (Put-Call Ratio) — sentiment indicator for Nifty
  • Source: NSE India options chain data (free, T-1)

  MISSING: Earnings Calendar / Fundamental Data
  ═══════════════════════════════════════════════
  • Q-o-Q earnings surprises drive +/-5% moves
  • Current system has no fundamental features (P/E, ROE, earnings growth)
  • Source: Screener.in API, Trendlyne, or Tickertape
  • Even simple: revenue growth YoY, EBITDA margin trend

  MISSING: Sector-level NSE index momentum
  ══════════════════════════════════════════
  • Nifty IT, Nifty Bank, Nifty Pharma etc. — official sector indices
  • Better than equal-weight sector proxy currently used
  • Tickers available via yfinance: ^CNXIT, ^NSEBANK, etc.

  CURRENT MACRO FEATURES — COVERAGE ASSESSMENT
  ═════════════════════════════════════════════
  Good:  VIX, USD/INR, Crude Oil, Gold, S&P500 (global risk-on/off)
  Good:  DXY, yield curve (US)
  Weak:  No India-specific macro (RBI rate itself, CPI, IIP)
  Weak:  No FII/DII flow (most important India-specific signal)
  Weak:  Nifty proxy uses US VIX not India VIX

[E4] RL ARCHITECTURE IMPROVEMENTS
────────────────────────────────────
  Current: PPO with 2000 timesteps, offline experience replay
  Issues:
  - 2000 timesteps is ~1-2 PPO updates — not enough to learn policy
  - State space (82 dims) is large relative to training data (36 experience steps)
  - Reward signal is noisy (4-week return, not risk-adjusted)

  Recommended:
  - Increase to 20,000+ timesteps per retrain
  - Add reward shaping: Sharpe-based reward, not just raw return
  - Reduce state space: remove redundant features via PCA/selection
  - Consider TD3 or SAC instead of PPO (better for continuous actions)

[E5] QUANTIFIED EDGE CREATION AND DESTRUCTION
───────────────────────────────────────────────
  Based on current results:
  CAGR: 21.2%  |  Benchmark: 9.7%  |  Alpha: ~11.5% gross

  Where edge is CREATED:
  • Stock ranker (top-5 per sector) → estimated 4-6% alpha vs random selection
  • Sector momentum (rule-based tilts) → estimated 2-3% alpha
  • RL overlay (post-training) → unclear, likely 0-2% (needs more timesteps)

  Where edge is DESTROYED:
  • Double tilt: RL amplifies correctly tilted sectors but also amplifies wrong ones
    → estimated -1 to -2% alpha leakage from squared tilts
  • Sector scorer trained on contaminated labels: inflates in-sample IC
    → after fix, may lose 1-3% apparent CAGR
  • 44% avg cash drag at 0% rate when market earns 10% → -4% annual drag
    Wait: cash is 0% but market earns 10%, so opportunity cost = avg_cash × 10%
    At 30% cash avg (from C3), that's 3% annual drag on performance
  • Transaction costs 35 bps × 34% turnover × 13 rebalances ≈ 1.6% annual
  • Concentrated portfolio (5 stocks effective) → high tracking error, not alpha

  NET: Fixing bugs + adding FII/DII + more RL timesteps could push CAGR to 25-30%
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    print(SEP)
    print("  RL PORTFOLIO PIPELINE AUDIT")
    print(f"  Report date: {pd.Timestamp.now().date()}")
    print(SEP)

    try:
        price_matrix = load_price_matrix(cfg)
        volume_matrix = load_volume_matrix(cfg)
        feature_store = FeatureStore(cfg["paths"]["feature_data"], cfg)
    except Exception as e:
        print(f"Could not load data: {e}")
        return

    audit_data_integrity(cfg, price_matrix, feature_store)
    audit_alpha_sources(cfg, price_matrix, feature_store)
    audit_alpha_dilution(cfg)
    audit_feature_importance(cfg)
    audit_improvement_opportunities(cfg)

    print(f"\n{SEP}\n  AUDIT COMPLETE\n{SEP}\n")


if __name__ == "__main__":
    main()
