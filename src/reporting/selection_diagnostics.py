"""Selection diagnostics for stock-picking quality analysis."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _get_field(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _coerce_returns(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        out: dict[str, float] = {}
        for key, raw in value.items():
            if raw is None:
                continue
            try:
                out[str(key)] = float(raw)
            except (TypeError, ValueError):
                continue
        return out
    return {}


def _coerce_sector_returns(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, dict[str, float]] = {}
    for sector, raw in value.items():
        name = str(sector)
        if isinstance(raw, Mapping):
            out[name] = _coerce_returns(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            seq_map = {}
            for idx, item in enumerate(raw):
                try:
                    seq_map[str(idx)] = float(item)
                except (TypeError, ValueError):
                    continue
            out[name] = seq_map
    return out


def _normalize_selected_items(record: Any) -> list[dict[str, Any]]:
    selected = _get_field(record, "selected_stocks", "selected_names", default=[])
    raw_scores = _coerce_returns(
        _get_field(record, "raw_stock_scores", "stock_scores", default={})
    )
    next_returns = _coerce_returns(
        _get_field(
            record,
            "next_period_returns",
            "realized_returns",
            "forward_returns",
            default={},
        )
    )
    items: list[dict[str, Any]] = []
    if isinstance(selected, Mapping):
        iterator = selected.items()
    elif isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
        iterator = enumerate(selected)
    else:
        iterator = []

    for key, raw in iterator:
        if isinstance(raw, Mapping):
            ticker = raw.get("ticker") or raw.get("symbol") or raw.get("name") or key
            sector = raw.get("sector")
            score = raw.get("score", raw_scores.get(str(ticker)))
            fwd_return = raw.get(
                "forward_return",
                raw.get("next_period_return", next_returns.get(str(ticker))),
            )
        else:
            ticker = raw if not isinstance(raw, tuple) else raw[0]
            sector = None
            score = raw_scores.get(str(ticker))
            fwd_return = next_returns.get(str(ticker))

        if ticker is None:
            continue
        item = {"ticker": str(ticker), "sector": sector}
        item["score"] = _safe_float(score)
        item["forward_return"] = _safe_float(fwd_return)
        items.append(item)
    return items


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _compute_rank_ic(scores: dict[str, float], returns: dict[str, float]) -> float | None:
    common = sorted(set(scores) & set(returns))
    if len(common) < 2:
        return None
    score_series = pd.Series({ticker: scores[ticker] for ticker in common})
    return_series = pd.Series({ticker: returns[ticker] for ticker in common})
    ic = score_series.corr(return_series, method="spearman")
    return None if pd.isna(ic) else float(ic)


def _sector_dispersion(values: Sequence[float]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(finite) < 2:
        return None
    return float(np.std(finite, ddof=0))


def _top_bottom_spread(
    scores: dict[str, float],
    returns: dict[str, float],
    k: int,
) -> float | None:
    common = [
        (ticker, scores[ticker], returns[ticker])
        for ticker in sorted(set(scores) & set(returns))
        if scores.get(ticker) is not None and returns.get(ticker) is not None
    ]
    if len(common) < 2:
        return None
    k = max(1, min(int(k), len(common) // 2 if len(common) > 2 else 1))
    ordered = sorted(common, key=lambda item: (-item[1], item[0]))
    top = [row[2] for row in ordered[:k]]
    bottom = [row[2] for row in ordered[-k:]]
    if not top or not bottom:
        return None
    return float(np.mean(top) - np.mean(bottom))


def _within_sector_local_metrics(
    scores: dict[str, float],
    sectors: dict[str, str],
    sector_returns: dict[str, dict[str, float]],
    k: int = 5,
) -> dict[str, float | None]:
    per_sector: list[dict[str, float]] = []
    for sector, ret_map in sector_returns.items():
        common = [
            (ticker, scores[ticker], ret_map[ticker])
            for ticker in sorted(set(scores) & set(ret_map))
            if sectors.get(ticker) == sector
        ]
        if len(common) < 2:
            continue

        ordered = sorted(common, key=lambda item: (-item[1], item[0]))
        kk = max(1, min(int(k), len(common) // 2 if len(common) > 2 else 1))
        top = [row[2] for row in ordered[:kk]]
        bottom = [row[2] for row in ordered[-kk:]]
        if not top or not bottom:
            continue

        score_series = pd.Series([row[1] for row in common])
        return_series = pd.Series([row[2] for row in common])
        ic = score_series.corr(return_series, method="spearman")
        if pd.isna(ic):
            continue

        sector_rets = [row[2] for row in common]
        per_sector.append(
            {
                "sector": sector,
                "count": float(len(common)),
                "ic": float(ic),
                "top_bottom_spread": float(np.mean(top) - np.mean(bottom)),
                "top_k_minus_sector_median": float(np.mean(top) - float(np.median(sector_rets))),
            }
        )

    if not per_sector:
        return {
            "within_sector_ic": None,
            "within_sector_ic_weighted": None,
            "within_sector_top_bottom_spread": None,
            "within_sector_top_bottom_spread_weighted": None,
            "within_sector_top_k_minus_sector_median": None,
            "within_sector_top_k_minus_sector_median_weighted": None,
        }

    frame = pd.DataFrame(per_sector)
    weights = frame["count"].to_numpy()
    return {
        "within_sector_ic": float(frame["ic"].mean()),
        "within_sector_ic_weighted": float(np.average(frame["ic"], weights=weights)),
        "within_sector_top_bottom_spread": float(frame["top_bottom_spread"].mean()),
        "within_sector_top_bottom_spread_weighted": float(
            np.average(frame["top_bottom_spread"], weights=weights)
        ),
        "within_sector_top_k_minus_sector_median": float(frame["top_k_minus_sector_median"].mean()),
        "within_sector_top_k_minus_sector_median_weighted": float(
            np.average(frame["top_k_minus_sector_median"], weights=weights)
        ),
    }


def _extract_period_metrics(record: Any, previous_selected: set[str] | None) -> dict[str, Any]:
    selected_items = _normalize_selected_items(record)
    selected_tickers = [item["ticker"] for item in selected_items]
    selected_set = set(selected_tickers)
    selected_returns = [
        item["forward_return"] for item in selected_items if item["forward_return"] is not None
    ]
    selected_scores = {
        item["ticker"]: item["score"] for item in selected_items if item["score"] is not None
    }
    selected_returns_map = {
        item["ticker"]: item["forward_return"]
        for item in selected_items
        if item["forward_return"] is not None
    }

    universe_returns = _coerce_returns(
        _get_field(record, "universe_forward_returns", "candidate_returns", default={})
    )
    if not universe_returns:
        universe_returns = _coerce_returns(
            _get_field(record, "forward_returns_all", "all_forward_returns", default={})
        )

    sector_returns = _coerce_sector_returns(
        _get_field(record, "sector_forward_returns", "sector_returns", default={})
    )
    candidate_scores = _coerce_returns(
        _get_field(
            record,
            "candidate_stock_scores",
            "universe_stock_scores",
            "stock_scores",
            default={},
        )
    )
    candidate_sectors_raw = _get_field(
        record, "candidate_stock_sectors", "universe_stock_sectors", default={}
    )
    candidate_sectors = (
        {str(k): str(v) for k, v in candidate_sectors_raw.items() if k is not None and v is not None}
        if isinstance(candidate_sectors_raw, Mapping)
        else {}
    )

    excess_vs_sector: list[float] = []
    sector_dispersions: list[float] = []
    for item in selected_items:
        if item["forward_return"] is None or not item["sector"]:
            continue
        sector_map = sector_returns.get(str(item["sector"]), {})
        sector_values = [ret for ret in sector_map.values() if ret is not None]
        if not sector_values:
            continue
        excess_vs_sector.append(item["forward_return"] - float(np.median(sector_values)))
        dispersion = _sector_dispersion(sector_values)
        if dispersion is not None:
            sector_dispersions.append(dispersion)

    stability = None
    if previous_selected is not None:
        union = selected_set | previous_selected
        stability = float(len(selected_set & previous_selected) / len(union)) if union else 1.0

    universe_avg = _mean_or_none(list(universe_returns.values()))
    top_k_avg = _mean_or_none(selected_returns)
    top_bottom_spread = _top_bottom_spread(
        selected_scores,
        universe_returns or selected_returns_map,
        len(selected_returns) if selected_returns else len(selected_scores),
    )
    within_sector = _within_sector_local_metrics(
        candidate_scores or selected_scores,
        candidate_sectors,
        sector_returns,
    )

    return {
        "date": str(_get_field(record, "rebalance_date", "date", default="")),
        "selected_count": len(selected_tickers),
        "selected_stocks": "|".join(selected_tickers),
        "top_k_avg_forward_return": top_k_avg,
        "universe_avg_forward_return": universe_avg,
        "top_k_minus_universe": (
            top_k_avg - universe_avg if top_k_avg is not None and universe_avg is not None else None
        ),
        "top_k_minus_sector_median": _mean_or_none(excess_vs_sector),
        "intra_sector_dispersion": _mean_or_none(sector_dispersions),
        "top_bottom_spread": top_bottom_spread,
        "precision_at_k": (
            float(np.mean([ret > 0 for ret in selected_returns])) if selected_returns else None
        ),
        "rank_ic": _compute_rank_ic(selected_scores, universe_returns or selected_returns_map),
        "stability": stability,
        **within_sector,
    }, selected_set


def compute_selection_diagnostics(records: Sequence[Any] | None) -> dict[str, Any] | None:
    """Compute aggregate and per-period stock-selection diagnostics."""
    if not records:
        return None

    rows: list[dict[str, Any]] = []
    previous_selected: set[str] | None = None
    for record in records:
        row, previous_selected = _extract_period_metrics(record, previous_selected)
        rows.append(row)

    if not rows:
        return None

    frame = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "periods": int(len(frame)),
        "avg_selected_count": _safe_float(frame["selected_count"].mean()),
    }
    for column in [
        "top_k_avg_forward_return",
        "universe_avg_forward_return",
        "top_k_minus_universe",
        "top_k_minus_sector_median",
        "intra_sector_dispersion",
        "top_bottom_spread",
        "precision_at_k",
        "rank_ic",
        "stability",
        "within_sector_ic",
        "within_sector_ic_weighted",
        "within_sector_top_bottom_spread",
        "within_sector_top_bottom_spread_weighted",
        "within_sector_top_k_minus_sector_median",
        "within_sector_top_k_minus_sector_median_weighted",
    ]:
        if column in frame:
            value = frame[column].dropna().mean()
            summary[column] = None if pd.isna(value) else float(value)

    # IC stability metrics for horizon-shift gate
    if "within_sector_ic" in frame:
        ic_series = frame["within_sector_ic"].dropna()
        if len(ic_series) > 1:
            summary["within_sector_ic_std"] = float(ic_series.std())
            summary["within_sector_ic_positive_fraction"] = float((ic_series > 0).mean())
        if "date" in frame:
            frame["year"] = pd.to_datetime(frame["date"], errors="coerce").dt.year
            by_year = (
                frame.dropna(subset=["year", "within_sector_ic"])
                .groupby("year")["within_sector_ic"]
                .mean()
            )
            summary["within_sector_ic_by_year"] = {
                int(yr): round(float(ic), 6) for yr, ic in by_year.items()
            }

    return {
        "summary": summary,
        "per_rebalance": frame,
    }


def prepare_selection_diagnostics(selection_diagnostics: Any) -> dict[str, Any] | None:
    """Normalize diagnostics input from raw records or precomputed payloads."""
    if selection_diagnostics is None:
        return None

    if isinstance(selection_diagnostics, Mapping):
        summary = selection_diagnostics.get("summary")
        per_rebalance = selection_diagnostics.get("per_rebalance")
        if summary is not None or per_rebalance is not None:
            result: dict[str, Any] = {}
            if summary is not None:
                result["summary"] = dict(summary)
            if per_rebalance is not None:
                result["per_rebalance"] = (
                    per_rebalance.copy()
                    if isinstance(per_rebalance, pd.DataFrame)
                    else pd.DataFrame(per_rebalance)
                )
            return result

    if isinstance(selection_diagnostics, Sequence) and not isinstance(
        selection_diagnostics, (str, bytes)
    ):
        return compute_selection_diagnostics(selection_diagnostics)

    return None
