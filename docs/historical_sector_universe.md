# Historical 10-Year Sector Union Universe

## Goal
Build a survivorship-aware sector universe where each sector keeps the union of stocks that were relevant at any point in the trailing 10 years.

## What gets built
- `sector_historical_master.parquet/csv`: ticker-sector master with:
  - `ticker`, `sector`, `active_from`, `active_to`, `added_on`, `source`, `derivation_note`, `inclusion_reason`
- `sector_union_universe_10y.parquet/csv/json`: only tickers that passed relevance filters
- `historical_universe_diagnostics.md`: summary + validation checks

Default output directory:
- `data/processed/universe/historical_union_10y`

## Relevance rule (Layer A)
A stock enters the 10-year union if, within the lookback window, it satisfies at least one:
- market-cap proxy rank threshold (proxy uses rolling traded-value rank)
- liquidity threshold (`min_median_traded_value_cr`)
- broad index presence (optional file input)

This layer is slow-changing and is separate from alpha ranking.

## Rebalance-time eligibility (Layer B)
At rebalance date `t`, candidates are filtered by:
- membership in sector union
- `added_on <= t`
- enough price history by `t`
- active window (`active_from/active_to`) when enabled
- minimum median traded value near `t`

## How to run builder
```bash
./.venv/bin/python scripts/build_historical_sector_universe.py
```

Optional:
```bash
./.venv/bin/python scripts/build_historical_sector_universe.py --as-of 2026-04-17
```

## How to enable in backtests
In `config/base.yaml`:
- set `universe.mode: historical_union_10y`
- keep `universe.mode: static` for baseline behavior

No silent baseline change: static mode remains default.

## Key approximations
- True historical market-cap is approximated via traded-value rank when full cap history is unavailable.
- Historical sector mapping uses current mapping unless an override map is provided via:
  - `universe.historical_union.candidate_sector_map_file`
- Active windows default to price-availability inference if explicit listing/delisting dates are missing.
