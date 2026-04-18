# Feature Update Flow — Strategy & Protocol

## Why this document exists

In run_011 the backtest silently trained on a 36-column feature set while
the builder was producing 44 columns.  The feature store's `_metadata.json`
claimed data was current, so no rebuild was triggered, and the new features
were never used.  This document defines the canonical flow to prevent that.

---

## How the feature store works

```
artifacts/feature_store/
  _metadata.json          ← per-type: last_date + schema_hash
  macro/year=YYYY/        ← ~3 500 rows × ~60 cols per shard
  sector/year=YYYY/       ← ~55 000 rows × ~30 cols (one row per sector per day)
  stock/year=YYYY/month=MM/ ← ~30 000 rows × ~44 cols per shard
```

`build_or_append()` is called once at backtest startup.  For each of the
three feature types it checks `is_fresh()` before doing any work:

```
is_fresh(ft, as_of) = True   iff ALL three conditions hold:
  1. metadata.last_date >= as_of          (date coverage)
  2. at least one .parquet shard exists   (disk guard)
  3. metadata.schema_hash == hash(current shard columns)  (schema guard)
```

If any condition fails the entire type is rebuilt from scratch.

---

## Two failure modes that are now handled

### Failure mode A — manual shard deletion without clearing metadata

**What used to happen:** delete `artifacts/feature_store/stock/`, run backtest,
see "FeatureStore[stock] up-to-date — skipping".  Model trains on nothing.

**Fix (disk guard):** `is_fresh()` now calls `_shards_exist()`.  If no parquet
files are found it logs a warning, clears `last_date` from metadata, and
returns `False`.  The store rebuilds.

### Failure mode B — new features added to builder, old shards still cached

**What used to happen:** add `rsi_14`, `ma_50_200_ratio` etc. to
`StockFeatureBuilder.build()`, run backtest.  Store saw current `last_date`,
said "up-to-date", served 36-column shards.  LightGBM ranker never saw the
new columns.

**Fix (schema guard):** after every build `_append_stock/sector/macro` stores:

```json
"stock": {
  "last_date": "2026-04-17",
  "schema_hash": "a3f8c21d"          ← MD5 of sorted column list
}
```

On the next `is_fresh()` call the hash of the most recent shard's columns is
recomputed and compared.  Any mismatch triggers `invalidate(ft)` which removes
the shard directory and clears metadata — forcing a clean rebuild.

---

## The canonical flow for adding or changing features

### Step 1 — edit the builder

Change `src/features/stock_features.py` or `src/features/sector_features.py`.

### Step 2 — invalidate the affected feature type

```python
# Quick one-liner from a script or REPL:
from src.features.feature_store import FeatureStore
from src.config import load_config
cfg = load_config()
store = FeatureStore("artifacts/feature_store", cfg)
store.invalidate("stock")   # or "sector", "macro"
```

Or equivalently, just run:

```bash
python3 -c "
from src.features.feature_store import FeatureStore
from src.config import load_config
FeatureStore('artifacts/feature_store', load_config()).invalidate('stock')
"
```

The next `run_backtest.py` will automatically detect the missing shards and
rebuild with the new columns.

> **Note:** you do NOT need to touch `_metadata.json` manually.  The
> `invalidate()` method handles deletion + metadata clearing atomically.

### Step 3 — verify the rebuild happened

After the next backtest run, confirm the log shows:

```
FeatureStore[stock] computing 2013-01-01 → 2026-04-17
Stock features: (365540, 44) → ...         ← row + column count
FeatureStore[stock] persisted 340060 rows (schema=a3f8c21d)
```

And spot-check that new columns are present:

```bash
python3 -c "
import pandas as pd, pathlib
shard = sorted(pathlib.Path('artifacts/feature_store/stock').rglob('*.parquet'))[-1]
df = pd.read_parquet(shard)
print(sorted(df.columns.tolist()))
"
```

### Step 4 — run the backtest and record results in NEXT_STEPS.md

---

## Feature type rebuild times (approximate)

| Type   | Rows rebuilt | Wall time |
|--------|-------------|-----------|
| macro  | ~3 500      | ~0.3 s    |
| sector | ~56 000     | ~0.7 s    |
| stock  | ~365 000    | ~70 s     |

Stock rebuild is the expensive one — it runs the full price ffill + rolling
windows + sector-relative calculations over the entire 13-year history.

---

## What triggers a rebuild automatically vs manually

| Trigger | Auto-detected? | Action needed |
|---------|----------------|---------------|
| New columns added to builder | ✅ Yes (schema hash mismatch) | None — rebuild fires automatically |
| Shard files deleted manually | ✅ Yes (disk guard) | None — rebuild fires automatically |
| Column *renamed* (same count) | ✅ Yes (hash of names changes) | None |
| Column *values* changed (same names) | ❌ No | Call `invalidate(ft)` manually |
| Rolling-window length changed | ❌ No | Call `invalidate(ft)` manually |
| Bug fix in rolling calculation | ❌ No | Call `invalidate(ft)` manually |
| New price data downloaded | ✅ Partial — incremental append only adds new dates | Full history unchanged |

**Rule of thumb:** if you change *how* a feature is computed (not just *which*
features exist), call `invalidate()` before the next backtest.  If you only
add new features, the schema hash catches it automatically.

---

## Metadata format reference

```json
{
  "macro":  { "last_date": "2026-04-17", "schema_hash": "b1e4d9f2" },
  "sector": { "last_date": "2026-04-17", "schema_hash": "c7a3e01b" },
  "stock":  { "last_date": "2026-04-17", "schema_hash": "a3f8c21d" }
}
```

Never edit `_metadata.json` by hand.  Use `store.invalidate(ft)` or let the
auto-detection handle it.
