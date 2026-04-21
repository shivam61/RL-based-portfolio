# Stock Ranker Experiment Protocol

Use this checklist for every stock-ranker experiment. The goal is to avoid stale
feature-store shards, retrain the ranker on the exact feature set being tested,
and refresh all analysis artifacts after the run completes.

## 1. Freeze the control surface

Before changing anything, lock the experiment scope:

- keep the universe fixed
- keep taxonomy fixed unless the experiment is explicitly about taxonomy
- keep RL and optimizer settings unchanged unless the experiment is about those layers
- change one feature block at a time

Do not mix universe expansion, taxonomy changes, and feature changes in the same
run. That makes attribution ambiguous.

## 2. Invalidate the old stock feature cache

If the stock feature code, feature family set, or stock universe changed, invalidate
the cached stock feature shard before retraining:

- call `FeatureStore.invalidate("stock")`
- then rebuild features so the stock shard is regenerated from the current
  `src/features/stock_features.py`

This avoids the exact failure mode where the backtest uses an old cached shard
while the code already points at a pruned feature builder.

## 3. Rebuild features if the feature schema changed

If the feature set changed:

- rebuild the feature store
- verify `data/features/stock_features.parquet` reflects the expected columns
- verify the feature-store metadata hash changed for `stock`

You should expect a stock feature schema refresh whenever columns are added,
removed, or replaced with a new canonical family.

## 4. Rerun the stock-ranker training

Run the backtest in the mode you want to measure, usually `selection_only` first:

- `selection_only` for feature and ranking experiments
- `optimizer_only` only after the ranker itself looks better
- `full_rl` only after both selection and optimizer are acceptable

Wait for the run to finish completely. Do not inspect or export importance from
an in-progress run.

## 5. Refresh the run artifacts after completion

Once the run is finished, refresh the following artifacts from the completed
model state:

- `artifacts/reports/metrics.json`
- `artifacts/reports/nav_series.parquet`
- `artifacts/reports/rebalance_log.csv`
- `artifacts/reports/rebalance_log.parquet`
- `artifacts/reports/selection_diagnostics.json`
- `artifacts/reports/selection_rebalance_log.csv`
- `artifacts/reports/selection_rebalance_log.parquet`
- `artifacts/reports/stock_ranker_feature_importance.csv`
- `artifacts/reports/stock_ranker_feature_importance.json`
- `artifacts/reports/attribution.json`
- `artifacts/models/stock_ranker.pkl`

If you run with the normal reporting path, the main report generator refreshes
most of these automatically. If you use a no-report run, regenerate the stock
ranker importance explicitly with `scripts/export_stock_ranker_importance.py`
after the backtest finishes.

## 6. Verify the report is fresh

Before reviewing results, check that the report timestamps line up with the
latest run and that the stock-ranker importance is not stale.

The two common stale-artifact symptoms are:

- `artifacts/reports/stock_ranker_feature_importance.csv` still has the old
  row count or old feature families
- `artifacts/reports/metrics.json` still reflects an earlier run even though the
latest backtest logs show a different result

If either happens, rerun the report export or rerun the backtest with reporting
enabled.

## 7. Evaluate with the right metrics

Do not judge stock-ranker experiments only on portfolio CAGR. For ranking-layer
experiments, use the ranking diagnostics first.

### Primary ranking metrics

- within-sector IC
- within-sector top-bottom spread
- top-k vs sector median
- top-k vs universe
- stability
- average selected names
- turnover

### Secondary portfolio metrics

- CAGR
- Sharpe
- max drawdown
- final NAV

### Importance / attribution checks

- stock-ranker feature importance by sector
- grouped importance by feature family
- whether momentum features are being displaced by volatility/liquidity features

## 8. Keep the experiment log updated

After each completed run, record:

- experiment name
- feature block changes
- whether the stock cache was invalidated
- the exact run mode used
- whether the stock-ranker importance export was refreshed
- the primary ranking metrics
- the final keep / reject decision

This keeps the branch history reproducible and makes later comparisons easier.

## 9. Commit and push the checkpoint

After the run is complete and the artifacts are refreshed:

- commit the code and doc changes for that experiment
- push the branch to the remote before starting the next experiment
- do not let local and remote drift across experiment boundaries
- use the canonical SSH remote `git@github-personal:shivam61/RL-based-portfolio.git`
  for experiment checkpoints unless that transport is unavailable

The intended workflow is:

1. make the experiment change
2. invalidate and rebuild if needed
3. run the backtest to completion
4. refresh report artifacts
5. record the result in the worklog
6. commit the checkpoint
7. push the branch
8. start the next experiment from the pushed state
