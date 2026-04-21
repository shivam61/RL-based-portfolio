# Feature Branch Worklog

This file tracks measured work completed on the current feature branch.
Each task should record:
- scope
- code changes
- validation run
- outcome
- learning

## 2026-04-21

### Task: Freeze truth baseline and run stock-feature ablations
- Scope:
  - revert horizon blending and lock the stock-ranker truth baseline to a single 8W label horizon
  - add block-based stock feature selection so the ranker can be ablated without changing architecture
  - add deterministic seed wiring for reproducible selection-only runs
  - run the following frozen-window ablations on `2013-01-01 → 2016-12-31`:
    - baseline blocks: `absolute_momentum + risk + liquidity + trend`
    - `absolute_momentum` only
    - `sector_relative_momentum` only
    - `absolute_momentum + sector_relative_momentum`
    - `volatility_adjusted_momentum`
- Validation:
  - `./.venv/bin/python -m py_compile src/models/stock_ranker.py src/models/sector_scorer.py src/backtest/walk_forward.py scripts/run_backtest.py src/features/stock_features.py src/reporting/report.py scripts/export_stock_ranker_importance.py`
  - `selection_only` baseline -> `11.55% CAGR`, `0.32 Sharpe`, `-19.58% MaxDD`, `54.45% avg turnover`
  - `absolute_momentum` only -> `7.05% CAGR`, `0.06 Sharpe`, `-23.70% MaxDD`, `56.32% avg turnover`
  - `sector_relative_momentum` only -> `6.53% CAGR`, `0.03 Sharpe`, `-24.79% MaxDD`, `56.66% avg turnover`
  - `absolute_momentum + sector_relative_momentum` -> `8.49% CAGR`, `0.14 Sharpe`, `-22.07% MaxDD`, `55.87% avg turnover`
  - `volatility_adjusted_momentum` -> `7.62% CAGR`, `0.09 Sharpe`, `-22.52% MaxDD`, `59.44% avg turnover`
- Learning:
  - the 8W truth baseline is the strongest of the tested feature slices on this frozen window
  - absolute momentum by itself is not enough
  - sector-relative momentum by itself is not enough
  - combining absolute + sector-relative helps relative to either alone, but still underperforms the full baseline that retains risk/liquidity/trend context
  - volatility-adjusted momentum alone does not solve the ranking problem
  - the next useful step is to prune weaker blocks one at a time from the 8W truth baseline rather than reintroducing horizon or optimizer complexity

## 2026-04-20

### Task: Data / PIT correctness baseline
- Scope:
  - point-in-time universe membership in feature generation
  - normalized sector index construction
  - shorter gap fill semantics
  - feature-store logic invalidation
- Validation:
  - `./.venv/bin/pytest tests -q` -> `89 passed`
  - full corrected backtest -> `15.75% CAGR`, `0.70 Sharpe`, `-29.82% MaxDD`, `17.49% avg turnover`
- Learning:
  - PIT fixes materially improved trustworthiness and recovered performance versus the earlier post-accounting low point.
  - The strategy still underperforms the historic high-water-mark runs, so more issues remain in optimizer and universe construction.

### Task: Optimizer turnover correctness
- Scope:
  - align optimizer turnover accounting with one-way turnover
  - repair solver outputs that violate turnover budget instead of silently accepting them
  - add regression tests for liquidation-only turnover math and repair behavior
- Validation:
  - `./.venv/bin/pytest tests -q` -> `91 passed`
  - full corrected backtest -> `18.94% CAGR`, `0.94 Sharpe`, `-21.47% MaxDD`, `21.92% avg turnover`
- Learning:
  - Making turnover accounting consistent with execution materially improved the measured frontier.
  - The optimizer now behaves more defensibly, but turnover is still a policy tradeoff rather than a fully solved hard-control problem.

### Next Queue
- Historical universe reconstruction beyond the static roster:
  - expand point-in-time coverage beyond the current fixed config names
  - preserve date-valid listing, sector, and investability logic
  - measure impact against the current branch baseline before proceeding further

### Investigation: Historical universe expansion
- Outcome:
  - Local-only universe expansion was not selected for implementation in this cycle.
- Evidence:
  - `price_matrix.parquet` already matches the configured NSE roster except for one missing ticker data hole (`TATAMOTORS.NS`).
  - Current PIT logic already delays post-2013 listings via price-coverage plus 252-day history gating.
  - Official Nifty archive files appear to be the right long-term source for date-effective roster reconstruction, but they were not harvestable reliably enough from this environment for a measured overnight change.
- Learning:
  - A meaningful historical-universe upgrade likely needs an external monthly constituent source, not just more logic on top of the current local cache.
  - The next best measured branch task was `run_021` rather than forcing a low-impact metadata-only universe change.

### Task: run_021 — Sharpe / Calmar stock features
- Scope:
  - added `sharpe_1m`, `sharpe_3m`, `sharpe_12m`, and `calmar_3m` to the stock feature set
  - extended feature validation with schema, fill-rate, and construction-consistency checks
- Validation:
  - `./.venv/bin/pytest tests/test_feature_validation.py -q` -> `44 passed`
  - `./.venv/bin/pytest tests -q` -> `93 passed`
  - full backtest -> `10.00% CAGR`, `0.31 Sharpe`, `-22.67% MaxDD`, `27.27% avg turnover`
- Decision:
  - reject and revert
- Learning:
  - Explicit return-over-volatility features pushed the system toward cash-heavy, concentrated RL behavior and materially degraded the stock ranker path.
  - The branch baseline remains the optimizer-fixed state (`18.94% CAGR`, `0.94 Sharpe`) until the next stock-feature experiment is measured.

### Task: run_022 — Cross-sectional return ranks
- Scope:
  - added `ret_1m_rank`, `ret_3m_rank`, and `ret_12m_rank` as percentile ranks across the investable stock universe
  - extended feature validation with schema, bounds, fill-rate, and cross-sectional consistency checks
- Validation:
  - `./.venv/bin/pytest tests/test_feature_validation.py -q` -> `44 passed`
  - `./.venv/bin/pytest tests -q` -> `93 passed`
  - full backtest -> `12.20% CAGR`, `0.43 Sharpe`, `-33.01% MaxDD`, `31.47% avg turnover`
- Decision:
  - reject and revert
- Learning:
  - Regime-invariant rank features did not help the current stack; they increased turnover and deepened drawdown while reducing both CAGR and Sharpe.
  - The branch baseline remains the optimizer-fixed state (`18.94% CAGR`, `0.94 Sharpe`) before `run_023`.

### Task: run_023 — Sector-relative z-score
- Scope:
  - replaced sector-relative raw spreads with within-sector z-scores for `ret_1w`, `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m`, and `vol_3m`
  - added a direct validation test that checks `ret_1m_vs_sector` against the expected within-sector z-score formula
- Validation:
  - `./.venv/bin/pytest tests/test_feature_validation.py -q` -> `43 passed`
  - `./.venv/bin/pytest tests -q` -> `92 passed`
  - full backtest -> `11.73% CAGR`, `0.39 Sharpe`, `-34.98% MaxDD`, `30.79% avg turnover`
- Decision:
  - reject and revert
- Learning:
  - Making sector-relative features variance-scaled did not improve comparability in practice; it destabilized the ranker path and materially worsened turnover and drawdown.
  - Three consecutive stock-feature experiments failed against the optimizer-fixed baseline, which is a signal to pause feature churn and move back to data/foundation work.

### Task: Stage A universe hardening (inferred listing dates + large/mid focus)
- Scope:
  - inferred `listed_since` from first valid trade date in local price history when config metadata is absent
  - added optional inferred delisting support for long disappearances
  - applied a default `active_caps: [large, mid]` universe filter
  - added targeted universe tests for inferred listing dates and default cap filtering
- Validation:
  - `./.venv/bin/pytest tests/test_data.py::TestUniverseManager -q` -> `8 passed`
  - `./.venv/bin/pytest tests -q` -> `93 passed`
  - full backtest -> `16.67% CAGR`, `0.69 Sharpe`, `-30.69% MaxDD`, `24.34% avg turnover`
- Decision:
  - reject and revert
- Learning:
  - Tightening realism on the current ~100-stock roster without expanding historical breadth reduced both return and risk-adjusted quality.
  - This suggests the next useful universe project is broader historical coverage, not a stricter version of the current narrow roster.

### Task: Stage B universe broadening (current Nifty 200 large/mid roster)
- Scope:
  - expanded the equity universe from the hand-curated 99-stock roster to a generated 199-stock broad large/mid roster based on current Nifty 200 constituents
  - added `scripts/build_universe_from_nifty200.py` to make the config generation reproducible
  - stored source constituent files under `data/reference/indices/`
  - added integrity tests for the broadened universe config
  - fixed `scripts/download_data.py` stale earnings import so the equity-only refresh path works again
  - fixed `FeatureStore` logic hashing so stock/sector shards rebuild when the universe roster changes
  - normalized source-symbol issues during generation:
    - dropped `DUMMY*` mirror artifacts
    - remapped `ZOMATO` to `ETERNAL`
    - excluded `TATAMOTORS` from this pass because the post-demerger ticker/history path was not stable in the Yahoo-backed downloader
- Validation:
  - `./.venv/bin/pytest tests -q` -> `94 passed`
  - short smoke backtest (`2013-01-01` → `2015-06-30`) -> `6.88% CAGR`, `0.12 Sharpe`, `-2.85% MaxDD`, `14.88% avg turnover`
  - initial full backtest result (`14.91% CAGR`, `0.61 Sharpe`, `-25.45% MaxDD`, `24.60% avg turnover`) was invalid because the feature store reused stale stock/sector shards from the old universe roster
  - corrected full backtest after feature-store rebuild -> `17.64% CAGR`, `0.75 Sharpe`, `-33.50% MaxDD`, `33.71% avg turnover`
- Decision:
  - reject as new baseline
- Learning:
  - The first measurement caught a real infrastructure bug: feature-store freshness was not keyed on universe composition, so universe changes could silently produce invalid backtest reads.
  - After correcting that bug and rebuilding the store, the broadened roster still underperformed the branch baseline (`18.94% CAGR`, `0.94 Sharpe`, `-21.47% MaxDD`, `21.92% turnover`) on return, Sharpe, drawdown, and turnover.
  - The broader universe also materially increases runtime because stock-ranker training and RL retrain checkpoints scale with the larger candidate set.

### Task: sector taxonomy experiment (18-sector target, selection_only)
- Scope:
  - remapped the current roster into the proposed 18-sector taxonomy using currently available local price history
  - added clean extra names that already exist in the local price matrix where possible, including:
    - `TATAELXSI.NS`
    - `AUBANK.NS`
    - `HDFCAMC.NS`
    - `LUPIN.NS`
    - `ZYDUSLIFE.NS`
    - `ADANIGREEN.NS`
    - `NTPC.NS`
    - `POWERGRID.NS`
    - `NHPC.NS`
    - `JINDALSTEL.NS`
    - `ETERNAL.NS`
    - `PAYTM.NS`
  - kept the run isolated in `/tmp/sector18_eval so it did not touch the main branch artifacts`
- Validation:
  - `selection_only` short backtest (`2013-01-01` → `2015-06-30`) -> `5.67% CAGR`, `-0.02 Sharpe`, `-10.47% MaxDD`, `72.80% avg turnover`
  - selection diagnostics:
    - Avg selected names `19.5`
    - Top-k avg forward return `0.92%`
    - Top-k vs universe `+0.17%`
    - Top-k vs sector median `+0.01%`
    - Precision@k `48.62%`
    - Rank IC `0.193`
    - Selection stability `26.17%`
- Decision:
  - reject and do not proceed to optimizer-only or full-RL on this taxonomy in the current branch state
- Learning:
  - The proposed finer sector split is reasonable as a target taxonomy, but on the current local history it did not materially improve stock selection versus the existing control setup.
  - The raw ranking edge stayed small (`top-k vs universe` `+0.17%`, `top-k vs sector median` `+0.01%`) and the selection set remained thin (`avg selected names` `19.5`, `stability` `26.17%`), even though intra-sector dispersion was measurable (`0.0592`).
  - That is consistent with relabeling clusters rather than creating a meaningfully easier ranking problem.
  - The current 15-sector model remains the practical baseline until we have broader historical coverage for the missing names.
  - The evaluation now includes explicit intra-sector dispersion in the selection diagnostics, and the added-stock feature path was validated against the local price/volume history for the names actually used in this isolated run.

### Task: existing-sector universe expansion (added supported names only)
- Scope:
  - added the locally supported names into the existing sector buckets without changing the sector taxonomy:
    - `TATAELXSI.NS`, `KPITTECH.NS`
    - `AUBANK.NS`
    - `HDFCAMC.NS`
    - `TATACONSUM.NS`
    - `SONACOMS.NS`, `BALKRISIND.NS`
    - `LUPIN.NS`, `ZYDUSLIFE.NS`
    - `ADANIGREEN.NS`, `ATGL.NS`
    - `JINDALSTEL.NS`
    - `PHOENIXLTD.NS`
    - `SRF.NS`
  - validated that the added names already have local price and volume history, so no fresh download was required for this set
  - ran the isolated `selection_only` smoke backtest against the expanded roster in `/tmp/universe_additions_eval`
- Validation:
  - isolated `selection_only` short backtest (`2013-01-01` → `2015-06-30`) -> `8.13% CAGR`, `0.12 Sharpe`, `-12.06% MaxDD`, `72.85% avg turnover`
  - selection diagnostics:
    - Avg selected names `19.17`
    - Top-k avg forward return `1.11%`
    - Top-k vs universe `+0.23%`
    - Top-k vs sector median `-0.02%`
    - Precision@k `50.87%`
    - Rank IC `0.083`
    - Intra-sector dispersion `0.0577`
    - Top-bottom spread `-0.0006`
    - Selection stability `22.85%`
- Decision:
  - keep the roster additions in the live universe config for now, but do not treat this as a strong stock-selection win yet
- Learning:
  - The broader roster is valid and the added names are correctly wired through the data path.
  - The ranking diagnostics are still mixed: top-k edge improved modestly, but rank IC, stability, and top-bottom spread remain weak.
  - This is a better universe shape than a sector-taxonomy rewrite, but it still needs more evidence before we call it a durable alpha improvement.

### Task: isolated 150-name universe retry
- Scope:
  - expanded the current curated roster to 150 names by adding extra liquid names available in the processed price matrix
  - kept the experiment isolated in temp configs so it did not collide with the background full-RL run
  - measured the stock-selection layer independently with the new `selection_only` / `optimizer_only` split
- Validation:
  - `selection_only` short backtest (`2013-01-01` → `2015-06-30`) -> `4.13% CAGR`, `-0.11 Sharpe`, `-10.32% MaxDD`, `72.85% avg turnover`
  - `optimizer_only` short backtest (`2013-01-01` → `2015-06-30`) -> `8.53% CAGR`, `0.21 Sharpe`, `-6.72% MaxDD`, `51.45% avg turnover`
  - selection diagnostics improved slightly in the optimizer pass, but not enough to beat the earlier isolated 99-name baseline
- Decision:
  - reject and do not keep the 150-name roster as a new baseline
- Learning:
  - The current roster can be expanded, but a naive bump from 99 to 150 names does not improve the isolated stock-selection or optimizer layers.
  - This suggests the next meaningful universe step is not just more names, but more historically correct breadth.

### Task: run_022 retry — selection-only / optimizer-only cross-sectional ranks
- Scope:
  - reintroduced `ret_1m_rank`, `ret_3m_rank`, and `ret_12m_rank` under the new `selection_only` / `optimizer_only` split
  - measured the candidate independently from the RL overlay in an isolated temp workspace
- Validation:
  - `selection_only` short backtest (`2013-01-01` → `2015-06-30`) -> `4.93% CAGR`, `-0.06 Sharpe`, `-9.57% MaxDD`, `73.58% avg turnover`
  - `optimizer_only` short backtest (`2013-01-01` → `2015-06-30`) -> `7.38% CAGR`, `0.13 Sharpe`, `-5.91% MaxDD`, `50.12% avg turnover`
- Learning:
  - The cross-sectional rank idea did improve some selection diagnostics, but it did not improve the portfolio outcome versus the current baseline in either isolated mode.
  - `optimizer_only` remained better than `selection_only`, which means the optimizer is still adding value, but this candidate is not a clear keeper.
  - This is enough evidence to treat `run_022` as a reject and move on to the next stock-selection candidate.

### Task: run_024 — momentum acceleration
- Scope:
  - added `mom_accel_1m` and `mom_accel_3m` to the stock feature set
  - kept the new `selection_only` / `optimizer_only` split for isolated measurement
- Validation:
  - `selection_only` short backtest (`2013-01-01` → `2015-06-30`) -> `4.13% CAGR`, `-0.11 Sharpe`, `-10.32% MaxDD`, `72.85% avg turnover`
  - `optimizer_only` short backtest (`2013-01-01` → `2015-06-30`) -> `7.80% CAGR`, `0.14 Sharpe`, `-6.47% MaxDD`, `50.81% avg turnover`
- Learning:
  - Momentum acceleration improved over its own selection-only pass, which is the right direction, but it still did not beat the earlier isolated optimizer-only baseline.
  - The optimizer is still contributing more than raw selection alone, but this candidate is also a reject for now.
  - This feature family is closed and should not be revisited on this branch.

### Task: cross-sectional ranking feature generation
- Scope:
  - added sector-relative momentum, per-date z-scores, rank transforms, and volatility-adjusted momentum to the stock feature builder
  - widened the feature store invalidation path through a new stock feature logic hash so the expanded feature set rebuilds cleanly
  - kept the universe fixed and reused the current expanded roster as the control set
- Validation:
  - `scripts/build_features.py` rebuilt the feature store with the new logic hash
  - focused regression tests passed:
    - `tests/test_data.py::TestFeatures::test_stock_features_cross_sectional_transforms`
    - `tests/test_data.py::TestFeatures::test_taxonomy_additions_have_price_and_feature_coverage`
  - stored stock feature parquet now includes:
    - `mom_3m_vol_adj`
    - `mom_3m_vol_adj_z`
    - `mom_3m_vol_adj_rank`
    - `ret_3m_z`
    - `ret_3m_rank`
    - `ret_12m_z`
    - `ret_12m_rank`
    - `mom_12m_skip1m_z`
    - `mom_12m_skip1m_rank`
- Learning:
  - This is the right layer to attack next because the issue is ranking signal quality, not taxonomy.
  - The expanded universe and the stock-selection diagnostics can now be tested on a richer, more discriminative feature set without changing sectors again.

### Task: local-sector signal diagnostics
- Scope:
  - froze the current expanded universe as the control set
  - added candidate-level score propagation into the walk-forward selection diagnostics
  - added within-sector IC and within-sector top-bottom spread, plus sector-median separation, aggregated equal-weight and by candidate count
- Validation:
  - `tests/test_selection_diagnostics.py` now checks the new local-sector metrics
  - `selection_diagnostics.json` can now report:
    - `within_sector_ic`
    - `within_sector_ic_weighted`
    - `within_sector_top_bottom_spread`
    - `within_sector_top_bottom_spread_weighted`
    - `within_sector_top_k_minus_sector_median`
    - `within_sector_top_k_minus_sector_median_weighted`
- Learning:
  - The ranking layer is now measurable both globally and locally inside each sector.
  - This is the right diagnostic split for deciding whether the model should be read as a flat cross-sectional ranker or a sector-local ranker inside a hierarchical selection pipeline.

### Task: stock ranker minimal raw v2
- Scope:
  - hard-prune the stock feature builder to the six raw canonical features only:
    - `ret_3m`
    - `mom_12m_skip1m`
    - `mom_accel_3m_6m`
    - `vol_3m`
    - `amihud_1m`
    - `ma_50_200_ratio`
  - retire all `_z`, `_rank`, `_resid`, `_vs_sector`, benchmark-relative, and duplicate-horizon transforms from the stock-ranker experiment series
  - update the schema tests to match the new contract
- Rationale:
  - the ranker is still fragmenting momentum signal across too many correlated transforms
  - the next test is whether a concentrated raw set can finally move within-sector IC and top-bottom spread

### Task: stock horizon shift [NEXT]
- Scope:
  - keep the raw-minimal stock contract fixed
  - vary only the stock label horizon at 4W / 8W / 12W
  - expose the horizon as `stock_model.fwd_window_days` plus a `run_backtest.py` CLI override
- Rationale:
  - the ranker has little to no 4W signal; the next hypothesis is that the signal is slower-moving momentum rather than a broken model wiring path
