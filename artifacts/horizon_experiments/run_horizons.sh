#!/bin/bash
set -e
REPO=/home/shivamguptanit/RL-based-portfolio
PYTHON=$REPO/.venv/bin/python
OUT=$REPO/artifacts/horizon_experiments
LOG=$OUT/run.log

cd $REPO
export PYTHONPATH=$REPO
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== HORIZON EXPERIMENT START ===" > $LOG
date >> $LOG

for DAYS in 28 56 84; do
  echo "" >> $LOG
  echo "--- horizon=${DAYS}D ---" >> $LOG
  date >> $LOG
  $PYTHON scripts/run_backtest.py \
    --mode selection_only \
    --stock-fwd-window-days $DAYS \
    --start 2013-01-01 \
    --end 2016-12-31 \
    --no-baselines \
    >> $LOG 2>&1
  echo "exit: $?" >> $LOG
  # preserve results before next run overwrites them
  cp $REPO/artifacts/reports/selection_diagnostics.json $OUT/selection_diagnostics_${DAYS}d.json 2>/dev/null || echo "no selection_diagnostics for ${DAYS}d" >> $LOG
  cp $REPO/artifacts/reports/metrics.json $OUT/metrics_${DAYS}d.json 2>/dev/null || echo "no metrics for ${DAYS}d" >> $LOG
  echo "saved horizon=${DAYS}D results" >> $LOG
  date >> $LOG
done

echo "" >> $LOG
echo "=== HORIZON EXPERIMENT COMPLETE ===" >> $LOG
date >> $LOG
