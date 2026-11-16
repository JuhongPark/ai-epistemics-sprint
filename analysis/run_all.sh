#!/bin/sh
# Rebuild every derived output from the raw snapshot, in dependency order, from a clean slate.
# Usage: sh analysis/run_all.sh   (about 10-15 minutes; seeds are fixed)
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
rm -rf ../data/derived && mkdir -p ../data/derived
for s in 01_replicate_scoring 02_h1_inflation 03_h2_crossround 04_h3_consequence 04b_noise_bootstrap \
         10_round_anchor 05_policy_sim 07_robustness 08_round_effects 09_backtest 11_board_selection \
         12_confounds 13_prospective 06_figures; do
  echo "== $s"
  $PY $s.py > ../data/derived/log_$s.txt 2>&1 || { echo "FAILED: $s"; tail -20 ../data/derived/log_$s.txt; exit 1; }
done
echo "all scripts finished"
