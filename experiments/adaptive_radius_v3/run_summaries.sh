#!/usr/bin/env bash
set -euo pipefail
cd /home/ps/AIS_adaptive_radius_relation_v3_20261109
exec 9>summary_launcher.lock
flock -n 9 || { echo 'Summary launcher already active.'; exit 1; }
grep -q 'TEST_COMPLETE 108 RUNS BEST_AND_LAST' logs/test_all.log || { echo 'Tests incomplete'; exit 1; }
py=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pids=();keys=()
for task in relation; do
 for name in tcn gru transformer graph; do
  key="${task}_${name}"
  "$py" -u summarize_model_part.py --task "$task" --name "$name" --bootstrap 2000 > "logs/summary_${key}.log" 2>&1 & pids+=("$!");keys+=("$key")
 done
done
failed=0
for i in "${!pids[@]}"; do if wait "${pids[$i]}"; then echo "SUMMARY_OK ${keys[$i]}"; else echo "SUMMARY_FAILED ${keys[$i]}"; failed=1; fi; done
if ((failed)); then echo 'SUMMARY_FAILED';exit 1;fi
echo 'SUMMARY_PARTS_COMPLETE'
