#!/usr/bin/env bash
set -euo pipefail
cd /home/ps/AIS_adaptive_sync_20261110
exec 9>test_launcher.lock
flock -n 9 || { echo 'Test launcher already active; refusing duplicate.'; exit 1; }
test -s frozen_manifest.json || { echo 'Missing freeze; refuse test scoring.'; exit 1; }
if find runs -name test_last_scores.npz -print -quit | grep -q .; then echo 'Existing test outputs; refuse overwrite.'; exit 1; fi
mkdir -p logs
gpus=(0 1 2 5 6 7);i=0;pids=();keys=()
for task in relation; do
 for name in tcn gru transformer graph; do
  for seed in 0 1 2; do
   gpu=${gpus[$((i%6))]};i=$((i+1));key="${task}_${name}_${seed}"
   bash run_test_worker.sh "$task" "$name" "$seed" "$gpu" > "logs/eval_${key}.log" 2>&1 & pids+=("$!");keys+=("$key")
  done
 done
done
failed=0
for i in "${!pids[@]}"; do if wait "${pids[$i]}"; then echo "TEST_WORKER_OK ${keys[$i]}"; else echo "TEST_WORKER_FAILED ${keys[$i]}"; failed=1; fi; done
if ((failed)); then echo 'TEST_FAILED';exit 1;fi
echo 'TEST_COMPLETE 84 RUNS BEST_AND_LAST'
