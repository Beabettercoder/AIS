#!/usr/bin/env bash
set -euo pipefail
cd /home/ps/AIS_adaptive_radius_relation_v2p1_20261109
exec 9>train_launcher.lock
flock -n 9 || { echo 'Training launcher already active; refusing duplicate.'; exit 1; }
if [[ -d runs ]]; then echo 'Existing experiment results found; refusing overwrite.'; exit 1; fi
test -s reports/pretrain_audit.json;test -s reports/relation_source_views_audit.json;test -s reports/normalization_independence.json
mkdir -p logs
gpus=(0 1 2 5 6 7);i=0;pids=();keys=()
for task in single relation; do
 for name in tcn gru transformer graph; do
  for seed in 0 1 2; do
   gpu=${gpus[$((i%6))]};i=$((i+1));key="${task}_${name}_${seed}"
   bash run_train_worker.sh "$task" "$name" "$seed" "$gpu" > "logs/train_${key}.log" 2>&1 & pids+=("$!");keys+=("$key")
  done
 done
done
failed=0
for i in "${!pids[@]}"; do
 if wait "${pids[$i]}"; then echo "WORKER_OK ${keys[$i]}"; else echo "WORKER_FAILED ${keys[$i]}"; failed=1; fi
done
if ((failed)); then echo 'TRAIN_FAILED';exit 1;fi
echo 'TRAIN_COMPLETE 216 RUNS'
