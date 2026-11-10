#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
root=/home/ps/AIS_published_detectors_20261107/recent_original_late
for method in tft_gcn simad dmrad carla; do
  log="$root/logs/train_${method}.log"
  until grep -q "TRAINING_COMPLETE $method" "$log" 2>/dev/null; do
    if grep -q 'Traceback (most recent call last)' "$log" 2>/dev/null; then echo "TRAINING FAILED: $method"; exit 1; fi
    sleep 30
  done
done
"$PY" -u freeze_recent_late_bridge.py
manifest="$root/frozen_manifest.json"
eval_one() {
  local method=$1 gpu=$2
  for seed in 0 1 2; do
    for arm in raw causal_bundle f0_bundle f1_bundle; do
      for checkpoint in best_val last; do
        echo "TEST START $method $arm seed=$seed $checkpoint"
        "$PY" -u train_recent_late_bridge.py --method "$method" --arm "$arm" --seed "$seed" --gpu "$gpu" --mode test --checkpoint "$checkpoint" --manifest "$manifest"
      done
    done
  done
}
eval_one tft_gcn 0 & p0=$!
eval_one simad 1 & p1=$!
eval_one dmrad 2 & p2=$!
eval_one carla 5 & p5=$!
wait "$p0";wait "$p1";wait "$p2";wait "$p5"
"$PY" -u summarize_recent_late_bridge.py --bootstrap 2000
echo 'RECENT_LATE_COMPLETE 48 RUNS BEST_AND_LAST'
