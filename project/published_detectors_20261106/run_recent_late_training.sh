#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
method=${1:?method};gpu=${2:?gpu};root=/home/ps/AIS_published_detectors_20261107/recent_original_late
test -s "$root/bridge_gate_lock.json" || { echo '桥接冻结与核验未完成，拒绝启动'; exit 1; }
for seed in 0 1 2; do
  for arm in raw causal_bundle f0_bundle f1_bundle; do
    echo "START $method $arm seed=$seed gpu=$gpu"
    "$PY" -u train_recent_late_bridge.py --method "$method" --arm "$arm" --seed "$seed" --gpu "$gpu" --epochs 25 --batch-size 128 --native-aux-weight .1
    echo "DONE $method $arm seed=$seed"
  done
done
echo "TRAINING_COMPLETE $method"
