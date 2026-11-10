#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
method=${1:?method};gpu=${2:-0}
for arm in raw causal_bundle f0_bundle f1_bundle; do
    echo "SMOKE $method $arm"
    "$PY" -u train_recent_late_bridge.py --method "$method" --arm "$arm" --gpu "$gpu" --smoke
    "$PY" -u train_recent_late_bridge.py --method "$method" --arm "$arm" --gpu "$gpu" --smoke --mode test --checkpoint best_val
    "$PY" -u train_recent_late_bridge.py --method "$method" --arm "$arm" --gpu "$gpu" --smoke --mode test --checkpoint last
done
echo "SMOKE DONE $method"
