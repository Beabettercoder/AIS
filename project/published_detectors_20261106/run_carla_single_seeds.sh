#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
arm=${1:?arm}; gpu=${2:?gpu}
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
out=/home/ps/AIS_published_detectors_20261107/carla_all_windows
for seed in 0 1 2; do
  if [[ -s "$out/$arm/seed$seed/result.json" ]]; then continue; fi
  "$PY" -u train_carla_all_windows.py --arm "$arm" --seed "$seed" --device "cuda:$gpu" --out "$out"
done
