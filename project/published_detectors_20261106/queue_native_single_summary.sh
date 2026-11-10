#!/usr/bin/env bash
set -euo pipefail
base=/home/ps/AIS_published_detectors_20261107
while true; do
  complete=1
  for method in tft_gcn simad dmrad carla; do
    for arm in raw ordinary reliable; do
      for seed in 0 1 2; do
        if [[ "$method" == carla ]]; then path="$base/carla_all_windows/$arm/seed$seed/test_scores.npz"; else path="$base/single_all_windows/$method/$arm/seed$seed/test_scores.npz"; fi
        [[ -s "$path" ]] || complete=0
      done
    done
  done
  (( complete == 1 )) && break
  sleep 60
done
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python -u summarize_native_single.py --bootstrap 2000
