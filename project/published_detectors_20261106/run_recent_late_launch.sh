#!/usr/bin/env bash
# Manual launch only after review; preflight is mandatory and writes gate once.
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
root=/home/ps/AIS_published_detectors_20261107/recent_original_late
mkdir -p "$root/logs"
"$PY" -u recent_late_bridge_preflight.py --write
bash run_recent_late_training.sh tft_gcn 0 > "$root/logs/train_tft_gcn.log" 2>&1 & p0=$!
bash run_recent_late_training.sh simad 1 > "$root/logs/train_simad.log" 2>&1 & p1=$!
bash run_recent_late_training.sh dmrad 2 > "$root/logs/train_dmrad.log" 2>&1 & p2=$!
bash run_recent_late_training.sh carla 5 > "$root/logs/train_carla.log" 2>&1 & p5=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p5" || status=1
if (( status != 0 )); then echo 'A training worker failed; no freezing or heldout scoring.'; exit 1; fi
bash run_recent_late_finalize.sh > "$root/logs/finalize.log" 2>&1
