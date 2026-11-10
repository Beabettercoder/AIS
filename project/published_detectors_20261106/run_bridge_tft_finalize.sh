#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
base=/home/ps/AIS_bridge_20261107
py=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
export PYTHONUNBUFFERED=1
for part in core late extras; do
  until grep -q "BRIDGE_TFT_$part VALIDATION COMPLETE" "$base/logs/tft128_$part.log"; do
    if grep -q 'Traceback (most recent call last)' "$base/logs/tft128_$part.log"; then echo "TRAIN FAILED $part"; exit 1; fi
    sleep 30
  done
done
"$py" -u freeze_bridge_tft.py
"$py" -u summarize_bridge_tft.py --split val
run_tests() {
 local gpu=$1; shift
 for group in "$@"; do
  IFS=: read -r arm aux rho <<< "$group"
  for seed in 0 1 2; do
   echo "TEST START $arm aux=$aux rho=$rho seed=$seed gpu=$gpu"
   "$py" -u bridge_tft.py --arm "$arm" --aux "$aux" --rho-mode "$rho" --seed "$seed" --gpu "$gpu" --mode test
  done
 done
}
run_tests 5 'raw:0:count' 'gate:0:count' 'direct:0:count' & p1=$!
run_tests 6 'late:0:count' 'bundle:0:count' & p2=$!
run_tests 7 'original:0:count' 'original:0:legacy' 'direct:0.1:count' & p3=$!
wait "$p1"; wait "$p2"; wait "$p3"
"$py" -u summarize_bridge_tft.py --split test
echo 'TFT BRIDGE COMPLETE'
