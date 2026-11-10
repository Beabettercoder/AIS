#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
gpu=${1:?gpu}; part=${2:?part}
py=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
export PYTHONUNBUFFERED=1
case "$part" in
  core) groups=('raw:0:count' 'gate:0:count' 'direct:0:count');;
  late) groups=('late:0:count' 'bundle:0:count');;
  extras) groups=('original:0:count' 'original:0:legacy' 'direct:0.1:count');;
  *) echo 'invalid part'; exit 2;;
esac
for group in "${groups[@]}"; do
  IFS=: read -r arm aux rho <<< "$group"
  for seed in 0 1 2; do
    echo "START $arm aux=$aux rho=$rho seed=$seed"
    "$py" -u bridge_tft.py --arm "$arm" --aux "$aux" --rho-mode "$rho" --seed "$seed" --gpu "$gpu" --epochs 25
    echo "DONE $arm aux=$aux rho=$rho seed=$seed"
  done
done
echo "BRIDGE_TFT_$part VALIDATION COMPLETE"
