#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
method=${1:?method}; gpu=${2:?gpu}; prior=${3:?prior PID}
while kill -0 "$prior" 2>/dev/null; do sleep 20; done
for arm in raw gate ordinary reliable; do
  test -s "/home/ps/AIS_published_detectors_20261107/single_all_windows/$method/$arm/seed0/test_metrics.json" || { echo "seed0 $arm did not complete"; exit 1; }
done
for seed in 1 2; do bash run_all_windows.sh "$method" "$gpu" "$seed"; done
