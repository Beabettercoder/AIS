#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
out=/home/ps/AIS_bridge_20261107/tcn
"$PY" -c 'import hashlib,json,pathlib; p=pathlib.Path("/home/ps/AIS_bridge_20261107/tcn/frozen_validation.lock.json"); x=json.loads(p.read_text()); assert all(hashlib.sha256(pathlib.Path(k).read_bytes()).hexdigest()==v for k,v in (x["dependency_sha256"] | x["training_validation_data_sha256"]).items())'
arms=raw,gate,early,late,late_bundle,late_bundle_irregular,late_bundle_irregular_historical_rho
for seed in 0 1 2; do
  "$PY" -u bridge_tcn_test_entry.py --mode test --seed "$seed" --gpu "$seed" --arms "$arms" > "$out/test_seed$seed.log" 2>&1 &
  eval "p$seed=$!"
done
wait "$p0"; wait "$p1"; wait "$p2"
"$PY" -u bridge_tcn_summary.py > "$out/paired_test_summary.log" 2>&1
echo 'COMPLETE: 21 frozen runs, validation and previously exposed test evaluated'
