#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
out=/home/ps/AIS_bridge_20261107/tcn
for seed in 0 1 2; do
  p="$out/validation/late_bundle_irregular/seed$seed/history.json"
  until test -f "$p" && "$PY" -c 'import json,sys;assert len(json.load(open(sys.argv[1])))==25' "$p" 2>/dev/null; do
    if grep -q 'Traceback (most recent call last)' "$out/seed$seed.log" 2>/dev/null; then echo "seed$seed training failed; not freezing"; exit 1; fi
    sleep 15
  done
done
for seed in 0 1 2; do
  "$PY" -u bridge_tcn_20261107.py --seed "$seed" --gpu "$seed" --arms late_bundle_irregular_historical_rho > "$out/rho_seed$seed.log" 2>&1 &
  eval "p$seed=$!"
done
wait "$p0"; wait "$p1"; wait "$p2"
"$PY" -u bridge_tcn_freeze.py
"$PY" -c 'import hashlib,json,pathlib; p=pathlib.Path("/home/ps/AIS_bridge_20261107/tcn/frozen_validation.lock.json"); x=json.loads(p.read_text()); assert all(hashlib.sha256(pathlib.Path(k).read_bytes()).hexdigest()==v for k,v in x["dependency_sha256"].items())'
arms=raw,gate,early,late,late_bundle,late_bundle_irregular,late_bundle_irregular_historical_rho
for seed in 0 1 2; do
  "$PY" -u bridge_tcn_20261107.py --mode test --seed "$seed" --gpu "$seed" --arms "$arms" > "$out/test_seed$seed.log" 2>&1 &
  eval "p$seed=$!"
done
wait "$p0"; wait "$p1"; wait "$p2"
"$PY" -u bridge_tcn_summary.py > "$out/paired_test_summary.log" 2>&1
echo 'COMPLETE: 21 frozen runs, validation and previously exposed test evaluated'
