#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
OUT=/home/ps/AIS_published_detectors_20261107
mkdir -p "$OUT/logs"
run_all() {
  local m="$1" gpu="$2" mode="$3"
  for arm in C0 F0 F1; do
    echo "$(date -Is) ${mode} ${m} ${arm}"
    "$PY" -u train_relation_recent.py --method "$m" --arm "$arm" --seed 0 --gpu "$gpu" --epochs 25 --mode "$mode" --out "$OUT"
  done
}
run_all tft_gcn 0 train > "$OUT/logs/relation_tft_gcn_train.log" 2>&1 & a=$!
run_all simad 6 train > "$OUT/logs/relation_simad_train.log" 2>&1 & b=$!
run_all dmrad 7 train > "$OUT/logs/relation_dmrad_train.log" 2>&1 & c=$!
wait "$a"; wait "$b"; wait "$c"
echo "All nine training runs complete; freezing all checkpoints before four-view test."
run_all tft_gcn 0 test > "$OUT/logs/relation_tft_gcn_test.log" 2>&1 & a=$!
run_all simad 6 test > "$OUT/logs/relation_simad_test.log" 2>&1 & b=$!
run_all dmrad 7 test > "$OUT/logs/relation_dmrad_test.log" 2>&1 & c=$!
wait "$a"; wait "$b"; wait "$c"
echo "All nine previously-exposed test evaluations complete."
