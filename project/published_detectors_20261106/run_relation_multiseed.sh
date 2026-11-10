#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
OUT=/home/ps/AIS_published_detectors_20261107
LAUNCHER=$OUT/logs/relation_launcher.log
# Seed zero training and evaluations finish before further work starts.
while ! grep -q 'All nine previously-exposed test evaluations complete.' "$LAUNCHER"; do
  if ! pgrep -f '^bash run_relation_recent.sh$' >/dev/null; then echo 'seed0 stopped before completion'; exit 1; fi
  sleep 20
done
"$PY" summarize_relation_recent.py --seed 0 > "$OUT/logs/relation_summary_seed0.log" 2>&1
run_all() {
 local model="$1" gpu="$2" mode="$3" seed="$4"
 for arm in C0 F0 F1; do
  echo "$(date -Is) $mode $model $arm seed$seed"
  "$PY" -u train_relation_recent.py --method "$model" --arm "$arm" --seed "$seed" --gpu "$gpu" --epochs 25 --mode "$mode" --out "$OUT"
 done
}
for seed in 1 2; do
 run_all tft_gcn 0 train "$seed" > "$OUT/logs/relation_tft_gcn_train_seed${seed}.log" 2>&1 & a=$!
 run_all simad 6 train "$seed" > "$OUT/logs/relation_simad_train_seed${seed}.log" 2>&1 & b=$!
 run_all dmrad 7 train "$seed" > "$OUT/logs/relation_dmrad_train_seed${seed}.log" 2>&1 & c=$!
 wait "$a"; wait "$b"; wait "$c"
 echo "seed${seed}: all nine checkpoints frozen"
 run_all tft_gcn 0 test "$seed" > "$OUT/logs/relation_tft_gcn_test_seed${seed}.log" 2>&1 & a=$!
 run_all simad 6 test "$seed" > "$OUT/logs/relation_simad_test_seed${seed}.log" 2>&1 & b=$!
 run_all dmrad 7 test "$seed" > "$OUT/logs/relation_dmrad_test_seed${seed}.log" 2>&1 & c=$!
 wait "$a"; wait "$b"; wait "$c"
 "$PY" summarize_relation_recent.py --seed "$seed" > "$OUT/logs/relation_summary_seed${seed}.log" 2>&1
done
echo 'Relation seeds 0/1/2 complete.'
