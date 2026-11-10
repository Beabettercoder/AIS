#!/usr/bin/env bash
set -euo pipefail
cd /home/ps/AIS_adaptive_sync_20261110
exec 9>pipeline.lock
flock -n 9 || { echo PIPELINE_DUPLICATE;exit 1; }
py=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
while ! grep -q SYNC_BUILD_COMPLETE build_all.log; do
 if ! pgrep -f '[b]ash build_all.sh' >/dev/null; then echo BUILD_INTERRUPTED;exit 1;fi
 if grep -q BUILD_FAILED build_all.log; then echo BUILD_FAILED;exit 1;fi
 sleep 15
done
mkdir -p reports logs
"$py" -u audit_sync_smoke.py
"$py" -u audit_v4.py
"$py" -u audit_normalization_independence.py
for name in tcn gru transformer graph; do
 "$py" -u train_relation_v2.py --name "$name" --arm adaptive_credible_fallback --seed 0 --gpu 0 --epochs 1 --smoke > "logs/smoke_${name}.log" 2>&1
done
echo SMOKE_COMPLETE
bash run_train_all.sh > logs/train_all.log 2>&1
"$py" -u freeze_v4.py
bash run_test_all.sh > logs/test_all.log 2>&1
"$py" -u independent_results_audit.py
bash run_summaries.sh > logs/summaries.log 2>&1
"$py" -u summarize_field_v2.py --split test --draws 2000 > logs/field_summary.log 2>&1
echo PIPELINE_COMPLETE
