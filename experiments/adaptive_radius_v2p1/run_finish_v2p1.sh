#!/usr/bin/env bash
set -euo pipefail
cd /home/ps/AIS_adaptive_radius_relation_v2p1_20261109
exec 9>finish_launcher.lock
flock -n 9 || { echo 'Finalizer already active'; exit 1; }
py=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
while ! grep -q 'TRAIN_COMPLETE 216 RUNS' logs_train_launcher.log 2>/dev/null; do
 if grep -q 'TRAIN_FAILED' logs_train_launcher.log 2>/dev/null; then echo 'TRAIN_FAILED, no test'; exit 1; fi
 sleep 20
done
"$py" -u freeze_v2.py
"$py" -u summarize_field_v2.py --split test > logs/field_test.log 2>&1
bash run_test_all.sh > logs/test_all.log 2>&1
"$py" -u independent_results_audit.py > logs/result_audit.log 2>&1
bash run_summaries.sh > logs/summaries.log 2>&1
"$py" -u render_v2_report.py
echo 'EXPERIMENT_COMPLETE 216 RUNS BEST_AND_LAST'
