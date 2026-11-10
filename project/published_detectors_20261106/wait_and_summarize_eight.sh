#!/usr/bin/env bash
set -uo pipefail
root=/home/ps/AIS_published_detectors_20261107
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106 || exit 1
pid=$1
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
if ! grep -q '^QUEUE_FINISHED' "$root/logs/eight_matched_queue.log"; then
  echo '队列进程已退出但未记录正常结束，将核查并记录缺失任务。'
fi
/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python -u summarize_eight_matched.py
rc=$?
if [ "$rc" -eq 0 ]; then echo '汇总完成'; else echo "汇总失败或任务未完成，状态码 $rc"; fi
exit "$rc"
