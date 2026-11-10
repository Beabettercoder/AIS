#!/usr/bin/env bash
set -euo pipefail
base=/home/ps/AIS_published_detectors_20261107
for method in tft_gcn simad dmrad carla; do
    log="$base/logs/supervised_${method}.log"
    until grep -q "完成 ${method} 监督双任务三种子" "$log" 2>/dev/null; do
        if grep -q 'Traceback (most recent call last)' "$log" 2>/dev/null; then echo "$method 失败，汇总未启动"; exit 1; fi
        sleep 60
    done
done
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python -u summarize_supervised.py --bootstrap 2000
