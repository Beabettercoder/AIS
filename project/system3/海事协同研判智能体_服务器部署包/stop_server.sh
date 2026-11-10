#!/bin/bash
# ============================================================
# 海事协同研判智能体系统 - Linux 服务器一键停止脚本
# ============================================================

PORT=8000
PID=$(lsof -t -i:$PORT)
if [ -n "$PID" ]; then
    echo "[停止] 正在停止监听在端口 $PORT 的服务 (PID: $PID)..."
    kill -15 $PID
    sleep 2
    if kill -0 $PID 2>/dev/null; then
        kill -9 $PID
    fi
    echo "[成功] 服务已停止。"
else
    echo "[提示] 未发现运行在端口 $PORT 上的智能体服务。"
fi
