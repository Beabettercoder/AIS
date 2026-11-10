#!/bin/bash
# ============================================================
# 海事协同研判智能体系统 - Linux 服务器一键启动脚本
# ============================================================

cd "$(dirname "$0")"

# 检查虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# 检查端口占用
PORT=8000
PID=$(lsof -t -i:$PORT)
if [ -n "$PID" ]; then
    echo "[提示] 端口 $PORT 已被进程 $PID 占用，正在停止旧进程..."
    kill -9 $PID
    sleep 1
fi

echo "[启动] 正在启动海事协同研判智能体服务..."
mkdir -p storage
nohup python3 -m uvicorn maritime_agent.api:app --host 0.0.0.0 --port $PORT --workers 1 > storage/server.stdout.log 2> storage/server.stderr.log &

NEW_PID=$!
echo "[成功] 服务已在后台平稳运行！"
echo "  - 进程 PID: $NEW_PID"
echo "  - 监听端口: $PORT"
echo "  - 访问地址: http://<你的服务器IP>:$PORT/ui"
echo "  - 查看日志: tail -f storage/server.stdout.log"
