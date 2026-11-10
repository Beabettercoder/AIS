#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/sda/HaiShangTS/sysetm3
APP="$ROOT/current"
if [[ ! -f "$APP/src/maritime_agent/api.py" || ! -x "$APP/.venv/bin/python" ]]; then
  echo "current release is incomplete: $APP" >&2
  exit 1
fi
if ss -ltn | awk '$4 ~ /:8000$/ { found=1 } END { exit !found }'; then
  echo "Port 8000 is already in use; refusing to start." >&2
  exit 1
fi
cd "$APP"
mkdir -p storage
export PYTHONPATH="$APP/src"
exec "$APP/.venv/bin/python" -m uvicorn maritime_agent.api:app --host 0.0.0.0 --port 8000 --workers 1 >> storage/server.stdout.log 2>> storage/server.stderr.log
