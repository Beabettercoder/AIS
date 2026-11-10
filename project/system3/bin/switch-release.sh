#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/sda/HaiShangTS/sysetm3
NAME="$1"
if [[ ! "$NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then echo "invalid release name" >&2; exit 2; fi
TARGET="releases/$NAME"
if [[ "$NAME" == "legacy" ]]; then TARGET=legacy; fi
if [[ ! -f "$ROOT/$TARGET/src/maritime_agent/api.py" || ! -x "$ROOT/$TARGET/.venv/bin/python" ]]; then
  echo "release is incomplete: $ROOT/$TARGET" >&2; exit 2
fi
OLD="$(readlink "$ROOT/current" 2>/dev/null || true)"
if [[ -z "$OLD" ]]; then OLD=legacy; fi
restart_app() {
  tmux kill-session -t sysetm3-app 2>/dev/null || true
  tmux new-session -d -s sysetm3-app "exec '$ROOT/bin/start-current.sh'"
}
activate() {
  local target="$1" tmp="$ROOT/.current.next.$$"
  ln -s "$target" "$tmp"
  mv -Tf "$tmp" "$ROOT/current"
  restart_app
  for i in $(seq 1 30); do
    if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null; then return 0; fi
    sleep 2
  done
  return 1
}
if activate "$TARGET"; then
  echo "Active release: $TARGET"
  exit 0
fi
echo "Health check failed for $TARGET; restoring $OLD" >&2
if activate "$OLD"; then echo "Rollback succeeded: $OLD" >&2; else echo "Rollback failed; inspect storage/server.stderr.log" >&2; fi
exit 1
