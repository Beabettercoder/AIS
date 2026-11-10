#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/sda/HaiShangTS/sysetm3
BASE="$ROOT/海事协同研判智能体_服务器部署包"
NAME="$1"
SOURCE="$2"
if [[ ! "$NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then echo "invalid release name" >&2; exit 2; fi
DEST="$ROOT/releases/$NAME"
if [[ -e "$DEST" ]]; then echo "release already exists: $DEST" >&2; exit 2; fi
if [[ ! -f "$SOURCE/src/maritime_agent/api.py" ]]; then echo "source is not an app package: $SOURCE" >&2; exit 2; fi
mkdir -p "$DEST"
cp -a "$SOURCE/src" "$DEST/"
for FILE in requirements.txt pyproject.toml verify_health.py README_服务器部署与运行指南.md; do
  [[ ! -f "$SOURCE/$FILE" ]] || cp -a "$SOURCE/$FILE" "$DEST/"
done
[[ ! -d "$SOURCE/tests" ]] || cp -a "$SOURCE/tests" "$DEST/"
cp -a "$BASE/.venv" "$DEST/.venv"
ln -s ../../shared/.env "$DEST/.env"
ln -s ../../shared/storage "$DEST/storage"
if [[ -f "$DEST/requirements.txt" ]]; then
  "$DEST/.venv/bin/python" -m pip install --disable-pip-version-check -r "$DEST/requirements.txt"
fi
PYTHONPATH="$DEST/src" "$DEST/.venv/bin/python" -c 'from maritime_agent.api import app; print(app.title, app.version)'
echo "Staged release: $DEST"
echo "Activate with: $ROOT/bin/switch-release.sh $NAME"
