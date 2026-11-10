"""Record a stage command's exit status without changing its arguments."""
from pathlib import Path
import argparse
import datetime as dt
import json
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('command required after --')
    logdir = args.stage / 'execution'
    logdir.mkdir(exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    marker = logdir / f'{args.name}.json'
    if marker.exists():
        raise FileExistsError(marker)
    record = dict(name=args.name, command=command, started_utc=started, status='running')
    marker.write_text(json.dumps(record, indent=2), encoding='utf-8')
    t0 = time.perf_counter()
    try:
        with (logdir / f'{args.name}.log').open('x', encoding='utf-8') as log:
            proc = subprocess.run(command, cwd=args.stage, stdout=log, stderr=subprocess.STDOUT, check=False)
        code = proc.returncode
        record.update(status='completed' if code == 0 else 'failed', exit_code=code)
    except Exception as exc:
        code = 1
        record.update(status='failed', exit_code=code, error=repr(exc))
    record.update(elapsed_seconds=time.perf_counter() - t0,
                  finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    marker.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return code


if __name__ == '__main__':
    sys.exit(main())
