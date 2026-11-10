"""Stage, verify, and reversibly update only the system3 screening plugin."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request

SYSTEM = Path('/mnt/sda/WangZijia/AIS/system3')
APP = (SYSTEM / 'current').resolve()
WORK = SYSTEM / 'shared/chart-ui-v31'
STAGE = WORK / 'stage'
PLUGIN = APP / 'src/anomaly_workbench'
PYTHON = APP / '.venv/bin/python'
SCAN_ID = '0aff9f4b7b304b5595ca9216be5e0424'
RECORD_ID = 'AIS-368212230-1720389600'


def environment():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(STAGE) + os.pathsep + str(APP / 'src')
    env['ANOMALY_WORKBENCH_RUNTIME'] = str(WORK / 'test-runtime')
    return env


def stage():
    with tarfile.open(WORK / 'sqlite-live.tar.gz') as archive:
        members = archive.getmembers()
        for member in members:
            p = Path(member.name)
            if member.issym() or member.islnk() or not member.isfile() or p.is_absolute() or '..' in p.parts or p.parts[0] != 'anomaly_workbench':
                raise ValueError('Unexpected release member')
        STAGE.mkdir(parents=True, exist_ok=True)
        archive.extractall(STAGE, members=members)
    manifest = json.loads((STAGE / 'anomaly_workbench/RELEASE_MANIFEST.json').read_text())
    for row in manifest['files']:
        assert hashlib.sha256((STAGE / 'anomaly_workbench' / row['path']).read_bytes()).hexdigest() == row['sha256']
    print('STAGE', manifest['version'], len(manifest['files']), flush=True)
    tests = [STAGE / ('anomaly_workbench/tests/' + name) for name in ['test_live_api.py','test_chart_explanations.py','test_review_policy.py','test_recurrent_patterns.py','test_event_episodes.py','test_review_batch.py','test_stopping_context.py','test_queue_views.py','test_evidence_snapshot.py','test_maneuver_intent.py','test_fast_record_api.py']]
    subprocess.run([str(PYTHON), '-m', 'pytest', '-q', *(str(p) for p in tests)], cwd=APP, env=environment(), check=True)


def probe():
    sys.path[:0] = [str(STAGE), str(APP / 'src')]
    from anomaly_workbench import chart_explanations as charts
    root = SYSTEM / 'shared/anomaly-workbench-runtime/jobs' / SCAN_ID
    with gzip.open(root / 'result.json.gz', 'rt') as f:
        scan = json.load(f)
    url = f'http://127.0.0.1:8000/anomaly-workbench/api/scans/{SCAN_ID}/records/{RECORD_ID}'
    with urllib.request.urlopen(url, timeout=120) as response:
        item = json.load(response)
    facts, reference = charts.chart_evidence(scan, item)
    # Run once without hiding validation errors, then use the production wrapper.
    readings = charts._ask_model(facts, reference)
    print('REAL_MODEL_READINGS', json.dumps(readings, ensure_ascii=False), flush=True)
    for key in charts.PURPOSES:
        assert isinstance(readings[key], str)
    print('REAL_MODEL_PROBE_PASS', flush=True)


def deploy():
    source = STAGE / 'anomaly_workbench'
    previous = json.loads((PLUGIN / 'RELEASE_MANIFEST.json').read_text())
    latest = json.loads((source / 'RELEASE_MANIFEST.json').read_text())
    assert latest['version'] == '5.0-maneuver-context-and-fast-selection'
    previous_by_name = {r['path']: r for r in previous['files']}
    changed = []
    for row in latest['files']:
        current = PLUGIN / row['path']
        if current.is_file() and hashlib.sha256(current.read_bytes()).hexdigest() == row['sha256']:
            continue
        old = previous_by_name.get(row['path'])
        if old:
            assert hashlib.sha256(current.read_bytes()).hexdigest() == old['sha256'], f'Concurrent change: {row["path"]}'
        else:
            assert not current.exists(), f'Untracked existing file: {row["path"]}'
        changed.append(row['path'])
    changed.append('RELEASE_MANIFEST.json')
    pid = int((SYSTEM / 'logs/ui.pid').read_text())
    process = Path('/proc') / str(pid)
    cmd = process.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert 'uvicorn maritime_agent.api:app' in cmd and '--port 8000 ' in cmd
    assert Path(os.readlink(process / 'cwd')).resolve() == APP
    env = {}
    for raw in process.joinpath('environ').read_bytes().split(b'\0'):
        key, sep, value = raw.partition(b'=')
        if sep:
            env[key.decode()] = value.decode()
    backup = WORK / ('backup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    backup.mkdir(parents=True)
    for name in changed:
        before = PLUGIN / name
        if before.exists():
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(before, target)
    (backup / 'changed-files.json').write_text(json.dumps(changed))
    for name in changed:
        target = PLUGIN / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.chart-update')
        temporary.write_bytes((source / name).read_bytes())
        temporary.replace(target)
    os.kill(pid, signal.SIGTERM)
    for _ in range(80):
        if not process.exists():
            break
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=.2):
                pass
        except Exception:
            break
        time.sleep(.25)
    log = (SYSTEM / 'logs/ui.stdout.log').open('ab')
    err = (SYSTEM / 'logs/ui.stderr.log').open('ab')
    args = [str(PYTHON), '-m', 'uvicorn', 'maritime_agent.api:app', '--host', '127.0.0.1', '--port', '8000', '--workers', '1']
    child = subprocess.Popen(args, cwd=APP, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=err, start_new_session=True)
    (SYSTEM / 'logs/ui.pid').write_text(str(child.pid))
    for _ in range(100):
        if child.poll() is not None:
            raise RuntimeError('UI failed to restart; backup: ' + str(backup))
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1) as response:
                assert response.status == 200
            print('DEPLOYED', child.pid, 'CHANGED', changed, 'BACKUP', str(backup), flush=True)
            return
        except Exception:
            time.sleep(.3)
    raise RuntimeError('UI health timeout; backup: ' + str(backup))


if __name__ == '__main__':
    {'stage': stage, 'probe': probe, 'deploy': deploy}[sys.argv[1]]()
