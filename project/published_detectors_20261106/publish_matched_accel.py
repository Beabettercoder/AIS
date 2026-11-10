#!/usr/bin/env python3
"""Safely publish a completed independent result to a live serial queue.

All publishers take the same flock; queue is paused only for inspection and
the no-replace rename. A separate watchdog resumes the queue if this process
is killed without running its finally block.
"""
import ctypes,fcntl,os,signal,subprocess,sys,time,json
from pathlib import Path

root=Path('/home/ps/AIS_published_detectors_20261107')
queue_pid=int(sys.argv[1]);model,arm,seed=sys.argv[2:5]
src=root/'eight_matched_accel'/'relation'/model/arm/('seed'+seed)
dst=root/'eight_matched'/'relation'/model/arm/('seed'+seed)
for name in ('last.pt','best_val.pt','test_scores.npz','test_metrics.json'):
    if not (src/name).is_file():raise SystemExit('incomplete source: '+str(src/name))

def interrupted(signum,frame):raise RuntimeError('publisher interrupted')
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
libc=ctypes.CDLL(None,use_errno=True);fn=libc.renameat2
fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
AT_FDCWD=-100;RENAME_NOREPLACE=1
with (root/'logs/eight_publish.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    os.kill(queue_pid,signal.SIGSTOP)
    # Watchdog: if this publisher is killed before finally, unfreeze the queue.
    watchdog='import os,signal,sys,time\np=int(sys.argv[1]); q=int(sys.argv[2])\nwhile True:\n try: os.kill(p,0)\n except ProcessLookupError:\n  try: os.kill(q,signal.SIGCONT)\n  except ProcessLookupError: pass\n  break\n time.sleep(.5)'
    subprocess.Popen([sys.executable,'-c',watchdog,str(os.getpid()),str(queue_pid)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    try:
        for _ in range(100):
            status=Path(f'/proc/{queue_pid}/status').read_text()
            if any(line.startswith('State:') and '\tT' in line for line in status.splitlines()):break
            time.sleep(.01)
        else:raise RuntimeError('queue did not stop')
        log=(root/'logs/eight_matched_queue.log').read_text(errors='replace')
        marker=f'START relation {model} {arm} seed={seed}'
        if marker in log.splitlines() or dst.exists():
            print('NOT_PUBLISHED already reserved or exists',str(dst));sys.exit(0)
        # Inspect exact active train AND test argument tuples; never stage if one
        # exists, even when it has not yet created its output directory.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:argv=(proc/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            args=[x.decode(errors='replace') for x in argv if x]
            if not any(x.endswith('train_eight_matched.py') for x in args):continue
            def arg(name):
                try:return args[args.index(name)+1]
                except (ValueError,IndexError):return None
            if (arg('--task'),arg('--name'),arg('--arm'),arg('--seed'))==('relation',model,arm,seed):
                print('NOT_PUBLISHED matching train/test process active',proc.name);sys.exit(0)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if fn(AT_FDCWD,os.fsencode(src),AT_FDCWD,os.fsencode(dst),RENAME_NOREPLACE)!=0:
            code=ctypes.get_errno();raise OSError(code,os.strerror(code))
        print('PUBLISHED',str(dst),flush=True)
    finally:
        try:os.kill(queue_pid,signal.SIGCONT)
        except ProcessLookupError:pass
