"""Offline live overview of training progress. Does not load data or torch."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


def read_json(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError): return None


def number(value,digits=2):
    return '-' if value is None else f'{value:.{digits}f}'


def bar(done,total,width=20):
    value=min(1.,max(0.,done/max(total,1)))
    fill=int(width*value)
    return '['+'#'*fill+'-'*(width-fill)+f'] {100*value:5.1f}%'


def last_event(path):
    try:
        with path.open('rb') as f:
            f.seek(0,2); end=f.tell(); f.seek(max(0,end-8192))
            lines=f.read().decode('utf-8',errors='replace').splitlines()
        return json.loads(lines[-1]) if lines else None
    except (OSError,ValueError): return None


def snapshot(root,expected):
    roots=[root] if isinstance(root,Path) else root
    rows=[]
    recent=[]
    paths=sorted({p.resolve() for r in roots for p in r.rglob('progress.json')})
    for path in paths:
        row=read_json(path)
        if row and all(k in row for k in ('model','seed','arm','stage','total_epochs')):
            rows.append(row)
            ev=last_event(path.with_name('events.jsonl'))
            if ev: recent.append((ev.get('utc',''),row,ev))
    rows.sort(key=lambda x:(x['model'],x['seed'],x['arm']))
    completed=sum(r['stage']=='complete' for r in rows)
    running=sum(r['stage'] not in ('complete','error','interrupted') for r in rows)
    failed=sum(r['stage']=='error' for r in rows)
    text=[f'AIS forecasting pilot | {datetime.now():%Y-%m-%d %H:%M:%S}',
          'Directories: '+', '.join(str(r.resolve()) for r in roots),
          f'Complete {completed}/{expected} | Active/stale {running} | Errors {failed} | '
          f'Not started {max(0,expected-len(rows))}',
          'Completed runs '+bar(completed,expected),
          'Model         Seed Arm Stage              Epoch     Batch     Loss       Val ADE/FDE (m)  Update age',
          '-'*114]
    for r in rows:
        stage=r['stage']
        try:
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(r['updated_utc'])).total_seconds()
        except (KeyError,ValueError): age=0
        e=f"{r.get('completed_epochs',0)}/{r['total_epochs']}"
        b=f"{r.get('batch',0)}/{r.get('total_batches',0)}"
        text.append(f"{r['model']:<13} {r['seed']:<4} {r['arm']:<3} {stage:<18} {e:<9} {b:<9} "
                    f"{number(r.get('train_loss'),6):<10} "
                    f"{number(r.get('val_ADE_m')):>7}/{number(r.get('val_FDE_m')):<7} {age:6.0f}s")
        if r.get('message'): text.append('  '+r['message'])
        if stage in ('training','validation','epoch_complete','final_evaluation'):
            fractional=r.get('completed_epochs',0)
            if stage=='training': fractional+=r.get('batch',0)/max(r.get('total_batches',1),1)
            text.append('  Epoch progress '+bar(fractional,r['total_epochs']))
    if not rows: text.append('Waiting for first progress.json. Run train.py in another terminal.')
    if recent:
        _,r,ev=max(recent,key=lambda t:t[0])
        text.append(f"\nLatest event: {r['model']} seed {r['seed']} {r['arm']} | {ev['event']} | {ev['utc']}")
    text.append('\nAn old update can indicate slow evaluation or a stopped process; status alone does not prove liveness.')
    text.append('Each run also has events.jsonl and history.csv. Ctrl+C closes only this viewer.')
    return '\n'.join(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,nargs='+',required=True,
                   help='One or more output roots; nested run directories are scanned recursively')
    p.add_argument('--interval',type=float,default=1.)
    p.add_argument('--expected',type=int,default=48,help='Expected runs across all terminals')
    p.add_argument('--once',action='store_true')
    args=p.parse_args()
    if args.interval<=0: p.error('interval must be positive')
    try:
        while True:
            if sys.stdout.isatty() and not args.once: print('\033[2J\033[H',end='')
            print(snapshot(args.out,args.expected),flush=True)
            if args.once: break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nViewer closed; training processes continue.')


if __name__=='__main__': main()
