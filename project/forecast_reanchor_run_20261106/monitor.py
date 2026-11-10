"""Continuous read-only progress display for the complete research workflow."""
import argparse,time,sys
from pathlib import Path
from common import read,ROOT
sys.path.insert(0,str(ROOT/'frozen_pilot'));from watch import snapshot

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('work'));p.add_argument('--once',action='store_true');a=p.parse_args()
    try:
        while True:
            if sys.stdout.isatty():print('\033[2J\033[H',end='')
            state=a.work/'pipeline_status.json'
            print('Pipeline:',read(state) if state.exists() else 'Not started')
            print('48 model slots = 36 newly trained + 12 reused Raw models')
            print(snapshot(a.work/'runs',48))
            if (a.work/'test_progress.json').exists():print('TEST:',read(a.work/'test_progress.json'))
            if a.once:break
            time.sleep(1)
    except KeyboardInterrupt:print('Viewer closed; training is unaffected.')
