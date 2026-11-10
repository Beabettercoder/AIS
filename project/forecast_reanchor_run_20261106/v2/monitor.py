import argparse,json,time,sys
from pathlib import Path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path(__file__).resolve().parents[1]/'work_v2');p.add_argument('--once',action='store_true');a=p.parse_args()
    try:
        while True:
            if sys.stdout.isatty():print('\033[2J\033[H',end='')
            file=a.work/'progress.json'
            try:print(json.dumps(json.loads(file.read_text()),ensure_ascii=False,indent=2))
            except (OSError,ValueError):print('Waiting for v2 pipeline')
            if a.once:break
            time.sleep(1)
    except KeyboardInterrupt:pass
