import argparse,json,subprocess,sys
from pathlib import Path
O=Path('/mnt/sda/WangZijia/AIS/outputs/heading_evidence_pilot_20261030')
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--jobs',required=True);a=p.parse_args()
for name,seed in json.loads(a.jobs):
 cmd=[sys.executable,str(O/'train_heading_evidence.py'),'--name',name,'--seed',str(seed),'--gpu',str(a.gpu)]
 print('START',cmd,flush=True);r=subprocess.run(cmd,check=False)
 if r.returncode:print('FAILED',name,seed,r.returncode,flush=True);sys.exit(r.returncode)
 print('FINISH',name,seed,flush=True)
print('WORKER COMPLETE',a.gpu,flush=True)
