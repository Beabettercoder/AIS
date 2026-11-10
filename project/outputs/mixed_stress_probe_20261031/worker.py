from pathlib import Path
import argparse,subprocess,sys,json
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/mixed_stress_probe_20261031')
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--job',action='append',required=True);a=p.parse_args()
for j in a.job:
 name,seed=j.split(':')
 if a.gpu==4:raise SystemExit('GPU4 reserved')
 subprocess.run([sys.executable,str(OUT/'train_mixed.py'),'--name',name,'--seed',seed,'--gpu',str(a.gpu)],check=True)
print('WORKER COMPLETE',a.gpu,flush=True)
