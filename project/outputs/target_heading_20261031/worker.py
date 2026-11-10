from pathlib import Path
import os,sys,subprocess
out=Path('/mnt/sda/WangZijia/AIS/outputs/target_heading_20261031')
gpu=int(sys.argv[1]);jobs={3:[('tcn',0),('gru',0)],5:[('tcn',1),('transformer',0)],6:[('tcn',2)]}[gpu]
for name,seed in jobs:
 c=[sys.executable,str(out/'train_target.py'),'--name',name,'--seed',str(seed),'--gpu',str(gpu),'--balance','window']
 print('START',name,seed,'GPU',gpu,flush=True)
 p=subprocess.run(c)
 if p.returncode:raise SystemExit(p.returncode)
 print('COMPLETE',name,seed,flush=True)
