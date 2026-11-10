from pathlib import Path
import json,subprocess,sys,os
O=Path('/mnt/sda/WangZijia/AIS/outputs/mixed_stress_probe_20261031')
J={2:['tcn:0','tcn:1'],3:['tcn:2','gru:0'],5:['itransformer:0'],6:['itransformer:1'],7:['itransformer:2','transformer:0']}
if (O/'launch.json').exists() or (O/'runs').exists():raise SystemExit('Existing outputs: refuse to overwrite')
procs=[]
for gpu,jobs in J.items():
 args=[sys.executable,str(O/'worker.py'),'--gpu',str(gpu)]
 for job in jobs:args.extend(['--job',job])
 log=(O/('gpu'+str(gpu)+'.log')).open('x');p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=str(O));log.close();procs.append(dict(gpu=gpu,pid=p.pid,jobs=jobs))
(O/'launch.json').write_text(json.dumps(procs,indent=2),encoding='utf8');print(json.dumps(procs))
