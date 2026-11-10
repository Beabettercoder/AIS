"""User-authorized graph acceleration; trainer creates atomic output directory."""
import json,subprocess,time,hashlib,concurrent.futures
from pathlib import Path
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python';record=P/'graph_accelerated_launch.json';result=P/'graph_accelerated_result.json';assert not record.exists() and not result.exists()
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for seed in range(3):assert not (P/'runs'/'graph'/f'seed{seed}').exists()
record.write_text(json.dumps({'script_sha256':sh(P/'train_single.py'),'protocol_sha256':sh(P/'frozen_train_protocol.json'),'model':'graph','seeds':[0,1,2],'gpus':[3,5,6],'epochs':25,'safety':'trainer mkdir(exist_ok=False) claims each target; later queued attempt must refuse overwrite'},indent=2))
def run(job):
 seed,gpu=job;st=time.time(); log=P/f'graph_seed{seed}_accelerated.log'
 with log.open('x') as f:r=subprocess.run([PY,str(P/'train_single.py'),'--name','graph','--seed',str(seed),'--gpu',str(gpu)],stdout=f,stderr=subprocess.STDOUT)
 out={'seed':seed,'gpu':gpu,'exit_code':r.returncode,'seconds':round(time.time()-st,2),'training_log_sha256':sh(log)};print('ACCELERATED_GRAPH',out,flush=True);return out
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:allr=list(pool.map(run,[(0,3),(1,5),(2,6)]))
result.write_text(json.dumps({'results':allr,'launch_sha256':sh(record)},indent=2))
