"""User-authorized acceleration: single seed2 process, atomic target directory prevents overwrite."""
import json,subprocess,time,hashlib
from pathlib import Path
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');target=P/'runs/neuralcde/seed2';record=P/'cde2_accelerated_launch.json';result=P/'cde2_accelerated_result.json';assert not target.exists() and not record.exists() and not result.exists()
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
script=P/'train_single.py';manifest={'script_sha256':sh(script),'frozen_protocol_sha256':sh(P/'frozen_train_protocol.json'),'model':'neuralcde','seed':2,'gpu':0,'epochs':25,'safety':'trainer mkdir(exist_ok=False) reserves target atomically; later queued attempt must refuse overwrite'};record.write_text(json.dumps(manifest,indent=2))
start=time.time()
with (P/'cde2_accelerated.log').open('x') as log:
 r=subprocess.run(['/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python',str(script),'--name','neuralcde','--seed','2','--gpu','0'],stdout=log,stderr=subprocess.STDOUT)
result.write_text(json.dumps({'exit_code':r.returncode,'wall_seconds':time.time()-start,'target':str(target),'launch_sha256':sh(record)},indent=2))
print('ACCELERATED_CDE2',r.returncode,flush=True)
