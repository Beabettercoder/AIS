from pathlib import Path
import json,subprocess,time,sys
O=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/relation');P='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
while not (O/'eval_launch.json').exists():
 p=O/'pipeline_status.json'
 if p.exists() and json.loads(p.read_text()).get('phase')=='failed':print('Upstream failed; no dataset audit',flush=True);sys.exit(1)
 time.sleep(10)
with (O/'dataset_audit.log').open('x') as f:r=subprocess.run([P,str(O/'scripts/dataset_audit.py')],stdout=f,stderr=subprocess.STDOUT)
if r.returncode:print('DATASET AUDIT FAILED',r.returncode,flush=True);sys.exit(r.returncode)
print('DATASET AUDIT COMPLETE',flush=True)
