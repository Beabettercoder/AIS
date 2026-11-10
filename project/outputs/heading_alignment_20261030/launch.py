import subprocess,sys,os,json,hashlib
from pathlib import Path
out=Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030')
py='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
assert not (out/'launch.json').exists()
lock={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'train_heading.py',out/'test_heading.py']}
(out/'code_lock.json').write_text(json.dumps(lock,indent=2))
workers=[]
for name,gpu in [('tcn',1),('gru',2),('transformer',5),('itransformer',7)]:
 log=(out/(name+'.log')).open('w')
 p=subprocess.Popen([py,str(out/'worker.py'),name,str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=out)
 workers.append({'model':name,'gpu':gpu,'pid':p.pid})
(out/'launch.json').write_text(json.dumps(workers,indent=2))
print(json.dumps(workers))
