import hashlib,json,subprocess
from pathlib import Path
out=Path('/mnt/sda/WangZijia/AIS/outputs/neighbor_gate_safe_20261030')
p=json.loads((out/'frozen_protocol.json').read_text())
for k,n in [('builder','builder.py'),('train','train.py')]:assert hashlib.sha256((out/n).read_bytes()).hexdigest()==p['sha256'][k]
assert not any((out/(s+'_relation_field.npz')).exists() for s in ['train','val'])
results=[]
for split in ['train','val']:
 f=(out/('build_'+split+'.log')).open('ab')
 proc=subprocess.Popen(['/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python','builder.py','--split',split],cwd=out,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 results.append({'split':split,'pid':proc.pid})
(out/'build_launch.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
