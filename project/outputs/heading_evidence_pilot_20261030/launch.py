import json,subprocess,sys
from pathlib import Path
O=Path('/mnt/sda/WangZijia/AIS/outputs/heading_evidence_pilot_20261030')
assert (O/'frozen_protocol.json').exists() and not (O/'launch.json').exists() and not (O/'runs').exists()
jobs={5:[['tcn',0],['gru',0],['itransformer',0]],6:[['tcn',1],['itransformer',1]],7:[['tcn',2],['transformer',0],['itransformer',2]]}
res={}
for gpu,tasks in jobs.items():
 f=(O/f'gpu{gpu}.log').open('w');cmd=[sys.executable,str(O/'worker.py'),'--gpu',str(gpu),'--jobs',json.dumps(tasks)]
 proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 f.close();res[gpu]={'pid':proc.pid,'jobs':tasks}
(O/'launch.json').write_text(json.dumps(res,indent=2),encoding='utf8');print(json.dumps(res))
