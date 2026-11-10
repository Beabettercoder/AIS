from pathlib import Path
import json,hashlib,subprocess,os
p=Path('/mnt/sda/WangZijia/AIS/outputs/itrans_stability_20261030/extended75')
h=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
spec={'decision':'frozen before training to inspect potential late learning with equal budget','justification':'F0 seed0/1 near ln2 at 25 while seed2 learned; F1 seeds learned late; no test-set used','seeds':[0,1,2],'gpu':[2,5,7],'arms':['F0','F1'],'epoch':75,'checkpoint':'fixed epoch75 no validation selection','optimizer':'same as pilot AdamW and training RNG initialization, from scratch','control':'paired F0/F1 same initialization and same steps each seed','comparisons':'fixed 3-seed mean AP differences and target MMSI bootstrap over paired predictions, all 4 views','scope':'existing exposed development validation; no test','script_sha256':h(p/'train_75.py')}
(p/'frozen_protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
for seed,gpu in zip([0,1,2],[2,5,7]):
 env=os.environ.copy();env['STABILITY_SEED']=str(seed)
 with (p/f'seed{seed}.log').open('w') as log:
  proc=subprocess.Popen(['/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python',str(p/'train_75.py'),'--name','itransformer','--gpu',str(gpu),'--variant','aligned_embed','--arms','F0,F1'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 print(seed,proc.pid,flush=True)
