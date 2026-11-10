from pathlib import Path
import json,hashlib,subprocess,os
p=Path('/mnt/sda/WangZijia/AIS/outputs/itrans_stability_20261030')
h=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
spec={'scope':'pre-frozen seeds 0,1,2 paired iTransformer validation stability','parent_seed0':'/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030/runs/aligned_embed/itransformer','new_seeds':[1,2],'GPU':[2,5],'variant':'aligned_embed','arms':['C0','F0','F1'],'epochs':25,'batch_order':'same as original seed0; deterministic 300000+epoch','metrics':'Base, natural, suffix, combined; paired target MMSI clustered bootstrap; pooled AP','test':'no access','script_sha256':h(p/'train_stability.py')}
(p/'frozen_protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
for seed,gpu in zip([1,2],[2,5]):
 env=os.environ.copy();env['STABILITY_SEED']=str(seed)
 with (p/f'seed{seed}.log').open('w') as log:
  proc=subprocess.Popen(['/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python',str(p/'train_stability.py'),'--name','itransformer','--gpu',str(gpu),'--variant','aligned_embed','--arms','C0,F0,F1'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 print(seed,proc.pid,flush=True)
