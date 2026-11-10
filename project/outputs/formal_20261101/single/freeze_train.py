import json,hashlib,sys,shutil,platform
from pathlib import Path
import numpy as np,torch
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single')
PREV=Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028')
sys.path.insert(0,str(OUT));import train_single as exp

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
train=exp.load('train');val=exp.load('val')
assert len(train['y'])==7840 and len(val['y'])==1680
assert not set(train['mmsi'])&set(val['mmsi'])
norm=exp.fit(train)
config={'title':'single-vessel full requested experiment','requested':'2026-11-01 user explicitly authorizes without previous performance gate','models':['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'],'seeds':[0,1,2], 'arms':['raw','causal_bundle','f0_bundle','f1_bundle'],'epochs':25,'batch':128,'order':'default_rng(300000+epoch), same across seeds and arms, matches original pilot','optimizer':'AdamW lr=.002 wd=.01; Mamba declared no-decay tags only','normalizer':'pooled raw/gated target + pooled F0/F1 valid field on training only','objective':'original_mainline_pilot_20261029 adapted step35 bundle, BCE raw; graph no neighbor node, iTransformer/PatchTST smooth time proxies','eval':'final epoch 25; previously exposed test, never label as blind; thresholds chosen on validation negatives before test','runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},'data':{'train':len(train['y']),'val':len(val['y']),'target_mmsi_overlap':0},'normalizer_arrays':{k:v.tolist() for k,v in norm.items()}}
deps=[OUT/'train_single.py',OUT/'eight_backbone_adapter.py',OUT/'verify_reuse.py',PREV/'pilot_train.py',PREV/'pilot_features.py',PREV/'pilot_historical_loss.py',ROOT/'models.py',PREV/'rho_train.npz',PREV/'rho_val.npz',ROOT/'data/individual/train.npz',ROOT/'data/individual/val.npz',ROOT/'extra_data/field_train.npz',ROOT/'extra_data/field_val.npz',OUT/'seed0_reuse_audit.json']
config['sha256']={str(p):sha(p) for p in deps}
assert not (OUT/'frozen_train_protocol.json').exists()
(OUT/'frozen_train_protocol.json').write_text(json.dumps(config,indent=2))
reuse=json.loads((OUT/'seed0_reuse_audit.json').read_text()); copied=[]
for name in ['tcn','gru','transformer','itransformer']:
 d=OUT/'runs'/name/'seed0';d.mkdir(parents=True,exist_ok=False)
 shutil.copy2(PREV/'runs'/name/'protocol.json',d/'protocol_source_pilot.json')
 for arm in config['arms']:
  assert reuse[name][arm]['eligible_reuse'];src=PREV/'runs'/name/arm;dst=d/arm;shutil.copytree(src,dst)
  assert sha(dst/'final.pt')==reuse[name][arm]['old_checkpoint_sha256']
  assert sha(dst/'final_val.npz')==reuse[name][arm]['old_predictions_sha256']
  copied.append({'name':name,'seed':0,'arm':arm,'path':str(dst),'checkpoint_sha':sha(dst/'final.pt'),'prediction_sha':sha(dst/'final_val.npz')})
 (d/'reused_pilot.json').write_text(json.dumps({'source':str(PREV/'runs'/name),'audit_sha':sha(OUT/'seed0_reuse_audit.json'),'note':'architecture/initialization/validation predictions verified byte-identical in own run'},indent=2))
(OUT/'reused_16_manifest.json').write_text(json.dumps(copied,indent=2));print('FROZEN_AND_REUSED',len(copied),sha(OUT/'frozen_train_protocol.json'))
