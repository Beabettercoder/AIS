"""Matched 12-slot factorial arms; train/validation only, no test reads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=ROOT/'factorial_ablation_v1_20261027'
sys.path.insert(0,str(ROOT/'causal_gate_main_v1_20261027'))
import train_controls as old
hm=old.hm;th=old.th
ARMS=['raw_bundle','raw_common','raw_common_bundle','causal_bundle','old_bundle','comp_bundle',
      'raw_bundle_self','causal_bundle_self','causal_common_bundle_self',
      'raw_bundle_bgz','causal_bundle_bgz','causal_common_bundle_bgz']
BASE={'raw_bundle':'raw','raw_common_bundle':'raw_common','causal_bundle':'causal','old_bundle':'old','comp_bundle':'comp',
      'raw_bundle_self':'raw','causal_bundle_self':'causal','causal_common_bundle_self':'causal_common',
      'raw_bundle_bgz':'raw','causal_bundle_bgz':'causal','causal_common_bundle_bgz':'causal_common'}
hm.ARMS.clear();hm.ARMS.update({a:dict(key=a,nf=12,graph=False,recipe=(a!='raw_common'),masked=False) for a in ARMS})
original_loss=old.original_loss
def controlled_loss(model,batch,class_weights):
 if not model.config['recipe']:
  return old.controlled_loss(model,batch,class_weights)
 if model.arm.endswith('_self'):
  batch={**batch,'ep':batch['ep_self']}
 elif model.arm.endswith('_bgz'):
  batch={**batch,'ep':batch['ep_bgz']}
 return original_loss(model,batch,class_weights)
th.training_loss=controlled_loss
original_tensors=th.to_tensors
def with_conditions(d,x):
 out=original_tensors(d,x);out['ep_self']=torch.from_numpy(d['ep_self']);out['ep_bgz']=torch.from_numpy(d['ep_bgz']);return out
th.to_tensors=with_conditions
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_data():
 data,proofs=old.build_data()
 for split,d in data.items():
  with np.load(OUT/(split+'_conditions.npz'),allow_pickle=False) as c:
   assert np.array_equal(d['window_id'],c['window_id']) and np.array_equal(d['mask'],c['mask'])
   d['ep_self']=c['self_ep'].copy();d['ep_bgz']=c['background_ep'].copy()
  f=d['features'];raw=f['raw'];common=f['causal_common'][...,-1]
  f['raw_common']=raw.copy();f['raw_common'][...,-1]=common
  for arm,base in BASE.items():f[arm]=f[base].copy()
  for arm in ARMS:
   assert f[arm].shape==(len(d['y']),20,12) and np.isfinite(f[arm]).all()
  for arm,base in BASE.items():assert np.array_equal(f[arm],f[base])
 return data,proofs

def main():
 p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['tcn','gru'],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--preflight',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);device=torch.device(a.device)
 assert json.loads((OUT/'protocol.json').read_text(encoding='utf8'))['arms']==ARMS
 data,proofs=load_data();stats={k:th.fit_normalizer(data['train']['features'][k],data['train']['mask']) for k in ARMS}
 training={'epochs':25,'batch_size':128,'learning_rate':.002,'weight_decay':.01,'policy':'all'}
 if a.preflight:
  report={}
  for arm in ARMS:
   th.seed_all(0);model=hm.HistoricalModel(a.backbone,arm).to(device);x=th.normalized(data['train']['features'][arm],data['train']['mask'],stats[arm],False);z=with_conditions(data['train'],x)
   b=th.batch(z,slice(0,8),device);loss,logs=controlled_loss(model,b,torch.ones(5,device=device));loss.backward();assert torch.isfinite(loss)
   report[arm]={'loss':float(loss.detach().cpu()),'logs':logs,'train_parameters':sum(t.numel() for t in model.parameters()),'ep_mode':'self' if arm.endswith('_self') else 'bg_standardized' if arm.endswith('_bgz') else 'historical_background' if arm!='raw_common' else 'none'}
  p=OUT/('preflight_'+a.backbone+'.json');p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print('PREFLIGHT_OK',a.backbone,flush=True);return
 worker=OUT/'runs'/a.backbone;worker.mkdir(parents=True,exist_ok=False)
 config={'protocol_sha256':sha(OUT/'protocol.json'),'script_sha256':sha(Path(__file__)),'prior_script_sha256':sha(ROOT/'causal_gate_main_v1_20261027/train_controls.py'),
         'train_conditions_sha256':sha(OUT/'train_conditions.npz'),'val_conditions_sha256':sha(OUT/'val_conditions.npz'),'sources':proofs,'device':torch.cuda.get_device_name(device)}
 (worker/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf8');configsha=sha(worker/'configuration.json')
 for seed in (0,1,2):
  for arm in ARMS:
   th.train_one(a.backbone,arm,seed,data,stats[arm],training,device,worker/f'seed_{seed}'/arm,sha(OUT/'protocol.json'),configsha)
 print('ALL_COMPLETE',a.backbone,flush=True)
if __name__=='__main__':main()
