"""Run matched 12-slot BCE and historical-bundle models on train/val only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
SRC=ROOT/'gating_training_v2_20261027/data'
FIELD=ROOT/'robust_background_v1_20261027'
OUT=ROOT/'causal_gate_main_v1_20261027'
sys.path.insert(0,str(ROOT/'gating_training_v2_20261027/historical'))
import historical_model as hm
arms=['raw','old','comp','causal','causal_common','affine','mean','median','weighted_median','causal_common_bundle','affine_bundle','weighted_median_bundle']
hm.ARMS.clear();hm.ARMS.update({a:dict(key=a,nf=12,graph=False,recipe=a.endswith('_bundle'),masked=False) for a in arms})
import train_historical as th
original_loss=hm.training_loss
def controlled_loss(model,batch,class_weights):
 if model.config['recipe']: return original_loss(model,batch,class_weights)
 pb,_,_,_=model(batch['x'],batch['mask'],batch['nb_seq'],batch['nb_xy'],batch['nb_mask'],batch['nb_w'],batch['rf'])
 loss=F.binary_cross_entropy_with_logits(pb,batch['y']);v=float(loss.detach())
 return loss,dict(binary=v,type_ce=0.,js=0.,aug_binary=0.,hidden_smooth=0.,epsilon_mean=0.,total=v)
th.training_loss=controlled_loss

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build_data():
 all_data={}; proofs=[]
 metas=th.META; keys=('y','label_class','mask','ep','rf','nb_seq','nb_xy','nb_w','nb_mask')+metas
 for split in ('train','val'):
  parts=[]
  for r in th.REGIONS:
   p=SRC/r/(split+'.npz');q=OUT/'gate_data'/r/(split+'.npz');s=FIELD/r/(split+'.npz')
   with np.load(p,allow_pickle=False) as z:d={k:z[k] for k in z.files}
   with np.load(q,allow_pickle=False) as z:g={k:z[k] for k in z.files}
   with np.load(s,allow_pickle=False) as z:f={k:z[k] for k in z.files}
   assert np.array_equal(d['window_id'],g['window_id']) and np.array_equal(d['window_id'],f['window_id'])
   w=d['w_self'].astype(np.float32);v=d['kinematic_valid'].astype(np.float32);common=f['common_valid'].astype(np.float32);zero=np.zeros_like(w)
   zero2=np.zeros((*w.shape,2),np.float32)
   def pack(base,res,weight,valid,avail):return np.concatenate((base,res,weight[...,None],valid[...,None],avail[...,None]),axis=-1).astype(np.float32)
   features={}
   features['raw']=pack(d['raw_legacy7'],zero2,zero,zero,zero)
   for name,base in [('old',d['old_gate9'][...,:7]),('comp',d['comp_gate9'][...,:7]),('causal',g['causal7'])]:features[name]=pack(base,zero2,w,v,zero)
   base=g['causal7'];features['causal_common']=pack(base,zero2,w,v,common)
   for name in ('affine','mean','median','weighted_median'):
    res=f['residual_'+name+'_mps'].astype(np.float32)
    assert np.all(res[~f['common_valid']]==0.)
    features[name]=pack(base,res,w,v,common)
   for name in ('causal_common','affine','weighted_median'):features[name+'_bundle']=features[name].copy()
   d['features']=features
   for k in metas:d[k]=d[k].astype(bool) if k in ('observation_noise','target_noise_effective','neighbor_noise_effective','observation_noise_effective') else d[k].astype(str)
   parts.append(d);proofs.append({'region':r,'split':split,'input_sha256':sha(p),'gate_sha256':sha(q),'field_sha256':sha(s)})
  all_data[split]={k:np.concatenate([d[k] for d in parts]) for k in keys}
  all_data[split]['features']={k:np.concatenate([d['features'][k] for d in parts]) for k in arms}
  assert len(all_data[split]['y'])=={'train':7840,'val':1680}[split]
 for key in ['mmsi','window_id','base_window_id']:
  assert not set(all_data['train'][key])&set(all_data['val'][key])
 for split,d in all_data.items():
  for k in arms:assert d['features'][k].shape==(len(d['y']),20,12) and np.isfinite(d['features'][k]).all()
  for base in ('causal_common','affine','weighted_median'):
   assert np.array_equal(d['features'][base],d['features'][base+'_bundle'])
 return all_data,proofs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--backbone',required=True,choices=['tcn','gru']);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);device=torch.device(args.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 protocol=json.loads((OUT/'training_protocol.json').read_text());assert protocol['arms']==arms
 data,proofs=build_data();stat={k:th.fit_normalizer(data['train']['features'][k],data['train']['mask']) for k in arms}
 config={'protocol_sha256':sha(OUT/'training_protocol.json'),'script_sha256':sha(Path(__file__)),'proofs':proofs,'backbone':args.backbone,'device':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu','seeds':[0,1,2]}
 worker=OUT/'runs'/args.backbone;worker.mkdir(parents=True,exist_ok=False)
 (worker/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf8');configsha=sha(worker/'configuration.json')
 training={'epochs':25,'batch_size':128,'learning_rate':.002,'weight_decay':.01,'policy':'all'}
 for seed in (0,1,2):
  for arm in arms:
   th.train_one(args.backbone,arm,seed,data,stat[arm],training,device,worker/f'seed_{seed}'/arm,sha(OUT/'training_protocol.json'),configsha)
 print('ALL_COMPLETE',args.backbone,flush=True)
if __name__=='__main__':main()
