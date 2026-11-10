"""Training-only z-scores for historic background and observable target ep."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OLD=ROOT/'gating_training_v2_20261027/data'
NEW=ROOT/'causal_gate_main_v1_20261027'
LAB=ROOT/'labeling/labeled_v1/features_scene'
OUT=ROOT/'factorial_ablation_v1_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

parts={s:[] for s in ('train','val')};sources=[]
for split in parts:
 for rg in REGIONS:
  op=OLD/rg/(split+'.npz');lp=LAB/rg/(split+'.npz')
  with np.load(op,allow_pickle=False) as z:
   mask=z['mask'].copy();raw=z['raw_legacy7'].copy();w=z['w_self'].copy();ids=z['window_id'].copy();bg=z['ep'].copy()
  with np.load(lp,allow_pickle=False) as z:
   assert np.array_equal(ids,z['window_id'])
   tim=z['time_s'].copy();err=z['e_kin_mps'].copy();kin=z['kinematic_valid'].copy();iv=z['time_interval_valid'].copy()
  speed=np.linalg.norm(raw[...,2:4],axis=-1);dt=np.diff(tim,axis=1);dv=np.abs(np.diff(speed,axis=1));acc=np.zeros_like(speed)
  ok=mask[:,1:]&mask[:,:-1]&iv[:,1:]&np.isfinite(dt)&(dt>0)&np.isfinite(dv)
  acc[:,1:]=np.divide(dv,dt,out=np.zeros_like(dv,dtype=np.float64),where=ok).astype(np.float32)
  selfx=np.stack((speed,acc,err,w),axis=-1).astype(np.float32)
  valid=np.zeros_like(selfx,dtype=bool);valid[...,0]=mask&np.isfinite(speed);valid[:,1:,1]=ok
  valid[...,2]=mask&kin&np.isfinite(err);valid[...,3]=mask&kin&np.isfinite(w)
  selfx[~valid]=0.;bv=np.broadcast_to(mask[...,None],bg.shape).copy()
  assert np.isfinite(selfx).all() and np.isfinite(bg).all()
  parts[split].append({'ids':ids,'self':selfx,'self_valid':valid,'background':bg,'background_valid':bv,'mask':mask})
  sources.append({'region':rg,'split':split,'prior_sha256':sha(op),'observed_sha256':sha(lp)})

def merge(split,key):return np.concatenate([x[key] for x in parts[split]])
stats={}
for typ in ('self','background'):
 x=merge('train',typ);v=merge('train',typ+'_valid');assert x.shape==(7840,20,4)
 channels=[]
 for k in range(4):
  a=x[...,k][v[...,k]].astype(float);assert len(a)>0
  channels.append({'mean':float(a.mean()),'std':float(max(a.std(),1e-6)), 'eligible_count':len(a),'minimum':float(a.min()),'maximum':float(a.max())})
 stats[typ]=channels
for split in ('train','val'):
 out={'window_id':merge(split,'ids'),'mask':merge(split,'mask')}
 for typ in ('self','background'):
  x=merge(split,typ);valid=merge(split,typ+'_valid');mean=np.array([d['mean'] for d in stats[typ]],dtype=np.float32);sd=np.array([d['std'] for d in stats[typ]],dtype=np.float32)
  normalized=np.clip((x-mean)/sd,-5,5);normalized[~valid]=0.
  assert np.isfinite(normalized).all();out[typ+'_ep']=normalized.astype(np.float32);out[typ+'_valid']=valid
 p=OUT/(split+'_conditions.npz');assert not p.exists();np.savez_compressed(p,**out)
stats.update(protocol_sha256=sha(OUT/'protocol.json'),sources=sources,outputs={s:sha(OUT/(s+'_conditions.npz')) for s in ('train','val')})
(OUT/'condition_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print(json.dumps({'self':stats['self'],'background':stats['background'],'outputs':stats['outputs']},indent=2),flush=True)
