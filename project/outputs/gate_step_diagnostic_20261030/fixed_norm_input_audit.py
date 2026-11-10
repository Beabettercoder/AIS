from pathlib import Path
import importlib.util, json, hashlib, numpy as np
BASE=Path('/mnt/sda/WangZijia/AIS/outputs');OLD=BASE/'relation_repair_20261030';NEW=BASE/'neighbor_gate_safe_20261030';OUT=BASE/'gate_step_diagnostic_20261030'
s=importlib.util.spec_from_file_location('safe_train',NEW/'train.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def read(path):
 with np.load(path,allow_pickle=False) as f: d={k:f[k] for k in f.files}
 d['mask']=np.arange(20)[None,:]<d['length'][:,None]
 return d
def prep(x,cm,mask,mu,sd):
 x=(x-mu)/sd;x[:,:,8:]*=cm[...,None];x[~mask]=0
 return x.astype(np.float32)
old_train=read(OLD/'train_relation_field.npz')
mu,sd=m.fit(old_train)
report={'scope':'fixed normalizer originally fit on training base old F0/F1; no test labels','normalizer_mean':mu.tolist(),'normalizer_std':sd.tolist(),'splits':{}}
for split in ('train','val'):
 a=old_train if split=='train' else read(OLD/'val_relation_field.npz');b=read(NEW/(split+'_relation_field.npz'))
 out={}
 for v in m.VIEWS:
  out[v]={}
  for arm in ('C0','F0','F1'):
   xa,ma=m.features(a,v,arm);xb,mb=m.features(b,v,arm)
   x=prep(xa,ma,a['mask'],mu,sd);y=prep(xb,mb,b['mask'],mu,sd)
   err=float(np.max(np.abs(x-y)));count=int(np.sum(np.any(x!=y,axis=-1)))
   if arm in ('C0','F0'):assert np.array_equal(x,y),(split,v,arm,err,count)
   out[v][arm]={'max_abs':err,'changed_steps':count,'masks_equal':bool(np.array_equal(ma,mb))}
 report['splits'][split]=out
(OUT/'fixed_norm_input_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
