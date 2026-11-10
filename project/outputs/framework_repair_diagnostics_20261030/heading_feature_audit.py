import importlib.util,sys,copy,json,hashlib
from pathlib import Path
import numpy as np
root=Path('/mnt/sda/WangZijia/AIS/outputs'); path=root/'heading_alignment_20261030/train_heading.py';sp=importlib.util.spec_from_file_location('heading',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def audit(split):
 d=m.read(split); y=d['y'].copy(); out={}
 for view in m.VIEWS:
  ref0,mask0=m.features(d,view,'C0'); assert np.count_nonzero(ref0[:,:,m.TARGET:])==0
  for arm in ('F0','F1'):
   a,valid=m.features(d,view,arm);assert np.isfinite(a).all() and np.count_nonzero(a[~d['mask']])==0 and np.count_nonzero(a[:,:,m.TARGET:][~valid])==0
   z=dict(d); z['y']=1-y; other,_=m.features(z,view,arm);assert np.array_equal(a,other)
   dd={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in d.items()}; theta=0.83;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
   for key in ('causal7',):
    dd[key][:,:,2:4]=np.einsum('ij,btj->bti',R,dd[key][:,:,2:4]);dd[key][:,:,5:7]=np.einsum('ij,btj->bti',R,dd[key][:,:,5:7])
   k=view+'__field'; dd[k][:,:,:,:2]=np.einsum('ij,batj->bati',R,dd[k][:,:,:,:2]); dd[k][:,:,:,2:4]=np.einsum('ij,batj->bati',R,dd[k][:,:,:,2:4])
   b,_=m.features(dd,view,arm);err=np.max(np.abs(a[:,:,24:]-b[:,:,24:])); assert err<1e-5,(split,view,arm,err)
   out[view+'_'+arm]={'valid':int(valid.sum()),'derived_rotation_max_error':float(err)}
 return out
tr=m.read('train'); mu,sd=m.fit(tr);assert np.isfinite(mu).all() and np.isfinite(sd).all() and (sd>0).all()
r={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checks':'label perturbation, missing and padding zero, coordinate rotation for derived channels only, finite training normalizer','train':audit('train'),'val':audit('val')};p=root/'framework_repair_diagnostics_20261030/heading_feature_audit.json';p.write_text(json.dumps(r,indent=2));print(p.read_text())
