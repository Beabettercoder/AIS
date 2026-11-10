"""Numerical identity against original 35-D feature and Heading standardization implementation; no scoring."""
from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'hawaii_event_feasibility_20261031';sys.path.insert(0,str(R/'heading_alignment_20261030'));import train_heading as old
frozen=json.loads((O/'frozen_ood_models_thresholds_v2.json').read_text());
with np.load(O/'ood_inputs_v2.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
d={'mask':a['mask'],'time_s':a['time_s'],'causal7':a['causal7'],'base__field':a['field'],'base__count':a['count']}; result={}
protos=[json.loads(Path(m['model_protocol']).read_text()) for m in frozen['models'] if m['training']=='base'];mu=np.asarray(protos[0]['mean']);sd=np.asarray(protos[0]['std']);assert mu.shape==(40,) and sd.shape==(40,)
assert all(np.array_equal(mu,np.asarray(x['mean'])) and np.array_equal(sd,np.asarray(x['std'])) for x in protos)
for arm in ('C0','F0','F1'):
 x,cm=old.features(d,'base',arm);z=(x-mu[:35])/sd[:35];z[...,8:]*=cm[...,None];z[~a['mask']]=0; extra=np.zeros((*x.shape[:2],5),float)
 if arm!='C0':
  raw=a['heading'][:,0 if arm=='F0' else 1];act=np.stack((raw[...,2]>0,raw[...,2]>0,np.ones_like(raw[...,2],bool),raw[...,4]>0,np.ones_like(raw[...,4],bool)),axis=-1);extra=(raw-mu[35:])/sd[35:];extra[~act]=0;extra[~a['mask']]=0
 z=np.concatenate((z,extra),-1).astype(np.float32);assert np.array_equal(z,a[arm]);result[arm]={'max_abs_error':float(np.max(np.abs(z-a[arm]))),'exact_bitwise_equal':True}
assert np.array_equal(a['C0'][...,8:],np.zeros_like(a['C0'][...,8:])); assert np.isfinite(a['F0']).all() and np.isfinite(a['F1']).all()
raw=json.loads((O/'ood_windows_v2.json').read_text())['all_windows'];case2=[i for i,x in enumerate(raw) if x['name']=='MATSON NAVIGATOR'];assert len(case2)==5 and all(raw[i]['n_anchor_members']==0 for i in case2);assert all(np.count_nonzero(a['F0'][i,:,8:])==0 and np.count_nonzero(a['F1'][i,:,8:])==0 for i in case2)
out={'n_windows':len(raw),'original_code_feature_equivalence':result,'all_base_model_normalizers_equal':True,'n_base_model_protocols':len(protos),'case2_stationary':{'n_windows':len(case2),'n_zero_context':len(case2)},'source':str(R/'heading_alignment_20261030/train_heading.py'),'explanation':'No model score calculated in this test; F0/F1 use exact original feature construction and saved train-only normalizer, all Hawaii windows remain out-of-training-domain.'}
(O/'feature_parity_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
