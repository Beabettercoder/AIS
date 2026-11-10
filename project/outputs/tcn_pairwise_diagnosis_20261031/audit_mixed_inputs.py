"""Independent, read-only train/val-only audit of source-level exposure."""
from pathlib import Path
import sys, importlib.util, json, hashlib
import numpy as np
import torch
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
M=ROOT/'mixed_stress_probe_20261031'
spec=importlib.util.spec_from_file_location('mix',M/'train_mixed.py')
mix=importlib.util.module_from_spec(spec);spec.loader.exec_module(mix)
ref=mix.ref
tr=ref.read('train');va=ref.read('val')
for k in ('source','y','mmsi','region','case'):assert k in tr and k in va
assert not set(tr['mmsi'].astype(str)).intersection(va['mmsi'].astype(str))
assert not set(tr['source'].astype(str)).intersection(va['source'].astype(str))
mu,sd=ref.fit(tr);assert np.isfinite(mu).all() and np.isfinite(sd).all() and (sd>0).all()
baseline={}
for name in ('tcn','gru','transformer','itransformer'):
 for seed in ([0,1,2] if name in ('tcn','itransformer') else [0]):
  for arm in mix.ARMS:
   path=mix.historical(name,seed,arm); z=json.loads((path.parent/'protocol.json').read_text())
   assert np.array_equal(np.array(z['mean']),mu) and np.array_equal(np.array(z['std']),sd),(name,seed,arm)
   ref.seed(seed);model=ref.base.get_model(name,40,name=='itransformer')
   initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest()
   p=json.loads((path/'result.json').read_text())
   assert initial==p['init_sha256'] and p['last_epoch']==(75 if name=='itransformer' else 25)
   baseline[f'{name}_{seed}_{arm}']={'init':initial,'last_epoch':p['last_epoch']}

sources, inv, counts=np.unique(tr['source'].astype(str),return_inverse=True,return_counts=True)
assert np.all(counts==4)
groups=[np.flatnonzero(inv==i) for i in range(len(sources))]
assert all(len(np.unique(tr['case'][g]))==4 and tr['y'][g].sum()==2 for g in groups)
assignment=[]
for epoch in range(75):
 a=np.random.default_rng(920000+epoch).integers(0,3,size=len(sources),dtype=np.int64)
 as_rows=a[inv]
 assert all(len(np.unique(as_rows[g]))==1 for g in groups)
 # Because each source has two labels, every view remains exactly balanced.
 for view in range(3):
  ix=np.flatnonzero(as_rows==view)
  assert len(ix)%4==0 and tr['y'][ix].sum()*2==len(ix)
 assignment.append(np.bincount(a,minlength=3).tolist())
for split,data in [('train',tr),('val',va)]:
 for view in mix.VIEWS:
  for arm in mix.ARMS:
   x,m,t=ref.prep(data,view,arm,mu,sd,'cpu')
   assert torch.isfinite(x).all() and torch.isfinite(t).all()
   perm=dict(data);perm['y']=1-data['y']
   xx,mm,tt=ref.prep(perm,view,arm,mu,sd,'cpu')
   assert torch.equal(x,xx) and torch.equal(m,mm) and torch.equal(t,tt)

audit={'train_rows':len(tr['y']),'train_sources':len(sources),'train_mmsi':len(np.unique(tr['mmsi'])),'val_rows':len(va['y']),'val_sources':len(np.unique(va['source'])),'val_mmsi':len(np.unique(va['mmsi'])),'train_val_target_and_sources_disjoint':True,'counts_per_epoch_first5':assignment[:5],'all75_epochs_source_view_uniform_label_balance':True,'all_views_both_arms_label_flip_feature_equal':True,'all_baseline_normalizers_and_init_hashes_match':baseline,'limitations':'matches baseline configuration and initial state; distinct view inputs alter gradients; no test data; source-based batch applies only itransformer; background reuse possible'}
(ROOT/'tcn_pairwise_diagnosis_20261031/mixed_input_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
print(json.dumps(audit,indent=2))
