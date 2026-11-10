from pathlib import Path
import sys,json,importlib.util
import numpy as np,torch
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OUT=ROOT/'heading_independent_audit_20261030';RUN=ROOT/'heading_evidence_pilot_20261030'
spec=importlib.util.spec_from_file_location('candidate_train',RUN/'train_heading_evidence.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
tr=m.read('train');va=m.read('val');assert not set(tr['mmsi'].astype(str))&set(va['mmsi'].astype(str));assert not set(tr['source'].astype(str))&set(va['source'].astype(str))
mu,sd=m.fit(tr);a,b=m.base.fit(tr);np.testing.assert_array_equal(mu[:35],a);np.testing.assert_array_equal(sd[:35],b)
summary={};maps={'C0':'C0','F0_heading':'F0','F1_noheading':'F1','F1_heading':'F1'}
for split,d in [('train',tr),('val',va)]:
 for view in m.VIEWS:
  for arm,old in maps.items():
   x,mask,tm=m.prep(d,view,arm,mu,sd,'cpu');y,mask0,tm0=m.base.prep(d,view,old,mu[:35],sd[:35],'cpu');assert torch.equal(x[:,:,:35],y) and torch.equal(mask,mask0) and torch.equal(tm,tm0)
   assert torch.isfinite(x).all() and (x[~mask]==0).all()
   if arm in ('C0','F1_noheading'):assert (x[:,:,35:]==0).all()
   d2=dict(d);d2['y']=1-d['y'];xx,_,_=m.prep(d2,view,arm,mu,sd,'cpu');assert torch.equal(x,xx)
   summary[split+'_'+view+'_'+arm]={'samples':int(len(x)),'nonzero_heading_values':int((x[:,:,35:]!=0).sum()),'all_finite':True}
# Four cells are aligned and heading cache itself never stores labels or factors.
for split,d in [('train',tr),('val',va)]:
 for i in range(0,len(d['y']),4):
  assert len(set(d['source'][i:i+4].astype(str)))==1 and sorted(d['y'][i:i+4].tolist())==[0,0,1,1]
with np.load(ROOT/'heading_coverage_20261030/heading_train.npz') as h: assert not any(k in h.files for k in ('y','label','label_class','factor','source_sog','source_pos'))
out={'normalizer_first35_matches_original':True,'all_arms_first35_match_intended_control':True,'zero_added_features_in_noheading_and_C0':True,'all_arms_independent_of_label_flip':True,'train_val_target_mmsi_overlap':0,'summaries':summary,'note':'40-dimensional model has different architecture from 35-dimensional original; same-width comparisons only'}
(OUT/'model_inputs_independent.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in out.items() if k!='summaries'}))
