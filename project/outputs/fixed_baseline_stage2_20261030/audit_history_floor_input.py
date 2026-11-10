from pathlib import Path
import numpy as np,json,sys,importlib.util,hashlib
BASE=Path('/mnt/sda/WangZijia/AIS/outputs')
OLD=BASE/'relation_repair_20261030'
NEW=BASE/'low_speed_history_floor_20261030'
AUD=BASE/'fixed_baseline_stage2_20261030'
def mod(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
model=mod(NEW/'fixed_normalizer'/'train_fixed.py','candidate_train')
def read(path):
 with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in f.files}
 d['mask']=np.arange(20)[None,:]<d['length'][:,None]
 return d
views=('base','natural','suffix','combined_l100_m600')
report={}
for split in ('train','val'):
 if not (NEW/f'{split}_relation_field.npz').exists():continue
 a,b=read(OLD/f'{split}_relation_field.npz'),read(NEW/f'{split}_relation_field.npz')
 result={'records':len(a['y']),'source_count':len(np.unique(a['source'])),'mmsi_count':len(np.unique(a['mmsi'])),'identical':{},'views':{}}
 for k in ('source','case','y','length','mmsi','region','raw7','causal7','self_w','time_s','mask'):
  result['identical'][k]=bool(np.array_equal(a[k],b[k]))
  assert result['identical'][k],(split,k)
 for view in views:
  fa=a[view+'__field'];fb=b[view+'__field']
  v={'counts_equal':bool(np.array_equal(a[view+'__count'],b[view+'__count']))}
  assert v['counts_equal']
  va=fa[:,0,:,7]>0;vb=fb[:,0,:,7]>0
  v['mask_changes']=int(np.sum(va!=vb));v['mode_changes']=int(np.sum(fa[:,0,:,7]!=fb[:,0,:,7]))
  v['f0_physical_max_abs']=float(np.max(np.abs(fa[:,0,:,:6]-fb[:,0,:,:6])))
  v['f0_physical_changed_steps']=int(np.sum(np.any(np.abs(fa[:,0,:,:6]-fb[:,0,:,:6])>1e-6,axis=-1)))
  for arm in ('C0','F0','F1'):
   xa,ma=model.features(a,view,arm);xb,mb=model.features(b,view,arm)
   v[arm+'_feature_changed_steps']=int(np.sum(np.any(np.abs(xa-xb)>1e-6,axis=-1)))
   v[arm+'_feature_max_abs']=float(np.max(np.abs(xa-xb)))
   if arm=='C0':
    assert np.array_equal(xa,xb) and np.array_equal(ma,mb)
   v[arm+'_mask_changes']=int(np.count_nonzero(ma!=mb))
  result['views'][view]=v
 if split=='train':
  am,asd=model.fit(a);bm,bsd=model.fit(b)
  result['shared_norm_mean_abs_max']=float(np.max(np.abs(am-bm)))
  result['shared_norm_std_abs_max']=float(np.max(np.abs(asd-bsd)))
  result['normalizer_mean_old']=am.tolist();result['normalizer_mean_safe']=bm.tolist()
  result['normalizer_std_old']=asd.tolist();result['normalizer_std_safe']=bsd.tolist()
 report[split]=result
if 'train' in report and 'val' in report:
 a=read(NEW/'train_relation_field.npz');b=read(NEW/'val_relation_field.npz')
 report['identity_overlap']={'target_mmsi':len(set(a['mmsi'])&set(b['mmsi'])),'source':len(set(a['source'])&set(b['source']))}
 assert report['identity_overlap']['target_mmsi']==0 and report['identity_overlap']['source']==0
(AUD/'history_floor_input_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))

