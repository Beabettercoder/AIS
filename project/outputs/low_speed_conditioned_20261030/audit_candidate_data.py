"""Independent matched data/schema audit; train/val only, no labels used in features."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
P=Path('/mnt/sda/WangZijia/AIS/outputs')
s=importlib.util.spec_from_file_location('original_heading',P/'heading_alignment_20261030/train_heading.py'); h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
out={};views=('base','natural','suffix','combined_l100_m600')
for split in ('train','val'):
 oldfile=P/'relation_repair_20261030'/f'{split}_relation_field.npz';newfile=P/'low_speed_history_floor_20261030'/f'{split}_relation_field.npz'
 if not newfile.exists():continue
 with np.load(oldfile,allow_pickle=False) as z:o={k:z[k] for k in z.files}
 with np.load(newfile,allow_pickle=False) as z:n={k:z[k] for k in z.files}
 for d in (o,n):d['mask']=np.arange(20)[None,:]<d['length'][:,None]
 assert all(np.array_equal(o[k],n[k]) for k in ['source','case','y','mmsi','region','length','time_s','raw7','causal7','self_w','mask'])
 info={'samples':len(o['y']),'sources':len(np.unique(o['source'])),'targets':len(np.unique(o['mmsi'])),'views':{}}
 for v in views:
  assert np.array_equal(o[v+'__count'],n[v+'__count'])
  fo=o[v+'__field'];fn=n[v+'__field']; mask=o['mask']
  modechanged=(fo[:,0,:,7]!=fn[:,0,:,7])&mask
  info['views'][v]={'count_equal':True,'f0_raw_changed_values':int(np.sum((fo[:,0,:,:6]!=fn[:,0,:,:6])&mask[...,None])),'f0_modes_changed_steps':int(modechanged.sum()),'f1_raw_changed_steps':int(np.sum(np.any((fo[:,1,:,:]!=fn[:,1,:,:]),axis=2)&mask))}
  for arm in ('C0','F0'):
   x0,c0=h.features(o,v,arm);x1,c1=h.features(n,v,arm)
   info['views'][v][arm+'_feature_changed_elements']=int(np.sum(x0!=x1));info['views'][v][arm+'_context_mask_equal']=bool(np.array_equal(c0,c1))
 out[split]=info
if len(out)==2:
 with np.load(P/'low_speed_history_floor_20261030/train_relation_field.npz') as a,np.load(P/'low_speed_history_floor_20261030/val_relation_field.npz') as b:
  out['target_overlap']=len(np.intersect1d(np.unique(a['mmsi']),np.unique(b['mmsi'])));out['source_overlap']=len(np.intersect1d(np.unique(a['source']),np.unique(b['source'])))
(P/'low_speed_conditioned_20261030/candidate_data_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
