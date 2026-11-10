import json, hashlib
from pathlib import Path
import numpy as np
O=Path('/mnt/sda/WangZijia/AIS/outputs')
D=O/'mixed_field_mechanism_20261031'
z=np.load(O/'relation_repair_20261030/val_relation_field.npz')
d={k:z[k] for k in z.files}
N=len(d['y']); y=d['y']; src=d['source'].astype(str); mids=d['mmsi'].astype(str)
mask=np.arange(20)[None,:] < d['length'][:,None]
views=['natural','suffix','combined_l100_m600']
unique=np.unique(src)
src_idx={s:np.flatnonzero(src==s) for s in unique}
assert all(len(ix)==4 and y[ix].sum()==2 and len(np.unique(mids[ix]))==1 for ix in src_idx.values())
models={}
for arm in ('F0_heading','F1_heading'):
 arrays=[]
 for seed in (0,1,2):
  p=O/'mixed_stress_probe_20261031'/'runs'/'tcn'/f'seed{seed}'/arm/'val_scores.npz'
  v=np.load(p)
  for k in ('y','mmsi','source','case','region'):
   assert np.array_equal(v[k],d[k])
  arrays.append({k:v[k] for k in v.files if k.endswith('_score')})
 models[arm]=arrays
def ranks(score):
 out=[]
 for s in unique:
  ix=src_idx[s]; p=score[ix][y[ix]==1];n=score[ix][y[ix]==0]
  out.append(((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])).mean())
 return np.array(out)
OLD=O/'balanced_global_v2/relation_dual_pressure_v2_20261027'
man=json.loads((OLD/'dataset_manifest_combined.json').read_text())
lowgroup={}
for row in man['samples']:
 if row['version']!='relaxed' or row['split']!='val':continue
 with np.load(OLD/row['files']['base']['path'],allow_pickle=False) as data:
  source_speed=data['source_neighbor_sog_kn'];off=data['member_offsets']
  lowgroup[row['window_id']]=any(source_speed[int(off[m['member_index']])+int(m['indices']['suffix']['report_index'])]<2 for m in row['stress']['members'])
assert set(lowgroup)==set(unique)
out={'scope':'Exploratory existing development validation; F0/F1 field deviations measured relative to each respective Base, which is not clean truth. 4 synthetic paired cases and shared neighbors are not independent; no test labels read.','sources':len(unique),'targets':len(np.unique(mids)),'windows':N,'views':{}}
for view in views:
 b=d['base__field'];v=d[view+'__field']
 valid=mask & (b[:,0,:,7]>0) & (v[:,0,:,7]>0) & (b[:,1,:,7]>0) & (v[:,1,:,7]>0)
 mag=np.linalg.norm(v[:,:,:,0:2]-b[:,:,:,0:2],axis=-1)
 affected=(mag[:,0,:]>1e-5) | (mag[:,1,:]>1e-5)
 fields={'valid_steps':int(valid.sum()),'changed_any_steps':int((affected&valid).sum())}
 for arm in (0,1):
  x=mag[:,arm,:][valid];xa=mag[:,arm,:][valid&affected]
  fields[f'F{arm}']={'mean_dev_all_mps':float(x.mean()),'median_dev_all_mps':float(np.median(x)),'p90_dev_all_mps':float(np.quantile(x,.9)),'mean_dev_changed_mps':float(xa.mean()),'fraction_changed':float((x>1e-5).mean())}
 ranks_model={a:np.mean(np.stack([ranks(x[view+'_score']) for x in models[a]]),axis=0) for a in models}
 rankd=ranks_model['F1_heading']-ranks_model['F0_heading']
 nreal=np.array([np.mean(d[view+'__count'][src_idx[s]][mask[src_idx[s]]]) for s in unique])
 targetsp=np.array([np.median(d['causal7'][src_idx[s],:,4][mask[src_idx[s]]]) for s in unique])
 # group source by median target speed, not baseline clean truth; no gating claims.
 groups={}
 for name,g in [('all',np.ones(len(unique),bool)),('mean_neighbors_lt4',nreal<4),('mean_neighbors_ge4',nreal>=4),('suffix_selected_baseline_low_neighbor',np.array([lowgroup[s] for s in unique])),('observed_target_median_lt2kn',targetsp<2),('observed_target_median_ge2kn',targetsp>=2)]:
  if g.sum()==0:continue
  ss=unique[g]; rr=rankd[g]
  smids={mids[src_idx[s][0]] for s in ss}
  group_mmsi=np.array([mids[src_idx[s][0]] for s in ss]);gu,gi=np.unique(group_mmsi,return_inverse=True)
  grng=np.random.default_rng(20261031);gb=[]
  for _ in range(2000):
   gw=np.bincount(grng.integers(0,len(gu),size=len(gu)),minlength=len(gu))[gi]
   gb.append(float(np.average(rr,weights=gw)*100))
  groups[name]={'sources':int(len(ss)),'targets':len(smids),'same_source_pairwise_F1_minus_F0_pp':float(rr.mean()*100),'MMSI_bootstrap_95CI_pp':np.quantile(gb,[.025,.975]).tolist(),'improved_sources':int((rr>1e-9).sum()),'worse_sources':int((rr< -1e-9).sum()),'unchanged_sources':int((np.abs(rr)<=1e-9).sum())}
 scores={}
 for arm in models:
  shifts=np.stack([np.abs(x[view+'_score']-x['base_score']) for x in models[arm]])
  scores[arm]={'mean_absolute_score_change':float(shifts.mean()),'p90_absolute_score_change':float(np.quantile(shifts,.9))}
 source_mmsi=np.array([mids[src_idx[s][0]] for s in unique])
 u, inv=np.unique(source_mmsi,return_inverse=True)
 rng=np.random.default_rng(20261031)
 boots=[]
 for _ in range(2000):
  draw=rng.integers(0,len(u),size=len(u))
  weights=np.bincount(draw,minlength=len(u))[inv]
  boots.append(float(np.average(rankd,weights=weights)*100))
 rankci=np.quantile(boots,[.025,.975]).tolist()
 out['views'][view]={'field':fields,'mixed_tcn':{'same_source_pairwise_F0':float(ranks_model['F0_heading'].mean()),'same_source_pairwise_F1':float(ranks_model['F1_heading'].mean()),'same_source_pairwise_delta_pp_95CI':rankci,'groups':groups,'score_shifts':scores}}
(D/'field_mechanism.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(out,indent=2,ensure_ascii=False))







