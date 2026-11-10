"""Fixed-model paired bootstrap on target-heading evidence; validation only."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from sklearn.metrics import average_precision_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OUT=ROOT/'target_heading_20261031';OLD=ROOT/'balanced_global_v2/relation_dual_pressure_v2_20261027'
sys.path.insert(0,str(OUT))
import train_target as t
d=t.load('val');src=d['source'].astype(str);mask=d['mask'];feat=d['target_heading_feature'];available=(feat[...,2]>0)&mask
manifest=json.loads((OLD/'dataset_manifest_combined.json').read_text())
low={}
for r in manifest['samples']:
 if r['version']!='relaxed' or r['split']!='val':continue
 with np.load(OLD/r['files']['base']['path'],allow_pickle=False) as z:sog=z['source_target_sog_kn']
 low[r['window_id']]=np.isfinite(sog).any() and bool(np.any(np.isfinite(sog)&(sog<2)))
assert all(s in low for s in src)
groups={'all':np.ones(len(src),bool),'target_heading_available':available.any(1),'target_heading_unavailable':~available.any(1),'raw_source_target_sog_lt2':np.array([low[s] for s in src]),'raw_target_low_and_heading_available':np.array([low[s] for s in src])&available.any(1)}
RESULT={};rng=np.random.default_rng(20261031);VIEWS=t.VIEWS;ARMS=t.ARMS
for name in ('tcn','gru','transformer','itransformer'):
 seeds=(0,1,2) if name in ('tcn','itransformer') else (0,)
 balance='source' if name=='itransformer' else 'window'
 p={};run={}
 for s in seeds:
  for a in ARMS:
   folder=OUT/'runs'/balance/name/f'seed{s}'/a
   rr=json.loads((folder/'result.json').read_text());run[s,a]=rr
   assert rr['last_epoch']==(75 if name=='itransformer' else 25)
   with np.load(folder/'val_scores.npz',allow_pickle=False) as z:p[s,a]={k:z[k] for k in z.files}
   assert rr['prediction_sha256']==t.sha(folder/'val_scores.npz') and rr['weights_sha256']==t.sha(folder/'final.pt')
 for s in seeds:assert len({run[s,a]['init_sha256'] for a in ARMS})==1
 ref=p[seeds[0],ARMS[0]]
 for q in p.values():
  for k in ('y','case','source','mmsi','region'):np.testing.assert_array_equal(ref[k],q[k])
 np.testing.assert_array_equal(src,ref['source'].astype(str))
 ids,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True);draw=rng.integers(0,len(ids),size=(2000,len(ids)))
 output={'seeds':list(seeds),'batch_balance':balance,'groups':{},'epoch':75 if name=='itransformer' else 25,'scope':'fixed models, target MMSI paired bootstrap; reused background, exposed validation; no seed-population/multiple-comparison guarantee','run_loss_final':{f'{s}_{a}':run[s,a]['loss_history'][-1] for s in seeds for a in ARMS}}
 for group,select in groups.items():
  out={'windows':int(select.sum()),'sources':len(np.unique(src[select])),'mmsi':len(np.unique(ref['mmsi'][select])),'views':{}}
  if select.sum()==0 or len(np.unique(ref['y'][select]))<2:output['groups'][group]=out;continue
  for view in VIEWS:
   scores={k:v[view+'_score'][select] for k,v in p.items()};yy=ref['y'][select];ii=inv[select]
   vals={k:float(average_precision_score(yy,v)) for k,v in scores.items()};cache={}
   for k,score in scores.items():
    o=np.argsort(-score,kind='mergesort');v=score[o];ends=np.r_[np.flatnonzero(np.diff(v)!=0),len(v)-1];cache[k]=(o,ends,yy[o].astype(float))
   def ap(k,w):
    o,ends,y=cache[k];z=w[o].astype(float);tp=np.cumsum(z*y)[ends];ct=np.cumsum(z)[ends]
    if tp[-1]<=0:return np.nan
    return float(np.sum(np.diff(np.r_[0.,tp])*np.divide(tp,ct,out=np.zeros_like(tp),where=ct>0))/tp[-1])
   for k in scores:assert abs(ap(k,np.ones(len(yy)))-vals[k])<1e-12
   pairs={'target_heading_gain':('F1_target','F1_no_target'),'reliability_config':('F1_target','F0_target'),'relative_to_self':('F1_target','C0_target')}
   per={label:[vals[s,a]-vals[s,b] for s in seeds] for label,(a,b) in pairs.items()};boots={k:[] for k in pairs}
   for sample in draw:
    w=np.bincount(sample,minlength=len(ids))[ii];v={k:ap(k,w) for k in scores}
    for label,(a,b) in pairs.items():
     diffs=[v[s,a]-v[s,b] for s in seeds]
     if np.isfinite(diffs).all():boots[label].append(float(np.mean(diffs)))
   out['views'][view]={'per_seed_AP':{f'{s}_{a}':v for (s,a),v in vals.items()},'mean_AP':{a:float(np.mean([vals[s,a] for s in seeds])) for a in ARMS},'per_seed_diff':per,'mean_diff':{k:float(np.mean(v)) for k,v in per.items()},'bootstrap95':{k:(list(map(float,np.quantile(v,[.025,.975]))) if len(v)>100 else None) for k,v in boots.items()},'bootstrap_valid':{k:len(v) for k,v in boots.items()}}
  output['groups'][group]=out
 RESULT[name]=output
 print(name,'done',flush=True)
(OUT/'paired_summary.json').write_text(json.dumps(RESULT,indent=2),encoding='utf8')
