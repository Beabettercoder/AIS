"""Paired target-MMSI bootstrap on saved predictions, using no test samples."""
from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.metrics import average_precision_score
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'heading_evidence_pilot_20261030';OLD=R/'balanced_global_v2/relation_dual_pressure_v2_20261027'
VIEWS=('base','natural','suffix','combined_l100_m600');ARMS=('C0','F0_heading','F1_noheading','F1_heading')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((OLD/'dataset_manifest_combined.json').read_text());rows=[r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='val']
low={}
for r in rows:
 with np.load(OLD/r['files']['base']['path'],allow_pickle=False) as z:
  source=z['source_neighbor_sog_kn'];offset=z['member_offsets'];
  low[r['window_id']]=any(source[int(offset[m['member_index']])+int(m['indices']['suffix']['report_index'])]<2 for m in r['stress']['members'])
with np.load(O/'val_heading_features.npz',allow_pickle=False) as z:
 src=z['source'].astype(str);h=z['base__F0'];valid=np.any(h[:,:,2]>0,axis=1)
headsource={s:bool(np.any(valid[src==s])) for s in np.unique(src)}
groups={'all':np.ones(len(src),bool),'source_any_heading':np.array([headsource[s] for s in src]),'source_no_heading':np.array([not headsource[s] for s in src]),'suffix_selected_baseline_low_neighbor':np.array([low[s] for s in src])}
result={};rng=np.random.default_rng(20261030)
for name in ('tcn','gru','transformer','itransformer'):
 seeds=(0,1,2) if name in ('tcn','itransformer') else (0,);pred={};runs={}
 for seed in seeds:
  for arm in ARMS:
   p=O/'runs'/name/f'seed{seed}'/arm;rr=json.loads((p/'result.json').read_text());
   with np.load(p/'val_scores.npz',allow_pickle=False) as z: pred[seed,arm]={k:z[k] for k in z.files}
   runs[seed,arm]=rr;assert rr['last_epoch']==(75 if name=='itransformer' else 25)
 for seed in seeds:assert len({runs[seed,a]['init_sha256'] for a in ARMS})==1
 ref=pred[0,'C0'];assert np.array_equal(ref['source'].astype(str),src)
 for obj in pred.values():
  for k in ('y','case','source','mmsi','region'):np.testing.assert_array_equal(ref[k],obj[k])
 ids,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True);draw=rng.integers(0,len(ids),size=(2000,len(ids)))
 output={'seeds':list(seeds),'mmsi':len(ids),'scope':'exploratory validation; fixed models and target-MMSI resampling; uncorrected; background reuse','views':{},'groups':{},'historical_35_reference':{}}
 for seed in seeds:
  if name=='itransformer':p=R/'itrans_stability_20261030'/'extended75'/'runs'/f'seed{seed}'/'aligned_embed'/'itransformer'/'F1'
  elif seed==0:p=R/'heading_alignment_20261030'/'runs'/'aligned_embed'/name/'F1'
  else:p=R/'tcn_suffix_stability_20261030'/'runs'/'aligned_embed'/f'seed{seed}'/'tcn'/'F1'
  r=json.loads((p/'result.json').read_text())
  with np.load(p/'val_scores.npz',allow_pickle=False) as z:
   for key in ('y','case','source','mmsi','region'):np.testing.assert_array_equal(ref[key],z[key])
  output['historical_35_reference'][str(seed)]={'views':r['views'],'prediction_sha256':sha(p/'val_scores.npz'),'scope':'different input width and initialization; contextual reference only'}
 for group,select in groups.items():
  entry={'n_windows':int(select.sum()),'n_sources':int(len(np.unique(src[select]))),'n_mmsi':int(len(np.unique(ref['mmsi'][select]))),'views':{}}
  if select.sum()==0 or len(np.unique(ref['y'][select]))<2: output['groups'][group]=entry;continue
  for view in VIEWS:
   scores={k:v[view+'_score'][select] for k,v in pred.items()};yy=ref['y'][select];ii=inv[select];values={k:float(average_precision_score(yy,p)) for k,p in scores.items()};cache={}
   for k,p in scores.items():
    order=np.argsort(-p,kind='mergesort');v=p[order];ends=np.r_[np.flatnonzero(np.diff(v)!=0),len(v)-1];cache[k]=(order,ends,yy[order].astype(float))
   def ap(k,w):
    order,ends,y=cache[k];z=w[order].astype(float);tp=np.cumsum(z*y)[ends];ct=np.cumsum(z)[ends]
    if tp[-1]<=0:return np.nan
    return float(np.sum(np.diff(np.r_[0.,tp])*np.divide(tp,ct,out=np.zeros_like(tp),where=ct>0))/tp[-1])
   for k in scores:assert abs(ap(k,np.ones(len(yy)))-values[k])<1e-12
   pairs={'heading_vs_noheading':('F1_heading','F1_noheading'),'reliability_config':('F1_heading','F0_heading'),'heading_vs_C0':('F1_heading','C0')}
   diffs={label:[values[s,a]-values[s,b] for s in seeds] for label,(a,b) in pairs.items()};boot={k:[] for k in pairs}
   for sample in draw:
    w=np.bincount(sample,minlength=len(ids))[ii];av={k:ap(k,w) for k in scores}
    for key,(a,b) in pairs.items():
     vals=[av[s,a]-av[s,b] for s in seeds]
     if np.isfinite(vals).all():boot[key].append(float(np.mean(vals)))
   entry['views'][view]={'values':{f'{s}_{a}':v for (s,a),v in values.items()},'mean':{a:float(np.mean([values[s,a] for s in seeds])) for a in ARMS},'per_seed_diff':diffs,'mean_diff':{k:float(np.mean(v)) for k,v in diffs.items()},'bootstrap95':{k:(list(map(float,np.quantile(v,[.025,.975]))) if len(v)>100 else None) for k,v in boot.items()},'bootstrap_valid':{k:len(v) for k,v in boot.items()}}
  output['groups'][group]=entry
 result[name]=output
 print(name, {k:v['n_windows'] for k,v in output['groups'].items()},flush=True)
(O/'paired_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
