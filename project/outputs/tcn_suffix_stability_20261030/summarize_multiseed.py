from pathlib import Path
import json, hashlib
import numpy as np
from sklearn.metrics import average_precision_score

base=Path('/mnt/sda/WangZijia/AIS/outputs')
old=base/'heading_alignment_20261030/runs/aligned_embed/tcn'
out=base/'tcn_suffix_stability_20261030'
views=('base','natural','suffix','combined_l100_m600')
arms=('C0','F0','F1')
files={s:{a:(old/a if s==0 else out/'runs/aligned_embed'/('seed'+str(s))/'tcn'/a) for a in arms} for s in (0,1,2)}
allpred={}; reports={};keys=('y','case','source','mmsi','region');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for s in (0,1,2):
 allpred[s]={};reports[s]={}
 for a in arms:
  with np.load(files[s][a]/'val_scores.npz') as z: allpred[s][a]={k:z[k] for k in z.files}
  reports[s][a]=json.loads((files[s][a]/'result.json').read_text())
  for k in keys:np.testing.assert_array_equal(allpred[s][a][k],allpred[0]['C0'][k])
 assert len({reports[s][a]['init_sha256'] for a in arms})==1
 assert all(reports[s][a]['last_epoch']==25 and reports[s][a]['seed']==s for a in arms)
ref=allpred[0]['C0']; ids,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True); rng=np.random.default_rng(20261029);draw=rng.integers(0,len(ids),size=(2000,len(ids)))
res={'scope':'Previously exposed development validation; 3 fixed models, grouped bootstrap samples targets, does not capture seed population uncertainty; no multiplicity correction.','seeds':[0,1,2],'target_mmsi':len(ids),'cases':len(ref['y']),'views':{},'hashes':{}}
for s in (0,1,2):
 res['hashes'][str(s)]={a:{'initial':reports[s][a]['init_sha256'],'prediction':sha(files[s][a]/'val_scores.npz'),'checkpoint':sha(files[s][a]/'final.pt')} for a in arms}
for view in views:
 values={str(s):{a:float(average_precision_score(ref['y'],allpred[s][a][view+'_score'])) for a in arms} for s in (0,1,2)}
 diffs={str(s):{k:values[str(s)]['F1']-values[str(s)][other] for k,other in [('f1_f0','F0'),('f1_c0','C0')]} for s in (0,1,2)}
 bootstrap={k:[] for k in ('f1_f0','f1_c0')};by_seed={str(s):{k:[] for k in bootstrap} for s in (0,1,2)}
 for sample in draw:
  weight=np.bincount(sample,minlength=len(ids))[inv]
  for s in (0,1,2):
   ap={a:average_precision_score(ref['y'],allpred[s][a][view+'_score'],sample_weight=weight) for a in arms}
   by_seed[str(s)]['f1_f0'].append(ap['F1']-ap['F0']);by_seed[str(s)]['f1_c0'].append(ap['F1']-ap['C0'])
  for k in bootstrap:bootstrap[k].append(np.mean([by_seed[str(s)][k][-1] for s in (0,1,2)]))
 res['views'][view]={'per_seed':values,'per_seed_differences':diffs,'mean':{a:float(np.mean([values[str(s)][a] for s in (0,1,2)])) for a in arms},'mean_differences':{k:float(np.mean([diffs[str(s)][k] for s in (0,1,2)])) for k in bootstrap},'bootstrap95':{k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in bootstrap.items()},'by_seed_bootstrap95':{str(s):{k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in by_seed[str(s)].items()} for s in (0,1,2)}}
 print(view,res['views'][view]['mean_differences'],res['views'][view]['bootstrap95'],flush=True)
(out/'three_seed_summary.json').write_text(json.dumps(res,indent=2),encoding='utf8')
