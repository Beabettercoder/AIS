from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.metrics import average_precision_score
p=Path('/mnt/sda/WangZijia/AIS/outputs/itrans_stability_20261030')
old=Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030/runs/aligned_embed/itransformer')
views=['base','natural','suffix','combined_l100_m600']; arms=['C0','F0','F1'];data={};runs={}
for seed in (0,1,2):
 data[seed]={};runs[seed]={}
 for arm in arms:
  path=old/arm if seed==0 else p/'runs'/('seed'+str(seed))/'aligned_embed'/'itransformer'/arm
  with np.load(path/'val_scores.npz',allow_pickle=False) as z:data[seed][arm]={k:z[k] for k in z.files}
  runs[seed][arm]=json.loads((path/'result.json').read_text())
  ref=data[0]['C0']
  for key in ['y','source','case','mmsi','region']:np.testing.assert_array_equal(data[seed][arm][key],ref[key])
 for arm in arms: assert runs[seed][arm]['init_sha256']==runs[seed]['C0']['init_sha256']
ref=data[0]['C0']; ids,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True);rng=np.random.default_rng(20261029);draw=rng.integers(0,len(ids),(2000,len(ids)))
summary={'scope':'three fixed model seeds on reused exposed development validation; bootstrap resamples MMSI, not model training seeds; no multiplicity adjustment','training':'seeds0,1,2; same 25 epochs / optimizer and matching initialization within each seed; shared data/order schedule','mmsi_count':int(len(ids)),'views':{},'runs':{}}
for seed in (0,1,2):
 summary['runs'][str(seed)]={arm:{'init':runs[seed][arm]['init_sha256'],'first_loss':runs[seed][arm]['loss_history'][0],'last_loss':runs[seed][arm]['loss_history'][-1],'loss_history':runs[seed][arm]['loss_history']} for arm in arms}
for view in views:
 z={'ap':{str(s):{a:runs[s][a]['views'][view]['pooled_AP'] for a in arms} for s in (0,1,2)},'bootstrap':{}}
 for name,refarm in [('F1_minus_F0','F0'),('F1_minus_C0','C0')]:
  dif=[]
  for draw_i in draw:
   wt=np.bincount(draw_i,minlength=len(ids))[inv]; ds=[]
   for s in (0,1,2):
    a=average_precision_score(ref['y'],data[s]['F1'][view+'_score'],sample_weight=wt)
    b=average_precision_score(ref['y'],data[s][refarm][view+'_score'],sample_weight=wt)
    ds.append(a-b)
   dif.append(np.mean(ds))
  pts=[z['ap'][str(s)]['F1']-z['ap'][str(s)][refarm] for s in (0,1,2)]
  z['bootstrap'][name]={'per_seed':pts,'average':float(np.mean(pts)),'ci_low':float(np.quantile(dif,.025)),'ci_high':float(np.quantile(dif,.975))}
 summary['views'][view]=z
(p/'stability_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
