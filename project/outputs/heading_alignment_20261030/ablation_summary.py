import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
root=Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030')
variants=['old_noembed','old_embed','aligned_noembed','aligned_embed']
views=['base','natural','suffix','combined_l100_m600']
data={}
for v in variants:
 with np.load(root/'runs'/v/'itransformer'/'F1'/'val_scores.npz',allow_pickle=False) as z:data[v]={k:z[k] for k in z.files}
for v in variants[1:]:
 for k in ('y','source','case','mmsi','region'):np.testing.assert_array_equal(data[v][k],data[variants[0]][k])
base=data[variants[0]];ids,inv=np.unique(base['mmsi'].astype(str),return_inverse=True);rng=np.random.default_rng(20261029);draw=rng.integers(0,len(ids),size=(2000,len(ids)))
pairs=[('aligned_embed','aligned_noembed'),('aligned_embed','old_embed'),('old_embed','old_noembed'),('aligned_noembed','old_noembed')]
results={}
for view in views:
 scores={v:data[v][view+'_score'] for v in variants}; ap={v:average_precision_score(base['y'],s) for v,s in scores.items()};dif={a+'__'+b:[] for a,b in pairs}
 for sample in draw:
  weight=np.bincount(sample,minlength=len(ids))[inv];a={v:average_precision_score(base['y'],s,sample_weight=weight) for v,s in scores.items()}
  for x,y in pairs:dif[x+'__'+y].append(a[x]-a[y])
 results[view]={'ap':ap,'paired_difference':{x+'__'+y:{'difference':float(ap[x]-ap[y]),'lo':float(np.quantile(dif[x+'__'+y],.025)),'hi':float(np.quantile(dif[x+'__'+y],.975))} for x,y in pairs}}
 print(view,'done',flush=True)
(root/'ablation_summary.json').write_text(json.dumps({'scope':'fixed seed model paired target MMSI clusters, 2000 draws, unadjusted exploratory','views':results},indent=2),encoding='utf8')
