import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
BASE=Path('/mnt/sda/WangZijia/AIS/outputs');out=BASE/'eight_backbone_independent_audit_20261030'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
views=('base','natural','suffix','combined_l100_m600'); seeds=(0,1,2)
tcn=BASE/'tcn_suffix_stability_20261030';it=BASE/'itrans_stability_20261030';old=BASE/'heading_alignment_20261030/runs/aligned_embed'
def folder(task,s,arm):
 if task=='tcn25':return (old/'tcn'/arm) if s==0 else tcn/'runs/aligned_embed'/f'seed{s}'/'tcn'/arm
 if task=='itrans25':return (old/'itransformer'/arm) if s==0 else it/'runs'/f'seed{s}'/'aligned_embed/itransformer'/arm
 return it/'extended75/runs'/f'seed{s}'/'aligned_embed/itransformer'/arm

def ap_fast(y,p,weight):
 o=np.argsort(-p,kind='mergesort');p=p[o];y=y[o];w=weight[o];tp=np.cumsum(y*w);tot=np.cumsum(w);last=np.r_[np.flatnonzero(p[1:]!=p[:-1]),len(p)-1];tp=tp[last];tot=tot[last];prec=np.divide(tp,tot,out=np.zeros_like(tp,dtype=float),where=tot>0);return float((np.diff(np.r_[0,tp])/tp[-1]*prec).sum())
result={};ref=None
for task in ('tcn25','itrans25','itrans75'):
 data={};reps={}
 for s in seeds:
  data[s]={};reps[s]={}
  arms=('F0','F1') if task=='itrans75' else ('C0','F0','F1')
  for arm in arms:
   f=folder(task,s,arm);a=json.loads((f/'result.json').read_text());z=np.load(f/'val_scores.npz',allow_pickle=False);d={k:z[k] for k in z.files};z.close()
   if ref is None:ref=d
   for k in ('y','source','case','mmsi','region'):np.testing.assert_array_equal(d[k],ref[k])
   assert a['seed']==s and a['last_epoch']==(75 if task=='itrans75' else 25) and len(a['loss_history'])==a['last_epoch']
   assert sha(f/'val_scores.npz')==a['prediction_sha256'] and sha(f/'final.pt')==a['weights_sha256']
   for v in views:
    ap=average_precision_score(d['y'],d[v+'_score']);auc=roc_auc_score(d['y'],d[v+'_score'])
    assert abs(ap-a['views'][v]['pooled_AP'])<1e-10 and abs(auc-a['views'][v]['AUROC'])<1e-10
    assert abs(ap_fast(d['y'],d[v+'_score'],np.ones(len(d['y'])))-ap)<1e-10
   data[s][arm]=d;reps[s][arm]=a
  assert len({r['init_sha256'] for r in reps[s].values()})==1
 ids, inv=np.unique(ref['mmsi'].astype(str),return_inverse=True);rng=np.random.default_rng(20261029);draw=rng.integers(0,len(ids),(2000,len(ids)))
 summaries={}
 for view in views:
  per=[average_precision_score(ref['y'],data[s]['F1'][view+'_score'])-average_precision_score(ref['y'],data[s]['F0'][view+'_score']) for s in seeds]
  delta=[]
  for sample in draw:
   w=np.bincount(sample,minlength=len(ids))[inv]
   delta.append(np.mean([ap_fast(ref['y'],data[s]['F1'][view+'_score'],w)-ap_fast(ref['y'],data[s]['F0'][view+'_score'],w) for s in seeds]))
  summaries[view]={'per_seed_pp':[x*100 for x in per],'mean_pp':np.mean(per)*100,'ci95_pp':(np.quantile(delta,[.025,.975])*100).tolist()}
 result[task]={'models':len(seeds)*len(arms),'epoch':75 if task=='itrans75' else 25,'views':summaries,'mmsi':len(ids),'windows':len(ref['y']),'verified':'SHA256, per-seed init, labels, MMSI, source, region, case, sklearn AP/AUROC, 2000 target-cluster draws; fixed trained models'}
 print(task,json.dumps(summaries['suffix']),flush=True)
(out/'independent_smallrun_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
