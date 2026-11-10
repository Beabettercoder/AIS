import numpy as np,json,hashlib
from pathlib import Path
from sklearn.metrics import average_precision_score,roc_auc_score
root=Path('/mnt/sda/WangZijia/AIS/outputs'); A=root/'itrans_sourcebalanced_20261031'; B=root/'target_heading_20261031'; O=root/'review_results_20261031';O.mkdir(exist_ok=True)
V=['base','natural','suffix','combined_l100_m600']; ta=['C0_target','F0_target','F1_no_target','F1_target']; ia=['C0','F0_heading','F1_noheading','F1_heading']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ap_weighted(score,y,W):
 order=np.argsort(-score,kind='stable');s=score[order]; yy=y[order]; ww=W[:,order]; boundaries=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]; pos=np.add.reduceat(ww*yy,boundaries,axis=1);tot=np.add.reduceat(ww,boundaries,axis=1);cp=np.cumsum(pos,axis=1);ct=np.cumsum(tot,axis=1);tp=cp[:,-1];precision=np.divide(cp,ct,out=np.zeros_like(cp),where=ct>0);return np.divide((pos*precision).sum(1),tp,out=np.full(len(W),np.nan),where=tp>0)
def audit_dir(root,kind):
 result={}; rng=np.random.default_rng(20261031 if kind=='target' else 20261030)
 for name in (['tcn','gru','transformer','itransformer'] if kind=='target' else ['itransformer']):
  seeds=(0,1,2) if name in ('tcn','itransformer') else (0,);arms=ta if kind=='target' else ia; P={}; meta={}
  for seed in seeds:
   for arm in arms:
    folder=(root/'runs'/('source' if name=='itransformer' else 'window')/name/f'seed{seed}'/arm) if kind=='target' else (root/'runs'/name/f'seed{seed}'/arm)
    r=json.loads((folder/'result.json').read_text());assert r['last_epoch']==(75 if name=='itransformer' else 25) and len(r['loss_history'])==r['last_epoch'];assert sha(folder/'val_scores.npz')==r['prediction_sha256'];assert sha(folder/'final.pt')==r['weights_sha256']; meta[seed,arm]=r
    with np.load(folder/'val_scores.npz',allow_pickle=False) as z:P[seed,arm]={k:z[k] for k in z.files}
  ref=P[seeds[0],arms[0]]
  for p in P.values():
   for key in ('y','case','source','mmsi','region'):np.testing.assert_array_equal(p[key],ref[key])
  for seed in seeds:assert len(set(meta[seed,a]['init_sha256'] for a in arms))==1
  y=ref['y'].astype(int);ids,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True); draws=rng.integers(0,len(ids),size=(2000,len(ids))); weights=np.array([np.bincount(x,minlength=len(ids))[inv] for x in draws],dtype=float)
  views={}
  for view in V:
   ap={}; auc={};boot={}
   for key,p in P.items():
    score=p[view+'_score'];r=meta[key]['views'][view];ap[key]=average_precision_score(y,score);auc[key]=roc_auc_score(y,score);assert abs(ap[key]-r['pooled_AP'])<1e-12 and abs(auc[key]-r['AUROC'])<1e-12
    boot[key]=ap_weighted(score,y,weights)
    for i in (0,100,1000):assert abs(boot[key][i]-average_precision_score(y,score,sample_weight=weights[i]))<1e-12
   pairs= {'target_heading_gain':('F1_target','F1_no_target'),'reliability_config':('F1_target','F0_target'),'relative_to_self':('F1_target','C0_target')} if kind=='target' else {'heading_vs_noheading':('F1_heading','F1_noheading'),'reliability_config':('F1_heading','F0_heading'),'heading_vs_C0':('F1_heading','C0')}
   views[view]={}
   for label,(a,b) in pairs.items():
    delta=np.mean([boot[s,a]-boot[s,b] for s in seeds],axis=0); views[view][label]={'mean':float(np.mean([ap[s,a]-ap[s,b] for s in seeds])),'interval':np.nanquantile(delta,[.025,.975]).tolist()}
  result[name]={'models':len(P),'n':len(y),'mmsi':len(ids),'views':views};print(kind,name,'ok',flush=True)
 return result
out={'sourcebalanced':audit_dir(A,'sourcebalanced'),'target':audit_dir(B,'target')}
for kind,path in [('sourcebalanced',A),('target',B)]:
 summary=json.loads((path/'paired_summary.json').read_text())
 for model,data in out[kind].items():
  for view,pairs in data['views'].items():
   expected=summary[model]['groups']['all']['views'][view]
   for name,value in pairs.items():
    assert abs(value['mean']-expected['mean_diff'][name])<1e-12,(kind,model,view,name,'mean')
    assert np.allclose(value['interval'],expected['bootstrap95'][name],atol=1e-12),(kind,model,view,name,value['interval'],expected['bootstrap95'][name])
(O/'independent_metrics.json').write_text(json.dumps(out,indent=2));print('all matches',flush=True)
