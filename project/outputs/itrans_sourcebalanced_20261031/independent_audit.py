from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np,torch
from sklearn.metrics import average_precision_score, roc_auc_score
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'itrans_sourcebalanced_20261031'
s=importlib.util.spec_from_file_location('train',O/'train.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
tr,va=m.read('train'),m.read('val');mu,sd=m.fit(tr);dev=torch.device('cuda:2');torch.set_num_threads(2)
ref=None;checks=[];pred={};init=[]
for seed in (0,1,2):
 for arm in m.ARMS:
  d=O/'runs'/'itransformer'/f'seed{seed}'/arm;r=json.loads((d/'result.json').read_text());model=m.base.get_model('itransformer',40,True).to(dev);model.load_state_dict(torch.load(d/'final.pt',map_location=dev,weights_only=True));model.eval()
  with np.load(d/'val_scores.npz') as z:p={k:z[k] for k in z.files}
  if ref is None:ref={k:p[k] for k in ('y','mmsi','region','source','case')}
  for k in ref:np.testing.assert_array_equal(p[k],ref[k])
  assert r['last_epoch']==75 and len(r['loss_history'])==75
  hh=hashlib.sha256((d/'final.pt').read_bytes()).hexdigest();ph=hashlib.sha256((d/'val_scores.npz').read_bytes()).hexdigest();assert hh==r['weights_sha256'] and ph==r['prediction_sha256']
  if arm!='C0':assert r['loss_history'][-1]<.65, (seed,arm,r['loss_history'][-1])
  for view in m.VIEWS:
   x,mask,t=m.prep(va,view,arm,mu,sd,dev);pieces=[]
   with torch.no_grad():
    for i in range(0,len(x),256):pieces.extend(torch.sigmoid(model(x[i:i+256],mask[i:i+256],t[i:i+256])['logits']).cpu().numpy())
   score=np.asarray(pieces);err=float(np.max(np.abs(score-p[view+'_score'])));assert err==0,(seed,arm,view,err)
   ap=average_precision_score(ref['y'],score);auc=roc_auc_score(ref['y'],score);assert abs(ap-r['views'][view]['pooled_AP'])<1e-12 and abs(auc-r['views'][view]['AUROC'])<1e-12
  init.append((seed,arm,r['init_sha256']));pred[seed,arm]=p;checks.append({'seed':seed,'arm':arm,'loss75':r['loss_history'][-1],'base_AP':r['views']['base']['pooled_AP'],'prediction_sha':ph,'checkpoint_sha':hh})
for seed in (0,1,2):assert len({h for s,a,h in init if s==seed})==1
# Reimplement weighted AP for independent 2,000 cluster bootstraps; fixed output group=all.
y=ref['y'].astype(float);_,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True);n=inv.max()+1;rng=np.random.default_rng(20261030);draw=rng.integers(n,size=(2000,n));summary=json.loads((O/'paired_summary.json').read_text())['itransformer']['groups']['all']['views'];bootstrap={}
for view in m.VIEWS:
 cache={}
 for key,p in pred.items():
  score=p[view+'_score'];order=np.argsort(-score,kind='mergesort');end=np.r_[np.flatnonzero(np.diff(score[order])!=0),len(score)-1];cache[key]=(order,end,y[order])
 def ap(k,weights):
  order,end,yy=cache[k];w=weights[order];tp=np.cumsum(w*yy)[end];den=np.cumsum(w)[end]
  if tp[-1]==0:return np.nan
  return float((np.diff(np.r_[0.,tp])*np.divide(tp,den,out=np.zeros_like(tp),where=den>0)).sum()/tp[-1])
 sample=[]
 for dr in draw:
  w=np.bincount(dr,minlength=n)[inv]; ds=[ap((seed,'F1_heading'),w)-ap((seed,'F0_heading'),w) for seed in (0,1,2)]
  if np.isfinite(ds).all():sample.append(np.mean(ds))
 q=np.quantile(sample,[.025,.975]);expected=summary[view]['bootstrap95']['reliability_config'];np.testing.assert_allclose(q,expected,rtol=0,atol=1e-12)
 bootstrap[view]={'95CI':q.tolist(),'count':len(sample)}
(O/'independent_audit.json').write_text(json.dumps({'models':checks,'views_replayed':len(checks)*4,'GPU_replay_max_abs':0,'bootstraps':bootstrap},indent=2),encoding='utf8')
print('audit passed:',len(checks),'models,',len(checks)*4,'exact GPU prediction replays; 4 bootstrap matches',flush=True)
