"""Independent relation formal result audit; requires frozen checkpoint and complete outputs."""
from pathlib import Path
import json,hashlib,time,numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'formal_20261101/relation';D=R/'formal_20261101/independent_audit';D.mkdir(exist_ok=True)
models=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph']; arms=['C0','F0','F1'];views=['base','natural','suffix','combined_l100_m600']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
lockpath=O/'heldout.lock.json';assert lockpath.exists(),'Checkpoint lock missing'; lock=json.loads(lockpath.read_text());assert len(lock['models'])==72
summary=json.loads((O/'summary.json').read_text());mismatches=[];max_metric_error=0.;max_ci_error=0.;pairsets={};details={};timeaudit={'checkpoint_lock_mtime':lockpath.stat().st_mtime,'test_input_mtime':(O/'test_relation_field.npz').stat().st_mtime,'test_created_after_freeze':(O/'test_relation_field.npz').stat().st_mtime>=lockpath.stat().st_mtime};assert timeaudit['test_created_after_freeze']
def weighted_ap_boot(y,score,inv,draws):
 order=np.argsort(-score,kind='mergesort'); s=score[order];yy=y[order].astype(float);inv=inv[order];starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1];result=[]
 for at in range(0,len(draws),100):
  w=draws[at:at+100][:,inv].T.astype(float);tp=np.add.reduceat(w*yy[:,None],starts,axis=0);mass=np.add.reduceat(w,starts,axis=0);cumtp=np.cumsum(tp,axis=0);cum=np.cumsum(mass,axis=0);precision=np.divide(cumtp,cum,out=np.zeros_like(cumtp),where=cum>0);total=cumtp[-1];z=np.divide((tp*precision).sum(0),total,out=np.full(len(total),np.nan),where=total>0);result.extend(z)
 return np.asarray(result)
for split in ('val','test'):
 details[split]={}
 for name in models:
  A={}; metrics={};records=[]
  for seed in range(3):
   init=[]
   for arm in arms:
    folder=O/'runs'/name/f'seed{seed}'/arm;model=folder/'final.pt';pred=folder/('val_scores.npz' if split=='val' else 'test_scores.npz');result=folder/('result.json' if split=='val' else 'test_result.json');jr=json.loads(result.read_text());key=f'{name}/seed{seed}/{arm}';lk=lock['models'][key]
    assert sha(model)==lk['model_sha256'];assert sha(folder/'val_scores.npz')==lk['val_sha256'];assert sha(pred)==jr['prediction_sha256'];assert (O/'runs'/name/f'seed{seed}'/'protocol.json').exists()
    init.append(json.loads((folder/'result.json').read_text())['init_sha256'])
    with np.load(pred,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    A[(seed,arm)]=a;metrics[(seed,arm)]=jr['views']
    for view in views:
     y=a['y'];p=a[view+'_score'];ap=float(average_precision_score(y,p));au=float(roc_auc_score(y,p));regions=a['region'];ma=float(np.mean([average_precision_score(y[regions==r],p[regions==r]) for r in np.unique(regions)]));rec=jr['views'][view]
     max_metric_error=max(max_metric_error,abs(ap-rec['pooled_AP']),abs(au-rec['AUROC']),abs(ma-rec['macro_AP']))
     if split=='test':
      for b in ('0.01','0.05'):
       th=lk['thresholds'][b];mask=y>0;rate=rec['FAR_'+b];assert rate['threshold']==th;assert abs(float((p[~mask]>th).mean())-rate['actual_FAR'])<1e-12;assert abs(float((p[mask]>th).mean())-rate['recall'])<1e-12
   assert len(set(init))==1
  ref=A[(0,'C0')]
  for a in A.values():
   for k in ('y','mmsi','region','source','case'):assert np.array_equal(a[k],ref[k]),(split,name,k)
  pairsets[split]=(set(ref['mmsi'].astype(str)),set(ref['source'].astype(str)))
  ids,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True);rng=np.random.default_rng(20261101);draws=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=2000);y=ref['y']
  errci=0.
  for v in views:
   aa={arm:np.nanmean(np.array([weighted_ap_boot(y,A[(seed,arm)][v+'_score'],inv,draws) for seed in range(3)]),axis=0) for arm in arms}
   for control in ('C0','F0'):
    q=np.nanpercentile(aa['F1']-aa[control],[2.5,97.5]);refci=summary['results'][split][name]['views'][v]['paired']['F1_minus_'+control]['mmsi_cluster95'];errci=max(errci,float(np.max(np.abs(q-refci))))
  max_ci_error=max(max_ci_error,errci);details[split][name]={'models_verified':9,'views_verified':36,'max_abs_bootstrap_interval_diff':errci,'sample_count':len(y),'targets':len(ids),'sources':len(set(ref['source'].astype(str)))}
  print('AUDITED',split,name,'CI',errci,flush=True)
assert max_metric_error<1e-9 and max_ci_error<1e-9,(max_metric_error,max_ci_error)
trainp=R/'relation_repair_20261030/train_relation_field.npz'
with np.load(trainp,allow_pickle=False) as z:tr=(set(z['mmsi'].astype(str)),set(z['source'].astype(str)))
for a,b in [(tr,pairsets['val']),(tr,pairsets['test']),(pairsets['val'],pairsets['test'])]:assert not a[0]&b[0] and not a[1]&b[1]
result={'status':'passed','scope':'independent AP AUROC macro AP paired grouped AP bootstrap, threshold, source and identity pairing; no new training','metrics':details,'max_abs_metric_diff':max_metric_error,'max_abs_ci_diff':max_ci_error,'source_and_target_mmsi_overlap':'zero among train,val,test','time_freeze_check':timeaudit,'heldout_lock_sha256':sha(lockpath),'summary_sha256':sha(O/'summary.json'),'caveats':['test previously exposed; not blind','background vessel/report overlaps remain','bootstrap fixed models only, not seed population','no multiple-comparison correction']}
p=D/'relation_independent_audit.json';assert not p.exists();p.write_text(json.dumps(result,indent=2));print('PASS',json.dumps({'max_metric':max_metric_error,'max_ci':max_ci_error,'output':str(p)}),flush=True)
