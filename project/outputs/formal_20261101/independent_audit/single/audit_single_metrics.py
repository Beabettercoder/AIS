"""Independent single-agent result verification; only read test after training freeze."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single'); O=P.parent/'independent_audit/single';RAW=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028/data/individual');names=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];arms=['raw','causal_bundle','f0_bundle','f1_bundle']; B=2000

def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as q:
  for b in iter(lambda:q.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def grouped_aps(y,s,inv,draws):
 idx=np.argsort(-s,kind='stable');s=s[idx];y=y[idx];inv=inv[idx];ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];result=[]
 for b in range(0,len(draws),200):
  w=draws[b:b+200][:,inv].T.astype(np.float64);tp=np.cumsum(w*y[:,None],axis=0)[ends];tot=np.cumsum(w,axis=0)[ends];inc=np.diff(np.vstack([np.zeros((1,tp.shape[1])),tp]),axis=0);precision=np.divide(tp,tot,out=np.zeros_like(tp),where=tot>0);denom=tp[-1];v=np.divide((precision*inc).sum(0),denom,out=np.full(len(denom),np.nan),where=denom>0);result.extend(v)
 return np.asarray(result)

def eq(a,b,eps=1e-12):
 if a is None or b is None:return a is None and b is None
 return np.isfinite(a) and np.isfinite(b) and abs(float(a)-float(b))<=eps

L=P/'frozen_checkpoint_thresholds.json';assert L.is_file();frozen=json.loads(L.read_text());assert len(frozen['arms'])==96;summary=json.loads((P/'single_formal_summary.json').read_text());raw={}
for sp in ['val','test']:
 with np.load(RAW/f'{sp}.npz',allow_pickle=False) as z:raw[sp]={k:z[k] for k in z.files}
report={'scope':'independent validation of fixed-25 single-vessel 8 backbones, 3 seeds, 4 arms; test sources previously exposed','frozen_sha256':sha(L),'freeze_mtime_utc':datetime.datetime.fromtimestamp(L.stat().st_mtime,datetime.timezone.utc).isoformat(),'checked_models':0,'checked_scores':0,'checked_strata':0,'tested_ci':0,'max_metric_abs_error':0.,'max_ci_abs_error':0.,'postfreeze_test_mtime_violations':[],'unconverged_candidates':[],'split_counts':{}}
for sp in ('val','test'):
 data=raw[sp]; y=data['y']; ids=data['window_id'].astype(str);mmsi=data['mmsi'].astype(str);reg=data['region'].astype(str);report['split_counts'][sp]={'windows':len(y),'mmsi':len(np.unique(mmsi)),'regions':len(np.unique(reg))}
 assert len(y)==1680 and len(set(ids))==1680
 for n in names:
  unique,inv=np.unique(mmsi,return_inverse=True);rng=np.random.default_rng(20261101);draw=rng.integers(0,len(unique),size=(B,len(unique)));counts=np.stack([np.bincount(x,minlength=len(unique)) for x in draw]);bs={};per={}
  for arm in arms:
   calc=[];per[arm]=[]
   for seed in (0,1,2):
    folder=P/'runs'/n/f'seed{seed}';p=folder/arm;rec=json.loads((p/'result.json').read_text());entry=frozen['arms'][f'{n}/{seed}/{arm}'];assert len(rec['history'])==25 and rec['history'][-1]['epoch']==25 and rec['seed']==seed
    assert sha(p/'final.pt')==entry['checkpoint_sha256'] and sha(p/'final_val.npz')==entry['validation_sha256']
    init=[json.loads((folder/a/'result.json').read_text())['init'] for a in arms];assert len(set(init))==1
    v=np.load(p/'final_val.npz',allow_pickle=False);neg=np.sort(v['score'][v['y']==0])
    for b,t in entry['thresholds'].items():
     index=min(len(neg)-1,max(0,int(np.ceil((1-float(b))*len(neg)))-1));assert eq(float(neg[index]),t['threshold']) and eq(float((neg>neg[index]).mean()),t['val_FAR'])
    if sp=='val':z={k:v[k] for k in v.files}
    else:
     pred=folder/'test'/f'{arm}_scores.npz';rj=json.loads((folder/'test/results.json').read_text())['metrics'][arm];assert sha(pred)==rj['prediction_sha256'] and sha(p/'final.pt')==rj['checkpoint_sha256']
     if pred.stat().st_mtime<L.stat().st_mtime:report['postfreeze_test_mtime_violations'].append(str(pred))
     with np.load(pred,allow_pickle=False) as zz:z={k:zz[k] for k in zz.files}
    for k,expect in [('y',y),('mmsi',mmsi),('region',reg),('window_id',ids)]:
     ok=np.array_equal(z[k],expect) if k=='y' else np.array_equal(z[k].astype(str),np.asarray(expect).astype(str))
     assert ok,(n,seed,arm,sp,k)
    s=z['score'];pooled=float(average_precision_score(y,s));auc=float(roc_auc_score(y,s));macro=float(np.mean([average_precision_score(y[reg==r],s[reg==r]) for r in np.unique(reg)]));per[arm].append({'pooled_AP':pooled,'macro_AP':macro,'AUROC':auc});calc.append(grouped_aps(y,s,inv,counts))
    ss=summary['models'][n]['validation' if sp=='val' else 'test']['performance'][arm]
    for k,vv in [('pooled_AP',pooled),('macro_AP',macro),('AUROC',auc)]:
     expected=ss[k]['seeds'][seed];err=abs(expected-vv);report['max_metric_abs_error']=max(report['max_metric_abs_error'],err);assert err<1e-12
    if sp=='test':
     assert eq(rj['AP'],pooled) and eq(rj['AUROC'],auc) and eq(rj['macro_AP'],macro)
     for b,t in entry['thresholds'].items():
      hit=s>t['threshold'];a=rj['operating_points'][b];assert eq(a['actual_test_FAR'],hit[y==0].mean()) and eq(a['test_recall'],hit[y>0].mean()) and eq(a['threshold'],t['threshold'])
     for col in ('noise_type','noise_scope','observation_noise','observation_noise_effective','target_noise_effective','neighbor_noise_effective'):
      if col not in data:continue
      values=data[col].astype(str)
      for value,metric in rj['strata_'+col].items():
       m=values==value;assert int(m.sum())==metric['n'] and int(y[m].sum())==metric['positive']
       if 0<y[m].sum()<m.sum():assert eq(average_precision_score(y[m],s[m]),metric['AP']) and eq(roc_auc_score(y[m],s[m]),metric['AUROC'])
       else:assert metric['AP'] is None and metric['AUROC'] is None
       report['checked_strata']+=1
    if arm!='raw' and rec['history'][-1]['loss']>.68:report['unconverged_candidates'].append({'name':n,'seed':seed,'arm':arm,'loss':rec['history'][-1]['loss'],'base_validation_ap':rec['final']['pooled_AP']})
    report['checked_scores']+=1
   bs[arm]=np.array(calc)
  for arm in arms:
   target=summary['models'][n]['validation' if sp=='val' else 'test']['performance'][arm]['pooled_AP_mmsi_CI'];got=np.nanpercentile(bs[arm].mean(axis=0),[2.5,97.5]);err=float(np.max(np.abs(got-target)));report['max_ci_abs_error']=max(report['max_ci_abs_error'],err);assert err<1e-12
  for arm in ('raw','causal_bundle','f0_bundle'):
   target=summary['models'][n]['validation' if sp=='val' else 'test']['paired_pooled_AP']['f1_minus_'+arm]['CI95'];got=np.nanpercentile((bs['f1_bundle']-bs[arm]).mean(axis=0),[2.5,97.5]);err=float(np.max(np.abs(got-target)));report['max_ci_abs_error']=max(report['max_ci_abs_error'],err);assert err<1e-12;report['tested_ci']+=1
  print('OK',sp,n,flush=True)
assert not(report['postfreeze_test_mtime_violations']);report['checked_models']=96
path=O/'independent_single_metrics_audit.json';assert not path.exists();path.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
