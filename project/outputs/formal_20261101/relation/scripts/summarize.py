from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
O=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/relation');names=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];views=['base','natural','suffix','combined_l100_m600'];arms=['C0','F0','F1'];report={}
def grouped_aps(y,sc,inv,draws):
 idx=np.argsort(-sc,kind='stable');s=sc[idx];y=y[idx];inv=inv[idx];ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];out=[]
 for a in range(0,len(draws),200):
  w=draws[a:a+200][:,inv].T.astype(np.float64);pw=w*y[:,None];tp=np.cumsum(pw,axis=0)[ends];tot=np.cumsum(w,axis=0)[ends];inc=np.diff(np.vstack([np.zeros((1,tp.shape[1])),tp]),axis=0);prec=np.divide(tp,tot,out=np.zeros_like(tp),where=tot>0);total=tp[-1];out.extend(np.divide((inc*prec).sum(0),total,out=np.full(len(total),np.nan),where=total>0))
 return np.array(out)
for split in ('val','test'):
 report[split]={}
 for name in names:
  data={};metrics={}
  for s in (0,1,2):
   for arm in arms:
    f=O/'runs'/name/f'seed{s}'/arm;pf=f/('val_scores.npz' if split=='val' else 'test_scores.npz');rf=f/('result.json' if split=='val' else 'test_result.json')
    with np.load(pf,allow_pickle=False) as z:data[(s,arm)]={k:z[k] for k in z.files}
    metrics[(s,arm)]=json.loads(rf.read_text(encoding='utf8'))['views']
  ref=data[(0,'C0')]
  for z in data.values():
   for k in ('y','mmsi','region','source','case'):assert np.array_equal(z[k],ref[k]),(name,k)
  vessels,inv=np.unique(ref['mmsi'],return_inverse=True);rs=np.random.default_rng(20261101);draws=rs.multinomial(len(vessels),np.ones(len(vessels))/len(vessels),size=2000)
  obj={'n_windows':len(ref['y']),'n_target_mmsi':len(vessels),'n_sources':len(np.unique(ref['source'])),'per_region':{},'views':{}}
  for reg in np.unique(ref['region']):
   m=ref['region']==reg;obj['per_region'][str(reg)]={'windows':int(m.sum()),'sources':len(np.unique(ref['source'][m])),'mmsi':len(np.unique(ref['mmsi'][m])),'positives':int(ref['y'][m].sum())}
  if 'observed_head_sog_kn' in ref:
   obj['observed_initial_speed_groups']={}
   for label,lo,hi in [('below_2kn',0,2),('2_to_5kn',2,5),('at_least_5kn',5,float('inf'))]:
    g=(ref['observed_head_sog_kn']>=lo)&(ref['observed_head_sog_kn']<hi)
    d={'windows':int(g.sum()),'target_mmsi':len(np.unique(ref['mmsi'][g])),'sources':len(np.unique(ref['source'][g])),'positive':int(ref['y'][g].sum()),'views':{}}
    if g.sum() and 0<ref['y'][g].sum()<g.sum():
     for view in views:
      d['views'][view]={arm:float(np.mean([average_precision_score(ref['y'][g],data[(seed,arm)][view+'_score'][g]) for seed in (0,1,2)])) for arm in arms}
    obj['observed_initial_speed_groups'][label]=d
  for v in views:
   obj['views'][v]={'arms':{},'paired':{}}
   draws_by_arm={}
   for arm in arms:
    rows=[metrics[(s,arm)][v] for s in (0,1,2)];aps=[float(x['pooled_AP']) for x in rows]
    obj['views'][v]['arms'][arm]={'pooled_AP_mean':float(np.mean(aps)),'pooled_AP_sd':float(np.std(aps,ddof=1)),'seeds':rows}
    draws_by_arm[arm]=np.mean([grouped_aps(ref['y'].astype(float),data[(s,arm)][v+'_score'],inv,draws) for s in (0,1,2)],axis=0)
   for control in ('C0','F0'):
    delta=np.array([metrics[(s,'F1')][v]['pooled_AP']-metrics[(s,control)][v]['pooled_AP'] for s in (0,1,2)]);boot=draws_by_arm['F1']-draws_by_arm[control]
    obj['views'][v]['paired']['F1_minus_'+control]={'mean_AP_diff':float(delta.mean()),'per_seed':delta.tolist(),'mmsi_cluster95':np.nanpercentile(boot,[2.5,97.5]).tolist()}
  report[split][name]=obj;print('SUMMARIZED',split,name,flush=True)
out={'protocol':'fixed seed0/1/2; 2000 target-MMSI cluster bootstraps of per-seed mean AP; samples not seed variability; no multiple comparison correction','test_caveat':'prior test exposed; background boats may overlap; controlled labels','results':report}
p=O/'summary.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8');print(p)
