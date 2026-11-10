from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OLD=ROOT/'balanced_global_v2/relation_dual_pressure_v2_20261027';RUN=ROOT/'heading_evidence_pilot_20261030';OUT=ROOT/'heading_independent_audit_20261030';S=json.loads((RUN/'paired_summary.json').read_text()); V=('base','natural','suffix','combined_l100_m600');ARMS=('C0','F0_heading','F1_noheading','F1_heading');rng=np.random.default_rng(20261030)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={}
manifest=json.loads((OLD/'dataset_manifest_combined.json').read_text());rows=[r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='val'];low={}
for r in rows:
 with np.load(OLD/r['files']['base']['path'],allow_pickle=False) as z:
  source=z['source_neighbor_sog_kn'];offset=z['member_offsets'];low[r['window_id']]=any(source[int(offset[m['member_index']])+int(m['indices']['suffix']['report_index'])]<2 for m in r['stress']['members'])
with np.load(RUN/'val_heading_features.npz',allow_pickle=False) as z:
 src=z['source'].astype(str);h=z['base__F0'];valid=np.any(h[:,:,2]>0,axis=1)
heads={s:bool(np.any(valid[src==s])) for s in np.unique(src)};groups={'all':np.ones(len(src),bool),'source_any_heading':np.array([heads[s] for s in src]),'source_no_heading':np.array([not heads[s] for s in src]),'suffix_selected_baseline_low_neighbor':np.array([low[s] for s in src])}
for name in ('tcn','gru','transformer','itransformer'):
 seeds=(0,1,2) if name in ('tcn','itransformer') else (0,);pred={};runs={};ref=None;maxerr=0
 for seed in seeds:
  for arm in ARMS:
   p=RUN/'runs'/name/f'seed{seed}'/arm;rr=json.loads((p/'result.json').read_text());z=np.load(p/'val_scores.npz',allow_pickle=False);a={k:z[k].copy() for k in z.files};z.close();pred[seed,arm]=a;runs[seed,arm]=rr
   assert rr['seed']==seed and rr['last_epoch']==(75 if name=='itransformer' else 25) and len(rr['loss_history'])==rr['last_epoch'];assert rr['prediction_sha256']==sha(p/'val_scores.npz') and rr['weights_sha256']==sha(p/'final.pt')
   if ref is None:ref=a
   for k in ('y','case','source','mmsi','region'):assert np.array_equal(ref[k],a[k])
   for v in V:
    ap=average_precision_score(a['y'],a[v+'_score']);auc=roc_auc_score(a['y'],a[v+'_score']);err=max(abs(ap-rr['views'][v]['pooled_AP']),abs(auc-rr['views'][v]['AUROC']));assert err<1e-10;maxerr=max(maxerr,err)
 for seed in seeds:assert len(set(runs[seed,arm]['init_sha256'] for arm in ARMS))==1
 assert np.array_equal(ref['source'].astype(str),src);ids,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True);draw=rng.integers(0,len(ids),size=(2000,len(ids)));intervals={};groupstats={}
 for group,select in groups.items():
  groupstats[group]={'n_windows':int(select.sum()),'n_sources':len(np.unique(src[select])),'n_mmsi':len(np.unique(ref['mmsi'][select]))}
  reported=S[name]['groups'][group]
  for k,val in groupstats[group].items():assert val==reported[k],(name,group,k,val,reported[k])
  if select.sum()==0 or len(np.unique(ref['y'][select]))<2:continue
  for view in V:
   scores={(seed,arm):pred[seed,arm][view+'_score'][select] for seed in seeds for arm in ARMS};yy=ref['y'][select].astype(float);ii=inv[select];cache={}
   for k,p in scores.items():
    order=np.argsort(-p,kind='stable');v=p[order];ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1];cache[k]=(order,ends)
    ap=average_precision_score(yy,p);assert abs(ap-reported['views'][view]['values'][f'{k[0]}_{k[1]}'])<1e-10
   def ap(k,w):
    o,end=cache[k];wg=w[o];pos=np.cumsum(wg*yy[o])[end];tot=np.cumsum(wg)[end]
    if pos[-1]<=0:return np.nan
    return float(np.sum(np.diff(np.r_[0.,pos])*np.divide(pos,tot,out=np.zeros_like(pos),where=tot>0))/pos[-1])
   pairs={'heading_vs_noheading':('F1_heading','F1_noheading'),'reliability_config':('F1_heading','F0_heading'),'heading_vs_C0':('F1_heading','C0')};boots={k:[] for k in pairs}
   for sample in draw:
    w=np.bincount(sample,minlength=len(ids))[ii];av={k:ap(k,w) for k in scores}
    for label,(a,b) in pairs.items():
     d=[av[s,a]-av[s,b] for s in seeds]
     if np.isfinite(d).all():boots[label].append(float(np.mean(d)))
   for label,values in boots.items():
    ci=list(map(float,np.quantile(values,[.025,.975]))) if len(values)>100 else None;original=reported['views'][view]['bootstrap95'][label]
    assert (ci is None and original is None) or np.allclose(ci,original,atol=1e-12),(name,group,view,label,ci,original)
   intervals[group+'_'+view]='matched'
 out[name]={'seeds':list(seeds),'n_models':len(pred),'max_AP_AUROC_error':maxerr,'paired_initial_hash_equal':True,'sample_identity_equal':True,'group_stats':groupstats,'bootstraps':intervals}
 print(name,'passed',flush=True)
(OUT/'results_independent.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('ALL_PASSED')
