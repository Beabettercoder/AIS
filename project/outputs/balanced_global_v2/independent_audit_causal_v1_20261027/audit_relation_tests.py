from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,sys
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');out=root/'independent_audit_causal_v1_20261027';dirs=[root/'relation_test_v1_20261027',root/'relation_test_expanded_v1_20261027'];views={'base':('file','sha256'),'existing_speed':('existing_file','existing_sha256'),'high_speed':('high_file','high_sha256')}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):
 with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def weighted(x,w):
 if np.sum(w)<=1e-12:return float(np.median(x))
 order=np.argsort(x,kind='stable');return float(x[order[min(np.searchsorted(np.cumsum(w[order]),np.sum(w)/2,side='left'),len(x)-1)]])
def stats(z,case):
 t=z['target_t'][case];s=z['target_sog_kn'][case];n=len(t);anchor=max(2,n//2-1);pre=np.arange(anchor+1);post=np.arange(anchor+1,n)[-3:];d=float(np.mean(s[post])/max(float(np.mean(s[pre])),.5)-1);ds=[];aw=[];mw=[];arm=[];mrm=[]
 for l,h in zip(z['member_offsets'][:-1],z['member_offsets'][1:]):
  l=int(l);h=int(h);nt=z['neighbor_t'][case,l:h];p=z['neighbor_pos'][case,l:h];ns=z['neighbor_sog_kn'][case,l:h];c=z['neighbor_cog_deg'][case,l:h];ang=np.deg2rad(c);v=ns[:,None]*(1852/3600)*np.column_stack((np.sin(ang),np.cos(ang)));dt=np.diff(nt);valid=np.r_[False,(dt>0)&(dt<=300)&np.isfinite(p[:-1]).all(1)&np.isfinite(p[1:]).all(1)&np.isfinite(ns[:-1])&np.isfinite(ns[1:])&np.isfinite(c[:-1])&np.isfinite(c[1:])];idx=np.flatnonzero(valid);w=np.zeros(len(nt));mix=np.zeros(len(nt));pv=(p[idx]-p[idx-1])/dt[idx-1,None];rv=(v[idx]+v[idx-1])/2;e=np.linalg.norm(pv-rv,axis=1);rel=e/(np.minimum(np.linalg.norm(pv,axis=1),np.linalg.norm(rv,axis=1))+.3);w[idx]=np.exp(-np.maximum(0,e-2.5)**2/8);mix[idx]=np.minimum(w[idx],np.exp(-np.maximum(0,rel-1)**2/2))
  ix=np.searchsorted(nt,t,side='right')-1;safe=np.maximum(ix,0);present=(ix>=0)&(t>=nt[safe])&(t-nt[safe]<=300);ai=pre[present[pre]];bi=post[present[post]]
  if len(ai)<2 or len(bi)<2:continue
  j=safe[ai];k=safe[bi];ds.append(float(ns[k].mean()/max(float(ns[j].mean()),.5)-1));aw.append(float(np.sqrt(w[j].mean()*w[k].mean())));mw.append(float(np.sqrt(mix[j].mean()*mix[k].mean())))
  def ratio(wt):
   aa=float(np.average(ns[j],weights=wt[j])) if wt[j].sum()>1e-12 else float(ns[j].mean());bb=float(np.average(ns[k],weights=wt[k])) if wt[k].sum()>1e-12 else float(ns[k].mean());return bb/max(aa,.5)-1
  arm.append(ratio(w));mrm.append(ratio(mix))
 ds=np.asarray(ds);aw=np.asarray(aw);mw=np.asarray(mw)
 return {'target_only':abs(d),'neighbor_only':abs(ds.mean()),'mean':abs(d-ds.mean()),'median':abs(d-np.median(ds)),
  'absolute_weighted_mean':abs(d-(np.average(ds,weights=aw) if aw.sum()>1e-12 else ds.mean())),
  'mixed_weighted_mean':abs(d-(np.average(ds,weights=mw) if mw.sum()>1e-12 else ds.mean())),
  'absolute_weighted_median':abs(d-weighted(ds,aw)),'mixed_weighted_median':abs(d-weighted(ds,mw)),
  'absolute_report_weighted_median':abs(d-np.median(arm)),'mixed_report_weighted_median':abs(d-np.median(mrm))}

audits=[];manifests=[];maxstat=0.;maxmetric=0.;maxboots=0.;maxsup=0.;counts=[]
for folder in dirs:
 manifest=json.loads((folder/'test_manifest.json').read_text());res=json.loads((folder/'test_results.json').read_text());lock=json.loads((folder/'freeze_lock.json').read_text());manifests.append(manifest)
 assert res['test_manifest_sha256']==sha(folder/'test_manifest.json') and res['lock_sha256']==sha(folder/'freeze_lock.json')
 for path,digest in lock['files'].items():assert sha(path)==digest
 raw=csvread(folder/'test_statistics.csv');nn=csvread(folder/'test_neural_predictions.csv');lookup={(x['view'],x['source'],int(x['case'])):x for x in raw};sources=set();ids=set();cases=0
 for item in manifest['samples']:
  sources.add(item['window_id']);ids.add(item['mmsi']);loaded={}
  for view,(filekey,hashkey) in views.items():
   path=folder/'cases'/item[filekey];assert sha(path)==item[hashkey]
   with np.load(path,allow_pickle=False) as z:loaded[view]={k:z[k] for k in z.files}
   z=loaded[view]
   if view=='base':
    assert str(z['target_mmsi'])==item['mmsi'] and str(z['source_window_id'])==item['window_id'];assert np.array_equal(z['y'],np.bitwise_xor(z['meta_target_factor'],z['meta_cohort_factor']))
    assert len(set(map(str,z['cohort_mmsi'])))==len(z['cohort_mmsi']) and item['mmsi'] not in set(map(str,z['cohort_mmsi']))
   for case in range(4):
    row=lookup[(view,item['window_id'],case)];assert int(row['y'])==int(loaded['base']['y'][case]) and row['mmsi']==item['mmsi']
    for name,val in stats(z,case).items():maxstat=max(maxstat,abs(float(row[name])-val))
    cases+=1
  base=loaded['base']
  for view,proofkey in [('existing_speed','old_stress'),('high_speed','high_stress')]:
   changed=base['neighbor_sog_kn'].copy()
   for member in item[proofkey]['members']:
    if member['applied']:
     for q in member['flat_indices']:changed[:,q]=np.minimum(changed[:,q]*member['factor'],102.1)
   assert np.array_equal(changed,loaded[view]['neighbor_sog_kn'])
   for k in ('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_cog_deg'):
    assert np.array_equal(base[k],loaded[view][k])
 assert len(sources)==len(manifest['samples']) and len(ids)==manifest['test_target_mmsi'] and cases==12*len(sources)
 for view in views:
  vr=[q for q in raw if q['view']==view];y=np.array([int(x['y']) for x in vr]);g=np.array([x['region'] for x in vr]);assert len(vr)==len(sources)*4
  for name,m in res['statistics'][view].items():
   s=np.array([float(q[name]) for q in vr]);ap=average_precision_score(y,s);au=roc_auc_score(y,s);region={a:average_precision_score(y[g==a],s[g==a]) for a in set(g)}
   maxmetric=max(maxmetric,abs(ap-m['pooled_AP']),abs(au-m['pooled_AUROC']),abs(np.mean(list(region.values()))-m['equal_region_AP']))
  for arch in res['neural']:
   for mode in res['neural'][arch]:
    for item in res['neural'][arch][mode][view]:
     rr=[q for q in nn if q['view']==view and q['architecture']==arch and q['mode']==mode and int(q['seed'])==item['seed']];s=np.array([float(q['score']) for q in rr]);yy=np.array([int(q['y']) for q in rr]);gg=np.array([q['region'] for q in rr]);assert len(rr)==len(vr)
     maxmetric=max(maxmetric,abs(average_precision_score(yy,s)-item['pooled_AP']),abs(roc_auc_score(yy,s)-item['pooled_AUROC']),abs(np.mean([average_precision_score(yy[gg==a],s[gg==a]) for a in set(gg)])-item['equal_region_AP']))
 draws=json.loads((folder/'test_bootstrap_draws.json').read_text());
 for name,v in draws.items():maxboots=max(maxboots,float(np.max(np.abs(np.quantile(v,[.025,.975])-res['bootstrap'][name]['ci95_percentile']))))
 # Independently replay 20 MMSI draws for the 3 stored statistics comparisons.
 subjects=np.array([q['mmsi'] for q in raw if q['view']=='base']);unique=np.unique(subjects);groups={m:np.flatnonzero(subjects==m) for m in unique};rng=np.random.default_rng(20261027)
 draws_ix=[np.concatenate([groups[m] for m in rng.choice(unique,size=len(unique),replace=True)]) for _ in range(20)]
 for view in views:
  vr=[q for q in raw if q['view']==view];y=np.array([int(q['y']) for q in vr]);b=np.array([float(q['median']) for q in vr])
  for name in ('absolute_weighted_median','mixed_report_weighted_median','absolute_report_weighted_median'):
   a=np.array([float(q[name]) for q in vr]);key=f'stat:{view}:{name}-median'
   for i,ix in enumerate(draws_ix):maxboots=max(maxboots,abs(average_precision_score(y[ix],a[ix])-average_precision_score(y[ix],b[ix])-draws[key][i]))
 counts.append({'directory':folder.name,'sources':len(sources),'target_mmsi':len(ids),'scored_cases':cases,'neural_prediction_rows':len(nn),'common_background':manifest['shared_background_mmsi'],'shared_background_reports':manifest['shared_background_source_reports']})

orig={x['window_id']:x for x in manifests[0]['samples']};expanded={x['window_id']:x for x in manifests[1]['samples']};assert set(orig).issubset(expanded)
equal_arrays=0
for wid in orig:
 for view,(fk,hk) in views.items():
  a=dirs[0]/'cases'/orig[wid][fk];b=dirs[1]/'cases'/expanded[wid][fk]
  with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
   assert set(x.files)==set(y.files)
   for k in x.files:assert np.array_equal(x[k],y[k]),(wid,view,k)
   equal_arrays+=1

folder=dirs[1];sup=json.loads((folder/'test_report_context_results.json').read_text());lock=json.loads((folder/'report_context_lock.json').read_text());assert sup['manifest_sha256']==sha(folder/'test_manifest.json') and sup['lock_sha256']==sha(folder/'report_context_lock.json')
for path,digest in lock['files'].items():assert sha(path)==digest
suprows=csvread(folder/'test_report_context_predictions.csv')
for arch in sup['models']:
 for view,items in sup['models'][arch].items():
  for item in items:
   rr=[q for q in suprows if q['architecture']==arch and q['view']==view and int(q['seed'])==item['seed']];yy=np.array([int(q['y']) for q in rr]);s=np.array([float(q['score']) for q in rr]);g=np.array([q['region'] for q in rr]);maxsup=max(maxsup,abs(average_precision_score(yy,s)-item['pooled_AP']),abs(roc_auc_score(yy,s)-item['pooled_AUROC']),abs(np.mean([average_precision_score(yy[g==a],s[g==a]) for a in set(g)])-item['equal_observed_region_AP']))
assert maxstat<1e-9 and maxmetric<1e-10 and maxboots<1e-10 and maxsup<1e-10
result={'status':'passed','test_versions':counts,'original_source_arrays_replayed_in_expanded':equal_arrays,'max_independent_stat_error':maxstat,'max_metric_error':maxmetric,'max_bootstrap_error':maxboots,'max_supplementary_metric_error':maxsup,
 'supplementary_prediction_rows':len(suprows),'noise_replay':'all source quartets and both stress views, via recorded changed member indices and multipliers',
 'bootstrap':'all saved intervals and 20 independent MMSI draws for 3 statistical comparisons per view/version',
 'limitations':'no full independent reconstruction of initial source eligibility; baseline test sources previously pilot-exposed; supplemental model was added after initial test was viewed and uses whole-window membership eligibility'}
(out/'audit_relation_tests.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
