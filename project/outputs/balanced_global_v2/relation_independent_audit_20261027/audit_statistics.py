"""Independently recompute all simple statistics, metrics and cluster bootstrap draws."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from independent_features import features
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');RUN=ROOT/'relation_dual_pressure_v2_20261027';OUT=ROOT/'relation_independent_audit_20261027'
ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['val','test'],required=True);args=ap.parse_args();split=args.split
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((RUN/'dataset_manifest_combined.json').read_text(encoding='utf8'))
summary=json.loads((RUN/f'statistics_summary_{split}.json').read_text(encoding='utf8'))
boot_saved=json.loads((RUN/f'statistics_bootstrap_{split}.json').read_text(encoding='utf8'))
assert summary['dataset_sha256']==sha(RUN/'dataset_manifest_combined.json')
with (RUN/f'statistics_{split}.csv').open(encoding='utf8',newline='') as f:saved=list(csv.DictReader(f))
index={(r['version'],r['view'],r['region'],r['source'],int(r['case'])):r for r in saved};assert len(index)==len(saved)
source=[r for r in manifest['samples'] if r['split']==split];methods=('target_only','neighbor_only','mean','median','absolute_weighted_median','absolute_report_weighted_median','mixed_report_weighted_median')
groups=defaultdict(list);max_stat=0.;n=0
for idx,item in enumerate(source):
 for view in ('base','natural','suffix'):
  p=RUN/item['files'][view]['path'];assert sha(p)==item['files'][view]['sha256']
  with np.load(p,allow_pickle=False) as f:z={k:f[k] for k in ('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets','y')}
  assert np.array_equal(z['y'],[0,1,1,0])
  for case in range(4):
   key=(item['version'],view,item['region'],item['window_id'],case);row=index.pop(key)
   assert row['mmsi']==item['mmsi'] and int(row['y'])==int(z['y'][case])
   _,S=features(z,case)
   for k in methods:
    err=abs(float(row[k])-S[k]);max_stat=max(max_stat,err);assert err<1e-9,(key,k,err)
   groups[(item['version'],view)].append(row);n+=1
 if (idx+1)%150==0:print('AUDIT_STATS',split,idx+1,'/',len(source),flush=True)
assert not index
def aps(scores,y,w):
 idx=np.argsort(-scores,kind='stable');scores=scores[idx];y=y[idx];w=w[:,idx]
 starts=np.r_[0,np.flatnonzero(scores[1:]!=scores[:-1])+1]
 mass=np.add.reduceat(w,starts,axis=1);pos=np.add.reduceat(w*y[None,:],starts,axis=1)
 cm=np.cumsum(mass,axis=1);cp=np.cumsum(pos,axis=1)
 return (np.divide(cp,cm,out=np.zeros_like(cp),where=cm>0)*pos).sum(axis=1)/np.maximum(cp[:,-1],1)
max_metric=0.;max_boot=0.;C=0
for (ver,view),rows in groups.items():
 y=np.array([int(r['y']) for r in rows]);rgn=np.array([r['region'] for r in rows]);mmsi=np.array([r['mmsi'] for r in rows]);ids,inv=np.unique(mmsi,return_inverse=True)
 report=summary['per_version'][ver][view];assert report['sources']==len(rows)//4 and report['target_mmsi']==len(ids)
 for m in methods:
  p=np.array([float(r[m]) for r in rows]);obs=(float(average_precision_score(y,p)),float(roc_auc_score(y,p)),float(np.mean([average_precision_score(y[rgn==r],p[rgn==r]) for r in np.unique(rgn)])))
  expect=report['metrics'][m]
  for x,z in zip(obs,(expect['pooled_AP'],expect['AUROC'],expect['macro_AP_present_regions'])):
   max_metric=max(max_metric,abs(x-z));assert abs(x-z)<1e-12
 rng=np.random.default_rng(20261027);counts=np.zeros((2000,len(ids)),np.float32)
 for i in range(2000):
  selected=rng.choice(ids,size=len(ids),replace=True);counts[i]=np.bincount(np.searchsorted(ids,selected),minlength=len(ids))
 weights=counts[:,inv];base=aps(np.array([float(r['median']) for r in rows]),y,weights)
 for m in ('absolute_weighted_median','absolute_report_weighted_median','mixed_report_weighted_median'):
  p=np.array([float(r[m]) for r in rows]);draws=aps(p,y,weights)-base;recorded=np.array(boot_saved[ver+'_'+view][m]);assert len(recorded)==2000
  err=float(np.max(np.abs(draws-recorded)));max_boot=max(max_boot,err);assert err<1e-10,(ver,view,m,err)
  ci=np.quantile(draws,[.025,.975]);exp=report['paired_mmsi_bootstrap'][m]['CI95'];assert np.max(np.abs(ci-exp))<1e-10;C+=1
out={'status':'passed','split':split,'sources':len(source),'evaluated_case_rows':n,'methods':list(methods),'max_observed_stat_error':max_stat,
     'max_metric_error':max_metric,'bootstrap_comparisons':C,'bootstrap_draws_each':2000,'max_individual_bootstrap_error':max_boot,
     'manifest_sha256':sha(RUN/'dataset_manifest_combined.json'),'statistics_sha256':sha(RUN/f'statistics_{split}.csv'),
     'helper_sha256':sha(Path(__file__).parent/'independent_features.py'),'script_sha256':sha(Path(__file__)),
     'scope':'reconstructed both original and relaxed observational statistics from NPZ; cluster bootstrap conditional on target MMSI and existing source distribution'}
with (OUT/f'statistics_audit_{split}.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
