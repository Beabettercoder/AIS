"""Frozen observed-only statistics and cluster bootstrap; test gated by model lock."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,sys
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');OUT=ROOT/'relation_dual_pressure_v2_20261027';PRIOR=ROOT/'relation_robust_v1_20261027';sys.path.insert(0,str(PRIOR));import run_relation as rr
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
orig=rr.weights
def absolute(*args):
 valid,wabs,wmix,e,rel=orig(*args);return valid,wabs,wabs,e,rel
def score(o,case):
 rr.weights=orig;s,_=rr.extract(o,case);rr.weights=absolute;a,_=rr.extract(o,case);rr.weights=orig
 return {'target_only':s['target_only'],'neighbor_only':s['neighbor_only'],'mean':s['mean'],'median':s['median'],
 'absolute_weighted_median':a['mixed_weighted_median'],'absolute_report_weighted_median':a['mixed_report_weighted_median'],
 'mixed_report_weighted_median':s['mixed_report_weighted_median']}
def metrics(y,p,regions):
 y=np.array(y);p=np.array(p);regions=np.array(regions);out={}
 for reg in sorted(set(regions)):
  q=regions==reg;out[str(reg)]=float(average_precision_score(y[q],p[q]))
 return {'pooled_AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),
 'macro_AP_present_regions':float(np.mean(list(out.values()))),'per_region_AP':out}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=['train','val','test']);args=ap.parse_args();s=args.split
 manifest=OUT/'dataset_manifest_combined.json';frozen_stat=json.loads((OUT/'statistics_lock.json').read_text())
 assert frozen_stat['dataset_sha256']==sha(manifest) and frozen_stat['script_sha256']==sha(Path(__file__)) and frozen_stat['prior_extract_sha256']==sha(PRIOR/'run_relation.py')
 allm=json.loads(manifest.read_text());samples=[x for x in allm['samples'] if x['split']==s]
 if s=='test':
  lock=OUT/'model_study/model_freeze_lock.json'
  if not lock.exists():raise RuntimeError('Test stats blocked until neural model freeze')
  frozen=json.loads(lock.read_text());assert frozen['dataset_sha256']==sha(manifest)
 methods=['target_only','neighbor_only','mean','median','absolute_weighted_median','absolute_report_weighted_median','mixed_report_weighted_median'];records=[]
 for n,item in enumerate(samples,1):
  for view in ('base','natural','suffix'):
   f=OUT/item['files'][view]['path'];assert sha(f)==item['files'][view]['sha256']
   with np.load(f,allow_pickle=False) as z:
    o=rr.observed(z);y=z['y'].astype(int)
   for case in range(4):
    st=score(o,case);records.append({'version':item['version'],'split':s,'view':view,'region':item['region'],'source':item['window_id'],'mmsi':item['mmsi'],'case':case,'y':int(y[case]),**st})
  if n%200==0:print('STATS',s,n,'/',len(samples),flush=True)
 path=OUT/f'statistics_{s}.csv'
 with path.open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 out={'status':'complete','split':s,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'dataset_sha256':sha(manifest),'protocol_sha256':sha(OUT/'protocol.json'),'script_sha256':sha(Path(__file__)),'per_version':{}}
 boot={}
 for ver in ['original','relaxed']:
  out['per_version'][ver]={};rows={v:[r for r in records if r['version']==ver and r['view']==v] for v in ['base','natural','suffix']}
  if not rows['base']:continue
  for view,rs in rows.items():
   y=np.array([r['y'] for r in rs]);region=[r['region'] for r in rs];subjects=np.array([r['mmsi'] for r in rs]);unique=np.unique(subjects)
   out['per_version'][ver][view]={'sources':len(rs)//4,'target_mmsi':len(unique),'metrics':{m:metrics(y,[r[m] for r in rs],region) for m in methods}}
   if s=='train':continue
   groups={u:np.flatnonzero(subjects==u) for u in unique};rng=np.random.default_rng(20261027);comp=['absolute_weighted_median','absolute_report_weighted_median','mixed_report_weighted_median'];scores={m:np.array([r[m] for r in rs]) for m in comp+['median']};draws={m:[] for m in comp}
   for _ in range(2000):
    ix=np.concatenate([groups[u] for u in rng.choice(unique,size=len(unique),replace=True)]);base=float(average_precision_score(y[ix],scores['median'][ix]))
    for m in comp:draws[m].append(float(average_precision_score(y[ix],scores[m][ix])-base))
   out['per_version'][ver][view]['paired_mmsi_bootstrap']={m:{'diff_AP':float(average_precision_score(y,scores[m])-average_precision_score(y,scores['median'])),'CI95':np.quantile(draws[m],[.025,.975]).tolist(),'draws':2000} for m in comp}
   boot[ver+'_'+view]=draws
   print('SUMMARY',s,ver,view,out['per_version'][ver][view]['sources'],out['per_version'][ver][view]['target_mmsi'],{m:round(out['per_version'][ver][view]['metrics'][m]['pooled_AP'],4) for m in methods},flush=True)
 with (OUT/f'statistics_summary_{s}.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
 if s!='train':
  with (OUT/f'statistics_bootstrap_{s}.json').open('x',encoding='utf8') as f:json.dump(boot,f,ensure_ascii=False)
 print('STATS_COMPLETE',s,flush=True)
if __name__=='__main__':main()
