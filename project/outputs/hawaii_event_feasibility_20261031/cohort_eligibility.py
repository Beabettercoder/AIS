"""Frozen Sandia January event-window eligibility audit. No labels or scores are read."""
from pathlib import Path
import sys,json,hashlib
from collections import Counter,defaultdict
import numpy as np,pandas as pd
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OUT=ROOT/'hawaii_event_feasibility_20261031'
sys.path.insert(0,str(ROOT/'balanced_global_v2/original_method_and_relation_v1_20261026/relation/vendor'))
from audit_neighborhood import Source,wrap,sha,NM
meta=json.loads((OUT/'canonical_context_v2.json').read_text());assert meta['canonical_sha256']==sha(OUT/'canonical_context_v2.parquet')
assert 1480000000<meta['time_range'][0]<1500000000
src=Source({'canonical_file':str(OUT/'canonical_context_v2.parquet'),'sha256':meta['canonical_sha256'],'core_bbox':meta['core_bbox']})
targets=pd.read_parquet(ROOT/'balanced_global_v2/release_20261026/selected_windows.parquet',columns=['mmsi','split']);blocked=set(targets.loc[targets['split'].isin(['train','val']),'mmsi'].astype(str))
md=json.loads((ROOT/'balanced_global_v2/relation_dual_pressure_v2_20261027/dataset_manifest.json').read_text())
blocked.update(str(s['mmsi']) for s in md['samples'] if s.get('split') in ('train','val'))
allowed=set(src.ids)-blocked
cand=json.loads((OUT/'candidate_targets.json').read_text())['windows']; result=[];counts=defaultdict(Counter)
for x in cand:
 name=x['name'];t=np.asarray(x['timestamps'],float);target=x['mmsi'];counts[name]['candidate_windows']+=1
 try:p,s,c=src.source_target(target,t)
 except KeyError:counts[name]['target_report_missing']+=1;continue
 a=max(2,len(t)//2-1);at=float(t[a]);prefix=bool(s[a]>=3 and np.median(s[:a+1])>=3 and a>=2 and t[-1]-at>=180)
 if not prefix:counts[name]['target_anchor_ineligible']+=1;continue
 counts[name]['target_anchor_eligible']+=1
 ids=src.latest(at,target,allowed);dist=np.linalg.norm(src.p[ids]-p[a],axis=1);angle=np.abs(wrap(src.c[ids]-c[a]));ds=np.abs(src.s[ids]-s[a]);ok=src.good[ids]&src.valid[ids]&(src.w[ids]>=.3)&(src.s[ids]>=3)&(dist<=5*NM)&(angle<=30)&(ds<=4.5)
 cohort=ids[ok];cohort=cohort[np.argsort(src.ids[cohort],kind='stable')][:8]
 if len(cohort)<3:counts[name]['fewer_than_three']+=1;continue
 mmsi=src.ids[cohort].tolist();hist=[];bad=False
 for m in mmsi:
  ii=src.history(m,t[0]-300,t[-1]);ii=ii[src.good[ii]]
  if len(ii)<3 or src.t[ii][-1]<at+181:bad=True;break
  nt=src.t[ii];idx=np.searchsorted(nt,t,side='right')-1;safe=np.maximum(idx,0);aligned=(idx>=0)&(t>=nt[safe])&(t-nt[safe]<=300)
  if aligned[:a+1].sum()<2 or aligned[a+1:][-3:].sum()<2:bad=True;break
  hist.append({'mmsi':m,'n_reports':len(ii),'pre_grid_count':int(aligned[:a+1].sum()),'suffix_grid_count':int(aligned[a+1:][-3:].sum())})
 if bad:counts[name]['member_history_or_alignment_ineligible']+=1;continue
 counts[name]['full_cohort_eligible']+=1
 result.append({**x,'anchor_step':a,'anchor_time_s':at,'cohort_mmsi':mmsi,'cohort_anchor_times_s':src.t[cohort].tolist(),'history_summary':hist,'coverage_certified':False})
summary={'protocol_sha256':sha(OUT/'frozen_protocol.json'),'protocol_addendum_sha256':sha(OUT/'frozen_protocol_addendum.json'),'candidate_targets_sha256':sha(OUT/'candidate_targets.json'),'canonical_sha256':sha(OUT/'canonical_context_v2.parquet'),'excluded_known_train_val_target_mmsi':len(blocked),'available_background_ids_after_exclusion':len(allowed),'counts':{k:dict(v) for k,v in counts.items()},'eligible_windows':result,'warnings':['Event intervals are event-level weak provenance, not per-window ground truth.','Original relation model target SOG threshold excludes low-speed event 2.','Coarse Hawaii bounding box only supplies projection origin and does not certify receiver coverage.','Excluding known historical train/val targets does not prove the background has never been seen.','Cohort requires future-in-window reports and is an offline task.','No model scores have been read or used.']}
(OUT/'cohort_eligibility.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary['counts'],indent=2),flush=True)
print('TOTAL_ELIGIBLE',len(result),flush=True)
