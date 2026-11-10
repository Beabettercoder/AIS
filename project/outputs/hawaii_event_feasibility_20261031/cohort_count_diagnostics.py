from pathlib import Path
import sys,json
from collections import Counter,defaultdict
import numpy as np,pandas as pd
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OUT=ROOT/'hawaii_event_feasibility_20261031'
sys.path.insert(0,str(ROOT/'balanced_global_v2/original_method_and_relation_v1_20261026/relation/vendor'))
from audit_neighborhood import Source,wrap,sha,NM
m=json.loads((OUT/'canonical_context_v2.json').read_text());src=Source({'canonical_file':str(OUT/'canonical_context_v2.parquet'),'sha256':m['canonical_sha256'],'core_bbox':m['core_bbox']})
targets=pd.read_parquet(ROOT/'balanced_global_v2/release_20261026/selected_windows.parquet',columns=['mmsi','split']);blocked=set(targets.loc[targets['split'].isin(['train','val']),'mmsi'].astype(str));md=json.loads((ROOT/'balanced_global_v2/relation_dual_pressure_v2_20261027/dataset_manifest.json').read_text());blocked.update(str(s['mmsi']) for s in md['samples'] if s.get('split') in ('train','val'))
allids=set(src.ids);allowed=allids-blocked;rows=[];summary={}
for x in json.loads((OUT/'candidate_targets.json').read_text())['windows']:
 t=np.asarray(x['timestamps'],float);target=x['mmsi'];p,s,c=src.source_target(target,t);a=max(2,len(t)//2-1);at=t[a]
 if not(s[a]>=3 and np.median(s[:a+1])>=3 and t[-1]-at>=180):continue
 ids=src.latest(at,target,allids);d=np.linalg.norm(src.p[ids]-p[a],axis=1);angle=np.abs(wrap(src.c[ids]-c[a]));ds=np.abs(src.s[ids]-s[a]);finite=src.good[ids]&src.valid[ids]&(src.w[ids]>=.3);sp=src.s[ids]>=3
 spatial=finite&sp&(d<=5*NM);direction=spatial&(angle<=30);joint=direction&(ds<=4.5)
 ids_original=src.ids[ids];unblocked=np.array([str(z) in allowed for z in ids_original])
 rows.append({'name':x['name'],'mmsi':target,'anchor_time_s':int(at),'tgt_speed_kn':float(s[a]),'n_latest':int(len(ids)),'n_finite_quality':int(finite.sum()),'n_5nm_speed':int(spatial.sum()),'n_plus_direction':int(direction.sum()),'n_plus_speed_match':int(joint.sum()),'n_after_known_identity_exclusion':int((joint&unblocked).sum()),'blocked_mmsi':ids_original[joint&~unblocked].tolist()})
for name in sorted(set(x['name'] for x in rows)):
 a=[x for x in rows if x['name']==name];summary[name]={'n':len(a),'stages':{key:{'mean':float(np.mean([x[key] for x in a])),'max':int(max(x[key] for x in a)),'n_ge3':sum(x[key]>=3 for x in a)} for key in ('n_latest','n_finite_quality','n_5nm_speed','n_plus_direction','n_plus_speed_match','n_after_known_identity_exclusion')}}
obj={'scope':'descriptive observations on all target-eligible windows; no model outputs or labels used','cutoff':'source-independent latest report per vessel <=300sec before fixed target anchor','summary':summary,'rows':rows}
(OUT/'cohort_count_diagnostics.json').write_text(json.dumps(obj,indent=2),encoding='utf8');print(json.dumps(summary,indent=2),flush=True)
