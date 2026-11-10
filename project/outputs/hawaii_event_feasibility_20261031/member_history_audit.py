"""Quantify omitted original whole-window cohort prerequisites for OOD windows; no models/scores."""
from pathlib import Path
import json,sys,numpy as np
from collections import Counter
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'hawaii_event_feasibility_20261031';sys.path.insert(0,str(R/'balanced_global_v2/original_method_and_relation_v1_20261026/relation/vendor'))
from audit_neighborhood import Source
m=json.loads((O/'canonical_context_v2.json').read_text());src=Source({'canonical_file':str(O/'canonical_context_v2.parquet'),'sha256':m['canonical_sha256'],'core_bbox':m['core_bbox']});wins=json.loads((O/'ood_windows_v2.json').read_text())['all_windows'];groups={};out=[]
for r in wins:
 t=np.asarray(r['timestamps'],float);a=max(2,len(t)//2-1);at=t[a];details=[]
 for member in r['cohort']:
  ii=src.history(member,t[0]-300,t[-1]);ii=ii[src.good[ii]];nt=src.t[ii];ix=np.searchsorted(nt,t,side='right')-1;safe=np.maximum(ix,0)
  aligned=np.zeros(len(t),bool) if len(nt)==0 else (ix>=0)&(t>=nt[safe])&(t-nt[safe]<=300)
  details.append({'mmsi':member,'num_reports':len(ii),'last_report_after_anchor_181s':bool(len(ii)>0 and nt[-1]>=at+181),'pre_alignment':int(aligned[:a+1].sum()),'last_three_post_alignment':int(aligned[a+1:][-3:].sum()),'whole_window_original_member_policy_pass':bool(len(ii)>=3 and nt[-1]>=at+181 and aligned[:a+1].sum()>=2 and aligned[a+1:][-3:].sum()>=2)})
 key=r['name'];g=groups.setdefault(key,Counter());g['windows']+=1;g['zero_member']+=int(len(details)==0);g['one_or_two_members']+=int(0<len(details)<3);g['any_member_policy_failed']+=int(any(not q['whole_window_original_member_policy_pass'] for q in details));g['members_total']+=len(details);g['members_failed']+=sum(not q['whole_window_original_member_policy_pass'] for q in details)
 out.append({'window_index':r['event_observed_window_index'],'name':key,'anchor_time_s':int(at),'members':details})
f={'scope':'No model scoring; matching existing offline source qualification, >=3 members is already false for every OOD window; these quantify additional membership deviations','summary':{k:dict(v) for k,v in groups.items()},'windows':out};(O/'member_history_audit.json').write_text(json.dumps(f,indent=2),encoding='utf8');print(json.dumps(f['summary'],indent=2))
