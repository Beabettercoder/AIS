import json
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/mnt/sda/WangZijia/AIS/outputs/hawaii_event_feasibility_20261031'); O=Path('/mnt/sda/WangZijia/AIS/outputs/hawaii_independent_audit_20261031'); a=pd.read_parquet(R/'canonical_context_v2.parquet');a['_i']=np.arange(len(a));a=a.sort_values(['mmsi','timestamp','_i'],kind='stable').drop_duplicates(['mmsi','timestamp'],keep='last').reset_index(drop=True); groups={str(m):g for m,g in a.groupby('mmsi',sort=False)}
wins=json.loads((R/'ood_windows_v2.json').read_text())['all_windows'];expected=json.loads((R/'member_history_audit.json').read_text());old={(w['window_index'],m['mmsi']):m for w in expected['windows'] for m in w['members']};out=[];bad=[]
for r in wins:
 t=np.asarray(r['timestamps'],dtype=float);anchor=max(2,len(t)//2-1);at=t[anchor]
 for m in r['cohort']:
  g=groups[m];g=g[(g.timestamp>=t[0]-300)&(g.timestamp<=t[-1])];good=np.isfinite(g.lat.to_numpy())&np.isfinite(g.lon.to_numpy())&np.isfinite(g.sog.to_numpy())&np.isfinite(g.cog.to_numpy())&(g.sog.to_numpy()>=0)&(g.sog.to_numpy()<102.2)&(g.cog.to_numpy()>=0)&(g.cog.to_numpy()<360); nt=g.timestamp.to_numpy(float)[good]; ix=np.searchsorted(nt,t,side='right')-1
  valid=np.zeros(len(t),bool) if not len(nt) else (ix>=0)&(t>=nt[np.maximum(ix,0)])&(t-nt[np.maximum(ix,0)]<=300)
  nr=len(nt);last=bool(nr and nt[-1]>=at+181);pre=int(valid[:anchor+1].sum());post=int(valid[anchor+1:][-3:].sum());passed=nr>=3 and last and pre>=2 and post>=2
  rec={'window':r['event_observed_window_index'],'mmsi':m,'num_reports':nr,'last':last,'pre':pre,'post':post,'pass':passed};out.append(rec);o=old[(rec['window'],m)]
  if (nr,last,pre,post,passed)!=(o['num_reports'],o['last_report_after_anchor_181s'],o['pre_alignment'],o['last_three_post_alignment'],o['whole_window_original_member_policy_pass']):bad.append(rec)
assert len(out)==len(old)==97 and not bad
ans={'independent_source':'Pandas canonical full data, own nearest report search and mask; no model score','windows':len(wins),'members_checked':len(out),'members_fail_original_history_policy':sum(not r['pass'] for r in out),'differences':bad,'caveat':'No OOD window has 3 qualifying members; per-member history passing does not make window eligible.'};(O/'independent_member_history.json').write_text(json.dumps(ans,indent=2),encoding='utf8');print(json.dumps(ans,indent=2))
