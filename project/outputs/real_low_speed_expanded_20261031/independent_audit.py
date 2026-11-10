#!/usr/bin/env python3
"""Independently rejoin a prespecified sample to canonical train AIS."""
from pathlib import Path
from collections import Counter
import hashlib,json,numpy as np,pandas as pd
ROOT=Path('/mnt/sda/WangZijia/AIS');REL=ROOT/'outputs/balanced_global_v2/release_20261026';OUT=ROOT/'outputs/real_low_speed_expanded_20261031'
def circ(a):
 z=np.exp(1j*np.deg2rad(a)).mean();return float(np.degrees(np.angle(z))%360),float(abs(z))
def delta(a,b):return (b-a+180)%360-180
df=pd.read_csv(OUT/'review_manifest.csv',dtype={'mmsi':str});spl=pd.read_parquet(REL/'global_all_mmsi_split.parquet',columns=['mmsi','split']);train=set(spl.loc[spl.split=='train','mmsi'].astype(str));nontrain=set(spl.loc[spl.split!='train','mmsi'].astype(str));ids=set(df.mmsi);assert ids<=train and not(ids&nontrain)
sample=df.assign(order=df.key.map(lambda x:hashlib.sha256(x.encode()).hexdigest())).sort_values('order').head(40)
counts=Counter();max_cog=0.;max_head=0.
for region,req in sample.groupby('region'):
 reg=json.loads((REL/'source_registry.json').read_text())['regions'][region]; people=sorted(req.mmsi.astype('int64').unique().tolist())
 raw=pd.read_parquet(reg['canonical_file'],columns=['mmsi','timestamp','sog','cog','heading'],filters=[('mmsi','in',people)])
 assert set(raw.mmsi.astype(str).unique())<=train
 raw['mmsi']=raw.mmsi.astype(str);raw['order']=np.arange(len(raw));raw=raw.sort_values(['mmsi','timestamp','order'],kind='stable').drop_duplicates(['mmsi','timestamp'],keep='last').set_index(['mmsi','timestamp'])
 for _,item in req.iterrows():
  t=json.loads(item.timestamps); g=raw.loc[[(item.mmsi,int(x)) for x in t]];assert len(g)==20
  s=g.sog.to_numpy(float);c=g.cog.to_numpy(float);h=g.heading.to_numpy(float);pre=np.arange(20//3);post=np.arange(20-20//3,20);pa,pr=circ(c[pre[s[pre]>=1]]);pb,ps=circ(c[post[s[post]>=1]]);d=delta(pa,pb);max_cog=max(max_cog,abs(d-float(item.cog_delta_deg)));assert abs(d-float(item.cog_delta_deg))<1e-8
  assert (item['group']=='turn_rule' and abs(d)>=30) or (item['group']=='straight_control' and abs(d)<=10)
  valid=np.isfinite(h)&(h>=0)&(h<360);a=pre[valid[pre]];b=post[valid[post]];yes=len(a)>=2 and len(b)>=2
  if yes:
   ha,hr=circ(h[a]);hb,hs=circ(h[b]);yes=min(hr,hs)>=.5
   if yes:dh=delta(ha,hb);max_head=max(max_head,abs(dh-float(item.heading_delta_deg)));assert abs(dh-float(item.heading_delta_deg))<1e-5
  assert yes==bool(item.heading_usable);counts[item['group']]+=1
def desc(g):
 return {'windows':len(g),'mmsi':int(g.mmsi.nunique()),'median_sog_kn':float(g.median_sog_kn.median()),'median_abs_cog_change_deg':float(g.cog_delta_deg.abs().median()),'regions':dict(Counter(g.region))}
missing={}
for label in ['turn_rule','straight_control']:
 g=df[df['group']==label];missing[label]={'heading_usable':desc(g[g.heading_usable]),'heading_unavailable':desc(g[~g.heading_usable])}
out={'scope':'train only; 40 deterministic hash-selected candidate/control windows rejoined to source','sample_windows':len(sample),'sample_by_label':dict(counts),'key_or_value_mismatches':0,'maximum_cog_difference_deg':max_cog,'maximum_heading_difference_deg':max_head,'target_ids_all_train':True,'unique_targets':len(ids),'max_windows_per_mmsi':int(df.groupby('mmsi').size().max()),'heading_availability_strata':missing}
(OUT/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
