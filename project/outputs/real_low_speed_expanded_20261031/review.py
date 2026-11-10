#!/usr/bin/env python3
"""Deterministic exploratory source-only train-split review, not ground truth."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,csv,heapq,math
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path('/mnt/sda/WangZijia/AIS');REL=ROOT/'outputs/balanced_global_v2/release_20261026'
OUT=ROOT/'outputs/real_low_speed_expanded_20261031'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def circ(x):
 z=np.exp(1j*np.deg2rad(x)).mean();return float(np.degrees(np.angle(z))%360),float(abs(z))
def delta(a,b):return float((b-a+180)%360-180)
def safe(x):return None if not math.isfinite(float(x)) else float(x)
def main():
 protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'));rule=json.loads((OUT/'rule_protocol.json').read_text(encoding='utf8'))
 assert sha(OUT/'rule_protocol.json')==protocol['rule_protocol_sha256']
 meta=json.loads((REL/'source_registry.json').read_text())['regions'];groups=pd.read_parquet(REL/'global_all_mmsi_split.parquet',columns=['mmsi','split'])
 assert not groups['mmsi'].duplicated().any();train=groups.loc[groups['split']=='train','mmsi'].astype('int64').tolist();allowed=set(train)
 assert not allowed.intersection(groups.loc[groups['split']!='train','mmsi'].astype('int64'))
 heaps=defaultdict(list);build={}; source_sha={}
 for region in protocol['regions']:
  info=meta[region];path=Path(info['canonical_file']);source_sha[region]=sha(path);assert source_sha[region]==info['sha256']
  df=pd.read_parquet(path,columns=['mmsi','timestamp','lon','lat','sog','cog','heading'],filters=[('mmsi','in',train)])
  assert set(df.mmsi.astype('int64').unique()).issubset(allowed)
  before=len(df);df['order']=np.arange(len(df));df=df.sort_values(['mmsi','timestamp','order'],kind='stable');duplicates=int(df.duplicated(['mmsi','timestamp'],keep=False).sum());df=df.drop_duplicates(['mmsi','timestamp'],keep='last')
  df['bin']=df.timestamp.astype('int64')//1800*1800
  cols=['timestamp','lon','lat','sog','cog','heading'];win_count=0; box=info['core_bbox']
  for (m,b),g in df.groupby(['mmsi','bin'],sort=False):
   if len(g)<20:continue
   a=g[cols].to_numpy(dtype=float);t=a[:,0];grid=t[0]+70*np.arange(20);ix=np.searchsorted(t,grid);l=np.maximum(ix-1,0);rr=np.minimum(ix,len(t)-1);ix=np.where(np.abs(t[l]-grid)<=np.abs(t[rr]-grid),l,rr)
   if np.unique(ix).size!=20 or np.any(np.abs(t[ix]-grid)>35) or t[ix[-1]]>=b+1800:continue
   q=a[ix];lon=q[:,1];lat=q[:,2]
   if not (np.isfinite(lon).all() and np.isfinite(lat).all() and np.all((lon>=box[0])&(lon<=box[2])&(lat>=box[1])&(lat<=box[3]))):continue
   m=int(m); assert m in allowed
   key=f'{region}|{m}|{int(b)}';priority=int(hashlib.sha256(key.encode()).hexdigest(),16);h=heaps[m];entry=(-priority,key,region,int(b),q.copy())
   if len(h)<6:heapq.heappush(h,entry)
   elif entry[0]>h[0][0]:heapq.heapreplace(h,entry)
   win_count+=1
  build[region]={'train_rows_materialized':before,'duplicate_key_rows':duplicates,'time_core_candidate_windows_before_cap':win_count,'source_sha256':source_sha[region]}
  print('REGION',region,build[region],flush=True)
 selected=[x for h in heaps.values() for x in h];counts=Counter();rec=[]
 for _,key,region,b,a in selected:
  m=int(key.split('|')[1]);t=a[:,0];lon=a[:,1];lat=a[:,2];s=a[:,3];c=a[:,4];head=a[:,5];n=len(t)
  if not (np.isfinite(s).all() and np.isfinite(c).all() and np.all((s>=0)&(c>=0)&(c<360))):counts['invalid_sog_cog']+=1;continue
  dt=np.diff(t)
  if not np.all((dt>0)&(dt<=300)):counts['gap']+=1;continue
  lat0=float(np.mean(lat));p=np.column_stack(((lon-lon[0])*111195*np.cos(np.deg2rad(lat0)),(lat-lat[0])*111195));disp=float(np.linalg.norm(p[-1]-p[0]));med=float(np.median(s))
  if not (.5<=med<=5 and (s<2).sum()>=2 and disp>=100):counts['outside_low_speed_scope']+=1;continue
  v=(s*.5144444444)[:,None]*np.column_stack((np.sin(np.deg2rad(c)),np.cos(np.deg2rad(c))));e=np.linalg.norm(np.diff(p,axis=0)/dt[:,None]-(v[1:]+v[:-1])/2,axis=1);e90=float(np.quantile(e,.9))
  if e90>4:counts['failed_kinematic_quality']+=1;continue
  k=n//3;pre=np.arange(k);post=np.arange(n-k,n);pv=pre[s[pre]>=1];qv=post[s[post]>=1]
  if len(pv)<2 or len(qv)<2:counts['insufficient_moving_cog']+=1;continue
  ca,cr=circ(c[pv]);cb,cs=circ(c[qv]);
  if min(cr,cs)<.5:counts['ambiguous_cog_direction']+=1;continue
  dc=delta(ca,cb)
  if abs(dc)>=30:group='turn_rule'
  elif abs(dc)<=10:group='straight_control'
  else:counts['intermediate_cog_change']+=1;continue
  valid=np.isfinite(head)&(head>=0)&(head<360);hp=pre[valid[pre]];hq=post[valid[post]];use=len(hp)>=2 and len(hq)>=2;dh=float('nan');diragree=float('nan')
  if use:
   ha,hr=circ(head[hp]);hb,hs=circ(head[hq]);use=min(hr,hs)>=.5
   if use:dh=delta(ha,hb);diragree=float(np.sign(dh)==np.sign(dc)) if abs(dc)>0 else float('nan')
  x=dict(key=key,window_id=hashlib.sha256(key.encode()).hexdigest()[:24],mmsi=str(m),region=region,group=group,start_utc=datetime.fromtimestamp(t[0],tz=timezone.utc).isoformat(),end_utc=datetime.fromtimestamp(t[-1],tz=timezone.utc).isoformat(),median_sog_kn=med,n_below2=int((s<2).sum()),displacement_m=disp,e90_mps=e90,cog_delta_deg=dc,n_heading_valid=int(valid.sum()),heading_usable=bool(use),heading_delta_deg=safe(dh),direction_agrees=safe(diragree),timestamps=json.dumps(t.astype('int64').tolist()),source_file=str(meta[region]['canonical_file']))
  rec.append(x);counts[group]+=1;counts['heading_usable_'+group]+=int(use)
 with (OUT/'review_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rec[0]) if rec else []);w.writeheader();w.writerows(rec)
 summary={}
 for label in ('turn_rule','straight_control'):
  vals=[r for r in rec if r['group']==label];v=[r for r in vals if r['heading_usable']]
  summary[label]=dict(windows=len(vals),mmsi=len({r['mmsi'] for r in vals}),heading_usable_windows=len(v),heading_usable_mmsi=len({r['mmsi'] for r in v}),by_region=dict(Counter(r['region'] for r in vals)),heading_abs_delta_median_deg=float(np.median([abs(r['heading_delta_deg']) for r in v])) if v else None,heading_delta_ge20=int(sum(abs(r['heading_delta_deg'])>=20 for r in v)),same_direction=int(sum(r['direction_agrees']==1 for r in v)))
 usable=[r for r in rec if r['heading_usable']];y=np.array([r['group']=='turn_rule' for r in usable]);pred=np.array([abs(r['heading_delta_deg']) for r in usable]);ids=np.array([r['mmsi'] for r in usable]);metric=None;boot=None
 if len(set(ids[y]))>=10 and len(set(ids[~y]))>=10:
  metric={'n':len(y),'positive':int(y.sum()),'prevalence':float(y.mean()),'auc':float(roc_auc_score(y,pred)),'ap':float(average_precision_score(y,pred))};rng=np.random.default_rng(20261031);unique=np.unique(ids);aa=[]
  for b in range(2000):
   c=Counter(rng.choice(unique,len(unique),replace=True));w=np.array([c[k] for k in ids],float)
   if w[y].sum() and w[~y].sum():aa.append(float(roc_auc_score(y,pred,sample_weight=w)))
  boot={'unit':'target MMSI','valid_reps':len(aa),'auc_95':[float(x) for x in np.quantile(aa,[.025,.975])]}
 report={'protocol_sha256':sha(OUT/'protocol.json'),'script_sha256':sha(Path(__file__)),'rule_protocol_sha256':sha(OUT/'rule_protocol.json'),'split_metadata_sha256':sha(REL/'global_all_mmsi_split.parquet'),'build':build,'train_mmsi_available':len(train),'train_mmsi_with_valid_time_core_window':len(heaps),'selected_windows_before_behavior_rules':len(selected),'filter_counts':dict(counts),'groups':summary,'heading_score_vs_cog_rule':metric,'bootstrap':boot,'caveats':['This protocol is an exploratory expansion chosen after the first controlled source review had too few controls','source quality rule may produce selection bias','COG rule defines candidates, not verified true maneuver or contamination','natural AIS contamination not labeled','multiple windows per MMSI and source coverage varies','Heading and COG sensor independence not verified']}
 (OUT/'review_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8');print('SUMMARY',json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
