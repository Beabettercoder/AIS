"""Train/validation-only neighborhood scan and label-blind quartet feasibility.

Use audited causal candidate identities; recover missing positions/speeds from
their recorded source observations and replay the recorded background corruption.
No classifier scores or test trajectory/label files are opened.
"""
from pathlib import Path
import argparse, collections, gzip, hashlib, json, time
import numpy as np
import pandas as pd

REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
NM=1852.;KNOT=1852/3600
CONFIGS=[dict(name=f'r{r:g}_h{h}_v{v:g}',radius_nm=r,heading_deg=h,min_speed_kn=v,sigma_nm=4.)
         for r in (8.,4.,3.5,3.) for h in (45,30) for v in (.5,2.)]
CONFIGS.append(dict(name='compact_joint_r3.5_h30_v2_sigma1.5',radius_nm=3.5,heading_deg=30,min_speed_kn=2.,sigma_nm=1.5))

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 text=json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
 with Path(p).open('x',encoding='utf-8') as f:f.write(text)
def vel(s,c):
 a=np.deg2rad(c);return s[:,None]*KNOT*np.column_stack((np.sin(a),np.cos(a)))
def kin(t,p,s,c):
 good=np.isfinite(p).all(1)&np.isfinite(s)&np.isfinite(c)&(s>=0)&(s<102.2)&(c>=0)&(c<360)
 dt=np.diff(t);valid=np.r_[False,good[:-1]&good[1:]&(dt>0)&(dt<=300)]
 e=np.zeros(len(t));v=vel(np.nan_to_num(s),np.nan_to_num(c));ii=np.flatnonzero(valid)
 e[ii]=np.linalg.norm((p[ii]-p[ii-1])/(t[ii]-t[ii-1])[:,None]-(v[ii]+v[ii-1])/2,axis=1)
 w=np.zeros(len(t));w[ii]=np.exp(-np.maximum(0,e[ii]-2.5)**2/8)
 return good,valid,e,w
def wrap(x):return (x+180)%360-180
def coverage_ok(p,box,radius):
 return box is None or (p[0]-radius>=box[0] and p[0]+radius<=box[2] and p[1]-radius>=box[1] and p[1]+radius<=box[3])
def geometry(p,w,radius):
 if len(p)<4:return False,'sparse',None,None
 w=w/w.sum();q=p-(w[:,None]*p).sum(0);cov=q.T@(w[:,None]*q)
 std=float(np.sqrt(max(0,float(np.linalg.eigvalsh(cov)[0]))))
 if std<50:return False,'degenerate_geometry',std,None
 X=np.column_stack((np.ones(len(p)),p/radius));A=X.T@(w[:,None]*X);condition=float(np.linalg.cond(A))
 return bool(np.isfinite(condition) and condition<=1e5),'valid' if np.isfinite(condition) and condition<=1e5 else 'ill_conditioned',std,condition if np.isfinite(condition) else None

class Source:
 def __init__(self,meta):
  path=Path(meta['canonical_file']);assert sha(path)==meta['sha256']
  a=pd.read_parquet(path,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
  box=meta['core_bbox'];lon=(box[0]+box[2])/2;lat=(box[1]+box[3])/2
  a['id']=a.mmsi.astype(str);a['order']=np.arange(len(a));a=a.sort_values(['id','timestamp','order'],kind='stable').drop_duplicates(['id','timestamp'],keep='last').reset_index(drop=True)
  self.ids=a.id.to_numpy();self.t=a.timestamp.to_numpy(float)
  self.p=np.column_stack(((a.lon.to_numpy()-lon)*111195*np.cos(np.deg2rad(lat)),(a.lat.to_numpy()-lat)*111195))
  self.s=a.sog.to_numpy(float);self.c=a.cog.to_numpy(float);self.lookup={(m,float(t)):i for i,(m,t) in enumerate(zip(self.ids,self.t))}
  self.ship={};self.good=np.zeros(len(a),bool);self.valid=np.zeros(len(a),bool);self.e=np.zeros(len(a));self.w=np.zeros(len(a))
  for m,g in a.groupby('id',sort=False):
   ii=g.index.to_numpy();self.ship[m]=ii
   self.good[ii],self.valid[ii],self.e[ii],self.w[ii]=kin(self.t[ii],self.p[ii],self.s[ii],self.c[ii])
  self.time_order=np.argsort(self.t,kind='stable');self.sorted_t=self.t[self.time_order]
 def observation(self,m,t,overlay):
  ix=self.lookup[(str(m),float(t))]
  if ix in overlay:return overlay[ix]
  return self.p[ix],self.s[ix],self.c[ix],self.valid[ix],self.e[ix],self.w[ix]
 def history(self,m,start,end):
  ids=self.ship[str(m)];ts=self.t[ids];return ids[(ts>=start)&(ts<=end)]
 def source_target(self,m,t):
  ii=np.array([self.lookup[(str(m),float(x))] for x in t]);return self.p[ii].copy(),self.s[ii].copy(),self.c[ii].copy()
 def latest(self,t,m,allowed):
  lo=np.searchsorted(self.sorted_t,t-300,'left');hi=np.searchsorted(self.sorted_t,t,'right');ii=self.time_order[lo:hi]
  if not len(ii):return ii
  _,rev=np.unique(self.ids[ii][::-1],return_index=True);ii=ii[len(ii)-1-rev]
  return ii[np.array([x!=str(m) and x in allowed for x in self.ids[ii]])]
 def overlay(self,md):
  out={};changes=md['neighbor_observation_noise']['changes'];end=md['decision_time_s'];start=end-1980
  for change in changes:
   if not change.get('effective'):continue
   allii=self.history(change['mmsi'],start,end);finite=self.good[allii];ii=allii[finite]
   t=self.t[ii];p=self.p[ii].copy();s=self.s[ii].copy();c=self.c[ii].copy();d=change['parameters'];kind=change['noise']
   if kind=='teleport':
    i=int(d['report_index']);assert t[i]==d['time_s'];p[i]+=d['offset_m']*np.array([np.sin(np.deg2rad(d['angle_deg'])),np.cos(np.deg2rad(d['angle_deg']))])
   elif kind=='speed_spike':
    i=int(d['report_index']);assert t[i]==d['time_s'];s[i:i+int(d['reports'])]=np.minimum(s[i:i+int(d['reports'])]*d['factor'],102.1)
   elif kind=='drift':
    u=(t-d['onset_s'])/d['duration_s'];env=np.where((u>=0)&(u<=1),.5-.5*np.cos(2*np.pi*np.clip(u,0,1)),0.)
    p+=d['offset_m']*env[:,None]*np.array([np.sin(np.deg2rad(d['angle_deg'])),np.cos(np.deg2rad(d['angle_deg']))])
   else:raise ValueError(kind)
   changed=np.any(p!=self.p[ii],1)|(s!=self.s[ii])|(c!=self.c[ii]);assert t[changed].tolist()==change['affected_time_s']
   # The augmentation API touches finite records, but the rebuilt background
   # index retains invalid records. Do not bridge across an invalid source row.
   ap=self.p[allii].copy();ass=self.s[allii].copy();ac=self.c[allii].copy();ap[finite]=p;ass[finite]=s;ac[finite]=c
   good,valid,e,w=kin(self.t[allii],ap,ass,ac)
   for j,k in enumerate(allii):out[int(k)]=(ap[j],ass[j],ac[j],valid[j],e[j],w[j])
  return out

def empty_stats():return dict(valid_steps=0,covered_steps=0,unknown_coverage_steps=0,plain_ge4_steps=0,trusted_ge3_steps=0,trusted_ge4_steps=0,kish_ge3_steps=0,affine_valid_steps=0,geometry_rejections=0,condition_rejections=0,total_trusted_count=0,total_kish=0.)
def audit_region(region,root,out,allowed):
 start=time.time();meta=json.loads((root/'release_20261026/source_registry.json').read_text())['regions'][region];src=Source(meta)
 report=dict(region=region,status='running',source=meta['canonical_file'],source_sha256=meta['sha256'],configs=CONFIGS,splits={},neighbor_reports_recovered=0,max_position_distance_error_m=0.,max_heading_error_deg=0.,max_gate_error=0.,method='Recorded causal latest-per-identity candidates joined to source observations; recorded neighbor corruptions replayed; no test arrays or labels',config_selection='Fixed geometric scan, no classifier scores')
 for split in ('train','val'):
  p=root/'labeling/labeled_v1/features_scene'/region/f'{split}.npz';ap=p.with_name(f'{split}_context_audit.jsonl.gz')
  with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in ['window_id','base_window_id','mmsi','mask','time_s','target_pos','target_sog_kn','target_cog_deg','context_valid']}
  totals={c['name']:empty_stats() for c in CONFIGS};rows=[];feasible=[];heading=collections.defaultdict(lambda:dict(transitions=0,prior_valid=0,lost_valid_context=0))
  with gzip.open(ap,'rt',encoding='utf-8') as f:
   for j,line in enumerate(f):
    md=json.loads(line);assert md['window_id']==a['window_id'][j];assert md['split']==split;overlay=src.overlay(md)
    n=int(a['mask'][j].sum());box=md['known_coverage_bbox_m'];window_counts=np.zeros(len(CONFIGS),int)
    for st in md['steps']:
     k=st['step'];assert k<n;when=a['time_s'][j,k];pos=a['target_pos'][j,k];sog=float(a['target_sog_kn'][j,k]);cog=float(a['target_cog_deg'][j,k])
     entries=st.get('neighbors',[]);positions=[];speeds=[];weights=[];angles=[];distances=[];plain=[]
     for ent in entries:
      m=ent['mmsi'];t=ent['time_s'];assert t<=when and when-t<=300 and m!=str(a['mmsi'][j]) and m in allowed[split]
      pp,ss,cc,kv,ee,ww=src.observation(m,t,overlay);dist=float(np.linalg.norm(pp-pos));angle=abs(float(wrap(cc-cog)))
      err=abs(dist-ent['distance_m']);report['max_position_distance_error_m']=max(report['max_position_distance_error_m'],err);assert err<1e-6
      if ent['heading_difference_deg'] is not None:
       err=abs(angle-ent['heading_difference_deg']);report['max_heading_error_deg']=max(report['max_heading_error_deg'],err);assert err<1e-8
      err=abs(ww-ent['reliability']);report['max_gate_error']=max(report['max_gate_error'],err);assert err<1e-8,(region,split,j,k,m,ww,ent['reliability'])
      assert bool(kv)==ent['kinematic_valid'],(region,split,j,k,m,t,kv,ent['kinematic_valid']);assert abs(ee-ent['e_kin_mps'])<1e-6
      ok=np.isfinite(pp).all() and np.isfinite(ss) and np.isfinite(cc) and 0.5<=ss<102.2 and 0<=cc<360 and angle<=45
      assert bool(ok)==ent['plain_accepted'];assert bool(ok and kv and ww>=.3)==ent['accepted']
      positions.append(pp-pos);speeds.append(ss);weights.append(ww);angles.append(angle);distances.append(dist);plain.append(bool(ok))
     report['neighbor_reports_recovered']+=len(entries)
     positions=np.asarray(positions).reshape(-1,2);speeds=np.asarray(speeds);weights=np.asarray(weights);angles=np.asarray(angles);distances=np.asarray(distances);plain=np.asarray(plain,bool)
     for ic,c in enumerate(CONFIGS):
      q=totals[c['name']];q['valid_steps']+=1;q['unknown_coverage_steps']+=int(box is None)
      covered=coverage_ok(pos,box,c['radius_nm']*NM);q['covered_steps']+=covered
      # The recorded target-heading policy is retained; source low-speed targets
      # have no candidate neighborhood, even under a different neighbor floor.
      candidate=plain&(distances<=c['radius_nm']*NM)&(angles<=c['heading_deg'])&(speeds>=c['min_speed_kn'])
      trusted=candidate&(weights>=.3);num=int(trusted.sum());dw=weights[trusted]*np.exp(-.5*(distances[trusted]/(c['sigma_nm']*NM))**2)
      neff=float(dw.sum()**2/np.square(dw).sum()) if len(dw) else 0.
      q['plain_ge4_steps']+=int(covered and candidate.sum()>=4);q['trusted_ge3_steps']+=int(covered and num>=3);q['trusted_ge4_steps']+=int(covered and num>=4);q['kish_ge3_steps']+=int(covered and neff>=3)
      q['total_trusted_count']+=num;q['total_kish']+=neff
      ok,why,_,_=geometry(positions[trusted],dw,c['radius_nm']*NM)
      q['geometry_rejections']+=int(covered and why=='degenerate_geometry');q['condition_rejections']+=int(covered and why=='ill_conditioned')
      valid=bool(covered and ok);q['affine_valid_steps']+=valid;window_counts[ic]+=valid
      if ic==0:assert valid==bool(a['context_valid'][j,k]),(region,split,j,k,why)
     if k:
      change=abs(float(wrap(a['target_cog_deg'][j,k]-a['target_cog_deg'][j,k-1])))
      label='lt15' if change<15 else '15to30' if change<30 else '30to45' if change<45 else 'ge45'
      h=heading[label];h['transitions']+=1;prior=bool(a['context_valid'][j,k-1]);h['prior_valid']+=prior;h['lost_valid_context']+=int(prior and not a['context_valid'][j,k])
    rows.append(dict(window_id=str(a['window_id'][j]),mmsi=str(a['mmsi'][j]),steps=n,valid_steps_by_config=window_counts.tolist()))
    # Quartet feasibility uses the unmodified source at one predetermined
    # midpoint anchor. The old binary behavior label is never consulted.
    t=a['time_s'][j,:n];tp,ts,tc=src.source_target(a['mmsi'][j],t);anchor=max(2,n//2-1);at=float(t[anchor]);ids=src.latest(at,a['mmsi'][j],allowed[split]);dist=np.linalg.norm(src.p[ids]-tp[anchor],axis=1);angle=np.abs(wrap(src.c[ids]-tc[anchor]))
    ok=src.good[ids]&src.valid[ids]&(src.w[ids]>=.3)&(src.s[ids]>=3)&(dist<=4*NM)&(angle<=30)&(np.abs(src.s[ids]-ts[anchor])<=3)
    cohort=ids[ok];cohort=cohort[np.argsort(src.ids[cohort],kind='stable')]
    prefix_ok=bool(ts[anchor]>=3 and np.median(ts[:anchor+1])>=3 and anchor>=2 and t[-1]-at>=180)
    accepted=prefix_ok and len(cohort)>=4
    feasible.append(dict(window_id=str(a['window_id'][j]),base_window_id=str(a['base_window_id'][j]),mmsi=str(a['mmsi'][j]),region=region,split=split,anchor_step=anchor,anchor_time_s=at,source_target_speed_kn=float(ts[anchor]),source_target_course_deg=float(tc[anchor]),source_target_position_m=tp[anchor].tolist(),source_timestamps=t.tolist(),cohort_mmsi=src.ids[cohort].tolist(),cohort_report_time_s=src.t[cohort].tolist(),cohort_count=len(cohort),prefix_ok=prefix_ok,accepted=bool(accepted)))
    if (j+1)%200==0:print('SCAN',region,split,j+1,flush=True)
  assert j+1==len(a['window_id'])
  for q in totals.values():
   den=q['valid_steps'];q['affine_valid_fraction']=q['affine_valid_steps']/den;q['covered_fraction']=q['covered_steps']/den;q['trusted_ge4_fraction']=q['trusted_ge4_steps']/den;q['mean_trusted_count']=q['total_trusted_count']/den;q['mean_kish_effective_count']=q['total_kish']/den
  report['splits'][split]=dict(windows=len(rows),valid_steps=int(a['mask'].sum()),config_results=totals,heading_change_association=dict(heading),relation_source_feasible=sum(x['accepted'] for x in feasible),source_npz_sha256=sha(p),source_audit_sha256=sha(ap))
  save(out/f'{region}_{split}_window_scan.json',rows);save(out/f'{region}_{split}_relation_feasibility.json',feasible)
 report.update(status='passed',elapsed_seconds=time.time()-start)
 save(out/f'{region}_neighborhood_audit.json',report)
 print('REGION_COMPLETE',region,report['neighbor_reports_recovered'],flush=True)
 return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--regions',nargs='+',default=REGIONS);args=ap.parse_args()
 args.out.mkdir(parents=True,exist_ok=True)
 groups=pd.read_parquet(args.root/'release_20261026/global_all_mmsi_split.parquet',columns=['mmsi','split'])
 # These are identity/split metadata only, not test trajectories or labels.
 targets=pd.read_parquet(args.root/'release_20261026/selected_windows.parquet',columns=['mmsi','split'])
 val=set(targets.loc[targets.split=='val','mmsi'].astype(str));test=set(targets.loc[targets.split=='test','mmsi'].astype(str));allids=set(groups.mmsi.astype(str));allowed={'train':allids-val-test,'val':allids-test}
 for region in args.regions:audit_region(region,args.root,args.out,allowed)

if __name__=='__main__':main()
