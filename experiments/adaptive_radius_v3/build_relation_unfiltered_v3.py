"""Rebuild relationship context from all causal geometric AIS candidates."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,time
import numpy as np
import pandas as pd
from radius_core import choose,ARMS,NM
from build_relation_v2 import ROOT,OLD,FROZEN,COHORT,STAGE,VIEWS,REGIONS,core,prior,sha,velocity

OUT=Path('/home/ps/AIS_adaptive_radius_relation_v3_20261109')
BAL=ROOT/'balanced_global_v2'
def load_module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
helper=load_module(BAL/'context_adapter_v1_20261026/neighborhood/audit_neighborhood.py','v3_ambient_source')
def allowed_for(split):
 pool=json.loads((COHORT/'candidate_pool_manifest.json').read_text())['target_ids'];targets={k:set(map(str,v)) for k,v in pool.items()}
 selected=pd.read_parquet(BAL/'release_20261026/selected_windows.parquet',columns=['mmsi','split']);orig={s:set(selected.loc[selected['split']==s,'mmsi'].astype(str)) for s in ('train','val','test')}
 allids=set(pd.read_parquet(BAL/'release_20261026/global_all_mmsi_split.parquet',columns=['mmsi'])['mmsi'].astype(str))
 return allids-set().union(*(targets[s]|orig[s] for s in ('train','val','test') if s!=split))
def coverage(meta,p,t,r):
 date=datetime.fromtimestamp(float(t),timezone.utc).strftime('%Y-%m-%d');box=meta.get('coverage_bbox_by_utc_day',{}).get(date)
 if not box:return True
 x=meta['core_bbox'];lon0=(x[0]+x[2])/2;lat0=(x[1]+x[3])/2;c=np.cos(np.deg2rad(lat0));lon=lon0+p[0]/(111195*c);lat=lat0+p[1]/111195
 return bool(box[0]<=lon-r*NM/(111195*c) and lon+r*NM/(111195*c)<=box[2] and box[1]<=lat-r*NM/111195 and lat+r*NM/111195<=box[3])
def collect_ambient(src,allowed,target,cohort,t):
 n=len(t);queries=[]
 for j,now in enumerate(t):
  phase=t[0] if j<n//2 else t[n//2];current=src.latest(float(now),target,allowed);records=[]
  for i in current:
   i=int(i);mid=str(src.ids[i])
   if mid in cohort or not src.good[i]:continue
   hs=src.history(mid,phase,float(now));hs=hs[src.good[hs]]
   if not len(hs):hs=np.asarray([i],int)
   vv=velocity(src.s[hs],src.c[hs]);ww=src.w[hs];v0=vv.mean(0);v1=np.average(vv,axis=0,weights=ww) if ww.sum()>1e-12 else v0
   records.append((mid,src.p[i].copy(),float(src.s[i]),float(src.c[i]),v0,v1,float(src.w[i]),bool(src.valid[i])))
  queries.append(records)
 return queries
def calc(o,base,case,target,ambient,meta):
 t=o['target_t'][case];p=o['target_pos'][case];tc=o['target_cog_deg'][case];tv=velocity(o['target_sog_kn'][case],tc);n=len(t);series=[]
 for mid,lo,hi in zip(o['cohort_mmsi'],o['member_offsets'][:-1],o['member_offsets'][1:]):
  lo,hi=int(lo),int(hi);nt=o['neighbor_t'][case,lo:hi];ns=o['neighbor_sog_kn'][case,lo:hi];nc=o['neighbor_cog_deg'][case,lo:hi];pos=o['neighbor_pos'][case,lo:hi]
  w,known=core.absolute_weights(nt,pos,ns,nc);idx=np.searchsorted(nt,t,side='right')-1
  series.append((str(mid),nt,ns,nc,pos,velocity(ns,nc),w,known,idx,base['neighbor_sog_kn'][case,lo:hi],base['neighbor_pos'][case,lo:hi]))
 out={a:np.zeros((n,8),np.float32) for a in ARMS};counts={a:np.zeros(n,np.int16) for a in ARMS};radii={a:np.zeros(n,np.float32) for a in ARMS}
 plain_count=np.zeros(n,np.int16);ambient_count=np.zeros(n,np.int16);loww_count=np.zeros(n,np.int16);hit=np.zeros(n,bool);coverage_known=np.zeros(n,bool)
 for j in range(n):
  P=[];V0=[];V1=[];W=[];K=[];I=[];phase=t[0] if j<n//2 else t[n//2]
  if not np.isfinite(tc[j]) or not 0<=tc[j]<360:continue
  def add(mid,pos,s,c,v0,v1,w,valid,ambient_member):
   if mid==str(target) or not np.isfinite(pos).all() or not np.isfinite(c) or not 0<=c<360 or not np.isfinite(s) or s<0:return
   if abs((float(c)-float(tc[j])+180)%360-180)>45:return
   rel=np.asarray(pos)-p[j]
   if np.linalg.norm(rel)>8*NM:return
   P.append(rel);V0.append(v0);V1.append(v1);W.append(w);K.append(valid);I.append(mid)
   if ambient_member:ambient_count[j]+=1
   if not valid or w<.3:loww_count[j]+=1
  for mid,pos,s,c,v0,v1,w,valid in ambient[j]:add(mid,pos,s,c,v0,v1,w,valid,True)
  for mid,nt,ns,nc,pos,vel,w,known,ix,bs,bp in series:
   q=int(ix[j])
   if q<0 or not 0<=t[j]-nt[q]<=300:continue
   selected=np.flatnonzero((nt>=phase)&(nt<=t[j]));selected=selected if len(selected) else np.asarray([q]);vv=vel[selected];ww=w[selected]
   if not np.isfinite(vv).all():raise ValueError('invalid cohort history')
   v0=vv.mean(0);v1=np.average(vv,axis=0,weights=ww) if ww.sum()>1e-12 else v0
   before=len(I);add(mid,pos[q],ns[q],nc[q],v0,v1,float(w[q]),bool(known[q]),False)
   base_eligible=(np.isfinite(bp[q]).all() and np.isfinite(nc[q]) and 0<=nc[q]<360 and abs((float(nc[q])-float(tc[j])+180)%360-180)<=45 and np.linalg.norm(bp[q]-p[j])<=8*NM)
   if (len(I)>before or base_eligible) and (np.any(ns[selected]!=bs[selected]) or np.any(pos[selected]!=bp[selected])):hit[j]=True
  date=datetime.fromtimestamp(float(t[j]),timezone.utc).strftime('%Y-%m-%d');coverage_known[j]=bool(meta.get('coverage_bbox_by_utc_day',{}).get(date))
  result,diag=choose(P,V0,V1,W,K,I,target,lambda r:coverage(meta,p[j],t[j],r));plain_count[j]=diag['plain_count_8nm']
  for arm,z in result.items():
   counts[arm][j]=z['count'];radii[arm][j]=z['radius_nm']
   if z['mode']:
    b=z['bg'];J=z['grad'];out[arm][j]=[*b,*(tv[j]-b),np.trace(J),J[1,0]-J[0,1],z['rho'],z['mode']]
 return out,counts,radii,plain_count,ambient_count,loww_count,hit,coverage_known
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=('train','val','test'),required=True);p.add_argument('--region',choices=REGIONS,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();dest=a.out/('smoke_relation' if a.limit else 'relation')/f'{a.split}_{a.region}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():raise FileExistsError(dest)
 meta=json.loads((BAL/'release_20261026/source_registry.json').read_text())['regions'][a.region];src=helper.Source(meta);allowed=allowed_for(a.split)
 manifest=COHORT/'dataset_manifest_combined.json';entries=sorted((r for r in json.loads(manifest.read_text())['samples'] if r['version']=='relaxed' and r['split']==a.split and r['region']==a.region),key=lambda r:(r['region'],r['window_id']))
 refpath=FROZEN/'test_relation_field.npz' if a.split=='test' else OLD/f'{a.split}_relation_field.npz'
 with np.load(refpath,allow_pickle=False) as z:ref={k:z[k] for k in ('source','case','y','length','mmsi','region','raw7','causal7','self_w','time_s')}
 idx=np.flatnonzero(ref['region'].astype(str)==a.region);idx=idx[:a.limit*4] if a.limit else idx;ref={k:v[idx] for k,v in ref.items()};entries=entries[:a.limit] if a.limit else entries
 if len(idx)!=4*len(entries):raise ValueError('quartet mismatch')
 out={k:v.copy() for k,v in ref.items()};out['sparse_source']=np.zeros(len(idx),bool);out['base_count8']=np.zeros((len(idx),20),np.int16)
 for view in VIEWS:
  for k in ('hit','coverage_known'):out[view+'__'+k]=np.zeros((len(idx),20),bool)
  for k in ('ambient_count','loww_count'):out[view+'__'+k]=np.zeros((len(idx),20),np.int16)
  for arm in ARMS:
   out[view+'__'+arm+'__field']=np.zeros((len(idx),20,8),np.float32);out[view+'__'+arm+'__count']=np.zeros((len(idx),20),np.int16);out[view+'__'+arm+'__radius']=np.zeros((len(idx),20),np.float32)
 keys=('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','y');start=time.time()
 for i,row in enumerate(entries):
  obs={}
  for view in ('base','natural','suffix'):
   file=COHORT/row['files'][view]['path']
   if sha(file)!=row['files'][view]['sha256']:raise ValueError('source hash mismatch')
   with np.load(file,allow_pickle=False) as z:obs[view]={k:z[k] for k in keys}
  comp={k:v.copy() for k,v in obs['base'].items()};comp['neighbor_sog_kn']=obs['suffix']['neighbor_sog_kn'].copy()
  for m in row['stress']['members']:
   lo=int(comp['member_offsets'][m['member_index']]);hi=int(comp['member_offsets'][m['member_index']+1]);k=lo+int(m['indices']['suffix']['report_index'])
   if not lo<=k<k+1<hi:raise ValueError('bad stress indices')
   comp['neighbor_pos'][:,k:k+2]+=600*prior.direction(row['window_id'],m['mmsi'])
  obs['combined_l100_m600']=comp
  target=str(row['mmsi']);base=obs['base'];cohort=set(map(str,base['cohort_mmsi']))
  if target in cohort or not cohort.issubset(allowed):raise ValueError('target or forbidden cross-split identity in intervention cohort')
  ambient=collect_ambient(src,allowed,target,cohort,base['target_t'][0])
  for case in range(4):
   at=4*i+case;n=int(ref['length'][at])
   if str(ref['source'][at])!=str(row['window_id']) or int(ref['case'][at])!=case or int(ref['y'][at])!=int(base['y'][case]):raise ValueError('quartet mismatch')
   if not np.array_equal(ref['time_s'][at,:n],base['target_t'][case]):raise ValueError('time mismatch')
   for view in VIEWS:
    fields,counts,radii,all_count,amb,low,hit,known=calc(obs[view],base,case,target,ambient,meta)
    out[view+'__hit'][at,:n]=hit;out[view+'__coverage_known'][at,:n]=known;out[view+'__ambient_count'][at,:n]=amb;out[view+'__loww_count'][at,:n]=low
    if view=='base':out['base_count8'][at,:n]=all_count
    for arm in ARMS:
     out[view+'__'+arm+'__field'][at,:n]=fields[arm];out[view+'__'+arm+'__count'][at,:n]=counts[arm];out[view+'__'+arm+'__radius'][at,:n]=radii[arm]
  n=int(ref['length'][4*i]);out['sparse_source'][4*i:4*i+4]=np.median(out['base_count8'][4*i,:n])<4
  if (i+1)%20==0:print('AMBIENT',a.split,a.region,i+1,'/',len(entries),'elapsed',round(time.time()-start,1),flush=True)
 np.savez_compressed(dest,**out);audit={'split':a.split,'region':a.region,'source_count':len(entries),'windows':len(idx),'sha256':sha(dest),'protocol_sha256':sha(Path(__file__).with_name('protocol.json')),'builder_sha256':sha(Path(__file__)),'operator_sha256':sha(Path(__file__).with_name('radius_core.py')),'reference_sha256':sha(refpath),'manifest_sha256':sha(manifest),'canonical_sha256':meta['sha256'],'candidate_policy':'all causal geometric source AIS members plus manipulated cohort, no credibility qualification in ordinary field','dataset_origin':'quartet source acceptance historically required a trusted cohort; ambient search itself is unfiltered','elapsed_seconds':time.time()-start}
 dest.with_suffix('.json').write_text(json.dumps(audit,indent=2));print('UNFILTERED_RELATION_COMPLETE',json.dumps(audit),flush=True)
if __name__=='__main__':main()
