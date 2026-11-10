"""Metadata-only global vessel split and exact balanced source-window selection.

Default is a dry run: only a JSON report is written. --build also writes selected
source-window manifests and the split assignment for every observed background ID.
No anomaly labels, neural predictions, or model metrics are used by this program.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import argparse, hashlib, json, math, sys, time
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

SPLITS=('train','val','test')
PROB=np.array([.7,.15,.15])

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def quota(n):
 ideal=n*PROB; out=np.floor(ideal).astype(int)
 for i in np.argsort(-(ideal-out),kind='stable')[:n-int(out.sum())]:out[i]+=1
 return out

def rank_strata(v):
 # Ties use MMSI ascending. This is deterministic metadata, never model output.
 return pd.qcut(v.rank(method='first'),q=min(4,len(v)),labels=False).astype(int)

def sources(args):
 if args.source_config:
  obj=json.loads(Path(args.source_config).read_text())
  return obj.get('regions',obj)
 result={}
 for p in sorted(args.root.glob('capacity_uniform_*.json'),key=lambda p:p.stat().st_mtime):
  result.update(json.loads(p.read_text()))
 return result

def build_problem(counts, seed, shipq, winq, limit):
 ids=np.array(sorted(set().union(*(set(v.index) for v in counts.values()))),dtype=np.int64)
 ids_ix={int(m):i for i,m in enumerate(ids)}; n=len(ids)
 nvar=3*n; yvars={}; dvars={}; strata={r:rank_strata(v) for r,v in counts.items()}
 for r,v in counts.items():
  for m in v.index:
   for s in range(3):yvars[r,int(m),s]=nvar;nvar+=1
  for q in sorted(strata[r].unique()):
   for s in range(3):dvars[r,int(q),s]=nvar;nvar+=1
 row=[];col=[];data=[];lb=[];ub=[]
 def add(index,values,lo,hi):
  index=list(index);values=list(values); k=len(lb)
  row.extend([k]*len(index));col.extend(index);data.extend(values);lb.append(lo);ub.append(hi)
 for i in range(n):add([3*i+s for s in range(3)],[1]*3,1,1)
 globalq=quota(n)
 for s in range(3):add([3*i+s for i in range(n)],[1]*n,globalq[s],globalq[s])
 for r,v in counts.items():
  for m in v.index:
   for s in range(3):add([yvars[r,int(m),s],3*ids_ix[int(m)]+s],[1,-1],-np.inf,0)
  for s in range(3):
   y=[yvars[r,int(m),s] for m in v.index]
   add(y,np.ones(len(y)),shipq[s],shipq[s])
   add(y,v.to_numpy(),winq[s],np.inf)
  for q in sorted(strata[r].unique()):
   members=v.index[strata[r]==q]; k=len(members)
   for s in range(3):
    # Keep candidate-vessel capacity strata proportionate in each global split.
    add([3*ids_ix[int(m)]+s for m in members],np.ones(k),math.floor(k*PROB[s]),math.ceil(k*PROB[s]))
    target=shipq[s]*k/len(v); ix=[yvars[r,int(m),s] for m in members]
    # Soft selected-stratum balance: relax only as much as the window quota needs.
    add(ix+[dvars[r,int(q),s]],list(np.ones(k))+[-1],-np.inf,target)
    add(ix+[dvars[r,int(q),s]],list(np.ones(k))+[1],target,np.inf)
  # Aggregate capped-capacity split balance, within one vessel contribution.
  for s in range(3):
   add([3*ids_ix[int(m)]+s for m in v.index],v.to_numpy(),max(0,v.sum()*PROB[s]-20),v.sum()*PROB[s]+20)
 A=coo_matrix((data,(row,col)),shape=(len(lb),nvar)).tocsc()
 objective=np.random.default_rng(seed).uniform(0,1,nvar)*1e-5
 integ=np.ones(nvar);upper=np.ones(nvar)
 for j in dvars.values():objective[j]=1;integ[j]=0;upper[j]=np.inf
 started=time.monotonic()
 result=milp(c=objective,integrality=integ,bounds=Bounds(0,upper),constraints=LinearConstraint(A,lb,ub),
             options={'time_limit':limit,'mip_rel_gap':.01,'presolve':True})
 solver={'status':int(result.status),'message':result.message,'seconds':time.monotonic()-started,
         'mip_gap':float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None,
         'objective':float(result.fun) if result.fun is not None else None,'scipy_version':scipy.__version__,
         'seed':seed,'optimization_scope':'metadata-only capacity strata; no labels or evaluation scores'}
 if result.x is None:return None,None,solver
 x=result.x.copy(); binary=integ==1
 assert np.max(np.abs(x[binary]-np.rint(x[binary])))<1e-5
 x[binary]=np.rint(x[binary]);values=A@x
 assert np.all(values>=np.array(lb)-1e-5) and np.all(values<=np.array(ub)+1e-5)
 assignment={int(m):SPLITS[int(np.argmax(x[i*3:i*3+3]))] for i,m in enumerate(ids)}
 selected={r:{s:sorted(int(m) for m in v.index if x[yvars[r,int(m),j]]>.5) for j,s in enumerate(SPLITS)} for r,v in counts.items()}
 solver['constraints_checked']=int(len(lb));solver['global_candidate_ship_counts']=dict(zip(SPLITS,map(int,globalq)))
 solver['selected_stratum_absolute_deviation']=float(sum(x[j] for j in dvars.values()))
 return assignment,selected,solver

def pick_windows(c,selected,winq,seed,region):
 chosen=[]; allocation=[]
 for split,needed in zip(SPLITS,winq):
  ships=selected[split]; groups={int(m):g.sort_values(['end_utc_s','window_id'],kind='stable') for m,g in c[c.mmsi.isin(ships)].groupby('mmsi')}
  cap={m:min(20,len(groups[m])) for m in ships};take={m:1 for m in ships}
  rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}:{region}:{split}'.encode()).digest()[:8],'little'))
  order=rng.permutation(ships).tolist();remaining=int(needed)-len(ships)
  assert remaining>=0 and sum(cap.values())>=needed
  # Max-min window allocation prevents a few prolific vessels dominating a split.
  while remaining:
   progress=False
   for m in order:
    if remaining and take[m]<cap[m]:take[m]+=1;remaining-=1;progress=True
   assert progress
  for m in ships:
   g=groups[m];k=take[m]
   bins=np.array_split(np.arange(len(g)),k)
   indices=[int(rng.choice(b)) for b in bins]
   sub=g.iloc[indices].copy();sub['split']=split;chosen.append(sub)
   allocation.append({'region':region,'split':split,'mmsi':m,'available_windows':len(g),'capped_capacity':cap[m],'selected_windows':k})
 return pd.concat(chosen,ignore_index=True),allocation

def audit(selected,canonical,source_info,shipq,winq):
 observations={s:set() for s in SPLITS};ships={s:set() for s in SPLITS};wids={s:set() for s in SPLITS}
 report={'regions':{},'scope':'Target-vessel, source-window and raw-target-observation separation. Full background identity groups are separately exported.'}
 for r,c in selected.items():
  raw=canonical[r]; info={}
  # Cache Arrow-backed columns once. Repeated DataFrame.iloc/to_numpy for tiny
  # windows can materialize the whole column thousands of times in pandas 3.
  raw_mmsi=raw.mmsi.to_numpy(dtype=np.int64)
  raw_time=raw.timestamp.to_numpy(dtype=np.int64)
  for si,s in enumerate(SPLITS):
   g=c[c.split==s]
   assert g.mmsi.nunique()==shipq[si] and len(g)==winq[si]
   assert g.groupby('mmsi').size().between(1,20).all()
   info[s]={'ships':int(g.mmsi.nunique()),'windows':len(g),'max_windows_per_ship':int(g.groupby('mmsi').size().max())}
   ships[s].update(map(int,g.mmsi));wids[s].update(g.window_id)
  for row in c.itertuples():
   ix=np.array(json.loads(row.row_indices),dtype=np.int64); ts=np.array(json.loads(row.timestamps),dtype=np.int64)
   assert 8<=len(ix)<=20 and len(ix)==len(ts)==row.n_obs
   assert np.all(np.diff(ts)>=60) and ts[-1]-ts[0]>=660 and ts[-1]-ts[0]<1380
   assert len(set(ix))==len(ix) and np.all(ix>=0) and np.all(ix<len(raw))
   assert np.all(raw_mmsi[ix]==row.mmsi) and np.array_equal(raw_time[ix],ts)
   expected=hashlib.sha256((str(row.mmsi)+':'+','.join(map(str,ts))).encode()).hexdigest()[:24]
   assert row.window_id==expected
   observations[row.split].update((int(row.mmsi),int(t)) for t in ts)
  for m,g in c.sort_values('start_utc_s').groupby('mmsi'):
   assert np.all(g.start_utc_s.to_numpy()[1:]>g.end_utc_s.to_numpy()[:-1]),f'Overlapping source intervals: {r}/{m}'
  assert not c.window_id.duplicated().any()
  report['regions'][r]=info
 intersections={}
 for a,b in [('train','val'),('train','test'),('val','test')]:
  intersections[a+'__'+b]={'target_mmsi':len(ships[a]&ships[b]),'source_windows':len(wids[a]&wids[b]),'target_observations':len(observations[a]&observations[b])}
  assert all(v==0 for v in intersections[a+'__'+b].values())
 allc=pd.concat(selected.values(),ignore_index=True)
 assert not allc.window_id.duplicated().any(),'Same source window selected in more than one region'
 report['cross_split_intersections']=intersections
 report['global_unique_selected_ships']={s:len(ships[s]) for s in SPLITS}
 report['global_unique_selected_observations']={s:len(observations[s]) for s in SPLITS}
 report['all_checks_passed']=True
 return report

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,default=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2'))
 ap.add_argument('--regions',nargs='+',required=True);ap.add_argument('--ships',type=int,default=300)
 ap.add_argument('--windows',type=int,default=2700);ap.add_argument('--seed',type=int,default=20261025)
 ap.add_argument('--out',type=Path,required=True);ap.add_argument('--source-config',type=Path)
 ap.add_argument('--build',action='store_true');ap.add_argument('--time-limit',type=float,default=180)
 args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
 if args.build and any((args.out/n).exists() for n in ['selected_windows.parquet','global_all_mmsi_split.parquet']):
  raise SystemExit('Refusing to overwrite a previously built selection; use a fresh output directory.')
 shipq,winq=quota(args.ships),quota(args.windows);config=sources(args)
 counts={};candidates={};canonical={};source_info={};all_observed=set();source_ids={}
 for region in args.regions:
  p=args.root/'candidates'/f'{region}.parquet';c=pd.read_parquet(p);c=c[c.duration_s>=660].copy()
  assert len(c) and set(c.region)=={region} and not c.window_id.duplicated().any()
  assert 'canonical_file' in config[region],f'Missing source path for {region}'
  src=Path(config[region]['canonical_file']);d=pd.read_parquet(src)
  source_info[region]={'canonical_file':str(src),'canonical_sha256':sha(src),'candidate_file':str(p),'candidate_sha256':sha(p),'raw_rows':len(d),'raw_unique_ids':int(d.mmsi.nunique())}
  # Source hash in the capacity inventory prevents stale row-index references.
  if config[region].get('sha256'):assert config[region]['sha256']==source_info[region]['canonical_sha256'],f'Stale source inventory for {region}'
  candidates[region]=c;canonical[region]=d;counts[region]=c.groupby('mmsi').size().clip(upper=20).sort_index()
  mids=set(map(int,d.mmsi.dropna().unique()));all_observed.update(mids);source_ids[region]=mids
  print('SOURCE_VERIFIED',region,'candidate_ships',len(counts[region]),'candidate_windows',len(c),flush=True)
 report={'mode':'build' if args.build else 'dryrun','created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'regions':args.regions,'requested_ships_per_region':args.ships,'requested_windows_per_region':args.windows,
         'ship_quota':dict(zip(SPLITS,map(int,shipq))),'window_quota':dict(zip(SPLITS,map(int,winq))),'max_windows_per_ship':20,'minimum_window_duration_s':660,'seed':args.seed,
         'source_files':source_info,'candidate_capacity':{r:{'ships':len(v),'capped_windows':int(v.sum()),'top_requested_ships_capped_windows':int(v.nlargest(args.ships).sum())} for r,v in counts.items()},
         'sampling_policy':'Joint global MMSI grouping and selected-vessel optimization; capacity quartile balance with explicit deviation, fixed-seed tie objective; max-min window allocation and seeded sampling from ordered time bins.'}
 infeasible=[r for r,v in counts.items() if len(v)<args.ships or v.nlargest(args.ships).sum()<args.windows]
 if infeasible:
  report.update(status='insufficient_raw_capacity',insufficient_regions=infeasible)
  (args.out/'selection_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return 2
 print('SOLVING_METADATA_GROUPS',flush=True)
 assignment,selected,solver=build_problem(counts,args.seed,shipq,winq,args.time_limit);report['solver']=solver
 if assignment is None:
  report['status']='no_feasible_assignment_obtained';(args.out/'selection_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return 3
 picked={};allocations=[]
 for r,c in candidates.items():
  picked[r],a=pick_windows(c,selected[r],winq,args.seed,r);allocations.extend(a)
 report['selected_capacity_strata']={}
 for r,v in counts.items():
  q=rank_strata(v);report['selected_capacity_strata'][r]={'candidate_counts':{str(int(k)):int(n) for k,n in q.value_counts().sort_index().items()},'selected_counts':{s:{str(int(k)):int(n) for k,n in q.loc[selected[r][s]].value_counts().sort_index().items()} for s in SPLITS}}
 print('VALIDATING_SELECTED_SOURCE_REPORTS',flush=True)
 validation=audit(picked,canonical,source_info,shipq,winq)
 # All remaining observed background IDs get a global group too. The total pool
 # has exact rounded 70/15/15 counts, including candidate target IDs.
 extras=np.array(sorted(all_observed-set(assignment)),dtype=np.int64)
 rng=np.random.default_rng(args.seed+1);extras=rng.permutation(extras)
 target_counts=np.array([sum(s==label for s in assignment.values()) for label in SPLITS])
 extra_quota=quota(len(all_observed))-target_counts;assert np.all(extra_quota>=0) and extra_quota.sum()==len(extras)
 start=0
 for label,k in zip(SPLITS,extra_quota):
  for m in extras[start:start+k]:assignment[int(m)]=label
  start+=int(k)
 group_rows=[{'mmsi':m,'split':assignment[m],'candidate_target':any(m in v.index for v in counts.values()),'observed_regions':json.dumps([r for r in args.regions if m in source_ids[r]])} for m in sorted(assignment)]
 group_df=pd.DataFrame(group_rows);assert len(group_df)==len(all_observed) and not group_df.mmsi.duplicated().any()
 for label in SPLITS:assert int((group_df.split==label).sum())==quota(len(all_observed))[SPLITS.index(label)]
 report.update(status='verified_build' if args.build else 'verified_dryrun',validation=validation,all_observed_global_groups=group_df.groupby('split').size().to_dict(),
               background_protocol='All observed IDs have exactly one global split. A strict inductive feature builder MUST restrict neighbor access to the same split. This sampler does not itself construct context features.')
 all_selected=pd.concat(picked.values(),ignore_index=True).sort_values(['region','split','mmsi','start_utc_s']).reset_index(drop=True)
 report['selected_manifest_content_sha256']=hashlib.sha256(all_selected.to_json(orient='records').encode()).hexdigest()
 report['global_group_content_sha256']=hashlib.sha256(group_df.to_json(orient='records').encode()).hexdigest()
 if args.build:
  all_selected.to_parquet(args.out/'selected_windows.parquet',index=False)
  group_df.to_parquet(args.out/'global_all_mmsi_split.parquet',index=False)
  pd.DataFrame(allocations).to_parquet(args.out/'vessel_window_allocation.parquet',index=False)
  for r,c in picked.items():c.to_parquet(args.out/f'{r}_windows.parquet',index=False)
  (args.out/'validation_report.json').write_text(json.dumps(validation,indent=2))
  report['output_hashes']={p.name:sha(p) for p in args.out.glob('*.parquet')}
 (args.out/'selection_report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k not in ['source_files','validation']},indent=2),flush=True)
 print('VERIFIED',len(all_selected),'selected windows; all observed identities',len(group_df),flush=True)
 return 0

if __name__=='__main__':sys.exit(main())
