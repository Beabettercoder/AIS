"""Freeze labels only after label-blind counterfactual physical acceptance.

Per domain: 1400 unique source windows, 260 target vessels, fixed 70/15/15
source groups. MILP uses metadata-only type/stratum distributions, never labels,
predictions, or test performance. y is allocated AFTER this selection.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

BASE=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
CF=BASE/'labeling/counterfactuals_v1';OUT=BASE/'labeling/labeled_v1'
SOURCE=BASE/'release_20261026';SEED=20261026
KINDS=['turn_back','deviation','stall','sinuous','speed_drop']
QUOTAS={'train':{'windows':980,'vessels':182,'latent_each':196,'positive':392},'val':{'windows':210,'vessels':39,'latent_each':42,'positive':84},'test':{'windows':210,'vessels':39,'latent_each':42,'positive':84}}

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def seed(*parts):return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8],'little')&((1<<63)-1)

def solve_cell(g,region,split,limit):
 g=g.sort_values('window_id',kind='stable').reset_index(drop=True);q=QUOTAS[split]
 vv=g.groupby('mmsi').size();N=len(g);ships=vv.index.to_numpy();m=len(ships);vix={int(x):N+i for i,x in enumerate(ships)}
 supply={k:int((g.latent_kind==k).sum()) for k in KINDS}
 info={'candidate_windows':N,'candidate_vessels':m,'candidate_type_windows':supply,'requested':q,'largest_requested_vessel_capacity_cap20':int(vv.clip(upper=20).nlargest(q['vessels']).sum())}
 if N<q['windows'] or m<q['vessels'] or min(supply.values())<q['latent_each'] or info['largest_requested_vessel_capacity_cap20']<q['windows']:
  return None,dict(info,status='insufficient_metadata_capacity')
 cells=list(g.groupby(['latent_kind','generation_stratum'],sort=True).indices.items())
 nvar=N+m+2*len(cells);rows=[];cols=[];data=[];lower=[];upper=[]
 def add(ix,values,lo,hi):
  r=len(lower);rows.extend([r]*len(ix));cols.extend(ix);data.extend(values);lower.append(lo);upper.append(hi)
 add(list(range(N)),[1]*N,q['windows'],q['windows'])
 add(list(range(N,N+m)),[1]*m,q['vessels'],q['vessels'])
 for vessel,ix in g.groupby('mmsi',sort=True).indices.items():
  ix=list(map(int,ix));y=vix[int(vessel)]
  add(ix+[y],[1]*len(ix)+[-20],-np.inf,0)
  add(ix+[y],[1]*len(ix)+[-1],0,np.inf)
 for kind in KINDS:
  ix=np.flatnonzero(g.latent_kind.to_numpy()==kind).tolist();add(ix,[1]*len(ix),q['latent_each'],q['latent_each'])
 objective=np.zeros(nvar);integer=np.zeros(nvar,np.int8);integer[:N+m]=1
 high=np.full(nvar,np.inf);high[:N+m]=1
 for j,((kind,stratum),ix) in enumerate(cells):
  ix=list(map(int,ix));a=N+m+2*j;b=a+1
  target=q['latent_each']*len(ix)/supply[kind]
  add(ix+[a,b],[1]*len(ix)+[-1,1],target,target)
  # Absolute metadata-stratum count loss; independent of labels and all scores.
  objective[a]=objective[b]=1.0
 objective[:N]=np.array([seed(SEED,'selection',w)/2**63 for w in g.window_id])*1e-6
 A=coo_matrix((data,(rows,cols)),shape=(len(lower),nvar)).tocsc()
 started=time.time();res=milp(c=objective,integrality=integer,bounds=Bounds(np.zeros(nvar),high),constraints=LinearConstraint(A,np.array(lower),np.array(upper)),options={'time_limit':limit,'mip_rel_gap':.01,'presolve':True})
 info['solver']={'status':int(res.status),'message':res.message,'seconds':time.time()-started,'mip_gap':None if getattr(res,'mip_gap',None) is None else float(res.mip_gap),'objective':None if res.fun is None else float(res.fun),'optimization_scope':'latent-type and nobs/SOG/ekin metadata strata only'}
 if res.x is None:return None,dict(info,status='no_feasible_solution_obtained')
 x=np.rint(res.x[:N+m]).astype(int);z=np.r_[x,res.x[N+m:]];values=A@z
 assert np.all(values>=np.array(lower)-1e-6) and np.all(values<=np.array(upper)+1e-6)
 selected=g.loc[x[:N]==1].copy();assert len(selected)==q['windows'] and selected.mmsi.nunique()==q['vessels']
 assert selected.groupby('mmsi').size().between(1,20).all()
 assert all((selected.latent_kind==k).sum()==q['latent_each'] for k in KINDS)
 info.update(status='feasible_verified',selected_windows=len(selected),selected_vessels=int(selected.mmsi.nunique()),max_windows_per_vessel=int(selected.groupby('mmsi').size().max()),candidate_strata={f'{k}/{s}':len(ix) for (k,s),ix in cells},selected_strata={f'{k}/{s}':int(n) for (k,s),n in selected.groupby(['latent_kind','generation_stratum']).size().items()})
 return selected,info

def stratified_pick(g,needed,namespace):
 """Largest-remainder class quotas within the frozen metadata strata."""
 cells=list(g.groupby('generation_stratum',sort=True).indices.items())
 targets=np.array([needed*len(ix)/len(g) for _,ix in cells]);counts=np.floor(targets).astype(int)
 left=needed-int(counts.sum());order=sorted(range(len(cells)),key=lambda j:(-(targets[j]-counts[j]),seed(SEED,namespace,cells[j][0])))
 counts[order[:left]]+=1;picked=[]
 for (stratum,ix),k in zip(cells,counts):
  ids=g.iloc[ix].window_id.tolist();ids.sort(key=lambda w:(seed(SEED,namespace,w),w));picked.extend(ids[:int(k)])
 assert len(picked)==needed
 return set(picked)

def assign_labels(selected):
 pieces=[]
 for region,regional in selected.groupby('region',sort=True):
  # train needs two ceil(+1) kinds, val/test four each. Excluding one train
  # extra from val and the other from test yields EXACT 112 positives/kind/domain.
  order=np.random.default_rng(seed(SEED,region,'positive_kind_extra')).permutation(5)
  extras={'train':{int(order[0]),int(order[1])},'val':set(range(5))-{int(order[0])},'test':set(range(5))-{int(order[1])}}
  for split,g in regional.groupby('split',sort=True):
   g=g.copy().reset_index(drop=True);q=QUOTAS[split];g['y']=np.int8(0)
   for k,kind in enumerate(KINDS):
    subset=g[g.latent_kind==kind].reset_index(drop=True)
    needed=q['positive']//5+int(k in extras[split])
    chosen=stratified_pick(subset,needed,f'{region}:{split}:{kind}:y')
    g.loc[g.window_id.isin(chosen),'y']=1
   assert g.y.sum()==q['positive']
   g['type']=np.where(g.y==1,g.latent_kind.map({k:i+1 for i,k in enumerate(KINDS)}),0).astype(np.int8)
   g['behavior_type']=np.where(g.y==1,g.latent_kind,'none');g['is_sham']=g.y.eq(0)
   g['observation_noise']=False;g['noise_kind']='none';g['noise_scope']='none'
   for y,ys in g.groupby('y',sort=True):
    needed=int(np.floor(len(ys)/3+.5));ids=ys.window_id.tolist();ids.sort(key=lambda w:(seed(SEED,'noise_selection',w),w))
    noisy=ids[:needed];g.loc[g.window_id.isin(noisy),'observation_noise']=True
    modes=['teleport','speed_spike','drift'];scopes=['target_only','neighbors_only','both']
    for factor,choices in [('noise_kind',modes),('noise_scope',scopes)]:
     ranked=sorted(noisy,key=lambda w:(seed(SEED,factor,w),w));offset=seed(SEED,region,split,int(y),factor,'offset')%3
     for j,w in enumerate(ranked):g.loc[g.window_id==w,factor]=choices[(j+offset)%3]
   g['noise_type']=g['noise_kind'];g['observation_noise_type']=g['noise_kind']
   g['noise_selection_seed']=[seed(SEED,w,'noise_selection') for w in g.window_id]
   g['noise_kind_seed']=[seed(SEED,w,'noise_kind') for w in g.window_id]
   g['noise_scope_seed']=[seed(SEED,w,'noise_scope') for w in g.window_id]
   g['group_normal_variation']=False;g['cell']=(2*g.y+g.observation_noise.astype(np.int8)).astype(np.int8)
   g['counterfactual_applied_params']=g.applied_params
   g['applied_params']=np.where(g.y==1,g.counterfactual_applied_params,'{}')
   g['base_window_id']=g.window_id;g['sample_id']=g.window_id
   g['labels_frozen']=True;g['label_version']='labeled_v1';g['label_seed']=SEED
   g['plan_status']='frozen_after_label_blind_physical_acceptance_and_metadata_selection'
   g['label_definition']='1: injected individual behavior; 0: sham, no injected behavior; original natural anomaly status unreviewed'
   g['group_challenge_scope']='separate paired view on same base IDs, not extra independent samples'
   pieces.append(g)
 result=pd.concat(pieces,ignore_index=True).sort_values(['region','split','mmsi','start_utc_s'],kind='stable').reset_index(drop=True)
 assert len(result)==11200 and result.y.sum()==4480 and not result.window_id.duplicated().any()
 assert result.groupby('mmsi').split.nunique().max()==1
 for region,g in result.groupby('region'):
  assert len(g)==1400 and g.mmsi.nunique()==260
  assert set(g[g.y==1].latent_kind.value_counts())=={112}
 for split,g in result.groupby('split'):
  for _,h in g.groupby(['region','latent_kind']):
   vals=h.y.value_counts();assert h.y.sum() in [QUOTAS[split]['positive']//5,QUOTAS[split]['positive']//5+1]
 return result

def validate_selected(result,parents):
 expected=parents.set_index('window_id');audit={'regions':{},'cross_split_intersections':{},'raw_target_observations':{}};ids={};obs={}
 for region,g in result.groupby('region',sort=True):
  rr={}
  for split,h in g.groupby('split',sort=True):
   q=QUOTAS[split];assert len(h)==q['windows'] and h.mmsi.nunique()==q['vessels'] and h.y.sum()==q['positive']
   assert h.groupby('mmsi').size().between(1,20).all() and h.counterfactual_accepted.all()
   assert all((h.latent_kind==k).sum()==q['latent_each'] for k in KINDS)
   noise={}
   for y,ys in h.groupby('y'):
    assert abs(int(ys.observation_noise.sum())-len(ys)/3)<=.5+1e-8
    noise[str(int(y))]={'windows':len(ys),'noisy':int(ys.observation_noise.sum()),'fraction':float(ys.observation_noise.mean()),'kind_counts':ys.noise_kind.value_counts().to_dict(),'scope_counts':ys.noise_scope.value_counts().to_dict()}
   rr[split]={'windows':len(h),'vessels':int(h.mmsi.nunique()),'positive':int(h.y.sum()),'negative':int((h.y==0).sum()),'positive_kinds':h.loc[h.y==1,'latent_kind'].value_counts().to_dict(),'latent_kinds':h.latent_kind.value_counts().to_dict(),'noise_by_y':noise}
  audit['regions'][region]=rr
 for row in result.itertuples():
  src=expected.loc[row.window_id];assert src.counterfactual_accepted
  for col in ['mmsi','split','latent_kind','onset_utc_s','behavior_seed','counterfactual_row','row_indices','timestamps']:
   assert getattr(row,col)==src[col],f'{row.window_id}: changed frozen field {col}'
  assert row.counterfactual_applied_params==src.applied_params
  assert (row.type==0 and row.applied_params=='{}') if row.y==0 else (row.type==KINDS.index(row.latent_kind)+1 and row.applied_params==src.applied_params)
 for split,g in result.groupby('split'):
  ids[split]=set(g.mmsi);obs[split]=set()
  for row in g.itertuples():obs[split].update((int(row.mmsi),int(t)) for t in json.loads(row.timestamps))
  audit['raw_target_observations'][split]=len(obs[split])
 for a,b in [('train','val'),('train','test'),('val','test')]:
  audit['cross_split_intersections'][a+'__'+b]={'target_mmsi':len(ids[a]&ids[b]),'target_observations':len(obs[a]&obs[b])}
  assert not ids[a]&ids[b] and not obs[a]&obs[b]
 audit['all_checks_passed']=True
 return audit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--build',action='store_true');ap.add_argument('--time-limit',type=float,default=45.);args=ap.parse_args()
 args.out.mkdir(parents=True,exist_ok=True)
 if args.build and (args.out/'label_manifest.parquet').exists():raise SystemExit('Refusing to overwrite an existing frozen label ledger')
 frames=[];sources={};reports={};selection={};chosen=[]
 for p in sorted(CF.glob('*/counterfactual_plan.parquet')):
  region=p.parent.name;g=pd.read_parquet(p);reportfile=p.parent/'counterfactual_report.json';r=json.loads(reportfile.read_text());npz=p.parent/'counterfactual_observations.npz'
  assert r['generator_sha256']=='d05df588da58120e78493a4703b6f8d9109d993d344315a51f1bcba9ad996882'
  assert r['labels_assigned'] is False
  g['counterfactual_plan_file']=str(p);g['counterfactual_plan_sha256']=digest(p);g['counterfactual_observations_file']=str(npz);g['counterfactual_observations_sha256']=digest(npz);g['generator_sha256']=r['generator_sha256'];g['physical_limits']=json.dumps(r['physical_limits'],sort_keys=True)
  a=g[g.counterfactual_accepted].copy();assert a.window_id.is_unique
  frames.append(a);sources[region]={'plan':str(p),'plan_sha256':digest(p),'observations':str(npz),'observations_sha256':digest(npz),'report_sha256':digest(reportfile),'candidate_windows':len(g),'accepted_windows':len(a),'training_only_budget_evidence':r['training_only_budget_evidence'],'budget_aggregation':r['budget_aggregation'],'physical_limits':r['physical_limits']}
 parents=pd.concat(frames,ignore_index=True);assert parents.region.nunique()==8 and not parents.window_id.duplicated().any()
 # Bottleneck groups first; each independent MILP preserves the inherited split.
 keys=[('ny','val'),('ny','train'),('kattegat','test')]
 keys += [(r,s) for r in sorted(parents.region.unique()) for s in ['train','val','test'] if (r,s) not in keys]
 feasible=True
 for region,split in keys:
  pool=parents[(parents.region==region)&(parents.split==split)]
  picked,rep=solve_cell(pool,region,split,args.time_limit);selection.setdefault(region,{})[split]=rep
  print('CELL',region,split,rep['status'],flush=True)
  if picked is None:feasible=False;break
  chosen.append(picked)
 report={'status':'metadata_selection_feasible' if feasible else 'metadata_selection_not_feasible','seed':SEED,'requested_windows_per_region':1400,'requested_vessels_per_region':260,'split_quotas':QUOTAS,'source_files':sources,'cells':selection,'selection_scope':'Only fixed physically accepted counterfactuals; no labels/predictions/test scores in selection','source_selection_sha256':digest(SOURCE/'selected_windows.parquet'),'global_all_mmsi_split_sha256':digest(SOURCE/'global_all_mmsi_split.parquet'),'freeze_script_sha256':digest(Path(__file__))}
 if feasible and args.build:
  result=assign_labels(pd.concat(chosen,ignore_index=True));audit=validate_selected(result,parents)
  result.to_parquet(args.out/'label_manifest.parquet',index=False)
  for region,g in result.groupby('region'):g.to_parquet(args.out/f'{region}_label_manifest.parquet',index=False)
  (args.out/'label_validation_report.json').write_text(json.dumps(audit,indent=2))
  report.update(status='frozen_labels_verified',rows=len(result),positive=int(result.y.sum()),negative=int((result.y==0).sum()),global_unique_target_mmsi=int(result.mmsi.nunique()),label_manifest_sha256=digest(args.out/'label_manifest.parquet'),validation_report_sha256=digest(args.out/'label_validation_report.json'),factor_design={'behavior_fraction':.4,'observation_noise_fraction':'1/3 rounded within each region/split/y; deviation <=0.5 window','noise_kind':['teleport','speed_spike','drift'],'noise_scope':['target_only','neighbors_only','both'],'factor_seed_spaces':'label, contamination selection, contamination kind and scope use distinct stable-hash namespaces','group_normal_variation':'not in primary label pool; separate paired challenge view'},labels_scope='Controlled injected anomaly vs accepted-counterfactual sham; natural anomaly status remains unreviewed',positive_kind_counts=result.loc[result.y==1,'latent_kind'].value_counts().to_dict())
 (args.out/'selection_report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k not in ['source_files','cells']},indent=2),flush=True)
 return 0 if feasible else 2

if __name__=='__main__':raise SystemExit(main())
