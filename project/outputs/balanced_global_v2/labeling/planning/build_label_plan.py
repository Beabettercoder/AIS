"""Source-only maneuver eligibility and deterministic controlled-label ledger.

No historical raw data, source split, or release file is changed. Controlled
labels indicate injected behavior; nominal backgrounds are NOT human-normal
ground truth. Threshold calibration uses selected TRAIN observations only.
"""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import pandas as pd

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
SOURCE=ROOT/'release_20261026'
OUT=ROOT/'labeling/planning'
KNOT=1852/3600
KINDS=['turn_back','deviation','stall','sinuous','speed_drop']
SEED=20261026

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def stable_seed(*parts):
 return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8],'little')&((1<<63)-1)

def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi

def geometry(arr,ii,lon0,lat0):
 t=arr['timestamp'][ii].astype(float);s=arr['sog'][ii].astype(float)
 cog=np.deg2rad(arr['cog'][ii].astype(float))
 xy=np.column_stack(((arr['lon'][ii]-lon0)*111195*np.cos(np.deg2rad(lat0)),(arr['lat'][ii]-lat0)*111195))
 v=KNOT*s[:,None]*np.column_stack((np.sin(cog),np.cos(cog)))
 dt=np.diff(t);dp=np.diff(xy,axis=0);distance=np.linalg.norm(dp,axis=1)
 err=np.linalg.norm(dp/dt[:,None]-(v[:-1]+v[1:])/2,axis=1)
 w=np.exp(-np.maximum(0,err-2.5)**2/8)
 pairvalid=(dt>0)&(dt<=300)
 accel=np.abs(np.diff(s))*KNOT/dt
 omega=np.abs(wrap(np.diff(cog)))/dt
 rot=np.rad2deg(omega);latacc=(s[:-1]+s[1:])/2*KNOT*omega
 return dict(t=t,s=s,cog=cog,xy=xy,v=v,dt=dt,distance=distance,error=err,w=w,pairvalid=pairvalid,acceleration_mps2=accel,rot_deg_s=rot,lateral_acceleration_mps2=latacc)

def analyze(row,arr,meta):
 ii=np.array(json.loads(row.row_indices),dtype=np.int64)
 assert np.all(arr['mmsi'][ii]==row.mmsi) and np.array_equal(arr['timestamp'][ii],json.loads(row.timestamps))
 box=meta['core_bbox'];g=geometry(arr,ii,(box[0]+box[2])/2,(box[1]+box[3])/2)
 t,s,xy=g['t'],g['s'],g['xy'];dt=g['dt'];n=len(t)
 span=t[-1]-t[0];net=np.linalg.norm(xy[-1]-xy[0]);path=g['distance'].sum()
 stationary=(s.max()<2) or (np.median(s)<.5 and net<300)
 intervals=[];strict_intervals=[]
 for j in range(3,n-2):
  prefix=s[:j];prefix_path=float(g['distance'][:j-1].sum())
  reported_path=float(np.sum((prefix[1:]+prefix[:-1])/2*KNOT*dt[:j-1]))
  prefix_net=float(np.linalg.norm(xy[j-1]-xy[0]))
  if stationary or np.median(prefix)<3 or (prefix>=3).sum()<3 or prefix[-1]<3 or prefix_path<500 or reported_path<500:continue
  lo=float(t[j-1]+1.0);hi=float(min(t[j]-1.,t[-1]-180.))
  if hi<lo:continue
  record={'lo_utc_s':lo,'hi_utc_s':hi,'prefix_obs':j,'post_obs':n-j,'prefix_path_m':prefix_path,'prefix_reported_distance_m':reported_path,'prefix_net_displacement_m':prefix_net,'prefix_median_sog_kn':float(np.median(prefix)),'anchor_sog_kn':float(prefix[-1])}
  intervals.append(record)
  prevalid=g['pairvalid'][:j-1]
  prebad=(g['w'][:j-1]<.3)&prevalid
  if prefix_net>=500 and prevalid.all() and not prebad.any():strict_intervals.append(record)
 valid=g['pairvalid'];v_errors=g['error'][valid]
 result={k:getattr(row,k) for k in ['window_id','region','split','mmsi','start_utc_s','end_utc_s','n_obs','row_indices','timestamps']}
 result.update(source_file=meta['canonical_file'],source_sha256=meta['sha256'],duration_s=float(span),original_path_m=float(path),original_net_displacement_m=float(net),median_sog_kn=float(np.median(s)),max_sog_kn=float(s.max()),stationary_excluded=bool(stationary),eligible=bool(intervals),strict_eligible=bool(strict_intervals),onset_intervals=json.dumps(intervals,separators=(',',':')),strict_onset_intervals=json.dumps(strict_intervals,separators=(',',':')),allowed_types=json.dumps(KINDS if intervals else []),kinematic_valid_fraction=float(valid.mean()),kinematic_low_trust_fraction=float(((g['w']<.3)&valid).sum()/max(1,valid.sum())),kinematic_error_p95_mps=float(np.quantile(v_errors,.95)) if len(v_errors) else None,max_acceleration_mps2=float(g['acceleration_mps2'][valid].max()) if valid.any() else None,max_rot_deg_s=float(g['rot_deg_s'][valid].max()) if valid.any() else None,max_lateral_acceleration_mps2=float(g['lateral_acceleration_mps2'][valid].max()) if valid.any() else None,background_annotation='unreviewed_observed_background')
 # Source-only TRAIN envelope; moderate speed, valid intervals, full confidence
 # under the existing 2.5 m/s gate dead zone. No generated or held-out values.
 budget_pairs=valid&(s[:-1]>=3)&(s[1:]>=3)&(s[:-1]<=40)&(s[1:]<=40)&(g['error']<=2.5)
 return result,{k:g[k][budget_pairs] for k in ['acceleration_mps2','rot_deg_s','lateral_acceleration_mps2']}

def totals(elig):
 result={}
 for (region,split),g in elig.groupby(['region','split'],sort=True):
  n=len(g);need=round(.4*n)
  result.setdefault(region,{})[split]={'windows':n,'target_ships':int(g.mmsi.nunique()),'required_injected_behavior_40_percent':need,'eligible_windows':int(g.eligible.sum()),'strict_eligible_windows':int(g.strict_eligible.sum()),'stationary_excluded':int(g.stationary_excluded.sum()),'base_feasible':int(g.eligible.sum())>=need,'strict_feasible':int(g.strict_eligible.sum())>=need,'max_kinematic_low_trust_fraction':float(g.kinematic_low_trust_fraction.max())}
 return result

def calibration(budgets):
 result={}
 for region,data in budgets.items():
  result[region]={}
  for key,parts in data.items():
   x=np.concatenate(parts) if parts else np.array([])
   result[region][key]={'pairs':len(x),'p50':float(np.quantile(x,.5)) if len(x) else None,'p99':float(np.quantile(x,.99)) if len(x) else None,'p995':float(np.quantile(x,.995)) if len(x) else None}
 return result

def build_plan(elig,out,strict,noise_num,noise_den):
 """Prepare latent counterfactuals BEFORE physical acceptance or binary labels.

 Every accepted future y=0 sham and y=1 observation must be selected from the
 SAME accepted counterfactual pool. This file intentionally has no final y.
 """
 pieces=[];counts={}
 for (region,split),group in elig.groupby(['region','split'],sort=True):
  g=group.copy();g['main_pool_candidate']=g.strict_eligible if strict else g.eligible
  g['latent_kind']='none';g['onset_utc_s']=np.nan;g['anchor_sog_kn']=np.nan;g['prefix_sog_kn']=np.nan
  g['latent_kind_index']=-1;g['sign']=0;g['prefix_obs']=0;g['post_obs']=0
  g['generation_stratum']='ineligible';g['eligibility_policy']='strict_prefix_qc' if strict else 'moving_prefix'
  ids=g.index[g.main_pool_candidate].to_numpy()
  for idx in ids:
   w=g.loc[idx,'window_id'];rng=np.random.default_rng(stable_seed(SEED,w,'onset_v1'))
   intervals=json.loads(g.loc[idx,'strict_onset_intervals' if strict else 'onset_intervals'])
   q=intervals[int(rng.integers(len(intervals)))];onset=float(rng.uniform(q['lo_utc_s'],q['hi_utc_s'])) if q['hi_utc_s']>q['lo_utc_s'] else float(q['lo_utc_s'])
   g.loc[idx,['onset_utc_s','anchor_sog_kn','prefix_sog_kn','prefix_obs','post_obs']]=[onset,q['anchor_sog_kn'],q['prefix_median_sog_kn'],q['prefix_obs'],q['post_obs']]
   nobs=int(g.loc[idx,'n_obs']);speed=q['prefix_median_sog_kn'];ekin=float(g.loc[idx,'kinematic_error_p95_mps'])
   nb='8_11' if nobs<=11 else ('12_15' if nobs<=15 else '16_20')
   sb='3_6' if speed<6 else ('6_10' if speed<10 else 'ge10')
   eb='lt025' if ekin<.25 else ('025_075' if ekin<.75 else ('075_25' if ekin<2.5 else 'ge25'))
   g.loc[idx,'generation_stratum']=f'n{nb}_s{sb}_e{eb}'
  # Exact global five-kind quotas; greedily minimize within-stratum imbalance.
  quotas=np.full(5,len(ids)//5,dtype=int)
  extra=np.random.default_rng(stable_seed(SEED,region,split,'latent_extra')).permutation(5)
  quotas[extra[:len(ids)%5]]+=1;remaining=quotas.copy()
  for stratum,h in g.loc[ids].groupby('generation_stratum',sort=True):
   ordered=sorted(h.index,key=lambda j:(stable_seed(SEED,g.loc[j,'window_id'],'latent_order'),g.loc[j,'window_id']))
   used=np.zeros(5,int);kindorder=np.random.default_rng(stable_seed(SEED,region,split,stratum,'latent_tie')).permutation(5)
   tie={int(k):i for i,k in enumerate(kindorder)}
   for idx in ordered:
    avail=np.flatnonzero(remaining>0)
    k=min(avail,key=lambda k:(used[k],tie[int(k)]))
    g.loc[idx,['latent_kind','latent_kind_index']]=[KINDS[k],int(k)]
    used[k]+=1;remaining[k]-=1
  assert remaining.sum()==0
  g['behavior_seed']=[stable_seed(SEED,w,kind,'generator_v1') for w,kind in zip(g.window_id,g.latent_kind)]
  g['noise_seed']=[stable_seed(SEED,w,'observation_v1') for w in g.window_id]
  g['group_seed']=[stable_seed(SEED,w,'group_v1') for w in g.window_id]
  g['label_assignment_seed']=[stable_seed(SEED,w,'labels_after_physical_acceptance_v1') for w in g.window_id]
  g['noise_assignment_u']=[np.random.default_rng(stable_seed(SEED,w,'noise_assignment_v1')).random() for w in g.window_id]
  g['requested_noise_type']=[['teleport','speed_spike','drift'][stable_seed(SEED,w,'noise_kind_v1')%3] for w in g.window_id]
  for idx in ids:
   if g.loc[idx,'latent_kind'] in ['turn_back','deviation','sinuous']:
    g.loc[idx,'sign']=int(np.random.default_rng(int(g.loc[idx,'behavior_seed'])).choice([-1,1]))
  g['role']=np.where(g.main_pool_candidate,'counterfactual_generation_candidate','nominal_extra_not_main_training')
  g['labels_frozen']=False;g['plan_status']='provisional_counterfactuals_before_binary_label_assignment'
  pieces.append(g)
  counts.setdefault(region,{})[split]={'source_windows':len(g),'counterfactual_candidates':len(ids),'candidate_mmsi':int(g.loc[ids,'mmsi'].nunique()),'latent_kinds':g.loc[ids,'latent_kind'].value_counts().to_dict(),'generation_strata':g.loc[ids,'generation_stratum'].value_counts().to_dict()}
 result=pd.concat(pieces,ignore_index=True)
 assert len(result)==21600 and not result.window_id.duplicated().any()
 assert 'behavior_anomaly' not in result and 'y' not in result
 for _,g in result.groupby('mmsi'):assert g.split.nunique()==1
 result.to_parquet(out/'label_plan.parquet',index=False)
 candidates=result[result.main_pool_candidate].reset_index(drop=True)
 candidates.to_parquet(out/'candidate_generation_plan.parquet',index=False)
 return {'rows':len(result),'counterfactual_candidates':len(candidates),'labels_frozen':False,'binary_y_present':False,'desired_noise_fraction_after_acceptance':f'{noise_num}/{noise_den}','counts':counts,'label_plan_sha256':sha(out/'label_plan.parquet'),'candidate_generation_plan_sha256':sha(out/'candidate_generation_plan.parquet')}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--build-plan',action='store_true');ap.add_argument('--strict',action='store_true');ap.add_argument('--noise-numerator',type=int,default=1);ap.add_argument('--noise-denominator',type=int,default=3);args=ap.parse_args()
 args.out.mkdir(parents=True,exist_ok=True)
 sel=pd.read_parquet(SOURCE/'selected_windows.parquet');registry=json.loads((SOURCE/'source_registry.json').read_text())['regions']
 rows=[];budgets={'all_train':{k:[] for k in ['acceleration_mps2','rot_deg_s','lateral_acceleration_mps2']}}
 for region,g in sel.groupby('region',sort=True):
  meta=registry[region];p=Path(meta['canonical_file']);assert sha(p)==meta['sha256']
  d=pd.read_parquet(p,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True);arr={k:d[k].to_numpy() for k in d.columns}
  budgets[region]={k:[] for k in budgets['all_train']}
  for row in g.itertuples():
   res,b=analyze(row,arr,meta);rows.append(res)
   if row.split=='train':
    for k,v in b.items():budgets[region][k].append(v);budgets['all_train'][k].append(v)
  print('ANALYZED',region,len(g),flush=True)
 elig=pd.DataFrame(rows);assert len(elig)==len(sel)==21600 and not elig.window_id.duplicated().any()
 elig.to_parquet(args.out/'eligibility.parquet',index=False)
 cap={'source_release':str(SOURCE),'source_selection_sha256':sha(SOURCE/'selected_windows.parquet'),'eligibility_file_sha256':sha(args.out/'eligibility.parquet'),'all_rows':len(elig),'all_base_feasible':all(v['base_feasible'] for r in totals(elig).values() for v in r.values()),'all_strict_feasible':all(v['strict_feasible'] for r in totals(elig).values() for v in r.values()),'regions':totals(elig),'training_only_physical_envelope':calibration(budgets),'budget_scope':'Empirical TRAIN-only 99%/99.5% envelopes under existing 2.5 m/s residual dead-zone filter; benchmark calibration, NOT independently validated ship-type safety limits','budget_pair_filters':{'same_selected_train_window':True,'dt_s':[0,300],'both_speed_kn':[3,40],'kinematic_error_max_mps':2.5},'noise_fraction_recommendation':'1/3 permits exact equal noise rates within normal/anomaly in every 1890/405/405 split; 1/4 does not divide the two behavior class counts exactly.'}
 (args.out/'capacity.json').write_text(json.dumps(cap,indent=2))
 plan=None
 if args.build_plan:plan=build_plan(elig,args.out,args.strict,args.noise_numerator,args.noise_denominator)
 protocol={'status':'provisional_counterfactual_plan_built' if plan else 'capacity_only_pending_assignment_choice','seed':SEED,'behavior_class_fraction_target_after_physical_acceptance':{'no_injected_behavior':.6,'injected_behavior':.4},'kind_order':KINDS,'class_quota_policy':'Latent five-kind assignment is metadata-stratified and near-equal BEFORE physical acceptance or binary labels. Future y0 sham and y1 must share the same accepted pool.','factors':['behavior_anomaly','observation_noise','group_normal_variation'],'background_semantics':'No injected behavior is a controlled label, not human-reviewed natural normal ground truth','onset_eligibility':{'minimum_prefix_observations':3,'prefix_moving_observations_sog_ge3':3,'prefix_median_sog_kn_min':3,'last_prefix_sog_kn_min':3,'prefix_path_m_min':500,'prefix_reported_distance_m_min':500,'post_observations_min':3,'post_duration_seconds_min':180,'strict_extra':'net displacement >=500 m, all prefix dt <=300 s, no prefix reliability <0.3'},'onset_sampling':'Uniformly choose one source-supported feasible between-report interval, then continuous uniform time within it; no fixed 3/5/10-min template','source_split_unchanged':True,'group_normal_variation_scope':'Separate paired challenge view, including coherent target+neighbors; same base IDs, not extra independent data','stratification':['region','split','n_obs bin','pre-onset median speed bin','original kinematic error p95 bin'],'plan':plan}
 (args.out/'protocol_plan.json').write_text(json.dumps(protocol,indent=2))
 print(json.dumps({'all_base_feasible':cap['all_base_feasible'],'all_strict_feasible':cap['all_strict_feasible'],'rows':len(elig),'output':str(args.out),'plan':None if plan is None else {k:v for k,v in plan.items() if k!='counts'}},indent=2),flush=True)

if __name__=='__main__':main()
