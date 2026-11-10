"""Verify every latent intervention before assigning positive/negative labels."""
from pathlib import Path
import argparse,gzip,hashlib,json,sys,time
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'generator'))
from maneuver_generator import generate_maneuver,sample_parameters,PhysicalLimits
ROOT=BASE.parent;SOURCE=ROOT/'release_20261026';OUT=BASE/'counterfactuals'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def realized_ok(kind,r):
 e=r['metadata']['realized']
 if kind=='turn_back':return e['max_heading_offset_deg']>=20
 if kind=='deviation':return e['max_heading_offset_deg']>=12
 if kind=='sinuous':return e['realized_sinuous_amplitude_deg']>=5 and e['realized_sinuous_peak_to_peak_deg']>=8
 return e['max_speed_change_kn']>=max(1.,.2*e['reference_speed_at_onset_kn'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--regions',nargs='*');ap.add_argument('--out',type=Path,default=BASE/'counterfactuals_v1');args=ap.parse_args();args.out.mkdir(exist_ok=True)
 plan=pd.read_parquet(BASE/'planning/candidate_generation_plan.parquet')
 cap=json.loads((BASE/'planning/capacity.json').read_text());env=cap['training_only_physical_envelope']['all_train']
 # Each training domain contributes an envelope rather than letting domains
 # with more retained reports determine a shared reported-profile budget.
 regional={k:v for k,v in cap['training_only_physical_envelope'].items() if k!='all_train'}
 domain_limit={key:max(v[key]['p995'] for v in regional.values()) for key in env}
 limits=PhysicalLimits(max_longitudinal_accel_mps2=domain_limit['acceleration_mps2'],max_turn_rate_deg_s=domain_limit['rot_deg_s'],max_lateral_accel_mps2=domain_limit['lateral_acceleration_mps2'],max_speed_kn=40.,max_duration_s=1800.)
 frozen_hash=sha(BASE/'generator/maneuver_generator.py')
 for region,g in plan.groupby('region',sort=True):
  if args.regions and region not in args.regions:continue
  dest=args.out/region;dest.mkdir(exist_ok=True)
  if (dest/'counterfactual_report.json').exists():raise RuntimeError(f'Already built {dest}')
  started=time.time();source_path=Path(g.source_file.iloc[0]);assert sha(source_path)==g.source_sha256.iloc[0]
  d=pd.read_parquet(source_path,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
  arr={k:d[k].to_numpy() for k in d.columns};reg=json.loads((SOURCE/'source_registry.json').read_text())['regions'][region]
  box=reg['core_bbox'];lat0=(box[1]+box[3])/2;lon0=(box[0]+box[2])/2
  positions=np.column_stack(((arr['lon']-lon0)*111195*np.cos(np.deg2rad(lat0)),(arr['lat']-lat0)*111195))
  data={k:[] for k in ['t','original_pos','original_sog','original_cog','smooth_pos','smooth_sog','smooth_cog','mask']};records=[];accepted_count=0;max_error=0
  with gzip.open(dest/'generation_audit.jsonl.gz','wt',encoding='utf-8',compresslevel=3) as audit:
   for row in g.itertuples():
    ii=np.array(json.loads(row.row_indices),dtype=np.int64);t=arr['timestamp'][ii].astype(float);pos=positions[ii];s=arr['sog'][ii].astype(float);c=arr['cog'][ii].astype(float);n=len(ii)
    assert np.all(arr['mmsi'][ii]==row.mmsi) and np.array_equal(t,json.loads(row.timestamps))
    params=sample_parameters(row.latent_kind,int(row.behavior_seed))
    if 'sign' in params:params['sign']=int(row.sign)
    r=generate_maneuver(t,pos,s,c,kind=row.latent_kind,onset_s=row.onset_utc_s,params=params,limits=limits)
    effect_ok=r['accepted'] and realized_ok(row.latent_kind,r)
    reason='accepted' if effect_ok else (r['reason'] if not r['accepted'] else 'insufficient_observed_effect')
    rec=row._asdict();rec.update(counterfactual_accepted=bool(effect_ok),rejection_reason=reason,counterfactual_row=len(records),applied_params=json.dumps(r['metadata']['applied_params'],separators=(',',':')),innovation_preservation_max_mps=r['metadata']['innovation_preservation_max_mps'],realized=json.dumps(r['metadata']['realized'],separators=(',',':')))
    records.append(rec);audit.write(json.dumps({'base_window_id':row.window_id,'region':region,'split':row.split,'mmsi':int(row.mmsi),'accepted':bool(effect_ok),'rejection_reason':reason,'generator':r['metadata']},separators=(',',':'))+'\n')
    max_error=max(max_error,r['metadata']['innovation_preservation_max_mps']);accepted_count+=int(effect_ok)
    values={'t':t,'original_pos':pos,'original_sog':s,'original_cog':c,'smooth_pos':r['pos'],'smooth_sog':r['sog_kn'],'smooth_cog':r['cog_deg'],'mask':np.ones(n,bool)}
    for k,v in values.items():
     pad=np.zeros((20,)+v.shape[1:],dtype=v.dtype);pad[:n]=v;data[k].append(pad)
  table=pd.DataFrame(records);table.to_parquet(dest/'counterfactual_plan.parquet',index=False)
  arrays={k:np.stack(v) for k,v in data.items()};arrays['window_id']=table.window_id.to_numpy(dtype='U24');np.savez_compressed(dest/'counterfactual_observations.npz',**arrays)
  eligible=table[table.counterfactual_accepted];capacity={}
  for split,h in eligible.groupby('split'):
   capacity[split]={'windows':len(h),'vessels':int(h.mmsi.nunique()),'kinds':h.latent_kind.value_counts().to_dict(),'strata':h.generation_stratum.value_counts().to_dict()}
  assert sha(BASE/'generator/maneuver_generator.py')==frozen_hash
  report={'region':region,'candidate_windows':len(table),'accepted_windows':accepted_count,'capacity':capacity,'rejections':table.loc[~table.counterfactual_accepted,'rejection_reason'].value_counts().to_dict(),'physical_limits':limits.__dict__,'training_only_budget_evidence':sha(BASE/'planning/capacity.json'),'generator_sha256':frozen_hash,'source_sha256':sha(source_path),'candidate_plan_sha256':sha(BASE/'planning/candidate_generation_plan.parquet'),'innovation_preservation_max_mps':max_error,'realized_effect_rules':{'turn_back_min_heading_deg':20,'deviation_min_heading_deg':12,'sinuous_min_amplitude_deg':5,'sinuous_min_peak_to_peak_deg':8,'speed_change_min_kn':'max(1,0.2*onset_speed)'},'labels_assigned':False,'elapsed_seconds':time.time()-started}
  report['budget_aggregation']='Maximum of each TRAIN domain p99.5; fixed before any model evaluation; no val/test calibration'
  (dest/'counterfactual_report.json').write_text(json.dumps(report,indent=2))
  print(region,json.dumps({'accepted_windows':accepted_count,'capacity':{s:{k:v for k,v in x.items() if k!='strata'} for s,x in capacity.items()},'rejections':report['rejections'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
