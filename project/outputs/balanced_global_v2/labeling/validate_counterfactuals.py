"""Independent all-window audit of the frozen counterfactuals_v1 pool.

Original reports, delta-velocity integration, vector innovations, maneuver
profiles, feasibility and train-only envelope quantiles are independently
recomputed below. The frozen generator is called ONLY to test its identity and
matched-hard API against these independent equations; it supplies no validator
physics calculations or acceptance decisions.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,sys,time
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
SOURCE=ROOT/'release_20261026'
FROZEN='d05df588da58120e78493a4703b6f8d9109d993d344315a51f1bcba9ad996882'
KINDS={'turn_back','deviation','stall','sinuous','speed_drop'}
KN=1852./3600.
METRICS={'acceleration_mps2':'max_longitudinal_accel_mps2',
         'rot_deg_s':'max_turn_rate_deg_s',
         'lateral_acceleration_mps2':'max_lateral_accel_mps2'}


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def require(condition,message):
 if not condition:raise ValueError(message)


def close(a,b,message,atol=1e-8,rtol=1e-10):
 require(np.asarray(a).shape==np.asarray(b).shape,message+' shape')
 require(np.allclose(a,b,atol=atol,rtol=rtol),message)


def velocity(s,c):
 angle=np.asarray(c)*np.pi/180.
 return np.asarray(s)[:,None]*KN*np.column_stack([np.sin(angle),np.cos(angle)])


def innovation(t,p,s,c):
 v=velocity(s,c)
 return np.diff(p,axis=0)/np.diff(t)[:,None]-(v[:-1]+v[1:])/2.


def pair_metrics(t,s,c):
 dt=np.diff(t);turn=(np.diff(c)+180.)%360.-180.
 omega=np.abs(turn)*np.pi/180./dt
 return {'acceleration_mps2':np.abs(np.diff(s))*KN/dt,
         'rot_deg_s':np.abs(turn)/dt,
         'lateral_acceleration_mps2':(s[:-1]+s[1:])*.5*KN*omega}


def stats(t,s,c):
 p=pair_metrics(t,s,c)
 result={METRICS[k]:float(v.max()) for k,v in p.items()}
 result.update(max_speed_kn=float(s.max()),min_speed_kn=float(s.min()))
 return result


def independent_profile(t,s,c,kind,onset,p,hard=False):
 def ramp(start,dur):
  if hard:return (t>=start).astype(float)
  u=np.clip((t-start)/dur,0.,1.)
  return 3*u*u-2*u*u*u
 progress=ramp(onset,p['duration_s']);ss=s.copy();offset=np.zeros(len(t))
 if kind=='turn_back':offset=p['sign']*p['delta_deg']*progress
 elif kind=='deviation':
  return_at=onset+p['duration_s']+p['hold_s']
  offset=p['sign']*p['delta_deg']*(progress-ramp(return_at,p['duration_s']))
 elif kind=='stall':ss=(1-progress)*s+progress*p['target_kn']
 elif kind=='speed_drop':ss=s*(1-(1-p['factor'])*progress)
 elif kind=='sinuous':
  end=onset+p['active_duration_s'];down_at=end if hard else end-p['ramp_down_s']
  envelope=progress*(1-ramp(down_at,p['ramp_down_s']))
  offset=p['sign']*p['amplitude_deg']*envelope*np.sin(2*np.pi*(t-onset)/p['period_s']+p['phase_rad'])
 else:raise ValueError(kind)
 return ss,(c+offset)%360.,offset,progress


def changed_position(t,pos,s,c,aug_s,aug_c):
 dv=velocity(aug_s,aug_c)-velocity(s,c)
 change=np.zeros_like(pos)
 change[1:]=np.cumsum((dv[:-1]+dv[1:])*.5*np.diff(t)[:,None],axis=0)
 return pos+change


def observed_effect(kind,t,s,aug_s,offset,onset):
 reference=float(np.interp(onset,t,s));post=offset[t>=onset]
 if kind=='turn_back':return np.abs(offset).max()>=20
 if kind=='deviation':return np.abs(offset).max()>=12
 if kind=='sinuous':return np.abs(post).max()>=5 and np.ptp(post)>=8
 return np.abs(aug_s-s).max()>=max(1.,.2*reference)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--counterfactuals',type=Path,default=BASE/'counterfactuals_v1');ap.add_argument('--regions',nargs='*');args=ap.parse_args()
 started=time.time();output=args.counterfactuals/'independent_counterfactual_audit.json'
 report={'schema':'independent_counterfactual_audit_v1','status':'running','regions':{},'errors':[],
         'verification_scope':'Every candidate, every stored source report, every accepted smooth and independently constructed matched hard control',
         'generator_import_scope':'Only identity/hard API calls; profiles, integration, innovations, constraints, quantiles and effect acceptance are independent here',
         'labels_scope':'Counterfactual pool has no final binary labels; acceptance is independently checked before any positive labels can be assigned',
         'source_dtypes':'Canonical float64 reports; identity is exact in value and dtype for these actual inputs'}
 try:
  genpath=BASE/'generator/maneuver_generator.py';require(sha(genpath)==FROZEN,'Unexpected generator version')
  sys.path.insert(0,str(genpath.parent));from maneuver_generator import generate_maneuver
  registry=json.loads((SOURCE/'source_registry.json').read_text())['regions']
  selected=pd.read_parquet(SOURCE/'selected_windows.parquet')
  group=pd.read_parquet(SOURCE/'global_all_mmsi_split.parquet');groupmap=dict(zip(group.mmsi.astype(int),group.split))
  candidates=pd.read_parquet(BASE/'planning/candidate_generation_plan.parquet')
  capacity=json.loads((BASE/'planning/capacity.json').read_text())
  require(sha(SOURCE/'selected_windows.parquet')==capacity['source_selection_sha256'],'Changed selected source file')
  report['input_hashes']={str(p.relative_to(ROOT)):sha(p) for p in [genpath,Path(__file__),SOURCE/'selected_windows.parquet',SOURCE/'global_all_mmsi_split.parquet',SOURCE/'source_registry.json',BASE/'planning/capacity.json',BASE/'planning/candidate_generation_plan.parquet',BASE/'build_counterfactuals.py']}
  common_columns=['window_id','region','split','mmsi','onset_utc_s','latent_kind','sign','behavior_seed','noise_seed','group_seed','label_assignment_seed','noise_assignment_u','requested_noise_type','source_file','source_sha256','row_indices','timestamps']
  envelopes={};pools={};global_ids=set();global_observations={s:set() for s in ['train','val','test']}
  counts=dict(candidates=0,accepted=0,rejected=0,source_reports=0,dense_profiles_checked=0,matched_hard_controls=0,identity_windows=0)
  maxima=dict(smooth_innovation_error_mps=0.,hard_innovation_error_mps=0.,delta_position_error_m=0.)
  for region in sorted(candidates.region.unique()):
   if args.regions and region not in args.regions:continue
   rs=time.time();dest=args.counterfactuals/region
   table=pd.read_parquet(dest/'counterfactual_plan.parquet')
   # NpzFile is lazy: cache each array once rather than decompressing an entire
   # region again for every individual window.
   with np.load(dest/'counterfactual_observations.npz',allow_pickle=False) as z:
    saved={k:z[k] for k in z.files}
   require(not table.window_id.duplicated().any(),'Repeated regional counterfactual ID')
   require(not set(table.columns)&{'y','label_binary','binary_label','behavior_anomaly'},'Final binary labels assigned before verification')
   require(not table.labels_frozen.any(),'Counterfactual plan incorrectly claims final labels')
   expected=candidates[candidates.region==region]
   pd.testing.assert_frame_equal(table[common_columns].set_index('window_id').sort_index(),expected[common_columns].set_index('window_id').sort_index())
   source_meta=registry[region];path=Path(source_meta['canonical_file']);require(sha(path)==source_meta['sha256'],'Changed canonical source '+region)
   require((table.source_file==str(path)).all() and (table.source_sha256==source_meta['sha256']).all(),'Wrong source pointer')
   source=pd.read_parquet(path,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
   a={k:source[k].to_numpy() for k in source.columns};b=source_meta['core_bbox'];lon0=(b[0]+b[2])/2;lat0=(b[1]+b[3])/2
   xy=np.column_stack(((a['lon']-lon0)*111195*np.cos(np.deg2rad(lat0)),(a['lat']-lat0)*111195))
   # Rebuild each TRAIN-domain envelope from ALL frozen source TRAIN windows.
   parts={k:[] for k in METRICS}
   for row in selected[(selected.region==region)&(selected.split=='train')].itertuples():
    ii=np.array(json.loads(row.row_indices),dtype=np.int64);t=a['timestamp'][ii].astype(float);s=a['sog'][ii].astype(float);c=a['cog'][ii].astype(float);p=xy[ii]
    err=np.linalg.norm(innovation(t,p,s,c),axis=1);dt=np.diff(t)
    choose=(dt>0)&(dt<=300)&(s[:-1]>=3)&(s[1:]>=3)&(s[:-1]<=40)&(s[1:]<=40)&(err<=2.5)
    pm=pair_metrics(t,s,c)
    for key,v in pm.items():parts[key].append(v[choose])
   envelopes[region]={}
   for key,values in parts.items():
    values=np.concatenate(values);require(len(values)>0,'No training calibration observations')
    q=float(np.quantile(values,.995));old=capacity['training_only_physical_envelope'][region][key]
    require(len(values)==old['pairs'],'Training pair count differs '+region+'/'+key)
    close(q,old['p995'],'TRAIN p99.5 mismatch '+region+'/'+key,atol=1e-10)
    envelopes[region][key]={'pairs':len(values),'p995':q}
   pools[region]=(table,saved,a,xy,source_meta,dest)
   report['regions'][region]={'candidates':len(table),'source_sha256':sha(path),'input_hashes':{p.name:sha(p) for p in [dest/'counterfactual_plan.parquet',dest/'counterfactual_observations.npz',dest/'generation_audit.jsonl.gz']},'source_loading_and_train_calibration_seconds':time.time()-rs}
  require(len(envelopes)==8 or args.regions,'Full audit requires eight regions')
  if len(envelopes)==8:
   limits={METRICS[key]:max(v[key]['p995'] for v in envelopes.values()) for key in METRICS}
  else:
   # Regional diagnostic does not redefine the frozen global TRAIN budget.
   limits={METRICS[key]:max(v[key]['p995'] for k,v in capacity['training_only_physical_envelope'].items() if k!='all_train') for key in METRICS}
  limits.update(max_speed_kn=40.,max_duration_s=1800.,tolerance=1e-8)
  report['independently_recomputed_train_envelopes']=envelopes;report['shared_budget']=limits
  for region,(table,saved,a,xy,source_meta,dest) in pools.items():
   rs=time.time();regional=dict(accepted=0,rejected=0,rejection_reasons={},source_reports=0,matched_hard_controls=0)
   require(np.array_equal(saved['window_id'],table.window_id.to_numpy()),'NPZ row order differs')
   with gzip.open(dest/'generation_audit.jsonl.gz','rt',encoding='utf-8') as stream:
    for row in table.itertuples():
     j=int(row.counterfactual_row);require(0<=j<len(table),'Bad counterfactual row')
     require(row.latent_kind in KINDS and bool(row.strict_eligible) and bool(row.main_pool_candidate),'Unknown/ineligible generation candidate')
     require(groupmap[int(row.mmsi)]==row.split,'Changed global identity split')
     require(row.window_id not in global_ids,'Repeated global source window');global_ids.add(row.window_id)
     ii=np.array(json.loads(row.row_indices),dtype=np.int64);t=a['timestamp'][ii].astype(float);p=xy[ii];s=a['sog'][ii].astype(float);c=a['cog'][ii].astype(float);n=len(ii)
     require(np.all(a['mmsi'][ii]==row.mmsi) and np.array_equal(t,json.loads(row.timestamps)),'Raw row identity/time mismatch')
     require(np.array_equal(saved['mask'][j],np.arange(20)<n),'Bad mask')
     for key,value in [('t',t),('original_pos',p),('original_sog',s),('original_cog',c)]:
      require(np.array_equal(saved[key][j,:n],value),'Original report changed: '+key)
      require(np.all(saved[key][j,n:]==0),'Source padding polluted')
     for key in ['smooth_pos','smooth_sog','smooth_cog']:
      require(np.isfinite(saved[key][j]).all() and np.all(saved[key][j,n:]==0),'Invalid output/padding')
     for tt in t:global_observations[row.split].add((int(row.mmsi),float(tt)))
     identity=generate_maneuver(t,p,s,c,kind='none')
     for key,value in [('t',t),('pos',p),('sog_kn',s),('cog_deg',c)]:
      require(identity[key].dtype==value.dtype and np.array_equal(identity[key],value),'Identity API altered canonical source')
     counts['identity_windows']+=1
     line=next(stream,None);require(line is not None,'Missing audit row');audit=json.loads(line);md=audit['generator']
     require(audit['base_window_id']==row.window_id and audit['mmsi']==row.mmsi and audit['split']==row.split,'Audit identity/order mismatch')
     require(md['schema']=='innovation_preserving_maneuver_v1' and md['kind']==row.latent_kind and md['profile']=='smooth','Unexpected intervention schema/kind')
     require(md['requested_onset_s']==row.onset_utc_s,'Changed onset')
     for key,value in limits.items():close(md['limits'][key],value,'Budget differs from TRAIN calibration: '+key,atol=1e-10)
     onset=float(row.onset_utc_s);idx=int(np.searchsorted(t,onset))
     require(3<=idx<=n-3 and t[-1]-onset>=180,'Insufficient before/after observations')
     require(t[idx-1]<onset<t[idx],'Continuous between-report onset is not recorded faithfully')
     require(md['onset_bracket_s']==[float(t[idx-1]),float(t[idx])],'Incorrect onset bracket')
     require(not md['sub_report_physical_onset_resolution_claimed'],'Unsupported sub-report timing claim')
     prefix=s[:idx];prefix_t=t[:idx];prefix_p=p[:idx]
     pre_err=np.linalg.norm(innovation(prefix_t,prefix_p,prefix,c[:idx]),axis=1)
     pre_w=np.exp(-np.maximum(0,pre_err-2.5)**2/8)
     require(np.median(prefix)>=3 and (prefix>=3).sum()>=3 and prefix[-1]>=3,'Prefix not underway')
     require(np.linalg.norm(prefix_p[-1]-prefix_p[0])>=500,'Prefix net displacement below500m')
     require(np.all(np.diff(prefix_t)<=300) and not np.any(pre_w<.3),'Unreliable prefix')
     applied=json.loads(row.applied_params);require(applied==md['applied_params'],'Ledger and audit parameter mismatch')
     requested=md['requested_params']
     for key in set(requested)-{'duration_s','ramp_up_s','period_s'}:
      require(applied[key]==requested[key],'Semantic parameter silently changed: '+key)
     require(0<applied['duration_s']<=limits['max_duration_s'] and applied['duration_s']>=requested['duration_s'],'Unrecorded duration adaptation')
     if 'sign' in applied:require(applied['sign']==int(row.sign),'Direction changed from plan')
     if row.latent_kind=='sinuous':
      require(applied['period_s']+1e-8>=4*np.median(np.diff(t)),'Sinuous reporting-period alias')
      require(applied['duration_s']==applied['ramp_up_s'],'Sinuous ramp alias mismatch')
     ss,cc,offset,progress=independent_profile(t,s,c,row.latent_kind,onset,applied)
     pp=changed_position(t,p,s,c,ss,cc)
     dense_t=np.unique(np.r_[t,np.arange(t[0],t[-1],1.),onset])
     dense_s=np.interp(dense_t,t,s);dense_c=np.interp(dense_t,t,np.rad2deg(np.unwrap(np.deg2rad(c))))%360
     dss,dcc,_,_=independent_profile(dense_t,dense_s,dense_c,row.latent_kind,onset,applied)
     observed_stats=stats(t,ss,cc);dense_stats=stats(dense_t,dss,dcc)
     for key,val in observed_stats.items():close(val,md['physical_observed'][key],'Wrong observed physics statistic '+key,atol=1e-8)
     for key,val in dense_stats.items():close(val,md['physical_dense'][key],'Wrong dense physics statistic '+key,atol=1e-8)
     physical_violations=sorted({key for key in ['max_longitudinal_accel_mps2','max_turn_rate_deg_s','max_lateral_accel_mps2','max_speed_kn'] if max(observed_stats[key],dense_stats[key])>limits[key]+limits['tolerance']})
     require(sorted(md['violations'])==physical_violations,'Physical budget rejection differs from independent calculation')
     generated=bool(md['behavior_injected'])
     require(generated==(not physical_violations),'Generated candidate physical status inconsistent')
     effect=observed_effect(row.latent_kind,t,s,ss,offset,onset)
     acceptance=generated and bool(effect)
     require(acceptance==bool(row.counterfactual_accepted)==bool(audit['accepted']),'Rejected/unknown candidate marked accepted')
     reason='accepted' if acceptance else ('physical_budget_exceeded' if not generated else 'insufficient_observed_effect')
     require(row.rejection_reason==reason==audit['rejection_reason'],'Wrong rejection reason')
     realized=md['realized'];require(realized==json.loads(row.realized),'Ledger realized effect differs')
     close(realized['max_heading_offset_deg'],float(np.abs(offset).max()),'Wrong realized heading effect')
     close(realized['max_speed_change_kn'],float(np.abs(ss-s).max()),'Wrong realized speed effect')
     reference=float(np.interp(onset,t,s));close(realized['reference_speed_at_onset_kn'],reference,'Wrong onset speed')
     if generated:
      for key,value in [('smooth_pos',pp),('smooth_sog',ss),('smooth_cog',cc)]:close(saved[key][j,:n],value,'Stored generated observations differ: '+key,atol=1e-8)
      source_i=innovation(t,p,s,c);smooth_i=innovation(t,saved['smooth_pos'][j,:n],saved['smooth_sog'][j,:n],saved['smooth_cog'][j,:n])
      err=float(np.linalg.norm(smooth_i-source_i,axis=1).max());require(err<1e-7,'Smooth source innovation not preserved')
      maxima['smooth_innovation_error_mps']=max(maxima['smooth_innovation_error_mps'],err)
      maxima['delta_position_error_m']=max(maxima['delta_position_error_m'],float(np.abs(saved['smooth_pos'][j,:n]-pp).max()))
      require(np.array_equal(saved['smooth_pos'][j,:idx],p[:idx]),'Before-onset position changed')
     else:
      for key,value in [('smooth_pos',p),('smooth_sog',s),('smooth_cog',c)]:require(np.array_equal(saved[key][j,:n],value),'Rejected generation did not return source')
     if acceptance:
      # The hard implementation is exercised using exactly the independently
      # verified applied smooth plan, source, onset and frozen noise seed.
      hs,hc,_,_=independent_profile(t,s,c,row.latent_kind,onset,applied,hard=True)
      hp=changed_position(t,p,s,c,hs,hc)
      hard=generate_maneuver(t,p,s,c,kind=row.latent_kind,onset_s=onset,params=applied,profile='hard_step',limits=limits,adapt=False)
      require(hard['accepted'] and not hard['metadata']['constraints_enforced'],'Hard comparison incorrectly uses smooth feasibility')
      require(hard['metadata']['applied_params']==applied and hard['metadata']['requested_onset_s']==onset,'Hard plan differs from smooth')
      for key,value in [('pos',hp),('sog_kn',hs),('cog_deg',hc)]:close(hard[key],value,'Hard semantic/integration mismatch: '+key)
      herr=float(np.linalg.norm(innovation(t,hp,hs,hc)-innovation(t,p,s,c),axis=1).max());require(herr<1e-7,'Hard source innovation not preserved')
      maxima['hard_innovation_error_mps']=max(maxima['hard_innovation_error_mps'],herr)
      counts['matched_hard_controls']+=1;regional['matched_hard_controls']+=1
     counts['candidates']+=1;counts['source_reports']+=n;counts['dense_profiles_checked']+=1;regional['source_reports']+=n
     which='accepted' if acceptance else 'rejected';counts[which]+=1;regional[which]+=1
     if not acceptance:regional['rejection_reasons'][reason]=regional['rejection_reasons'].get(reason,0)+1
    require(next(stream,None) is None,'Extra generation audit rows')
   report['regions'][region].update(regional,validation_seconds=time.time()-rs)
   print('INDEPENDENTLY_VERIFIED',region,json.dumps(regional),flush=True)
  for a,b in [('train','val'),('train','test'),('val','test')]:require(not global_observations[a]&global_observations[b],'Cross-split original target observations')
  require(counts['candidates']==len(candidates) or args.regions,'Not all planned candidates audited')
  require(sha(genpath)==FROZEN,'Frozen generator changed during audit')
  report.update(status='passed',counts=counts,maxima=maxima,rejected_rows_not_marked_accepted=True,
                original_source_reports_unchanged=True,all_dense_profiles_independently_checked=True,
                actual_dt_and_circular_course_used=True,matched_hard_same_source_onset_semantics=True,
                noise_plan_unchanged=True,noise_applied_at_this_stage=False,
                binary_labels_present=False,calibration_uses_train_only=True,
                all_planned_regions_checked=len(envelopes)==8,
                per_kind_candidate_counts=candidates.latent_kind.value_counts().to_dict())
 except Exception as err:
  report['status']='failed';report['errors'].append(f'{type(err).__name__}: {err}')
 report['elapsed_seconds']=time.time()-started
 output.write_text(json.dumps(report,indent=2))
 print(json.dumps({'status':report['status'],'report':str(output),'counts':report.get('counts'),'errors':report['errors']},indent=2),flush=True)
 return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
