"""Independent complete audit of labeled target, background and paired views.

Imports neither the feature builder nor its generator/context implementation.
Replays observations, changing-background fingerprints, causal candidate pools,
kinematic gates and the two affine estimators directly from frozen source data.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,time,sys
import numpy as np
import pandas as pd
from audit_math_v2 import (COLS,Background,close,stable,velocity,innovation,behavior,slowdown,
                        noise,noise_spec,replay_background,target_kin,verify_step)

BASE=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
LAB=BASE/'labeling/labeled_v1';SOURCE=BASE/'release_20261026'
VIEWS=['features_scene','features_strict','features_hard_scene','paired_group_scene']
SPLITS=['train','val','test']
EXPECTED_GEN='d05df588da58120e78493a4703b6f8d9109d993d344315a51f1bcba9ad996882'
PAIR_TARGET_KEYS=['raw','gated','mask','time_s','window_id','base_window_id','mmsi','y','type','target_pos','target_sog_kn','target_cog_deg','pre_noise_pos','pre_noise_sog_kn','pre_noise_cog_deg','e_kin_mps','self_reliability','self_gate_weight','kinematic_valid','position_increment_valid','time_interval_valid']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def bounds(meta,start,end,lon0,lat0):
 days=pd.date_range(pd.Timestamp(start-600,unit='s',tz='UTC').floor('D'),pd.Timestamp(end,unit='s',tz='UTC').floor('D'),freq='D')
 c=meta['coverage_bbox_by_utc_day']
 if any(d.strftime('%Y-%m-%d') not in c for d in days):return None
 a=np.asarray([c[d.strftime('%Y-%m-%d')] for d in days]);b=[a[:,0].max(),a[:,1].max(),a[:,2].min(),a[:,3].min()]
 if b[0]>=b[2] or b[1]>=b[3]:return None
 return np.array([(b[0]-lon0)*111195*np.cos(np.deg2rad(lat0)),(b[1]-lat0)*111195,(b[2]-lon0)*111195*np.cos(np.deg2rad(lat0)),(b[3]-lat0)*111195])
def physical(t,s,c):
 dt=np.diff(t);rot=np.abs((np.diff(c)+180)%360-180)/dt
 return {'max_longitudinal_accel_mps2':float((np.abs(np.diff(s))*1852/3600/dt).max()),'max_turn_rate_deg_s':float(rot.max()),'max_lateral_accel_mps2':float(((s[1:]+s[:-1])/2*1852/3600*np.deg2rad(rot)).max()),'max_speed_kn':float(s.max()),'min_speed_kn':float(s.min())}
def compare_dict(actual,expected,label,atol=1e-9):
 for k,v in expected.items():
  assert k in actual,(label,k)
  if isinstance(v,dict):compare_dict(actual[k],v,label+'/'+k,atol)
  elif isinstance(v,(float,np.floating)):close(actual[k],v,label+'/'+k,atol=atol,rtol=1e-9)
  else:assert actual[k]==v,(label,k,actual[k],v)

def audit_region(region,ledger,registry,groups,targets,args):
 started=time.time();meta=registry[region];box=meta['core_bbox'];lon0=(box[0]+box[2])/2;lat0=(box[1]+box[3])/2
 labelhash=sha(LAB/'label_manifest.parquet');source_path=Path(meta['canonical_file']);assert sha(source_path)==meta['sha256']
 canonical=pd.read_parquet(source_path,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
 raw={'mmsi':canonical.mmsi.to_numpy().astype(str),'time_s':canonical.timestamp.to_numpy(np.float64),'x_m':(canonical.lon.to_numpy()-lon0)*111195*np.cos(np.deg2rad(lat0)),'y_m':(canonical.lat.to_numpy()-lat0)*111195,'sog_kn':canonical.sog.to_numpy(np.float64),'cog_deg':canonical.cog.to_numpy(np.float64)}
 allids=set(groups.mmsi.astype(str));groupmap=dict(zip(groups.mmsi.astype(str),groups.split))
 allowed_scene={'train':allids-targets['val']-targets['test'],'val':allids-targets['test'],'test':allids}
 allowed_strict={s:set(groups.loc[groups.split==s,'mmsi'].astype(str)) for s in SPLITS}
 counter_dir=BASE/'labeling/counterfactuals_v1'/region;cf_path=counter_dir/'counterfactual_observations.npz';counter=np.load(cf_path,allow_pickle=False)
 report={'status':'running','region':region,'independent_implementation':True,'imports_generator_or_builder':False,'views':{},'source_hashes':{'canonical':sha(source_path),'labels':labelhash,'counterfactual_observations':sha(cf_path),'global_group_assignment':sha(SOURCE/'global_all_mmsi_split.parquet'),'source_registry':sha(SOURCE/'source_registry.json')},'strict_actual_identity_sets':{},'background_fingerprints_recomputed':0,'background_changes_verified':0,'raw_target_steps_verified':0,'neighbor_reports_verified':0,'max_pre_noise_innovation_difference_mps':0.,'errors':[]}
 bases={};background_cache={};peers={};all_strict={s:set() for s in SPLITS}
 for view in args.views:
  dest=LAB/view/region;fr_path=dest/'feature_report.json';fr=json.loads(fr_path.read_text());cfg=fr['context_config'];proof=fr['source_proof']
  assert fr['status']=='completed' and fr['region']==region and fr['view']==view
  expectedproof={'canonical_sha256':meta['sha256'],'counterfactual_plan_sha256':sha(counter_dir/'counterfactual_plan.parquet'),'counterfactual_observations_sha256':sha(cf_path),'counterfactual_report_sha256':sha(counter_dir/'counterfactual_report.json'),'generator_sha256':sha(BASE/'labeling/generator/maneuver_generator.py'),'context_module_sha256':sha(BASE/'context/context_features.py'),'labels_sha256':labelhash,'source_registry_sha256':sha(SOURCE/'source_registry.json'),'global_all_mmsi_split_sha256':sha(SOURCE/'global_all_mmsi_split.parquet'),'builder_sha256':sha(BASE/'labeling/build_labeled_features.py')}
  assert proof['generator_sha256']==EXPECTED_GEN
  for k,v in expectedproof.items():assert proof[k]==v,(region,view,k)
  result={};view_exposure={}
  for split in (['test'] if view=='paired_group_scene' else SPLITS):
   sub=ledger[ledger.split==split].sort_values(['mmsi','start_utc_s','window_id'],kind='stable').reset_index(drop=True)
   arrays_path=dest/f'{split}.npz';audit_path=dest/f'{split}_context_audit.jsonl.gz'
   assert sha(arrays_path)==fr['output_hashes'][split]['npz_sha256'] and sha(audit_path)==fr['output_hashes'][split]['audit_sha256']
   with np.load(arrays_path,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
   nrows=len(sub);assert len(arrays['y'])==nrows
   for arr,col in [('window_id','window_id'),('base_window_id','base_window_id'),('mmsi','mmsi'),('y','y'),('type','type'),('inttype','type'),('behavior_type','behavior_type'),('noise_type','noise_kind'),('noise_scope','noise_scope'),('observation_noise','observation_noise')]:assert np.array_equal(arrays[arr],sub[col].to_numpy()),(region,view,split,arr)
   assert np.all(arrays['region']==region) and np.all(arrays['group_normal_variation']==(view=='paired_group_scene'))
   assert arrays['time_s'].dtype==np.float64
   for name in ['target_pos','target_sog_kn','target_cog_deg','pre_noise_pos','pre_noise_sog_kn','pre_noise_cog_deg']:assert arrays[name].dtype==np.float64,name
   for name,v in arrays.items():
    if v.dtype.kind=='f':assert np.isfinite(v).all(),name
    if v.ndim>=2 and v.shape[1]==20:assert np.all(v[~arrays['mask']]==0),(region,view,split,name,'nonzero padding')
   peerk=(split,'features_scene')
   if view=='features_strict':
    peer=peers[peerk]
    for k in PAIR_TARGET_KEYS:assert np.array_equal(arrays[k],peer[k]),(region,split,k,'scene/strict target mismatch')
   if view=='features_hard_scene':
    peer=peers[peerk];sham=arrays['y']==0
    for k in PAIR_TARGET_KEYS+['context','plain_context','plain_affine_context']:
     assert np.array_equal(arrays[k][sham],peer[k][sham]),(region,split,k,'hard/smooth sham mismatch')
   allowed=allowed_strict[split] if view=='features_strict' else allowed_scene[split]
   basekey=('strict' if view=='features_strict' else 'scene',split)
   if basekey not in bases:
    use=np.isin(raw['mmsi'],np.array(sorted(allowed)));bases[basekey]=Background({k:v[use] for k,v in raw.items()},cfg)
   base=bases[basekey];exposure=set(sub.mmsi.astype(str));candidate_edges=0;changedsnapshots=0;group_members=0;changedreportsread=0;numericbg=0;group_noop_members=0;group_noop_targets=0
   with gzip.open(audit_path,'rt',encoding='utf-8') as af:
    for j,row in enumerate(sub.itertuples()):
     line=next(af,None);assert line is not None,'Missing context audit row';md=json.loads(line)
     assert md['window_id']==row.window_id and md['base_window_id']==row.base_window_id and md['y']==row.y and md['type']==row.type and md['view']==view and md['split']==split
     assert md['original_source_mmsi']==str(row.mmsi) and md['counterfactual_row']==row.counterfactual_row and groupmap[str(row.mmsi)]==split
     n=int(row.n_obs);mask=np.arange(20)<n;assert np.array_equal(arrays['mask'][j],mask)
     ci=int(row.counterfactual_row);t=counter['t'][ci,:n];p=counter['original_pos'][ci,:n];s=counter['original_sog'][ci,:n];c=counter['original_cog'][ci,:n]
     assert counter['window_id'][ci]==row.window_id and np.array_equal(t,json.loads(row.timestamps)) and np.array_equal(arrays['time_s'][j,:n],t)
     assert t[-1]==row.end_utc_s==md['decision_time_s'] and np.all(t<=row.end_utc_s)
     params=json.loads(row.counterfactual_applied_params);kind=row.latent_kind if row.y else 'none'
     pre_p,pre_s,pre_c=behavior(t,p,s,c,kind,row.onset_utc_s,params,hard=view=='features_hard_scene')
     bgmd=md['behavior_generation'];assert bgmd['kind']==kind
     if row.y:
      assert bgmd['profile']==('hard_step' if view=='features_hard_scene' else 'smooth') and bgmd['applied_params']==params
      if view!='features_hard_scene':
       close(pre_p,counter['smooth_pos'][ci,:n],'frozen smooth position',atol=1e-8,rtol=1e-12);close(pre_s,counter['smooth_sog'][ci,:n],'frozen smooth speed',atol=1e-10,rtol=1e-12);close(pre_c,counter['smooth_cog'][ci,:n],'frozen smooth course',atol=1e-10,rtol=1e-12)
     else:assert bgmd['profile']=='identity' and bgmd['is_sham']
     spec=noise_spec(t,row);compare_dict(md['requested_observation_noise'],spec,'noise plan')
     cachekey=(basekey,row.window_id,view=='paired_group_scene')
     core_noise={k:md['neighbor_observation_noise'].get(k) for k in ['requested','candidate_mmsi','selected_mmsi','changes','effective']}
     if cachekey in background_cache:
      overlay,bga=background_cache[cachekey];assert md['background_fingerprint']==bga['expected_fingerprint']
      assert bga['noise_plan_core']==core_noise,'Paired hard/smooth background-noise plans differ'
     else:
      overlay,bga=replay_background(base,t,p,s,c,row,spec,view=='paired_group_scene',md)
      bga['expected_fingerprint']=md['background_fingerprint'];bga['noise_plan_core']=core_noise;background_cache[cachekey]=(overlay,bga)
     if bga['fingerprint_recomputed']:changedsnapshots+=1
     numericbg+=int(bga['changed_background']);group_noop_members+=len(bga.get('group_operator_only_members',[]))
     if view=='paired_group_scene':
      gm=md['group_variation'];assert gm['requested']
      old_group_p=pre_p.copy();old_group_s=pre_s.copy();old_group_c=pre_c.copy()
      pre_p,pre_s,pre_c=slowdown(t,pre_p,pre_s,pre_c,gm['onset_s'],gm['factor'],gm['duration_s'])
      compare_dict(gm['target'],{'onset_s':gm['onset_s'],'factor':gm['factor'],'duration_s':gm['duration_s']},'shared target slowdown')
      assert gm['target']['changed'],'Common group operator was not activated for target'
      actual_group_target=not(np.array_equal(pre_p,old_group_p) and np.array_equal(pre_s,old_group_s) and np.array_equal(pre_c,old_group_c))
      group_noop_targets+=int(not actual_group_target)
      group_members+=len(gm['controlled_mmsi'])
     close(arrays['pre_noise_pos'][j,:n],pre_p,'pre-noise target position',atol=1e-8,rtol=1e-12)
     close(arrays['pre_noise_sog_kn'][j,:n],pre_s,'pre-noise target speed',atol=1e-10,rtol=1e-12)
     close(arrays['pre_noise_cog_deg'][j,:n],pre_c,'pre-noise target course',atol=1e-10,rtol=1e-12)
     preservation=float(np.abs(innovation(t,pre_p,pre_s,pre_c)-innovation(t,p,s,c)).max());assert preservation<1e-8
     close(md['pre_noise_innovation_preservation_max_mps'],preservation,'pre-noise innovation diagnostic',atol=1e-8,rtol=0)
     report['max_pre_noise_innovation_difference_mps']=max(report['max_pre_noise_innovation_difference_mps'],preservation)
     target_kind=spec['kind'] if spec['target_requested'] else 'none';post_p,post_s,post_c=noise(t,pre_p,pre_s,pre_c,target_kind,spec['target_parameters'])
     close(arrays['target_pos'][j,:n],post_p,'post-noise target position',atol=1e-8,rtol=1e-12);close(arrays['target_sog_kn'][j,:n],post_s,'post-noise target speed',atol=1e-10,rtol=1e-12);close(arrays['target_cog_deg'][j,:n],post_c,'post-noise target course',atol=1e-10,rtol=1e-12)
     compare_dict(md['physical_before_observation_noise'],physical(t,pre_s,pre_c),'physical before noise')
     compare_dict(md['physical_after_observation_noise'],physical(t,post_s,post_c),'physical after noise')
     effective=not(np.array_equal(pre_p,post_p) and np.array_equal(pre_s,post_s) and np.array_equal(pre_c,post_c))
     assert md['target_observation_noise']['observations_changed']==effective
     assert md['effective_observation_noise']==(effective or bool(bga['changed_noise_pairs']))
     kin=target_kin(t,post_p,post_s,post_c,cfg)
     for k in ['raw','gated','e_kin_mps','self_reliability','self_gate_weight','kinematic_valid','time_interval_valid','position_increment_valid']:close(arrays[k][j,:n],kin[k],k)
     bbox=bounds(meta,row.start_utc_s,row.end_utc_s,lon0,lat0);assert np.all(arrays['coverage_known'][j,:n]==(bbox is not None))
     bmask=np.ones(n,bool) if bbox is None else ((post_p[:,0]-cfg['radius_nm']*1852>=bbox[0])&(post_p[:,0]+cfg['radius_nm']*1852<=bbox[2])&(post_p[:,1]-cfg['radius_nm']*1852>=bbox[1])&(post_p[:,1]+cfg['radius_nm']*1852<=bbox[3]))
     assert np.array_equal(arrays['boundary_context_valid'][j,:n],bmask)
     assert len(md['steps'])==n;accepted=set();plain=set();pairs=set()
     for k,step in enumerate(md['steps']):
      assert step['step']==k and step['time_s']==t[k];step['target_identity']=str(row.mmsi)
      ids,ps,aa,bb=verify_step(base,overlay,step,t[k],post_p[k],post_s[k],post_c[k],kin['velocity'][k],kin['self_gate_weight'][k],kin['self_reliability'][k],kin['e_kin_mps'][k],kin['kinematic_valid'][k],bmask[k],kin['raw'][k],kin['gated'][k],cfg,arrays,j,k)
      assert ids<=allowed and str(row.mmsi) not in ids
      exposure.update(ids);pairs.update(ps);accepted.update(aa);plain.update(bb);candidate_edges+=len(ids)
     assert set(md['context_exposure_mmsi'])==accepted and set(md['plain_context_exposure_mmsi'])==plain
     actual_changed=pairs&bga['changed_noise_pairs'];declared_changed={(str(x['mmsi']),float(x['time_s'])) for x in md['neighbor_observation_noise']['changed_reports_read_in_context_audit']}
     assert actual_changed==declared_changed;changedreportsread+=len(actual_changed)
     report['raw_target_steps_verified']+=n
    assert next(af,None) is None,'Extra audit windows'
   view_exposure[split]=exposure
   if view=='features_strict':all_strict[split].update(exposure)
   result[split]={'windows':nrows,'target_vessels':int(sub.mmsi.nunique()),'positive':int(sub.y.sum()),'candidate_neighbor_reports_checked':candidate_edges,'background_fingerprints_recomputed':changedsnapshots,'numerically_changed_backgrounds_verified':numericbg,'changed_noise_reports_actually_read':changedreportsread,'group_operator_active_neighbor_memberships':group_members,'group_operator_only_no_numeric_change_members':group_noop_members,'group_operator_only_no_numeric_change_targets':group_noop_targets,'accessed_identity_count':len(exposure),'npz_sha256':sha(arrays_path),'audit_sha256':sha(audit_path)}
   assert fr['splits'][split]['audited_candidate_edges']==candidate_edges
   assert set(fr['all_accessed_identity_sets'][split])==exposure
   report['neighbor_reports_verified']+=candidate_edges;report['background_fingerprints_recomputed']+=changedsnapshots
   if view=='features_scene':peers[peerk]=arrays
   print('AUDITED_LABELED_SPLIT',region,view,split,nrows,candidate_edges,flush=True)
  report['views'][view]={'splits':result,'actual_cross_split_identity_intersections':{a+'__'+b:len(view_exposure[a]&view_exposure[b]) for a,b in [('train','val'),('train','test'),('val','test')] if a in view_exposure and b in view_exposure}}
 for a,b in [('train','val'),('train','test'),('val','test')]:assert not all_strict[a]&all_strict[b]
 report['strict_actual_identity_sets']={s:sorted(v) for s,v in all_strict.items()}
 report['status']='passed';report['elapsed_seconds']=time.time()-started
 return report

def selftest():
 cfg={'max_kinematic_gap_seconds':300,'gate_deadzone_mps':2.5,'gate_denominator_mps2':8,'radius_nm':8,'velocity_scale_mps':5*1852/3600,'window_seconds':1380}
 t=np.array([0.,70.,170.,560.,660.]);s=np.full(5,9.);c=np.array([359.,.5,1.,2.,2.5]);v=velocity(s,c);p=np.zeros((5,2));p[1:]=np.cumsum((v[:-1]+v[1:])/2*np.diff(t)[:,None],axis=0);p[1:]+=np.array([[1.,2.],[3.,1.],[2.,-1.],[5.,4.]])
 pp,ss,cc=behavior(t,p,s,c,'turn_back',80.,{'sign':-1,'delta_deg':150.,'duration_s':400.})
 close(innovation(t,pp,ss,cc),innovation(t,p,s,c),'selftest innovation',atol=1e-10,rtol=0)
 noise_p,noise_s,noise_c=noise(t,pp,ss,cc,'teleport',{'report_index':2,'offset_m':900.,'angle_deg':90.})
 assert not np.array_equal(noise_p,pp) and np.array_equal(noise_s,ss)
 kin=target_kin(t,noise_p,noise_s,noise_c,cfg);assert not kin['kinematic_valid'][3] and kin['self_gate_weight'][3]==1 and kin['raw'][3,4]>0
 assert np.all(kin['raw'][0,:2]==0) and not np.all(kin['raw'][0,2:4]==0)
 bad=kin['raw'].copy();bad[1,2]+=1
 try:close(bad,kin['raw'],'tamper');raise RuntimeError('Tampered raw was not rejected')
 except AssertionError:pass
 print('SELFTEST_PASSED innovation, noise, angle wrap, missing gap, neutral gate, tamper rejection',flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--regions',nargs='*');ap.add_argument('--views',nargs='+',default=VIEWS,choices=VIEWS);ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
 out=BASE/'labeling/validation';out.mkdir(parents=True,exist_ok=True)
 if args.self_test:selftest();return
 assert args.views[0]=='features_scene','Paired audits require main scene first'
 ledger=pd.read_parquet(LAB/'label_manifest.parquet');registry=json.loads((SOURCE/'source_registry.json').read_text())['regions'];groups=pd.read_parquet(SOURCE/'global_all_mmsi_split.parquet')
 original=pd.read_parquet(SOURCE/'selected_windows.parquet',columns=['mmsi','split']);targets={s:set(original.loc[original.split==s,'mmsi'].astype(str)) for s in SPLITS}
 for region,rows in ledger.groupby('region',sort=True):
  if args.regions and region not in args.regions:continue
  try:r=audit_region(region,rows,registry,groups,targets,args)
  except Exception as e:
   r={'status':'failed','region':region,'error':f'{type(e).__name__}: {e}'};(out/f'{region}_feature_audit.json').write_text(json.dumps(r,indent=2));raise
  r['verifier_sha256']=sha(Path(__file__));r['numerical_replay_module_sha256']=sha(Path(__file__).with_name('audit_math_v2.py'))
  (out/f'{region}_feature_audit.json').write_text(json.dumps(r,indent=2));print('REGION_LABELED_FEATURES_PASSED',region,r['neighbor_reports_verified'],flush=True)
 regions=sorted(ledger.region.unique());reports=[json.loads((out/f'{region}_feature_audit.json').read_text()) for region in regions if (out/f'{region}_feature_audit.json').exists()]
 if len(reports)==8 and all(r['status']=='passed' and set(r['views'])==set(VIEWS) for r in reports):
  exposure={s:set().union(*(set(r['strict_actual_identity_sets'][s]) for r in reports)) for s in SPLITS};cross={a+'__'+b:len(exposure[a]&exposure[b]) for a,b in [('train','val'),('train','test'),('val','test')]};assert not any(cross.values())
  final={'status':'passed','regions':8,'independent_source_windows':11200,'views_checked':VIEWS,'labels_sha256':sha(LAB/'label_manifest.parquet'),'raw_target_steps_recomputed_across_views':sum(r['raw_target_steps_verified'] for r in reports),'actual_neighbor_reports_checked':sum(r['neighbor_reports_verified'] for r in reports),'changed_background_fingerprints_recomputed':sum(r['background_fingerprints_recomputed'] for r in reports),'strict_all_accessed_identity_intersections':cross,'region_reports':{r['region']:sha(out/f'{r["region"]}_feature_audit.json') for r in reports},'imports_generator_or_builder':False}
  (LAB/'independent_labeled_features_audit.json').write_text(json.dumps(final,indent=2));print(json.dumps(final,indent=2),flush=True)

if __name__=='__main__':main()
