"""Build declared labeled views from a frozen counterfactual/label ledger.

No labels are inferred here. Noise decisions depend on the ledger and original
source observation timestamps, never on behavior labels or neighbor availability.
Every changed background is indexed again before any context feature is used.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,sys,time
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
SOURCE=ROOT/'release_20261026'
sys.path.insert(0,str(ROOT/'context'))
sys.path.insert(0,str(BASE/'generator'))
sys.path.insert(0,str(ROOT))
from context_features import BackgroundIndex,ContextConfig,build_window
from maneuver_generator import (apply_observation_noise,generate_maneuver,
                                innovation_vectors,physical_statistics,
                                velocity_vectors,PhysicalLimits)
from build_features import projected,coverage

VIEWS=('features_scene','features_strict','features_hard_scene','paired_group_scene')
SPLITS=('train','val','test')
EXPECTED_GENERATOR='d05df588da58120e78493a4703b6f8d9109d993d344315a51f1bcba9ad996882'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def stable_seed(seed,tag):
 return int.from_bytes(hashlib.sha256(f'{int(seed)}:{tag}'.encode()).digest()[:8],'little')

def copy_observations(d):return {k:np.asarray(v).copy() for k,v in d.items()}

def target_dict(t,pos,sog,cog,mmsi):
 return {'mmsi':np.repeat(str(mmsi),len(t)),'time_s':np.asarray(t,float).copy(),
         'x_m':np.asarray(pos[:,0],float).copy(),'y_m':np.asarray(pos[:,1],float).copy(),
         'sog_kn':np.asarray(sog,float).copy(),'cog_deg':np.asarray(cog,float).copy()}

def as_generator(d):
 return d['time_s'],np.column_stack((d['x_m'],d['y_m'])),d['sog_kn'],d['cog_deg']

def noise_spec(source,row):
 """One target/noise query plan shared by all feature protocols and profiles."""
 requested=bool(row.observation_noise);kind=str(row.noise_kind);scope=str(row.noise_scope)
 if kind not in ('none','teleport','speed_spike','drift'):raise ValueError(f'Unknown finalized noise {kind}')
 if scope not in ('none','target_only','neighbors_only','both'):raise ValueError(f'Unknown noise scope {scope}')
 if requested!=(kind!='none') or requested!=(scope!='none'):raise ValueError('Noise ledger inconsistent')
 rng=np.random.default_rng(stable_seed(row.noise_seed,'query_and_target_noise'))
 t=source['time_s'];ix=int(rng.integers(max(1,len(t)//2),len(t)))
 query=float(t[ix]);params={}
 if kind=='teleport':params={'report_index':ix,'offset_m':float(rng.uniform(500,1500)),'angle_deg':float(rng.uniform(0,360))}
 elif kind=='speed_spike':params={'report_index':ix,'reports':max(1,int(np.sum((t>=query)&(t<query+120)))),'factor':float(rng.uniform(3,5))}
 elif kind=='drift':params={'onset_s':query-150,'duration_s':300.,'offset_m':float(rng.uniform(500,1500)),'angle_deg':float(rng.uniform(0,360))}
 return {'requested':requested,'kind':kind,'scope':scope,'query_step':ix,'query_time_s':query,
         'seed':int(row.noise_seed),'target_parameters':params,
         'target_requested':requested and scope in ('target_only','both'),
         'neighbors_requested':requested and scope in ('neighbors_only','both')}

def apply_target_noise(target,spec):
 if not spec['target_requested']:
  return copy_observations(target),{'noise':'none','requested_for_target':False,'observations_changed':False,'parameters':{}}
 out=apply_observation_noise(*as_generator(target),noise=spec['kind'],seed=stable_seed(spec['seed'],'target_api'),params=spec['target_parameters'])
 after=target_dict(out['t'],out['pos'],out['sog_kn'],out['cog_deg'],target['mmsi'][0])
 metadata=dict(out['metadata'])
 metadata['observations_changed']=any(not np.array_equal(after[k],target[k]) for k in ['x_m','y_m','sog_kn','cog_deg'])
 metadata['requested_for_target']=True
 return after,metadata

def original_candidates(index,source,query_step,cfg):
 """Candidates use ONLY original target location/course and allowed source BG."""
 i=int(query_step);when=float(source['time_s'][i]);identity=str(source['mmsi'][0])
 ids=index.latest(when,identity,None)
 if not len(ids):return []
 x=index.data['x_m'][ids];y=index.data['y_m'][ids];s=index.data['sog_kn'][ids];c=index.data['cog_deg'][ids]
 distance=np.hypot(x-source['x_m'][i],y-source['y_m'][i])
 angle=np.abs((c-source['cog_deg'][i]+180)%360-180)
 valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(s)&np.isfinite(c)&(s>=cfg.min_heading_speed_kn)&(s<102.2)&(c>=0)&(c<360)&(distance<=cfg.radius_nm*1852)&(angle<=cfg.heading_limit_deg)
 return sorted([{'mmsi':str(index.data['mmsi'][j]),'time_s':float(index.data['time_s'][j]),
                 'query_time_s':when,'distance_m':float(di)} for j,di in zip(ids[valid],distance[valid])],key=lambda z:z['mmsi'])

def finite_ship_indices(local,identity):
 valid=(local['mmsi']==identity)
 for k in ['time_s','x_m','y_m','sog_kn','cog_deg']:valid&=np.isfinite(local[k])
 valid&=(local['sog_kn']>=0)&(local['sog_kn']<102.2)&(local['cog_deg']>=0)&(local['cog_deg']<360)
 return np.flatnonzero(valid)

def shared_deceleration(target,onset,factor,duration=300.):
 t,p,s,c=as_generator(target);u=np.clip((t-onset)/duration,0,1)
 multiplier=1-(1-factor)*(u*u*(3-2*u));s2=s*multiplier
 delta=velocity_vectors(s2,c)-velocity_vectors(s,c);offset=np.zeros_like(p)
 if len(t)>1:offset[1:]=np.cumsum(.5*(delta[1:]+delta[:-1])*np.diff(t)[:,None],axis=0)
 result=target_dict(t,p+offset,s2,c,target['mmsi'][0])
 preservation=float(np.abs(innovation_vectors(*as_generator(result))-innovation_vectors(t,p,s,c)).max()) if len(t)>1 else 0.
 return result,{'onset_s':float(onset),'factor':float(factor),'duration_s':float(duration),
                'innovation_preservation_max_mps':preservation,'observed_physical':physical_statistics(t,s2,c),
                'changed':bool(np.any(multiplier!=1))}

def augment_background(index,source,spec,row,cfg,group=False):
 """Return original immutable index or a freshly rebuilt causal local index."""
 group_meta={'requested':bool(group),'controlled_mmsi':[],'members':[]}
 noise_meta={'requested':spec['neighbors_requested'],'candidate_selection':'original target position/heading at preselected source query step; protocol-eligible source identities only',
             'candidate_mmsi':[],'selected_mmsi':[],'changes':[],'effective':False,'no_target_fallback':True}
 if not group and not spec['neighbors_requested']:return index,group_meta,noise_meta
 local=index.window_observations(float(source['time_s'][-1]));changed=False
 if group:
  rng=np.random.default_rng(stable_seed(row.group_seed,'shared_group_slowdown'));factor=float(rng.uniform(.65,.85));onset=float(row.onset_utc_s)
  reference=max(0,int(np.searchsorted(source['time_s'],onset,side='right'))-1)
  candidates=original_candidates(index,source,reference,cfg)
  group_meta.update(factor=factor,onset_s=onset,duration_s=300.,reference_source_step=reference,
                    semantics='Same smooth slowdown applied to target and source-local co-directional neighbors; behavior labels preserved')
  for entry in candidates:
   identity=entry['mmsi'];ii=finite_ship_indices(local,identity)
   if len(ii)<2:
    group_meta['members'].append({'mmsi':identity,'effective':False,'reason':'insufficient_finite_history'});continue
   d={k:v[ii] for k,v in local.items()};aug,md=shared_deceleration(d,onset,factor)
   for k in ['x_m','y_m','sog_kn','cog_deg']:local[k][ii]=aug[k]
   times=d['time_s'][d['time_s']>=onset].tolist();group_meta['members'].append({'mmsi':identity,'effective':md['changed'],'affected_time_s':times,**md})
   if md['changed']:group_meta['controlled_mmsi'].append(identity);changed=True
 if spec['neighbors_requested']:
  candidates=original_candidates(index,source,spec['query_step'],cfg)
  noise_meta['candidate_mmsi']=[c['mmsi'] for c in candidates]
  rng=np.random.default_rng(stable_seed(spec['seed'],'neighbor_selection'))
  count=min(len(candidates),max(1,int(np.ceil(.15*len(candidates))))) if candidates else 0
  chosen=sorted(rng.permutation(len(candidates))[:count].tolist())
  for which in chosen:
   entry=candidates[which];identity=entry['mmsi'];noise_meta['selected_mmsi'].append(identity)
   ii=finite_ship_indices(local,identity)
   if len(ii)<2:
    noise_meta['changes'].append({'mmsi':identity,'effective':False,'reason':'insufficient_finite_history'});continue
   before={k:v[ii].copy() for k,v in local.items()}
   anchor=int(np.searchsorted(before['time_s'],entry['time_s']))
   if anchor>=len(ii) or before['time_s'][anchor]!=entry['time_s']:
    noise_meta['changes'].append({'mmsi':identity,'effective':False,'reason':'original_query_report_unavailable'});continue
   nrng=np.random.default_rng(stable_seed(spec['seed'],'neighbor:'+identity));params={}
   if spec['kind']=='teleport':params={'report_index':anchor,'offset_m':float(nrng.uniform(500,1500)),'angle_deg':float(nrng.uniform(0,360))}
   elif spec['kind']=='speed_spike':
    params={'report_index':anchor,'reports':max(1,int(np.sum((before['time_s']>=entry['time_s'])&(before['time_s']<entry['time_s']+120)))),'factor':float(nrng.uniform(3,5))}
   else:params={'onset_s':entry['time_s']-150,'duration_s':300.,'offset_m':float(nrng.uniform(500,1500)),'angle_deg':float(nrng.uniform(0,360))}
   aug=apply_observation_noise(*as_generator(before),noise=spec['kind'],seed=stable_seed(spec['seed'],'neighbor_api:'+identity),params=params)
   after=target_dict(aug['t'],aug['pos'],aug['sog_kn'],aug['cog_deg'],identity)
   report_changed=np.zeros(len(ii),bool)
   for k in ['x_m','y_m','sog_kn','cog_deg']:
    report_changed|=before[k]!=after[k];local[k][ii]=after[k]
   actual_times=before['time_s'][report_changed].tolist();effective=bool(report_changed.any())
   noise_meta['changes'].append({'mmsi':identity,'effective':effective,'original_latest_query_report_s':entry['time_s'],
                                'target_query_time_s':spec['query_time_s'],'affected_time_s':actual_times,
                                'parameters':aug['metadata']['parameters'],'noise':spec['kind']})
   changed|=effective;noise_meta['effective']|=effective
 group_meta['source_background_fingerprint']=index.fingerprint
 if not changed:return index,group_meta,noise_meta
 rebuilt=BackgroundIndex(local,cfg)
 group_meta['augmented_background_fingerprint']=rebuilt.fingerprint
 assert rebuilt.fingerprint!=index.fingerprint
 return rebuilt,group_meta,noise_meta

def latent_target(z,row,view,limits):
 j=int(row.counterfactual_row);mask=z['mask'][j];n=int(mask.sum())
 assert np.array_equal(mask,np.arange(20)<n) and n==int(row.n_obs)
 assert str(z['window_id'][j])==str(row.window_id)
 t=z['t'][j,:n].copy();identity=str(row.mmsi)
 original=target_dict(t,z['original_pos'][j,:n],z['original_sog'][j,:n],z['original_cog'][j,:n],identity)
 assert np.array_equal(t,json.loads(row.timestamps))
 if int(row.y)==0:
  return original,copy_observations(original),{'kind':'none','profile':'identity','behavior_injected':False,'is_sham':True,'innovation_preservation_max_mps':0.}
 if view=='features_hard_scene':
  hard=generate_maneuver(*as_generator(original),kind=str(row.latent_kind),onset_s=float(row.onset_utc_s),
                        params=json.loads(row.counterfactual_applied_params),profile='hard_step',limits=limits,adapt=False)
  if not hard['accepted']:raise RuntimeError(f'Matched hard control failed: {row.window_id} {hard["reason"]}')
  assert hard['metadata']['applied_params']==json.loads(row.counterfactual_applied_params)
  target=target_dict(hard['t'],hard['pos'],hard['sog_kn'],hard['cog_deg'],identity);md=hard['metadata']
 else:
  target=target_dict(t,z['smooth_pos'][j,:n],z['smooth_sog'][j,:n],z['smooth_cog'][j,:n],identity)
  md={'kind':str(row.latent_kind),'profile':'smooth','applied_params':json.loads(row.counterfactual_applied_params),
      'behavior_injected':True,'realized':json.loads(row.realized),'innovation_preservation_max_mps':float(row.innovation_preservation_max_mps)}
 error=float(np.abs(innovation_vectors(*as_generator(target))-innovation_vectors(*as_generator(original))).max())
 assert error<1e-8
 md['checked_innovation_preservation_max_mps']=error
 return original,target,md

def build_region(region,ledger,registry,groups,original_targets,args,cfg):
 meta=registry[region];source_path=Path(meta['canonical_file']);assert sha(source_path)==meta['sha256']
 report_path=args.counterfactuals/region/'counterfactual_report.json';counter_report=json.loads(report_path.read_text())
 assert counter_report['generator_sha256']==EXPECTED_GENERATOR==sha(BASE/'generator/maneuver_generator.py')
 observations_path=args.counterfactuals/region/'counterfactual_observations.npz';plan_path=args.counterfactuals/region/'counterfactual_plan.parquet'
 assert sha(observations_path)==ledger.counterfactual_observations_sha256.iloc[0]
 assert sha(plan_path)==ledger.counterfactual_plan_sha256.iloc[0]
 # Materialize each compressed array ONCE; NpzFile.__getitem__ otherwise
 # decompresses the whole region array every time a tiny window is requested.
 with np.load(observations_path,allow_pickle=False) as loaded:z={k:loaded[k] for k in loaded.files}
 limits=PhysicalLimits(**counter_report['physical_limits'])
 d=pd.read_parquet(source_path,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
 box=meta['core_bbox'];lon0=(box[0]+box[2])/2;lat0=(box[1]+box[3])/2;data=projected(d,lon0,lat0)
 all_ids=set(groups.mmsi.astype(str));group_map=dict(zip(groups.mmsi.astype(str),groups.split))
 eligible_scene={'train':all_ids-original_targets['val']-original_targets['test'],
                 'val':all_ids-original_targets['test'],'test':all_ids}
 eligible_strict={s:set(groups.loc[groups.split==s,'mmsi'].astype(str)) for s in SPLITS}
 source_proof={'canonical_file':str(source_path),'canonical_sha256':sha(source_path),
               'counterfactual_plan_sha256':sha(plan_path),'counterfactual_observations_sha256':sha(observations_path),
               'counterfactual_report_sha256':sha(report_path),'generator_sha256':EXPECTED_GENERATOR,
               'context_module_sha256':sha(ROOT/'context/context_features.py'),
               'labels_sha256':sha(args.labels),'source_registry_sha256':sha(args.source/'source_registry.json'),
               'global_all_mmsi_split_sha256':sha(args.source/'global_all_mmsi_split.parquet'),
               'builder_sha256':sha(Path(__file__))}
 index_cache={};reports={}
 for view in args.views:
  dest=args.out/view/region;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'feature_report.json').exists():raise RuntimeError(f'Refusing completed output overwrite {dest}')
  started=time.time();splits=['test'] if view=='paired_group_scene' else SPLITS
  split_reports={};identity_sets={};output_hashes={}
  for split in splits:
   sub=ledger[ledger.split==split].sort_values(['mmsi','start_utc_s','window_id'],kind='stable').reset_index(drop=True)
   allowed=eligible_strict[split] if view=='features_strict' else eligible_scene[split]
   key=('strict' if view=='features_strict' else 'scene',split)
   if key not in index_cache:
    use=np.isin(data['mmsi'],np.array(sorted(allowed)));index_cache[key]=BackgroundIndex({k:v[use] for k,v in data.items()},cfg)
   index=index_cache[key];accum={};accessed=set();candidate_edges=0;effective_noise=0;effective_neighbors=0;read_changed=0;target_noise_count=0;group_members=0
   window_flags={'observation_noise_effective':[],'target_noise_effective':[],'neighbor_noise_effective':[]}
   physical_pre=[];physical_post=[];max_pre_innovation_error=0.
   audit_path=dest/f'{split}_context_audit.jsonl.gz'
   with gzip.open(audit_path,'wt',encoding='utf-8',compresslevel=3) as audit:
    for row in sub.itertuples():
     assert group_map[str(row.mmsi)]==split
     original,target,behavior_md=latent_target(z,row,view,limits)
     spec=noise_spec(original,row)
     changed_index,group_md,neighbor_md=augment_background(index,original,spec,row,cfg,group=view=='paired_group_scene')
     if view=='paired_group_scene':target,target_group_md=shared_deceleration(target,group_md['onset_s'],group_md['factor']);group_md['target']=target_group_md
     pre_noise=copy_observations(target)
     pre_physical=physical_statistics(pre_noise['time_s'],pre_noise['sog_kn'],pre_noise['cog_deg'])
     pre_error=float(np.abs(innovation_vectors(*as_generator(pre_noise))-innovation_vectors(*as_generator(original))).max())
     assert pre_error<1e-8
     max_pre_innovation_error=max(max_pre_innovation_error,pre_error)
     target,target_noise_md=apply_target_noise(target,spec)
     post_physical=physical_statistics(target['time_s'],target['sog_kn'],target['cog_deg'])
     physical_pre.append(pre_physical);physical_post.append(post_physical)
     bbox=coverage(meta,float(row.start_utc_s),float(row.end_utc_s),lon0,lat0)
     out=build_window(target,changed_index,decision_time_s=float(row.end_utc_s),eligible_neighbor_mmsi=allowed,known_coverage_bbox_m=bbox,split=split)
     n=len(target['time_s']);assert out['mask'].sum()==n and out['metadata']['usable_window']
     out['coverage_known']=np.full(cfg.max_steps,bbox is not None,dtype=bool)&out['mask']
     for name,col in [('target_pos',np.column_stack((target['x_m'],target['y_m']))),('target_sog_kn',target['sog_kn']),('target_cog_deg',target['cog_deg']),
                      ('pre_noise_pos',np.column_stack((pre_noise['x_m'],pre_noise['y_m']))),('pre_noise_sog_kn',pre_noise['sog_kn']),('pre_noise_cog_deg',pre_noise['cog_deg'])]:
      arr=np.zeros((cfg.max_steps,)+col.shape[1:],dtype=np.float64);arr[:n]=col;out[name]=arr
     for k,v in out.items():
      if isinstance(v,np.ndarray):accum.setdefault(k,[]).append(v)
     md=out['metadata'];md.update(window_id=str(row.window_id),base_window_id=str(row.base_window_id),
          sample_id=str(row.sample_id),region=region,view=view,y=int(row.y),type=int(row.type),behavior_type=str(row.behavior_type),
          requested_observation_noise=spec,original_source_mmsi=str(row.mmsi),
          counterfactual_row=int(row.counterfactual_row),behavior_generation=behavior_md,
          target_observation_noise=target_noise_md,neighbor_observation_noise=neighbor_md,group_variation=group_md,
          physical_before_observation_noise=pre_physical,physical_after_observation_noise=post_physical,
          pre_noise_innovation_preservation_max_mps=pre_error)
     changed_pairs={(str(e['mmsi']),float(t)) for e in neighbor_md['changes'] if e.get('effective') for t in e.get('affected_time_s',[])}
     read_pairs=set();accepted_read_pairs=set()
     for step in md['steps']:
      seen=set()
      for nb in step['neighbors']:
       identity=str(nb['mmsi']);pair=(identity,float(nb['time_s']))
       assert identity in allowed and identity!=str(row.mmsi) and identity not in seen
       assert nb['time_s']<=step['time_s'] and 0<=step['time_s']-nb['time_s']<=cfg.freshness_seconds
       seen.add(identity);accessed.add(identity);candidate_edges+=1
       if pair in changed_pairs:read_pairs.add(pair)
       if pair in changed_pairs and nb['accepted']:accepted_read_pairs.add(pair)
     neighbor_md['changed_reports_read_in_context_audit']=[{'mmsi':m,'time_s':t} for m,t in sorted(read_pairs)]
     neighbor_md['changed_reports_accepted_after_gate']=len(accepted_read_pairs)
     target_effective=bool(target_noise_md.get('observations_changed'))
     effective=target_effective or bool(neighbor_md['effective'])
     window_flags['observation_noise_effective'].append(effective)
     window_flags['target_noise_effective'].append(target_effective)
     window_flags['neighbor_noise_effective'].append(bool(neighbor_md['effective']))
     md['effective_observation_noise']=effective
     md['context_protocol']='strict_all_vessels' if view=='features_strict' else 'target_disjoint_causal_scene'
     audit.write(json.dumps(md,separators=(',',':'))+'\n')
     effective_noise+=int(effective);effective_neighbors+=int(neighbor_md['effective']);read_changed+=len(read_pairs);target_noise_count+=int(target_effective);group_members+=len(group_md['controlled_mmsi'])
   arrays={k:np.stack(v) for k,v in accum.items()}
   for k,v in arrays.items():
    if v.dtype.kind=='f':
     assert np.isfinite(v).all();arrays[k]=v.astype(np.float32) if k not in ['time_s','target_pos','target_sog_kn','target_cog_deg','pre_noise_pos','pre_noise_sog_kn','pre_noise_cog_deg'] else v
   arrays.update({k:np.asarray(v,dtype=bool) for k,v in window_flags.items()})
   arrays.update(y=sub.y.to_numpy(np.int8),inttype=sub.type.to_numpy(np.int8),type=sub.type.to_numpy(np.int8),
                 window_id=sub.window_id.to_numpy(dtype='U24'),base_window_id=sub.base_window_id.to_numpy(dtype='U24'),
                 mmsi=sub.mmsi.to_numpy(np.int64),region=sub.region.to_numpy(dtype='U32'),
                 behavior_type=sub.behavior_type.to_numpy(dtype='U24'),noise_type=sub.noise_kind.to_numpy(dtype='U24'),
                 noise_scope=sub.noise_scope.to_numpy(dtype='U24'),observation_noise=sub.observation_noise.to_numpy(bool),
                 group_normal_variation=np.full(len(sub),view=='paired_group_scene',bool))
   npz=dest/f'{split}.npz';np.savez_compressed(npz,**arrays)
   # Main scene/strict have identical target observations regardless of protocol
   # neighbor availability; comparison is done whenever the peer view is ready.
   if view in ['features_scene','features_strict']:
    peer=args.out/('features_strict' if view=='features_scene' else 'features_scene')/region/f'{split}.npz'
    if peer.exists():
     with np.load(peer,allow_pickle=False) as pp:
      for k in ['raw','gated','mask','time_s','window_id','y','inttype','target_pos','target_sog_kn','target_cog_deg']:
       assert np.array_equal(arrays[k],pp[k]),f'Target protocol discrepancy in {region}/{split}/{k}'
   mask=arrays['mask'];accessed.update(sub.mmsi.astype(str));identity_sets[split]=sorted(accessed)
   split_reports[split]={'windows':len(sub),'target_vessels':int(sub.mmsi.nunique()),'normal':int((sub.y==0).sum()),'anomaly':int((sub.y==1).sum()),
      'valid_target_steps':int(mask.sum()),'requested_observation_noise':int(sub.observation_noise.sum()),
      'effective_observation_noise':effective_noise,'target_noise_effective_windows':target_noise_count,
      'neighbor_noise_effective_windows':effective_neighbors,'changed_background_reports_read':read_changed,
      'group_controlled_neighbor_memberships':group_members,'field_valid_fraction':float(arrays['context_valid'][mask].mean()),
      'mean_valid_neighbor_count':float(arrays['valid_neighbor_count'][mask].mean()),'mean_plain_neighbor_count':float(arrays['plain_neighbor_count'][mask].mean()),
      'boundary_complete_fraction':float((arrays['boundary_context_valid']&arrays['coverage_known'])[mask].mean()),
      'audited_candidate_edges':candidate_edges,'all_accessed_identity_count':len(accessed),
      'pre_noise_innovation_preservation_max_mps':max_pre_innovation_error,
      'physical_before_noise':{k:{'p50':float(np.median([v[k] for v in physical_pre])),'max':float(max(v[k] for v in physical_pre))} for k in physical_pre[0]},
      'physical_after_noise':{k:{'p50':float(np.median([v[k] for v in physical_post])),'max':float(max(v[k] for v in physical_post))} for k in physical_post[0]}}
   output_hashes[split]={'npz_sha256':sha(npz),'audit_sha256':sha(audit_path)}
   print('FEATURE_SPLIT_COMPLETE',region,view,split,json.dumps(split_reports[split]),flush=True)
  cross={a+'__'+b:len(set(identity_sets[a])&set(identity_sets[b])) for a,b in [('train','val'),('train','test'),('val','test')] if a in identity_sets and b in identity_sets}
  if view=='features_strict':assert not any(cross.values())
  report={'schema':'labeled_innovation_preserving_features_v1','region':region,'view':view,
          'status':'completed','split_policy':'Only same global-split background identities' if view=='features_strict' else 'Train excludes all original held-out target IDs; val excludes all original test targets; test uses causal available scene; background IDs may recur',
          'label_definition':'y=controlled behavior intervention; no human-natural-anomaly ground-truth claim',
          'noise_policy':'Final ledger noise_kind/scope; target noise independent of y/profile/protocol/candidate availability; background selection uses original target pose and freshly rebuilt causal index',
          'group_policy':'test-only paired scene: common smooth deceleration of target plus original local co-directional source neighbors; then same observation noise; y unchanged' if view=='paired_group_scene' else 'no added group variation',
          'source_proof':source_proof,'projection':{'lon0':lon0,'lat0':lat0,'meters_per_degree':111195},
          'context_config':cfg.__dict__,'splits':split_reports,'all_accessed_identity_sets':identity_sets,
          'actual_identity_intersections':cross,'output_hashes':output_hashes,'elapsed_seconds':time.time()-started}
  (dest/'feature_report.json').write_text(json.dumps(report,indent=2));reports[view]=report
  print('FEATURE_REGION_COMPLETE',region,view,'seconds',round(time.time()-started,1),flush=True)
 return reports

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--labels',type=Path,default=BASE/'labeled_v1/label_manifest.parquet')
 ap.add_argument('--counterfactuals',type=Path,default=BASE/'counterfactuals_v1')
 ap.add_argument('--out',type=Path,default=BASE/'labeled_v1');ap.add_argument('--source',type=Path,default=SOURCE)
 ap.add_argument('--regions',nargs='*');ap.add_argument('--views',nargs='+',choices=VIEWS,default=list(VIEWS[:3]))
 args=ap.parse_args();ledger=pd.read_parquet(args.labels)
 assert set(ledger.y)=={0,1} and ledger.labels_frozen.all() and ledger.counterfactual_accepted.all()
 assert not ledger.window_id.duplicated().any()
 registry=json.loads((args.source/'source_registry.json').read_text())['regions']
 groups=pd.read_parquet(args.source/'global_all_mmsi_split.parquet');assert not groups.mmsi.duplicated().any()
 original=pd.read_parquet(args.source/'selected_windows.parquet',columns=['mmsi','split'])
 original_targets={s:set(original.loc[original.split==s,'mmsi'].astype(str)) for s in SPLITS}
 assert sha(BASE/'generator/maneuver_generator.py')==EXPECTED_GENERATOR
 cfg=ContextConfig()
 for region,g in ledger.groupby('region',sort=True):
  if args.regions and region not in args.regions:continue
  build_region(region,g,registry,groups,original_targets,args,cfg)

if __name__=='__main__':main()
