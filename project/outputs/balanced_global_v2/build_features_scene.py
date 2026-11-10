"""Alternative causal scene view on the SAME globally target-disjoint windows.

Training excludes all validation/test target vessels from context. Validation
excludes test target vessels. Test uses the available causal scene. Background
vessels may recur across sets: this is explicitly NOT all-input identity isolation.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, sys, time
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'context'))
from context_features import BackgroundIndex, ContextConfig, build_window

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def projected(d,lon0,lat0):
 return {'mmsi':d.mmsi.to_numpy().astype(str),'time_s':d.timestamp.to_numpy(np.float64),
  'x_m':(d.lon.to_numpy()-lon0)*111195*np.cos(np.deg2rad(lat0)),
  'y_m':(d.lat.to_numpy()-lat0)*111195,
  'sog_kn':d.sog.to_numpy(np.float64),'cog_deg':d.cog.to_numpy(np.float64)}

def coverage(meta,start,end,lon0,lat0):
 # Conservative intersection across every source day touched by the target and
 # the 600-second causal background/kinematic history. Never use observed extrema
 # as a claimed complete receiver footprint.
 days=pd.date_range(pd.Timestamp(start-600,unit='s',tz='UTC').floor('D'),pd.Timestamp(end,unit='s',tz='UTC').floor('D'),freq='D')
 if any(d.strftime('%Y-%m-%d') not in meta['coverage_bbox_by_utc_day'] for d in days):return None
 boxes=[meta['coverage_bbox_by_utc_day'][d.strftime('%Y-%m-%d')] for d in days]
 if not boxes:return None
 a=np.array(boxes);b=[a[:,0].max(),a[:,1].max(),a[:,2].min(),a[:,3].min()]
 if b[0]>=b[2] or b[1]>=b[3]:return None
 return ((b[0]-lon0)*111195*np.cos(np.deg2rad(lat0)),(b[1]-lat0)*111195,(b[2]-lon0)*111195*np.cos(np.deg2rad(lat0)),(b[3]-lat0)*111195)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--registry',type=Path,default=BASE/'source_registry.json');ap.add_argument('--regions',nargs='*');args=ap.parse_args()
 cfg=ContextConfig();registry=json.loads(args.registry.read_text())['regions']
 sel=pd.read_parquet(args.selection/'selected_windows.parquet');groups=pd.read_parquet(args.selection/'global_all_mmsi_split.parquet')
 assert not groups.mmsi.duplicated().any() and set(groups.split)=={'train','val','test'}
 assert set(sel.split)=={'train','val','test'} and not sel.window_id.duplicated().any()
 provenance={'selected_windows_sha256':sha(args.selection/'selected_windows.parquet'),'global_all_mmsi_split_sha256':sha(args.selection/'global_all_mmsi_split.parquet'),'build_features_script_sha256':sha(Path(__file__)),'source_registry_sha256':sha(args.registry)}
 group_map=dict(zip(groups.mmsi.astype(str),groups.split))
 targets={s:set(sel.loc[sel.split==s,'mmsi'].astype(str)) for s in ['train','val','test']}
 all_ids=set(groups.mmsi.astype(str))
 all_groups={'train':all_ids-targets['val']-targets['test'],'val':all_ids-targets['test'],'test':all_ids}
 sr=json.loads((args.selection/'selection_report.json').read_text());reports={}
 for region,windows in sel.groupby('region',sort=True):
  if args.regions and region not in args.regions:continue
  t0=time.time();meta=registry[region];p=Path(meta['canonical_file']);assert sha(p)==sr['source_files'][region]['canonical_sha256']
  dest=args.selection/'scene_features'/region;dest.mkdir(parents=True,exist_ok=True)
  assert not (dest/'feature_report.json').exists(),'Refusing to overwrite completed region features'
  d=pd.read_parquet(p).reset_index(drop=True);box=meta['core_bbox'];lon0=(box[0]+box[2])/2;lat0=(box[1]+box[3])/2
  data=projected(d,lon0,lat0);dgroup=np.array([group_map[m] for m in data['mmsi']]);stepcounts={};exposures={};file_info={}
  for split in ['train','val','test']:
   sub=windows[windows.split==split].reset_index(drop=True);allowed=all_groups[split]
   use=np.isin(data['mmsi'],np.array(sorted(allowed)))
   idx=BackgroundIndex({k:v[use] for k,v in data.items()},cfg)
   accum={};exposure=set();total_edges=0;cases=[]
   audit_path=dest/f'{split}_context_audit.jsonl.gz'
   with gzip.open(audit_path,'wt',encoding='utf-8',compresslevel=3) as af:
    for row in sub.itertuples():
     assert group_map[str(row.mmsi)]==split
     ii=np.array(json.loads(row.row_indices),dtype=np.int64);target={k:v[ii] for k,v in data.items()}
     assert np.all(target['mmsi']==str(row.mmsi)) and np.array_equal(target['time_s'],json.loads(row.timestamps))
     bbox=coverage(meta,row.start_utc_s,row.end_utc_s,lon0,lat0)
     r=build_window(target,idx,decision_time_s=row.end_utc_s,eligible_neighbor_mmsi=allowed,known_coverage_bbox_m=bbox,split=split)
     r['coverage_known']=np.full(cfg.max_steps,bbox is not None,dtype=bool)&r['mask']
     assert r['metadata']['usable_window'] and r['mask'].sum()==row.n_obs
     for k,v in r.items():
      if isinstance(v,np.ndarray):accum.setdefault(k,[]).append(v)
     md=r['metadata'];md['window_id']=row.window_id;md['region']=region;md['context_protocol']='target_disjoint_causal_scene'
     for step in md['steps']:
      seen=set()
      for nb in step['neighbors']:
       identity=nb['mmsi'];assert identity in allowed and identity!=str(row.mmsi) and identity not in seen
       assert nb['time_s']<=step['time_s'] and nb['age_s']<=cfg.freshness_seconds
       seen.add(identity);exposure.add(identity);total_edges+=1
     af.write(json.dumps(md,separators=(',',':'))+'\n')
   arrays={k:np.stack(v) for k,v in accum.items()};arrays.update(window_id=sub.window_id.to_numpy(dtype='U24'),mmsi=sub.mmsi.to_numpy(np.int64),start_utc_s=sub.start_utc_s.to_numpy(np.int64),end_utc_s=sub.end_utc_s.to_numpy(np.int64))
   for k,v in arrays.items():
    if v.dtype.kind=='f':assert np.isfinite(v).all();arrays[k]=v.astype(np.float32) if k!='time_s' else v
   npz=dest/f'{split}.npz';np.savez_compressed(npz,**arrays)
   mask=arrays['mask'];n=int(mask.sum())
   stepcounts[split]={'windows':len(sub),'target_vessels':int(sub.mmsi.nunique()),'raw_background_rows':int(use.sum()),'background_vessels':int(len(set(data['mmsi'][use]))),'valid_target_steps':n,'kinematic_valid_fraction':float(arrays['kinematic_valid'][mask].mean()),'known_coverage_fraction':float(arrays['coverage_known'][mask].mean()),'boundary_complete_fraction':float((arrays['boundary_context_valid']&arrays['coverage_known'])[mask].mean()),'field_valid_fraction':float(arrays['context_valid'][mask].mean()),'mean_valid_neighbor_count':float(arrays['valid_neighbor_count'][mask].mean()),'mean_plain_neighbor_count':float(arrays['plain_neighbor_count'][mask].mean()),'audited_candidate_edges':total_edges,'distinct_context_ids_read':len(exposure)}
   exposures[split]=sorted(exposure|set(sub.mmsi.astype(str)));file_info[split]={'npz_sha256':sha(npz),'audit_sha256':sha(audit_path)}
   print(region,split,json.dumps(stepcounts[split]),flush=True)
  intersections={a+'__'+b:len(set(exposures[a])&set(exposures[b])) for a,b in [('train','val'),('train','test'),('val','test')]}
  report={'region':region,'schema':'balanced_global_v2_observed_source_features','context_protocol':'target_disjoint_causal_scene','label_status':'Alternative features for the SAME unlabeled source windows, not additional independent samples','split_policy':'Targets globally disjoint. Train excludes all held-out target vessels from context; val excludes global test targets; test uses available causal scene. Background identities can recur across sets.','density_note':'Causal scene view preserves full observed traffic density at test; training context removes held-out targets. Compare against strict all-vessel-isolated view separately.','source':meta,'canonical_sha256':sha(p),'module_sha256':sha(BASE/'context/context_features.py'),'projection':{'name':'local equirectangular','lon0':lon0,'lat0':lat0,'meters_per_degree':111195},'coverage_policy':'Conservative intersection of source-day rectangles touched by window and 600s history; boundary absence distinct from natural sparsity','splits':stepcounts,'all_accessed_identity_sets':exposures,'actual_background_identity_intersections':intersections,'output_hashes':file_info,'elapsed_seconds':time.time()-t0}
  report['construction_provenance']=provenance
  (dest/'feature_report.json').write_text(json.dumps(report,indent=2));reports[region]=report
  print('REGION_COMPLETE',region,'seconds',round(time.time()-t0,1),flush=True)

if __name__=='__main__':main()
