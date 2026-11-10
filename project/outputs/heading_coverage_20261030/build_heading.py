from pathlib import Path
from collections import defaultdict,Counter
import numpy as np, pandas as pd, pyarrow.parquet as pq
import json,hashlib

ROOT=Path('/mnt/sda/WangZijia/AIS')
OLD=ROOT/'outputs/balanced_global_v2/relation_dual_pressure_v2_20261027'
OUT=ROOT/'outputs/heading_coverage_20261030'
registry=json.loads((ROOT/'outputs/balanced_global_v2/release_20261026/source_registry.json').read_text())['regions']
manifest=json.loads((OLD/'dataset_manifest_combined.json').read_text())
rows=sorted((r for r in manifest['samples'] if r.get('version')=='relaxed' and r.get('split') in ('train','val')),key=lambda r:(r['region'],r['window_id']))
assert len({r['window_id'] for r in rows})==len(rows)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def stat(group):
 n=len(group); valid=group['heading_valid']; low=group['source_sog'].lt(2)
 return dict(requests=n,unique_keys=int(group[['mmsi','timestamp']].drop_duplicates().shape[0]),unique_mmsi=int(group.mmsi.nunique()),joined=int(group.joined.sum()),valid_heading=int(valid.sum()),valid_rate=float(valid.mean()) if n else None,low_requests=int(low.sum()),low_valid=int((low&valid).sum()),low_valid_rate=float(valid[low].mean()) if low.any() else None)

allout={s:dict(window_id=[],region=[],target_mmsi=[],target_length=[],target_heading=[],target_valid=[],target_t=[],sample_neighbor_offsets=[0],neighbor_mmsi=[],neighbor_t=[],neighbor_heading=[],neighbor_valid=[],sample_member_offsets=[0],member_offsets=[],cohort_mmsi=[]) for s in ('train','val')}
stats={};global_checks=Counter();request_sets=defaultdict(set)
for region in ('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai'):
 selected=[r for r in rows if r['region']==region]; requests=[]; saved=[]; checks=Counter()
 for r in selected:
  ent=r['files']['base'];path=OLD/ent['path'];assert sha(path)==ent['sha256']
  with np.load(path,allow_pickle=False) as z:
   keys=('target_t','neighbor_t','target_cog_deg','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','target_sog_kn','neighbor_sog_kn','y')
   d={k:z[k].copy() for k in keys}
  assert str(r['mmsi']) not in d['cohort_mmsi'].astype(str)
  for key in ('target_t','neighbor_t','target_cog_deg','neighbor_cog_deg'):
   if not np.array_equal(d[key],np.broadcast_to(d[key][0],d[key].shape)): raise ValueError(f'four-cell mismatch {r["window_id"]} {key}')
  y=d['y']; assert sorted(y.tolist())==[0,0,1,1]
  for view in ('natural','suffix'):
   entv=r['files'][view]; vp=OLD/entv['path']; assert sha(vp)==entv['sha256']
   with np.load(vp,allow_pickle=False) as z:
    for key in ('target_t','neighbor_t','target_cog_deg','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','y'):
     if not np.array_equal(z[key],d[key]): raise ValueError(f'view mismatch {r["window_id"]} {view} {key}')
  checks['source_count']+=1; checks['four_cells_key_and_cog_equal']+=1;checks['natural_suffix_keys_equal']+=1
  checks['target_sog_varies']+=int(any(not np.array_equal(d['target_sog_kn'][0],d['target_sog_kn'][i]) for i in range(1,4)))
  checks['neighbor_sog_varies']+=int(any(not np.array_equal(d['neighbor_sog_kn'][0],d['neighbor_sog_kn'][i]) for i in range(1,4)))
  target_t=d['target_t'][0];neighbor_t=d['neighbor_t'][0];nm=d['neighbor_mmsi'].astype(str)
  assert np.all(np.isfinite(target_t)) and np.all(np.isfinite(neighbor_t)); assert np.all(target_t==target_t.astype('int64')) and np.all(neighbor_t==neighbor_t.astype('int64'))
  start=len(requests)
  for t in target_t: requests.append((str(r['mmsi']),int(t),r['split'],'target'))
  for m,t in zip(nm,neighbor_t):requests.append((str(m),int(t),r['split'],'neighbor'))
  saved.append((r,d,start,len(target_t),len(neighbor_t)))
  request_sets[(region,r['split'])].update((str(m),int(t)) for m,t in zip(nm,neighbor_t))
 path=Path(registry[region]['canonical_file']);names=pq.read_schema(path).names
 cols=[x for x in ('mmsi','timestamp','heading','cog','sog','nav_status','vessel_type') if x in names]
 q=pd.DataFrame(requests,columns=['mmsi','timestamp','split','role']); ids=sorted(set(map(int,q.mmsi)))
 frame=pd.read_parquet(path,columns=cols,filters=[('mmsi','in',ids)])
 frame['mmsi']=frame['mmsi'].astype(str);frame['timestamp']=frame['timestamp'].astype('int64')
 frame['original_order']=np.arange(len(frame)); dup=int(frame.duplicated(['mmsi','timestamp'],keep=False).sum())
 frame=frame.sort_values(['mmsi','timestamp','original_order'],kind='stable').drop_duplicates(['mmsi','timestamp'],keep='last')
 merged=q.merge(frame,on=['mmsi','timestamp'],how='left',indicator=True,validate='many_to_one',sort=False)
 heading=pd.to_numeric(merged['heading'] if 'heading' in merged else pd.Series(np.nan,index=merged.index),errors='coerce')
 valid=heading.ge(0)&heading.lt(360)&merged['_merge'].eq('both')
 merged['joined']=merged['_merge'].eq('both');merged['heading_valid']=valid;merged['source_sog']=pd.to_numeric(merged['sog'],errors='coerce') if 'sog' in merged else np.nan
 h=heading.to_numpy(dtype='float32'); v=valid.to_numpy();h[~v]=np.nan
 gs={}
 for split in ('train','val'):
  for role in ('target','neighbor'):
   z=merged[(merged['split']==split)&(merged.role==role)];gs[split+'_'+role]=stat(z)
 for r,d,start,nt,nn in saved:
  o=allout[r['split']];o['window_id'].append(r['window_id']);o['region'].append(region);o['target_mmsi'].append(str(r['mmsi']));o['target_length'].append(nt)
  hp=np.full(20,np.nan,np.float32);vp=np.zeros(20,bool);tp=np.zeros(20,np.float64);assert nt<=20
  hp[:nt]=h[start:start+nt];vp[:nt]=v[start:start+nt];tp[:nt]=d['target_t'][0]
  o['target_heading'].append(hp);o['target_valid'].append(vp);o['target_t'].append(tp)
  o['neighbor_mmsi'].extend(d['neighbor_mmsi'].astype(str));o['neighbor_t'].extend(d['neighbor_t'][0]);o['neighbor_heading'].extend(h[start+nt:start+nt+nn]);o['neighbor_valid'].extend(v[start+nt:start+nt+nn]);o['sample_neighbor_offsets'].append(o['sample_neighbor_offsets'][-1]+nn)
  o['cohort_mmsi'].extend(d['cohort_mmsi'].astype(str));o['member_offsets'].extend(d['member_offsets']);o['sample_member_offsets'].append(o['sample_member_offsets'][-1]+len(d['member_offsets']))
 overlap=len(request_sets[(region,'train')]&request_sets[(region,'val')])
 stats[region]=dict(canonical_file=str(path),canonical_sha256=sha(path),canonical_columns=names,source_duplicate_key_rows_before_dedup=dup,groups=gs,checks=dict(checks),neighbor_train_val_shared_key_count=overlap)
 print('REGION',region,'source',len(selected),'requests',len(q),'matched',int(merged.joined.sum()),'valid',int(valid.sum()),'duplicate',dup,flush=True)
for split,o in allout.items():
 arrays={k:np.asarray(v) for k,v in o.items()}; assert len(arrays['sample_neighbor_offsets'])==len(arrays['window_id'])+1
 assert arrays['sample_neighbor_offsets'][-1]==len(arrays['neighbor_heading'])==len(arrays['neighbor_valid'])==len(arrays['neighbor_t'])==len(arrays['neighbor_mmsi'])
 dest=OUT/f'heading_{split}.npz'; assert not dest.exists();np.savez_compressed(dest,**arrays)
 print('FILE',dest,len(arrays['window_id']),sha(dest),flush=True)
stats['_protocol']=dict(manifest_sha256=sha(OLD/'dataset_manifest_combined.json'),builder_sha256=sha(Path(__file__)),schema='target [N,20] plus concatenated original neighbor reports and sample offsets; all values mapped from original Base source MMSI and UTC seconds; NaN+mask',exclude_labels=True,train_val_only=True,heading_not_proven_independent_sensor=True,warning='shared heading is not verified physically consistent with synthetic trajectories; low-speed COG/heading difference not fault label')
(OUT/'coverage_full.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf8')
