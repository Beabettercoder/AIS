from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd

ROOT=Path('/mnt/sda/WangZijia/AIS')
BASE=ROOT/'outputs/balanced_global_v2/relation_dual_pressure_v2_20261027'
CACHE=ROOT/'outputs/heading_coverage_20261030'
OUT=ROOT/'outputs/heading_independent_audit_20261030'
manifest=json.loads((BASE/'dataset_manifest_combined.json').read_text())
registry=json.loads((ROOT/'outputs/balanced_global_v2/release_20261026/source_registry.json').read_text())['regions']
rows={r['window_id']:r for r in manifest['samples'] if r.get('version')=='relaxed' and r.get('split') in ('train','val')}
all_targets={}; total={'sources':0,'quartets':0,'keys':0,'selected_exact_heading':0,'selected_exact_invalid':0,'view_keys':0,'missing_heading':0}; lookup={}; selected={}
for split in ('train','val'):
 with np.load(CACHE/f'heading_{split}.npz',allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
 assert len(set(a['window_id'].tolist()))==len(a['window_id'])
 all_targets[split]=set(a['target_mmsi'].astype(str)); lookup[split]=a
 expected={r['window_id'] for r in rows.values() if r['split']==split};assert set(a['window_id'])==expected
 for i,w in enumerate(a['window_id'].astype(str)):
  r=rows[w]; assert r['split']==split and r['region']==a['region'][i] and str(r['mmsi'])==str(a['target_mmsi'][i]);n=int(a['target_length'][i]);
  start,end=a['sample_neighbor_offsets'][i:i+2];h=a['neighbor_heading'][start:end];hv=a['neighbor_valid'][start:end];nt=a['neighbor_t'][start:end];nm=a['neighbor_mmsi'][start:end]
  assert np.all(np.isnan(h[~hv])) and np.all(np.isfinite(h[hv])) and np.all((h[hv]>=0)&(h[hv]<360))
  assert np.all(np.isnan(a['target_heading'][i,:n][~a['target_valid'][i,:n]])) and not a['target_valid'][i,n:].any()
  with np.load(BASE/r['files']['base']['path'],allow_pickle=False) as z:
   assert np.array_equal(a['target_t'][i,:n],z['target_t'][0]); assert np.array_equal(nt,z['neighbor_t'][0]);assert np.array_equal(nm,z['neighbor_mmsi'].astype(str))
   assert np.array_equal(z['target_t'],np.broadcast_to(z['target_t'][0],z['target_t'].shape)); assert np.array_equal(z['neighbor_t'],np.broadcast_to(z['neighbor_t'][0],z['neighbor_t'].shape));assert sorted(z['y'].tolist())==[0,0,1,1]
   assert np.array_equal(z['target_cog_deg'],np.broadcast_to(z['target_cog_deg'][0],z['target_cog_deg'].shape));assert np.array_equal(z['neighbor_cog_deg'],np.broadcast_to(z['neighbor_cog_deg'][0],z['neighbor_cog_deg'].shape)); assert str(r['mmsi']) not in z['cohort_mmsi'].astype(str)
   ms,me=a['sample_member_offsets'][i:i+2];assert np.array_equal(a['member_offsets'][ms:me],z['member_offsets']);assert np.array_equal(a['cohort_mmsi'][ms-i:me-i-1],z['cohort_mmsi'].astype(str))
  for view in ('natural','suffix'):
   with np.load(BASE/r['files'][view]['path'],allow_pickle=False) as v:
    assert np.array_equal(nt,v['neighbor_t'][0]) and np.array_equal(nm,v['neighbor_mmsi'].astype(str));assert np.array_equal(a['target_t'][i,:n],v['target_t'][0]);assert sorted(v['y'].tolist())==[0,0,1,1]
   total['view_keys']+=1
  total['sources']+=1;total['quartets']+=4;total['keys']+=n+len(nt);total['missing_heading']+=int((~hv).sum())+int((~a['target_valid'][i,:n]).sum())
  # Two source windows per region/split, chosen by stable hash and independently joined.
  key=(split,r['region']); selected.setdefault(key,[]).append((hashlib.sha256(('audit:'+w).encode()).hexdigest(),i,w))
assert not all_targets['train']&all_targets['val']
for (split,region),lst in selected.items():
 a=lookup[split]; picks=sorted(lst)[:2]; requests=[]
 for _,i,w in picks:
  n=int(a['target_length'][i]);start,end=a['sample_neighbor_offsets'][i:i+2]
  requests.extend((str(a['target_mmsi'][i]),int(t),float(h),bool(ok)) for t,h,ok in zip(a['target_t'][i,:n],a['target_heading'][i,:n],a['target_valid'][i,:n]))
  requests.extend((str(m),int(t),float(h),bool(ok)) for m,t,h,ok in zip(a['neighbor_mmsi'][start:end],a['neighbor_t'][start:end],a['neighbor_heading'][start:end],a['neighbor_valid'][start:end]))
 ids=sorted(set(int(x[0]) for x in requests));path=Path(registry[region]['canonical_file']);frame=pd.read_parquet(path,columns=['mmsi','timestamp','heading'],filters=[('mmsi','in',ids)])
 frame['mmsi']=frame['mmsi'].astype(str);frame['timestamp']=frame['timestamp'].astype('int64');frame=frame.drop_duplicates(['mmsi','timestamp'],keep='last');source={(str(m),int(t)):h for m,t,h in zip(frame['mmsi'],frame['timestamp'],frame['heading'])}
 for m,t,h,ok in requests:
  val=source.get((m,t),np.nan)
  try: val=float(val)
  except (ValueError,TypeError):val=float('nan')
  isvalid=np.isfinite(val) and 0<=val<360
  assert ok==isvalid,(split,region,m,t,h,ok,val)
  if ok: assert abs(h-val)<1e-4;total['selected_exact_heading']+=1
  else: total['selected_exact_invalid']+=1
OUT.mkdir(parents=True,exist_ok=True);total['train_val_target_mmsi_overlap']=len(all_targets['train']&all_targets['val']);total['sampled_source_windows']=sum(min(2,len(x)) for x in selected.values());total['scope']='train+val only; heading separate AIS field, sensor independence and synthetic physical consistency not established'
(OUT/'full_heading_join_audit.json').write_text(json.dumps(total,indent=2),encoding='utf8');print(json.dumps(total))
