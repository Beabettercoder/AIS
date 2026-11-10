from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path('/mnt/sda/WangZijia/AIS')
OUT=ROOT/'outputs/independent_ais_evidence_20261030'
OLD=ROOT/'outputs/balanced_global_v2/relation_dual_pressure_v2_20261027'
REG=json.loads((ROOT/'outputs/balanced_global_v2/release_20261026/source_registry.json').read_text())['regions']
regions=['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai']
manifest=json.loads((OLD/'dataset_manifest_combined.json').read_text())
rows=[r for r in manifest['samples'] if r.get('version')=='relaxed' and r.get('split') in ('train','val')]
results={}
for region in regions:
    selected=[]
    for split in ('train','val'):
        subset=[r for r in rows if r['region']==region and r['split']==split]
        subset.sort(key=lambda r:hashlib.sha256(('extra-evidence:'+r['window_id']).encode()).hexdigest())
        selected.extend(subset[:10])
    requests=[]
    invariants={'target_cog_four_cells_equal':0,'neighbor_cog_four_cells_equal':0,'target_speed_modified_source':0,'neighbor_speed_modified_source':0,'base_natural_suffix_cog_equal':0}
    for r in selected:
        with np.load(OLD/r['files']['base']['path'],allow_pickle=False) as z:
            tc=z['target_cog_deg']; nc=z['neighbor_cog_deg']; ts=z['target_sog_kn']; ns=z['neighbor_sog_kn']
            invariants['target_cog_four_cells_equal']+=int(np.array_equal(tc,np.broadcast_to(tc[0],tc.shape)))
            invariants['neighbor_cog_four_cells_equal']+=int(np.array_equal(nc,np.broadcast_to(nc[0],nc.shape)))
            invariants['target_speed_modified_source']+=int(any(not np.array_equal(ts[0],ts[k]) for k in range(1,4)))
            invariants['neighbor_speed_modified_source']+=int(any(not np.array_equal(ns[0],ns[k]) for k in range(1,4)))
            for t in z['target_t'][0]: requests.append((str(r['mmsi']),int(t),'target',r['split']))
            for m,t in zip(z['neighbor_mmsi'].astype(str),z['neighbor_t'][0]): requests.append((str(m),int(t),'neighbor',r['split']))
            base_target_cog=tc.copy();base_neighbor_cog=nc.copy()
        same=True
        for view in ('natural','suffix'):
            with np.load(OLD/r['files'][view]['path'],allow_pickle=False) as v:
                same &= np.array_equal(v['target_cog_deg'],base_target_cog) and np.array_equal(v['neighbor_cog_deg'],base_neighbor_cog)
        invariants['base_natural_suffix_cog_equal']+=int(same)
    path=Path(REG[region]['canonical_file']);names=pq.read_schema(path).names
    cols=[x for x in ('mmsi','timestamp','heading','cog','sog','nav_status','vessel_type') if x in names]
    ids=sorted({int(m) for m,_,_,_ in requests})
    frame=pd.read_parquet(path,columns=cols,filters=[('mmsi','in',ids)])
    frame['mmsi']=frame['mmsi'].astype(str);frame['timestamp']=frame['timestamp'].astype('int64')
    frame['original_order']=np.arange(len(frame)); dups=int(frame.duplicated(['mmsi','timestamp'],keep=False).sum())
    frame=frame.sort_values(['mmsi','timestamp','original_order'],kind='stable').drop_duplicates(['mmsi','timestamp'],keep='last')
    q=pd.DataFrame(requests,columns=['mmsi','timestamp','role','split'])
    j=q.merge(frame,on=['mmsi','timestamp'],how='left',indicator=True,validate='many_to_one')
    heading=pd.to_numeric(j.get('heading',pd.Series(np.nan,index=j.index)),errors='coerce')
    cog=pd.to_numeric(j.get('cog',pd.Series(np.nan,index=j.index)),errors='coerce')
    sog=pd.to_numeric(j.get('sog',pd.Series(np.nan,index=j.index)),errors='coerce')
    valid=heading.ge(0)&heading.lt(360)&cog.ge(0)&cog.lt(360)&j['_merge'].eq('both')
    angle=((heading-cog+180)%360-180).abs()
    def stat(mask):
        z=angle[mask&valid];return {'count':int(len(z)),'median_deg':float(z.median()) if len(z) else None,'p90_deg':float(z.quantile(.9)) if len(z) else None}
    def val(s):return {str(k):int(v) for k,v in s.value_counts(dropna=False).head(8).items()}
    nav=j['nav_status'] if 'nav_status' in j else pd.Series(np.nan,index=j.index)
    group_stats={}
    for split in ('train','val'):
        for role in ('target','neighbor'):
            m=j['split'].eq(split)&j['role'].eq(role)
            group_stats[f'{split}_{role}']={'requests':int(m.sum()),'unique_mmsi':int(j.loc[m,'mmsi'].nunique()),'joined':int((m&j['_merge'].eq('both')).sum()),'heading_valid':int((m&valid).sum())}
    results[region]={'windows':len(selected),'train_windows':sum(r['split']=='train' for r in selected),'val_windows':sum(r['split']=='val' for r in selected),'canonical_columns':names,'requests':len(q),'unique_requested_keys':int(q.drop_duplicates(['mmsi','timestamp']).shape[0]),'joined':int(j['_merge'].eq('both').sum()),'source_rows_for_requested_mmsi':len(frame),'source_duplicate_key_rows_before_dedup':dups,'valid_heading_cog':int(valid.sum()),'heading_missing_or_invalid':int((~valid).sum()),'heading_cog_angle':{'low_sog_lt2_kn':stat(sog.lt(2)), 'moving_sog_ge5_kn':stat(sog.ge(5))},'groups':group_stats,'four_cell_consistency':invariants,'heading_raw_top':val(heading),'nav_raw_top':val(nav),'candidate_heading_evidence':'separate AIS field; not proven independent sensor','selection':'first 10 per region per train/val by SHA256; repeated windows/reports not independent'}
    print(region,json.dumps(results[region],ensure_ascii=False),flush=True)
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'sample_field_join_audit.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
