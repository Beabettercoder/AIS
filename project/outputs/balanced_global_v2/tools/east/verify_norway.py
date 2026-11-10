from pathlib import Path
import gzip,hashlib,json
import pandas as pd
R=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/east')
m=json.loads((R/'norway_api_acquisition.json').read_text());raw=Path(m['raw_path'])
assert hashlib.sha256(raw.read_bytes()).hexdigest()==m['sha256']
obj=json.loads(gzip.decompress(raw.read_bytes()));assert obj['success']
print('raw_metadata',{k:v for k,v in obj.items() if k!='data'},flush=True)
print('first_row',obj['data'][0],flush=True)
new=pd.read_parquet(R/'norway_20240104.parquet');old=pd.read_parquet(R/'norway_existing.parquet');full=pd.read_parquet(R/'norway_expanded.parquet')
assert len(new)==m['filtered_rows'] and len(full)==m['combined_rows']
assert new.timestamp.ge(int(pd.Timestamp('2024-01-04',tz='UTC').timestamp())).all()
assert new.timestamp.lt(int(pd.Timestamp('2024-01-05',tz='UTC').timestamp())).all()
assert new.lon.between(4.65,5.65).all() and new.lat.between(58.95,59.65).all()
assert not full.duplicated(['mmsi','timestamp']).any()
expected=pd.concat([old,new],ignore_index=True).sort_values(['mmsi','timestamp','source_file'],kind='stable').drop_duplicates(['mmsi','timestamp']).reset_index(drop=True)
pd.testing.assert_frame_equal(expected,full.reset_index(drop=True))
report={'passed':True,'raw_hash_verified':True,'gzip_crc_verified':True,'raw_json_success':True,'rows':len(full),'unique_mmsi':int(full.mmsi.nunique()),'new_date_exact_filter':True,'exact_union_verified':True,'no_duplicate_mmsi_timestamp':True,'all_dates':sorted(pd.to_datetime(full.timestamp,unit='s',utc=True).dt.strftime('%Y-%m-%d').unique().tolist()),'raw_observation_time_range':[min(r[1] for r in obj['data']),max(r[1] for r in obj['data'])],'warning':'API response contains adjacent-day observations. Only actual UTC timestamps within requested January 4 were retained.'}
(R/'norway_expansion_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
