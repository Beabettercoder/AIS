"""Uniform independent-source-window inventory; no synthetic labels or training."""
from pathlib import Path
import hashlib, json, argparse
import numpy as np
import pandas as pd

ROOT=Path('/mnt/sda/WangZijia/AIS')
OUT=ROOT/'outputs/balanced_global_v2'
REGIONS={
 'houston':('west/houston_raw_standard.parquet',[-95.1,29.2,-94.6,29.8]),
 'ny':('west/ny_raw_standard.parquet',[-74.35,40.35,-73.65,40.95]),
 'la':('west/la_raw_standard.parquet',[-118.55,33.6,-117.85,34.05]),
 'puget':('west/puget_raw_standard.parquet',[-124,47.5,-122.3,48.5]),
 'great_belt':('east/denmark_existing.parquet',[10.75,55.05,11.45,55.55]),
 'haugesund':('east/norway_expanded.parquet',[4.65,58.95,5.65,59.65]),
 'shanghai':('east/shanghai_standard.parquet',[121.4402,30.6,122.66998,31.41998]),
}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def normalize_shanghai():
 base=ROOT/'data/raw/external/mendeley_drntjjtcv8_v1_shanghai'
 frames=[]
 for p in sorted((base/'raw').glob('*.csv')):
  d=pd.read_csv(p).rename(columns={'longitude':'lon','latitude':'lat','hdg':'heading'})
  d['timestamp']=pd.to_datetime(d.pop('utc'),utc=True).dt.as_unit('ns').astype('int64')//10**9
  d['source_file']=str(p);d['source_row']=np.arange(len(d),dtype=np.int64)
  frames.append(d)
 d=pd.concat(frames,ignore_index=True).sort_values(['mmsi','timestamp','source_file'],kind='stable').drop_duplicates(['mmsi','timestamp']).reset_index(drop=True)
 assert d.timestamp.between(int(pd.Timestamp('2023-09-22',tz='UTC').timestamp()),int(pd.Timestamp('2023-09-24',tz='UTC').timestamp())-1).all()
 d.to_parquet(OUT/'east/shanghai_standard.parquet',index=False)

def candidates(d,region,bbox):
 valid=(d.mmsi.between(200000000,799999999)&d.sog.ge(0)&d.sog.lt(102.2)&d.cog.ge(0)&d.cog.lt(360)&d.lon.between(-180,180)&d.lat.between(-90,90))
 # Keep canonical indices so every selected report is directly recoverable.
 v=d[valid].sort_values(['mmsi','timestamp'],kind='stable').drop_duplicates(['mmsi','timestamp'])
 rows=[]
 for m,g in v.groupby('mmsi',sort=True):
  t=g.timestamp.to_numpy(np.int64); s=g.sog.to_numpy(); ids=g.index.to_numpy()
  core=g.lon.between(bbox[0],bbox[2]).to_numpy()&g.lat.between(bbox[1],bbox[3]).to_numpy()
  starts=np.r_[0,np.flatnonzero(np.diff(t)>1800)+1]; stops=np.r_[starts[1:],len(t)]
  for seg,(a,z) in enumerate(zip(starts,stops)):
   for begin in range(int(t[a]),int(t[z-1])+1,1380):
    lo=max(a,int(np.searchsorted(t,begin)));hi=min(z,int(np.searchsorted(t,begin+1380)))
    take=[];last=-10**15
    for j in range(lo,hi):
     if t[j]-last>=60:take.append(j);last=t[j]
    ix=np.array(take[-20:],dtype=np.int64)
    if len(ix)<8 or not core[ix[-1]] or (s[ix]>=2).sum()<4 or s[ix].max()<3:continue
    tt=t[ix]; rid=hashlib.sha256((str(m)+':'+','.join(map(str,tt))).encode()).hexdigest()[:24]
    rows.append({'window_id':rid,'region':region,'mmsi':int(m),'segment_id':int(seg),'start_utc_s':int(tt[0]),'end_utc_s':int(tt[-1]),'cell_start_utc_s':begin,'n_obs':len(ix),'duration_s':int(tt[-1]-tt[0]),'max_gap_s':int(np.diff(tt).max()),'all_points_core':bool(core[ix].all()),'moving_points_ge2':int((s[ix]>=2).sum()),'timestamps':json.dumps(tt.tolist()),'row_indices':json.dumps(ids[ix].tolist())})
 return pd.DataFrame(rows)

def capacity(c):
 v=c.groupby('mmsi').size()
 return {'ships':len(v),'windows':len(c),'ships_ge5':int((v>=5).sum()),'ships_ge10':int((v>=10).sum()),'top300_cap20':int(v.clip(upper=20).nlargest(300).sum()),'top300_cap30':int(v.clip(upper=30).nlargest(300).sum()),'all_cap20':int(v.clip(upper=20).sum())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--regions',nargs='*');ap.add_argument('--registry',type=Path);args=ap.parse_args()
 region_sources=REGIONS if args.registry is None else {r:(m['canonical_file'],m['core_bbox']) for r,m in json.loads(args.registry.read_text())['regions'].items()}
 (OUT/'candidates').mkdir(parents=True,exist_ok=True)
 if not (OUT/'east/shanghai_standard.parquet').exists():normalize_shanghai()
 report={}
 for region,(rel,bbox) in region_sources.items():
  if args.regions and region not in args.regions:continue
  p=OUT/rel;d=pd.read_parquet(p).reset_index(drop=True);assert d.timestamp.between(1500000000,1900000000).all()
  c=candidates(d,region,bbox);c.to_parquet(OUT/f'candidates/{region}.parquet',index=False)
  report[region]={'canonical_file':str(p),'sha256':sha(p),'rows':len(d),'reported_ids':int(d.mmsi.nunique()),'core_bbox':bbox,'observed_extent_bbox':[float(d.lon.min()),float(d.lat.min()),float(d.lon.max()),float(d.lat.max())],'row_index_semantics':'Zero-based physical Parquet row position after read; never persisted DataFrame index labels','time_start':int(d.timestamp.min()),'time_end':int(d.timestamp.max()),'base':capacity(c),'duration660':capacity(c[c.duration_s>=660]),'duration660_gap300':capacity(c[(c.duration_s>=660)&(c.max_gap_s<=300)]),'all_core_duration660_gap300':capacity(c[(c.duration_s>=660)&(c.max_gap_s<=300)&c.all_points_core])}
  print(region,json.dumps(report[region]),flush=True)
 (OUT/('capacity_uniform_'+('_'.join(args.regions) if args.regions else 'all')+'.json')).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
