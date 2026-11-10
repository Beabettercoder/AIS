"""Sample accepted source cohorts directly from canonical parquet without Source class."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json
import numpy as np
import pandas as pd

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');OUT=ROOT/'relation_independent_audit_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai');NM=1852.;KNOT=1852/3600
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();RUN=args.run
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((RUN/'dataset_manifest.json').read_text(encoding='utf8'));config=json.loads((RUN/'protocol.json').read_text(encoding='utf8'))
registry=json.loads((ROOT/'release_20261026/source_registry.json').read_text(encoding='utf8'))['regions']
pool=json.loads((RUN/'candidate_pool_manifest.json').read_text(encoding='utf8'));targets={s:set(pool['target_ids'][s]) for s in ('train','val','test')}
sel=pd.read_parquet(ROOT/'release_20261026/selected_windows.parquet',columns=['mmsi','split']);orig={s:set(sel.loc[sel.split==s,'mmsi'].astype(str)) for s in targets}
group=pd.read_parquet(ROOT/'release_20261026/global_all_mmsi_split.parquet',columns=['mmsi']);allids=set(group.mmsi.astype(str))
allowed={s:allids-set().union(*(targets[q]|orig[q] for q in targets if q!=s)) for s in targets}
selected=[]
for region in REGIONS:
 for ver in ('original','relaxed'):
  for split in ('train','val','test'):
   rows=[x for x in manifest['samples'] if x['region']==region and x['version']==ver and x['split']==split]
   rows.sort(key=lambda r:hashlib.sha256(('audit-source:'+ver+':'+r['window_id']).encode()).digest())
   selected.extend(rows[:3])
byregion=defaultdict(list)
for e in selected:byregion[e['region']].append(e)
metrics={'max_target_position_error_m':0.,'max_target_speed_error_kn':0.,'max_neighbor_position_error_m':0.,'max_neighbor_speed_error_kn':0.}
per=[]
for region in REGIONS:
 meta=registry[region];path=Path(meta['canonical_file']);assert sha(path)==meta['sha256']
 d=pd.read_parquet(path,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
 d['mid']=d.mmsi.astype(str);d['ordinal']=np.arange(len(d));d=d.sort_values(['mid','timestamp','ordinal'],kind='stable').drop_duplicates(['mid','timestamp'],keep='last').reset_index(drop=True)
 bbox=meta['core_bbox'];lon=(bbox[0]+bbox[2])/2;lat=(bbox[1]+bbox[3])/2
 d['x']=(d.lon-lon)*111195*np.cos(np.deg2rad(lat));d['y']=(d.lat-lat)*111195
 good=(np.isfinite(d.x)&np.isfinite(d.y)&np.isfinite(d.sog)&np.isfinite(d.cog)&(d.sog>=0)&(d.sog<102.2)&(d.cog>=0)&(d.cog<360))
 g=d.groupby('mid',sort=False);dt=d.timestamp-g.timestamp.shift(1);dx=d.x-g.x.shift(1);dy=d.y-g.y.shift(1);prev_good=good.groupby(d.mid,sort=False).shift(1).fillna(False).astype(bool)
 a=np.deg2rad(d.cog);vx=d.sog*KNOT*np.sin(a);vy=d.sog*KNOT*np.cos(a);pvx=vx.groupby(d.mid,sort=False).shift(1);pvy=vy.groupby(d.mid,sort=False).shift(1)
 d['good']=good;d['valid']=good&prev_good&(dt>0)&(dt<=300)
 e=np.sqrt((dx/dt-(vx+pvx)/2)**2+(dy/dt-(vy+pvy)/2)**2);d['w']=np.where(d.valid,np.exp(-np.maximum(0,e-2.5)**2/8),0.)
 byid={m:x for m,x in d.groupby('mid',sort=False)}
 for ent in byregion[region]:
  target=byid[ent['mmsi']];timestamps=np.array(ent['source_timestamps']);seq=target.set_index('timestamp').loc[timestamps]
  with np.load(RUN/ent['files']['base']['path'],allow_pickle=False) as z:base={k:z[k] for k in z.files}
  xy=seq[['x','y']].to_numpy();speed=seq.sog.to_numpy();course=seq.cog.to_numpy()
  ep=float(np.max(np.abs(xy-base['source_target_pos'])));es=float(np.max(np.abs(speed-base['source_target_sog_kn'])));assert ep<1e-6 and es<1e-9
  metrics['max_target_position_error_m']=max(metrics['max_target_position_error_m'],ep);metrics['max_target_speed_error_kn']=max(metrics['max_target_speed_error_kn'],es)
  assert np.array_equal(course,base['source_target_cog_deg'])
  t=timestamps;anchor=max(2,len(t)//2-1);when=float(t[anchor]);subset=d[(d.timestamp>=when-300)&(d.timestamp<=when)&(d.mid!=ent['mmsi'])]
  subset=subset.drop_duplicates('mid',keep='last');subset=subset[subset.mid.isin(allowed[ent['split']])]
  dist=np.sqrt((subset.x-xy[anchor,0])**2+(subset.y-xy[anchor,1])**2);angle=np.abs((subset.cog-course[anchor]+180)%360-180);difference=np.abs(subset.sog-speed[anchor]);cfg=config['versions'][ent['version']]
  q=subset[subset.good&subset.valid&(subset.w>=.3)&(subset.sog>=3)&(dist<=cfg['radius_nm']*NM)&(angle<=cfg['heading_deg'])&(difference<=cfg['delta_speed_kn'])]
  got=sorted(q.mid.tolist())[:cfg['max_members']];assert got==ent['cohort_mmsi'],(region,ent['version'],ent['window_id'],got,ent['cohort_mmsi'])
  offs=base['member_offsets']
  for j,mid in enumerate(got):
   m=byid[mid];h=m[(m.timestamp>=t[0]-300)&(m.timestamp<=t[-1])&m.good];lo=int(offs[j]);hi=int(offs[j+1]);assert np.array_equal(h.timestamp.to_numpy(float),base['source_neighbor_t'][lo:hi])
   ex=float(np.max(np.abs(h[['x','y']].to_numpy()-base['source_neighbor_pos'][lo:hi])));es=float(np.max(np.abs(h.sog.to_numpy()-base['source_neighbor_sog_kn'][lo:hi])));assert ex<1e-6 and es<1e-9
   metrics['max_neighbor_position_error_m']=max(metrics['max_neighbor_position_error_m'],ex);metrics['max_neighbor_speed_error_kn']=max(metrics['max_neighbor_speed_error_kn'],es)
   assert np.array_equal(h.cog.to_numpy(),base['source_neighbor_cog_deg'][lo:hi])
  per.append({'region':region,'version':ent['version'],'split':ent['split'],'window_id':ent['window_id'],'cohort_size':len(got)})
 print('RAW_COHORT_OK',region,len(byregion[region]),flush=True)
del d
out={'status':'passed','scope':'up to 3 SHA256-ranked accepted sources per region/version/split; directly reconstruct historical latest-per-identity and anchor cohort from canonical parquet without Source','samples':len(per),'coverage':per,'errors':metrics,'dataset_sha256':sha(RUN/'dataset_manifest.json'),'script_sha256':sha(Path(__file__)),
     'limitations':'sampling only; does not independently verify every candidate rejection or separate receiver coverage'}
with (OUT/(RUN.name+'_raw_cohort_audit.json')).open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k!='coverage'},ensure_ascii=False,indent=2),flush=True)
