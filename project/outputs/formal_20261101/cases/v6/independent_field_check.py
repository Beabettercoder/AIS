"""Independent re-evaluation of single-vessel event inputs from source data; no scores read."""
from pathlib import Path
import numpy as np,pandas as pd,hashlib,json,sys,importlib.util
R=Path('/mnt/sda/WangZijia/AIS/outputs');D=R/'formal_20261101/cases/v6';sys.path.insert(0,str(R/'balanced_global_v2/context_adapter_v1_20261026/neighborhood'))
from audit_neighborhood import Source,geometry,wrap,sha
fs=importlib.util.spec_from_file_location('field_independent',R/'framework_validation_v1_20261028/field_builder.py');fb=importlib.util.module_from_spec(fs);fs.loader.exec_module(fb)
miami=Path('/mnt/sda/WangZijia/AIS/data/interim/ais_miami_enriched.parquet');raw=R/'external_event_evidence_20261031/sara_regina_20240716_1200_2200_utc_noaa_raw.csv'
a=pd.read_parquet(miami,columns=['mmsi','timestamp','lat','lon','sog','cog']);a=a[a.mmsi.astype(str)!='334979000'];b=pd.read_csv(raw,dtype={'MMSI':str});b['timestamp']=pd.to_datetime(b['utc_iso'],utc=True).dt.as_unit('s').astype('int64');b=b.rename(columns={'MMSI':'mmsi','LAT':'lat','LON':'lon','SOG':'sog','COG':'cog'})[['mmsi','timestamp','lat','lon','sog','cog']];a=pd.concat([a,b],ignore_index=True);a['mmsi']=a.mmsi.astype(str)
for k in ('timestamp','lat','lon','sog','cog'):a[k]=pd.to_numeric(a[k],errors='coerce')
a=a.replace([np.inf,-np.inf],np.nan).dropna(subset=['timestamp']).copy();a['rowid']=np.arange(len(a));a=a.sort_values(['mmsi','timestamp','rowid'],kind='stable').drop_duplicates(['mmsi','timestamp'],keep='last');a=a.drop(columns='rowid');p=D/'sara_canonical_for_audit.parquet';assert not p.exists();a.to_parquet(p,index=False)
meta=[('MATSON',{'canonical_file':str(R/'hawaii_event_feasibility_20261031/canonical_context_v2.parquet'),'sha256':sha(R/'hawaii_event_feasibility_20261031/canonical_context_v2.parquet'),'core_bbox':[-159.,19.,-154.,23.]}),('SARA',{'canonical_file':str(p),'sha256':sha(p),'core_bbox':[-80.30,25.40,-79.80,26.20]})]
with np.load(D/'realcase_inputs.npz',allow_pickle=False) as z: x={k:z[k].copy() for k in z.files}
md=json.loads((D/'realcase_input_audit.json').read_text());out={}
for case,source in meta:
 src=Source(source);allids=set(src.ids);rows=md['cases'][case];count=0;fieldcheck=0;err=0.;errrho=0.;positions=[]
 for i,row in enumerate(rows):
  ts=np.array([pd.Timestamp(t).timestamp() for t in row['times_utc']],float);pp,ss,cc=src.source_target(row['mmsi'],ts)
  for j,t in enumerate(ts):
   q=src.latest(t,row['mmsi'],allids);distance=np.linalg.norm(src.p[q]-pp[j],axis=1);ng=int((distance<=8*1852).sum()); assert ng==row['neighbors_geo_per_step'][j]
   if ss[j]<.5:plain=q[:0];trusted=q[:0]
   else:
    plain=q[src.good[q]&(src.s[q]>=.5)&(distance<=8*1852)&(np.abs(wrap(src.c[q]-cc[j]))<=45)];trusted=plain[src.valid[plain]&(src.w[plain]>=.3)]
   assert len(plain)==row['neighbors_plain_per_step'][j] and len(trusted)==row['neighbors_trusted_per_step'][j]
   if len(trusted)>=4:
    positions_t=src.p[trusted]-pp[j];ww=src.w[trusted]*np.exp(-.5*np.sum(positions_t**2,axis=1)/fb.SIGMA**2);ok,_,_,_=geometry(positions_t,ww,8*1852)
   else:ok=False
   ff=x[case+'__field_valid'][i,j]
   if ff:
    assert ok and len(plain)>=4
    pos=src.p[plain]-pp[j];theta=np.deg2rad(src.c[plain]);vel=np.column_stack([src.s[plain]*np.sin(theta),src.s[plain]*np.cos(theta)])*1852/3600;w=np.where(src.valid[plain],src.w[plain],0.);d=np.exp(-.5*np.sum(pos**2,axis=1)/fb.SIGMA**2)
    coef=fb.independently_fit(pos,vel,d*w);grad=coef[1:,:].T/fb.R;want=np.r_[x[case+'__self'][i,j,2:4]-coef[0],np.trace(grad),grad[1,0]-grad[0,1]];got=x[case+'__field'][i,j];err=max(err,float(np.max(np.abs(want-got))));rho=min(1,len(w)/4)*w.mean();errrho=max(errrho,abs(float(x[case+'__rho'][i,j])-rho));fieldcheck+=1
   elif not ok:assert not ff
   count+=1
 out[case]={'windows':len(rows),'steps_checked':count,'affine_steps_checked':fieldcheck,'max_abs_field_diff':err,'max_abs_rho_diff':errrho}
 assert err<2e-5 and errrho<1e-6
print(json.dumps(out,indent=2))
(D/'independent_field_check.json').write_text(json.dumps(out,indent=2))
