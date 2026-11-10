"""Pre-score frozen supplemental rolling-window event inputs."""
from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,sys
import numpy as np,pandas as pd
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');D=ROOT/'formal_20261101/cases/v6';P=D/'rolling_protocol_prescore.json';assert not P.exists();assert not (D/'case_scores.json').exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
ctx=module(ROOT/'balanced_global_v2/context/context_features.py','rolling_context');fb=module(ROOT/'framework_validation_v1_20261028/field_builder.py','rolling_field');gate=module(ROOT/'balanced_global_v2/causal_gate_main_v1_20261027/gate_builder.py','rolling_gate');cfg=ctx.ContextConfig();cap=float(json.loads((ROOT/'balanced_global_v2/causal_gate_main_v1_20261027/gate_cap.json').read_text())['cap_mps'])
source=(D/'case_prepare_v6.py').read_text();tree=ast.parse(source);selected=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('normalize','prepare')];assert len(selected)==2;exec(compile(ast.Module(body=selected,type_ignores=[]),str(D/'case_prepare_v6.py'),'exec'),globals())
protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'supplementary overlapping 23-minute trailing windows; pre-model-score fixed and not independent events','rule':'For every valid observed target report, when at least 300 s since last selected decision and decision is within frozen case interval, take available reports no later than decision and >= decision−1380 s; thin backwards to >=60 s, retain most recent20, require >=8; no score-based choices. All model arms use identical windows.','MATSON_decision_utc':['2017-01-13T16:01:56+00:00','2017-01-13T19:58:54+00:00'],'SARA_decision_utc':['2024-07-16T12:00:00+00:00','2024-07-16T22:00:00+00:00'],'notes':['window samples overlap and are not independent','original non-overlap cases retained unchanged','SARA event time unknown, no incident onset assignment','timestamp filtering is causal for observations; receiver coverage not established'],'fixed_case_protocol_sha256':sha(D/'case_protocol_prescore.json'),'input_code_lock_sha256':sha(D/'case_input_code_lock.json'),'source_builder_sha256':sha(D/'case_prepare_v6.py')};P.write_text(json.dumps(protocol,indent=2))
haw=ROOT/'hawaii_event_feasibility_20261031/canonical_context_v2.parquet'; sar=ROOT/'external_event_evidence_20261031/sara_regina_20240716_1200_2200_utc_noaa_raw.csv';miami=Path('/mnt/sda/WangZijia/AIS/data/interim/ais_miami_enriched.parquet')
hdf=normalize(pd.read_parquet(haw,columns=['mmsi','timestamp','lat','lon','sog','cog']));raw=pd.read_csv(sar,dtype={'MMSI':str});raw['timestamp']=pd.to_datetime(raw['utc_iso'],utc=True).dt.as_unit('s').astype('int64');raw=normalize(raw.rename(columns={'MMSI':'mmsi','LAT':'lat','LON':'lon','SOG':'sog','COG':'cog'})[['mmsi','timestamp','lat','lon','sog','cog']]);m=pd.read_parquet(miami,columns=['mmsi','timestamp','lat','lon','sog','cog']);m=m[m.mmsi.astype(str)!='334979000'];m=normalize(pd.concat([m,raw],ignore_index=True))
def windows(df,mid,start,end):
 d=df[(df.mmsi==mid)&(df.sog>=0)&(df.sog<102.2)&(df.cog>=0)&(df.cog<360)&df.lat.between(-90,90)&df.lon.between(-180,180)].sort_values('timestamp');tt=d.timestamp.to_numpy(float);lo=pd.Timestamp(start).timestamp();hi=pd.Timestamp(end).timestamp();out=[];last=-1e20
 for q in tt[(tt>=lo)&(tt<=hi)]:
  if q-last<300:continue
  pool=tt[(tt>=q-1380)&(tt<=q)];chosen=[]
  for x in pool[::-1]:
   if not chosen or chosen[-1]-x>=60:chosen.append(float(x))
  chosen=chosen[:20][::-1]
  if len(chosen)>=8:out.append({'timestamps':chosen});last=q
 return out
hw=windows(hdf,'366792000',*protocol['MATSON_decision_utc']);sw=windows(raw,'334979000',*protocol['SARA_decision_utc']);assert hw and sw
ha,hm=prepare(hdf,'366792000',[-159.,19.,-154.,23.],hw,'MATSON');sa,sm=prepare(m,'334979000',[-80.30,25.40,-79.80,26.20],sw,'SARA');data={}
for prefix,x in [('MATSON',ha),('SARA',sa)]:
 for k,v in x.items():data[prefix+'__'+k]=v
np.savez_compressed(D/'rolling_inputs.npz',**data); (D/'rolling_input_audit.json').write_text(json.dumps({'protocol_sha256':sha(P),'input_npz_sha256':sha(D/'rolling_inputs.npz'),'cases':{'MATSON':hm,'SARA':sm},'report_18utc':'Reference event-table time, not exact pointwise label','all_windows_overlap':True},indent=2));print('ROLLING_COUNTS',len(hw),len(sw),'contains_event_18utc',[r['end_utc'] for r in hm if '17:58' in r['end_utc'] or '18:01' in r['end_utc']],flush=True)
