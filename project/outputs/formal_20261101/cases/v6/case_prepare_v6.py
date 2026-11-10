from pathlib import Path
import json,hashlib,importlib.util,sys,datetime
import numpy as np,pandas as pd
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OUT=ROOT/'formal_20261101/cases/v6';OUT.mkdir(parents=True,exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
ctxpath=ROOT/'balanced_global_v2/context/context_features.py';fieldpath=ROOT/'framework_validation_v1_20261028/field_builder.py';gatepath=ROOT/'balanced_global_v2/causal_gate_main_v1_20261027/gate_builder.py'
ctx=module(ctxpath,'case_source_context');fb=module(fieldpath,'case_field');gate=module(gatepath,'case_gate');cfg=ctx.ContextConfig();assert cfg.radius_nm==8 and cfg.min_heading_speed_kn==.5
haw=ROOT/'hawaii_event_feasibility_20261031/canonical_context_v2.parquet'; hawmeta=ROOT/'hawaii_event_feasibility_20261031/canonical_context_v2.json'; frozenhaw=ROOT/'hawaii_event_feasibility_20261031/observed_event_windows.json'
sar=ROOT/'external_event_evidence_20261031/sara_regina_20240716_1200_2200_utc_noaa_raw.csv';miami=Path('/mnt/sda/WangZijia/AIS/data/interim/ais_miami_enriched.parquet');capfile=ROOT/'balanced_global_v2/causal_gate_main_v1_20261027/gate_cap.json';cap=float(json.loads(capfile.read_text())['cap_mps'])
required=[ctxpath,fieldpath,gatepath,haw,hawmeta,frozenhaw,sar,miami,capfile];inputs={str(p):sha(p) for p in required}
protocol={'purpose':'single-agent frozen F1 model: two real-vessel OOD replay cases, no threshold selection from cases','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'architectures':['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'],'seeds':[0,1,2],'weights':'formal_20261101/single/runs/{architecture}/seed{seed}/f1_bundle/final.pt after all 24 locked','models':'all requested pretrained fixed25 F1 models, no event score based selection','target_channels':'cumulative relative positions in meters, vx/vy m/s, SOG knots, sin/cos COG; causal gate cap from training','field':'original individual pilot: 8nm, sigma4nm, target speed>=0.5kn, same direction<45, freshness<=300s, trusted geometry >=4 then common weighted F0/F1 affine; unsupported field = rho0, no invented zero-order','context_coverage':'receiver coverage not independently verified, unknown; background reported by available local source only','target_window_rule':'MATSON reuse previously frozen non-overlap 23min timestamps; SARA 12-22UTC non-overlap 23min bins anchored at first valid report, gap >1800s resets, >=60s spacing, last20, >=8 observations','sara_background':'miami_enriched.parquet; target from full NOAA raw per-event extract replaces duplicate target rows; projection uses original extraction bounding box lon[-80.3,-79.8] lat[25.4,26.2] from scripts/00a; neighborhood source may be geographically/time filtered','causality':'each target step only already available same-direction latest neighbor<=300s','time_axis':'window end timestamps UTC; SARA precise bridge-contact time unknown, MATSON 18:00 UTC official-dataset reported event with uncertainty','warnings':['real event interval is weak trajectory-level evidence, no per-window label','single-agent event-time scores do not establish recall or calibration','Hawaii target is 0.1kn mostly, out of training domain','source report preprocessed background not proof full receiver coverage','no score-based event/model selection'],'input_sha256':inputs}
p=OUT/'case_protocol_prescore.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf8')
def normalize(df):
 df=df.copy();df['mmsi']=df['mmsi'].astype(str)
 for k in ('timestamp','lat','lon','sog','cog'):df[k]=pd.to_numeric(df[k],errors='coerce')
 df=df.replace([np.inf,-np.inf],np.nan).dropna(subset=['timestamp'])
 # Keep invalid/latest background messages: select latest first, validate second, like the original index.
 df['_order']=np.arange(len(df));df=df.sort_values(['mmsi','timestamp','_order'],kind='stable').drop_duplicates(['mmsi','timestamp'],keep='last');return df
hdf=pd.read_parquet(haw,columns=['mmsi','timestamp','lat','lon','sog','cog']);hdf=normalize(hdf); hw=json.loads(frozenhaw.read_text())['windows']; hw=[x for x in hw if str(x['mmsi'])=='366792000'];assert len(hw)==5
raw=pd.read_csv(sar,dtype={'MMSI':str});raw['timestamp']=pd.to_datetime(raw['utc_iso'],utc=True).dt.as_unit('s').astype('int64')
raw=raw.rename(columns={'MMSI':'mmsi','LAT':'lat','LON':'lon','SOG':'sog','COG':'cog'})[['mmsi','timestamp','lat','lon','sog','cog']];raw=normalize(raw)
m=pd.read_parquet(miami,columns=['mmsi','timestamp','lat','lon','sog','cog']);m=m[m['mmsi'].astype(str)!='334979000'];m=normalize(pd.concat([m,raw],ignore_index=True))
# Original non-overlapping 23-min source-style windows. Retain all eligible under rule, no event score-dependent selection.
s=raw[(raw.sog>=0)&(raw.sog<102.2)&(raw.cog>=0)&(raw.cog<360)&raw.lat.between(-90,90)&raw.lon.between(-180,180)].sort_values('timestamp');times=s.timestamp.to_numpy(float); windows=[];seg=[]
for t in times:
 if seg and t-seg[-1]>1800:windows.append(seg);seg=[]
 seg.append(t)
if seg:windows.append(seg)
sw=[]
for seg in windows:
 first=seg[0];end=seg[-1];start=first
 while start<=end:
  vals=[t for t in seg if start<=t<start+1380];start+=1380;sel=[]
  for t in vals:
   if not sel or t-sel[-1]>=60:sel.append(t)
  sel=sel[-20:]
  if len(sel)>=8:sw.append({'timestamps':sel})
assert len(sw)>0

def prepare(df,mid,box,windows,case):
 lon=(box[0]+box[2])/2;lat=(box[1]+box[3])/2
 obs={'mmsi':df.mmsi.to_numpy(str),'time_s':df.timestamp.to_numpy(float),'x_m':(df.lon.to_numpy(float)-lon)*111195*np.cos(np.deg2rad(lat)),'y_m':(df.lat.to_numpy(float)-lat)*111195,'sog_kn':df.sog.to_numpy(float),'cog_deg':df.cog.to_numpy(float)}
 bg=ctx.BackgroundIndex(obs,cfg); lookup={(str(mid),float(t)):i for i,(mid,t) in enumerate(zip(bg.data['mmsi'],bg.data['time_s']))}; values=bg.values;data=bg.data
 rows=[]; xs=[]; masks=[];ts=[];fs=[];rhos=[];valids=[]
 for wi,win in enumerate(windows):
  tt=np.asarray(win['timestamps'],float); n=len(tt);assert 8<=n<=20
  ii=np.array([lookup[(mid,float(t))] for t in tt]);p=values['position'][ii];speed=data['sog_kn'][ii];course=data['cog_deg'][ii];vel=values['velocity'][ii]; localvalid=np.r_[False,(np.diff(tt)>0)&(np.diff(tt)<=300)];localw=np.ones(n);ee=np.linalg.norm(np.diff(p,axis=0)/np.diff(tt)[:,None]-(vel[1:]+vel[:-1])/2,axis=1);localw[1:]=np.where(localvalid[1:],np.exp(-np.maximum(0,ee-2.5)**2/8),1.)
  raw7=np.zeros((20,7),np.float32);raw7[:n]=np.column_stack([p-p[0],vel,speed,np.sin(np.deg2rad(course)),np.cos(np.deg2rad(course))]);tm=np.zeros(20);tm[:n]=tt;mask=np.arange(20)<n;w=np.zeros(20,dtype=np.float32);w[:n]=localw;valid=np.zeros(20,bool);valid[:n]=localvalid
  gated,_=gate.causal(raw7,w,mask,valid,tm,cap)
  for j in range(1,n):
   if w[j]<.2:gated[j,2:7]=gated[j-1,2:7]
  gated[~mask]=0;field=np.zeros((20,4),np.float32);rho=np.zeros(20,np.float32);fv=np.zeros(20,bool);ngeo=[];nplain=[];ntrust=[]; reasons=[];overlap=[]
  for j in range(n):
   cand=bg.latest(tt[j],mid,None); dist=np.linalg.norm(values['position'][cand]-p[j],axis=1);nge=int((dist<=8*1852).sum());c=ctx._context_at(bg,tt[j],mid,p[j],float(course[j]),float(speed[j]),None,cfg);es=c['neighbors'];plain=[q for q in es if q['plain_accepted']];ngeo.append(nge);nplain.append(len(plain));ntrust.append(c['count']);reasons.append(c['reason']);overlap.append([str(q['mmsi']) for q in plain])
   if not c['valid'] or len(plain)<4:continue
   jj=np.array([lookup[(str(q['mmsi']),float(q['time_s']))] for q in plain]);ps=values['position'][jj]-p[j];vs=values['velocity'][jj];ww=np.where(values['kinematic_valid'][jj],values['reliability'][jj],0.);dw=np.exp(-.5*np.sum(ps**2,axis=1)/fb.SIGMA**2);ids=data['mmsi'][jj]
   a=fb.fit_field(ps,vs,dw,ids,mid);b=fb.fit_field(ps,vs,dw*ww,ids,mid)
   if not (a['valid'] and b['valid']):continue
   B=b['coefficient']; grad=B[1:,:].T/fb.R;field[j,:]=np.r_[gated[j,2:4]-B[0],np.trace(grad),grad[1,0]-grad[0,1]];rho[j]=min(1,len(ww)/4)*float(ww.mean());fv[j]=True
  xs.append(gated);masks.append(mask);relative=tm-tm[0];relative[~mask]=0;ts.append(relative);fs.append(field);rhos.append(rho);valids.append(fv)
  rows.append({'case':case,'window_index':wi,'mmsi':mid,'first_utc':datetime.datetime.fromtimestamp(tt[0],datetime.timezone.utc).isoformat(),'end_utc':datetime.datetime.fromtimestamp(tt[-1],datetime.timezone.utc).isoformat(),'times_utc':[datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat() for t in tt],'count':n,'median_sog_kn':float(np.median(speed)),'max_sog_kn':float(np.max(speed)),'target_sog_lt2_n':int((speed<2).sum()),'max_report_gap_s':float(np.max(np.diff(tt))),'target_training_foundation_ok':bool((speed>=2).sum()>=4 and speed.max()>=3),'target_gap_policy_ok':bool(tt[-1]-tt[0]>=660 and np.diff(tt).max()<=300),'neighbors_geo_per_step':ngeo,'neighbors_plain_per_step':nplain,'neighbors_trusted_per_step':ntrust,'context_reason_per_step':reasons,'context_valid_steps':int(fv.sum()),'context_valid_per_step':fv[:n].tolist(),'rho_per_step':rho[:n].tolist(),'neighbor_ids_per_step':overlap,'receiver_coverage':'unknown'})
 return {'self':np.asarray(xs),'mask':np.asarray(masks),'time_s':np.asarray(ts),'field':np.asarray(fs),'rho':np.asarray(rhos),'field_valid':np.asarray(valids)},rows
ha,hm=prepare(hdf,'366792000',[-159.,19.,-154.,23.],hw,'MATSON');sa,sm=prepare(m,'334979000',[-80.30,25.40,-79.80,26.20],sw,'SARA')
N={}
for prefix,data in [('MATSON',ha),('SARA',sa)]:
 for k,v in data.items():N[prefix+'__'+k]=v
out=OUT/'realcase_inputs.npz'
if out.exists():raise FileExistsError(out)
np.savez_compressed(out,**N)
result={'protocol_sha256':sha(p),'input_npz_sha256':sha(out),'cases':{'MATSON':hm,'SARA':sm},'notes':['Field available only if original trusted geometry and both F0/F1 fits valid; no invented zero-order in individual model.','MATSON timestamps reuse previously frozen OOD windows; all are event association not pointwise label.','SARA interval not a collision label; exact event time unknown.','Coverage for both sources is unverified; no statement that the target has no nearby vessels.']}
(OUT/'realcase_input_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
for case,rows in result['cases'].items():print(case,'windows',len(rows),'observed_steps',sum(z['count'] for z in rows),'field_steps',sum(z['context_valid_steps'] for z in rows),'training_foundation_windows',sum(z['target_training_foundation_ok'] for z in rows),'max_observed_neighbor',max(max(z['neighbors_geo_per_step']) for z in rows),flush=True)
