"""Build original phase-cumulative relation-model inputs for frozen Hawaii OOD windows.
No target scores, incident labels, injected clean trajectory, or test labels are read.
"""
from pathlib import Path
from collections import Counter
import json,sys,hashlib,importlib.util
import numpy as np,pandas as pd
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OUT=ROOT/'hawaii_event_feasibility_20261031'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def module(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sys.path.insert(0,str(ROOT/'balanced_global_v2/original_method_and_relation_v1_20261026/relation/vendor'))
from audit_neighborhood import Source,wrap,NM
rel=module(ROOT/'relation_repair_20261030/builder.py','orig_rel_builder');heading=module(ROOT/'heading_evidence_pilot_20261030/feature_builder.py','orig_heading_builder')
frozen=json.loads((OUT/'frozen_ood_models_thresholds_v2.json').read_text());windows=json.loads((OUT/'observed_event_windows.json').read_text())['windows'];assert sha(OUT/'observed_event_windows.json')==frozen['windows_sha256']; assert sha(OUT/'canonical_context_v2.parquet')==frozen['context_parquet_sha256']
meta=json.loads((OUT/'canonical_context_v2.json').read_text()); assert meta['canonical_sha256']==frozen['context_parquet_sha256']
src=Source({'canonical_file':str(OUT/'canonical_context_v2.parquet'),'sha256':meta['canonical_sha256'],'core_bbox':meta['core_bbox']})
headers=pd.read_parquet(OUT/'canonical_context_v2.parquet',columns=['mmsi','timestamp','heading']).reset_index(drop=True);headers['id']=headers.mmsi.astype(str);headers['order']=np.arange(len(headers));headers=headers.sort_values(['id','timestamp','order'],kind='stable').drop_duplicates(['id','timestamp'],keep='last');heading_map={(str(r.id),float(r.timestamp)):r.heading for r in headers.itertuples()}
targets=pd.read_parquet(ROOT/'balanced_global_v2/release_20261026/selected_windows.parquet',columns=['mmsi','split']);blocked=set(targets.loc[targets['split'].isin(['train','val']),'mmsi'].astype(str));m=json.loads((ROOT/'balanced_global_v2/relation_dual_pressure_v2_20261027/dataset_manifest.json').read_text());blocked.update(str(s['mmsi']) for s in m['samples'] if s.get('split') in ('train','val'));allowed=set(src.ids)-blocked
N=len(windows);f=np.zeros((N,2,20,8),np.float32);cnt=np.zeros((N,20),np.int16);hs=np.zeros((N,2,20,5),np.float32);causal=np.zeros((N,20,7),np.float32);tt=np.zeros((N,20),np.float64);mask=np.zeros((N,20),bool);records=[]
for i,x in enumerate(windows):
 t=np.asarray(x['timestamps'],float);n=len(t);target=x['mmsi'];p,s,c=src.source_target(target,t);a=max(2,n//2-1);at=float(t[a]);prefix=bool(s[a]>=3 and np.median(s[:a+1])>=3 and t[-1]-at>=180)
 if prefix:
  ids=src.latest(at,target,allowed);d=np.linalg.norm(src.p[ids]-p[a],axis=1);ang=np.abs(wrap(src.c[ids]-c[a]));dv=np.abs(src.s[ids]-s[a]);keep=src.good[ids]&src.valid[ids]&(src.w[ids]>=.3)&(src.s[ids]>=3)&(d<=5*NM)&(ang<=30)&(dv<=4.5);cohort=ids[keep];cohort=cohort[np.argsort(src.ids[cohort],kind='stable')][:8];members=src.ids[cohort].tolist()
 else:members=[]
 nt=[];np_=[];ns=[];nc=[];hv=[];hm=[];offsets=[0]
 for member in members:
  idx=src.history(member,t[0]-300,t[-1]);idx=idx[src.good[idx]]
  for ix in idx:
   nt.append(src.t[ix]);np_.append(src.p[ix]);ns.append(src.s[ix]);nc.append(src.c[ix]);hh=heading_map[(str(member),float(src.t[ix]))]
   try:hh=float(hh)
   except (ValueError,TypeError):hh=float('nan')
   good=bool(np.isfinite(hh) and 0<=hh<360);hv.append(hh if good else np.nan);hm.append(good)
  offsets.append(len(nt))
 nt=np.asarray(nt,float);np_=np.asarray(np_,float).reshape(-1,2);ns=np.asarray(ns,float);nc=np.asarray(nc,float);hv=np.asarray(hv,float);hm=np.asarray(hm,bool)
 o={'target_t':t[None,:],'target_pos':p[None,:,:],'target_sog_kn':s[None,:],'target_cog_deg':c[None,:],'neighbor_t':nt[None,:],'neighbor_pos':np_[None,:,:],'neighbor_sog_kn':ns[None,:],'neighbor_cog_deg':nc[None,:],'cohort_mmsi':np.asarray(members),'member_offsets':np.asarray(offsets,int)}
 field,count=rel.calc(o,0,target); hfeatures=heading.heading_features(o,0,hv,hm,target,field[0,:,7]>0)
 v=rel.velocity(s,c);r=np.column_stack([p-p[0],v,s,np.sin(np.deg2rad(c)),np.cos(np.deg2rad(c))]).astype(np.float32);w,wvalid=rel.core.absolute_weights(t,p,s,c);cg,_=rel.gate.causal(r,w,np.ones(n,bool),wvalid,t,13.73566722869873)
 for j in range(1,n):
  if w[j]<.2:cg[j,2:7]=cg[j-1,2:7]
 f[i,:,:n,:]=field;cnt[i,:n]=count;hs[i,:,:n,:]=hfeatures;causal[i,:n]=cg;tt[i,:n]=t;mask[i,:n]=True
 hit=(field[0,:,7]>0); maxage=max((float(at-src.t[ix]) for ix in (cohort if prefix else [])),default=None)
 records.append({**x,'event_observed_window_index':i,'prefix_qualified':prefix,'n_anchor_members':len(members),'cohort':members,'max_anchor_report_age_s':maxage,'field_supported_steps':int(hit.sum()),'field_affine_steps':int((field[0,:,7]==2).sum()),'field_zero_order_steps':int((field[0,:,7]==1).sum()),'neighbor_heading_available_steps':int((hfeatures[1,:,2]>0).sum()),'out_of_training_domain':True,'no_field_due_low_speed':not prefix,'coverage_certified':False})
def feature35(d,arm):
 mask=d['mask'];b,t=mask.shape;out=np.zeros((b,t,35),np.float32);valid=np.zeros((b,t),bool);selfx=d['causal7'];tm=d['time_s'];v=selfx[:,:,2:4];sp=selfx[:,:,4];ss=selfx[:,:,5:7]
 out[:,:,:8]=np.concatenate((v,sp[...,None],ss,v-v[:,:1],(sp-sp[:,:1])[...,None]),axis=-1)
 if arm!='C0':
  f=d['base__field'][:,0 if arm=='F0' else 1];count=d['base__count'];bg=f[:,:,:2];valid=mask&(f[:,:,7]>0);r=v-bg
  first=np.argmax(valid,axis=1);rows=np.arange(b);anchorbg=bg[rows,first];anchorr=r[rows,first];at=tm[rows,first];dt=tm-at[:,None];can=valid&(dt>0)
  bgdelta=np.where(valid[...,None],bg-anchorbg[:,None,:],0);rdelta=np.where(valid[...,None],r-anchorr[:,None,:],0)
  ratebg=np.divide(bgdelta,dt[...,None],out=np.zeros_like(bgdelta),where=can[...,None]);rater=np.divide(rdelta,dt[...,None],out=np.zeros_like(rdelta),where=can[...,None]);support=np.minimum(count/4.,1.).astype(np.float32) if arm=='F0' else f[:,:,6]
  c=np.concatenate((bg,bgdelta,ratebg,r,rdelta,rater,f[:,:,4:6],support[...,None],(f[:,:,7]==2)[...,None].astype(np.float32)),axis=-1);out[:,:,8:24]=np.where(valid[...,None],c,0)
  sx=selfx[:,:,5];cy=selfx[:,:,6];dot=lambda z,x,y:z[:,:,0]*x+z[:,:,1]*y
  bg_long=dot(bg,sx,cy);bg_lat=dot(bg,cy,-sx);r_long=dot(r,sx,cy);r_lat=dot(r,cy,-sx);bmag=np.linalg.norm(bg,axis=-1);tmag=np.linalg.norm(v,axis=-1);delta_t=tm-tm[rows,first,None]
  db=bmag-bmag[rows,first,None];ds=tmag-tmag[rows,first,None];good=valid&(delta_t>0);rb=np.divide(db,delta_t,out=np.zeros_like(db,dtype=np.float64),where=good);rt=np.divide(ds,delta_t,out=np.zeros_like(ds,dtype=np.float64),where=good)
  vals=np.stack((bg_long,bg_lat,r_long,r_lat,bmag,tmag-bmag,db,ds,ds-db,rb,rt-rb),axis=-1);out[:,:,24:]=np.where(valid[:,:,None],vals,0)
 out[~mask]=0;return out,valid

d={'mask':mask,'causal7':causal,'time_s':tt,'base__field':f,'base__count':cnt}
arrays={}
proto=json.loads(Path(frozen['models'][0]['model_protocol']).read_text());mu=np.asarray(proto['mean'],float);sd=np.asarray(proto['std'],float);assert mu.shape==(40,) and sd.shape==(40,)
for arm in ('C0','F0','F1'):
 xx,cm=feature35(d,arm);z=(xx-mu[:35])/sd[:35];z[...,8:]*=cm[...,None];z[~mask]=0
 extra=np.zeros((N,20,5),np.float64)
 if arm!='C0':
  a=hs[:,0 if arm=='F0' else 1];active=np.stack((a[...,2]>0,a[...,2]>0,np.ones((N,20),bool),a[...,4]>0,np.ones((N,20),bool)),axis=-1);extra=(a-mu[35:])/sd[35:];extra[~active]=0;extra[~mask]=0
 z=np.concatenate((z,extra),axis=-1).astype(np.float32);assert np.isfinite(z).all();arrays[arm]=z
np.savez_compressed(OUT/'ood_inputs.npz',**arrays,mask=mask,time_s=tt,field=f,heading=hs)
(OUT/'ood_windows.json').write_text(json.dumps({'input_lock_sha256':sha(OUT/'frozen_ood_models_thresholds_v2.json'),'projection_bbox_origin':meta['core_bbox'],'all_windows':records,'inputs_sha256':sha(OUT/'ood_inputs.npz'),'counts_by_event':{name:sum(x['name']==name for x in records) for name in sorted(set(x['name'] for x in records))},'caveats':['all windows are outside original relation eligibility; case2 is stationary, none has >=3 aligned same-direction members','offline fixed cohort and phase segmentation; not online','no incident labels used as model input','receiver spatial coverage unknown','all window-level true classes unknown']},indent=2),encoding='utf8')
print('WINDOWS',N,'BY_EVENT',Counter(x['name'] for x in records),'ANCHOR_MEMBERS',Counter(x['n_anchor_members'] for x in records),'INPUT_SHA',sha(OUT/'ood_inputs.npz'),flush=True)

