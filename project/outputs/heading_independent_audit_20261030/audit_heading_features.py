from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OLD=ROOT/'balanced_global_v2/relation_dual_pressure_v2_20261027';CACHE=ROOT/'heading_coverage_20261030';FEAT=ROOT/'heading_evidence_pilot_20261030';FIELD=ROOT/'relation_repair_20261030';OUT=ROOT/'heading_independent_audit_20261030'
manifest=json.loads((OLD/'dataset_manifest_combined.json').read_text());rows=sorted([r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='val'],key=lambda r:(r['region'],r['window_id']))
with np.load(CACHE/'heading_val.npz',allow_pickle=False) as z:h={k:z[k].copy() for k in z.files}
with np.load(FIELD/'val_relation_field.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
with np.load(FEAT/'val_heading_features.npz',allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
hidx={str(w):i for i,w in enumerate(h['window_id'])};assert len(hidx)==len(h['window_id']);assert set(hidx)==set(r['window_id'] for r in rows)
for key in ('source','case','y','mmsi','region','length'):assert np.array_equal(d[key],f[key]),key
def weights(t,p,s,c):
 a=np.radians(c);v=np.column_stack((s*np.sin(a),s*np.cos(a)))*.514444444444;dt=np.diff(t);w=np.zeros(len(t));valid=(dt>0)&(dt<=300)&np.isfinite(p[:-1]).all(1)&np.isfinite(p[1:]).all(1)&np.isfinite(s[:-1])&np.isfinite(s[1:])&np.isfinite(c[:-1])&np.isfinite(c[1:]);e=np.linalg.norm(np.diff(p,axis=0)[valid]/dt[valid,None]-(v[1:][valid]+v[:-1][valid])/2,axis=1);w[1:][valid]=np.exp(-np.maximum(0,e-2.5)**2/8);return w
def calc(o,case,tidx,hh,hvalid,fieldvalid):
 if not fieldvalid:return np.zeros((2,5))
 t=o['target_t'][case];p=o['target_pos'][case];target_t=t[tidx];parts=[];phase=t[0] if tidx<len(t)//2 else t[len(t)//2]
 for lo,hi in zip(o['member_offsets'][:-1],o['member_offsets'][1:]):
  lo=int(lo);hi=int(hi);nt=o['neighbor_t'][case,lo:hi];pos=o['neighbor_pos'][case,lo:hi];s=o['neighbor_sog_kn'][case,lo:hi];c=o['neighbor_cog_deg'][case,lo:hi];q=np.searchsorted(nt,target_t,side='right')-1
  if q<0 or target_t-nt[q]>300:continue
  d=pos[q]-p[tidx];dist=np.exp(-np.dot(d,d)/(2*(4*1852.)**2));inds=np.flatnonzero((nt>=phase)&(nt<=target_t));inds=inds if len(inds) else np.array([q]);w=weights(nt,pos,s,c);hd=float(hh[lo+q]);valid=bool(hvalid[lo+q]) and np.isfinite(c[q]);angle=np.deg2rad(((hd-c[q]+180)%360)-180) if valid else 0;rateok=q>0 and hvalid[lo+q] and hvalid[lo+q-1] and 0<nt[q]-nt[q-1]<=300
  yaw=np.deg2rad(((hd-float(hh[lo+q-1])+180)%360)-180)/(nt[q]-nt[q-1]) if rateok else 0
  parts.append((dist,dist*w[inds].mean(),valid,np.sin(angle),np.cos(angle),rateok,yaw))
 res=np.zeros((2,5))
 if not parts:return res
 a=np.asarray(parts,float)
 for arm in range(2):
  wt=a[:,arm];total=wt.sum()
  if total<=1e-12:continue
  good=wt*a[:,2];den=good.sum();vg=wt*a[:,5];vd=vg.sum()
  if den>1e-12:res[arm,0]=np.dot(good,a[:,3])/den;res[arm,1]=np.dot(good,a[:,4])/den
  res[arm,2]=den/total
  if vd>1e-12:res[arm,3]=np.dot(vg,a[:,6])/vd
  res[arm,4]=vd/total
 return res
chosen=[]
for region in sorted(set(r['region'] for r in rows)):
 ix=[i for i,r in enumerate(rows) if r['region']==region];ix.sort(key=lambda i:hashlib.sha256(('heading-feature-audit:'+rows[i]['window_id']).encode()).hexdigest());chosen.extend(ix[:2])
counter=0;changed=0;maxerr=0.;maxfuture=0.;maxfuture_all=0.; views=('base','natural','suffix','combined_l100_m600')
for ri in chosen:
 row=rows[ri];hi=hidx[row['window_id']];start,end=h['sample_neighbor_offsets'][hi:hi+2];hh=h['neighbor_heading'][start:end].copy();hv=h['neighbor_valid'][start:end].copy();data={}
 for view in ('base','natural','suffix'):
  with np.load(OLD/row['files'][view]['path'],allow_pickle=False) as z:data[view]={k:z[k].copy() for k in ('target_t','target_pos','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets')}
  assert np.array_equal(data[view]['neighbor_t'][0],h['neighbor_t'][start:end])
 o=data['base'];combined={k:v.copy() for k,v in o.items()};combined['neighbor_sog_kn']=data['suffix']['neighbor_sog_kn'].copy()
 for m in row['stress']['members']:
  lo=int(combined['member_offsets'][m['member_index']]);q=lo+int(m['indices']['suffix']['report_index']);digest=hashlib.sha256(('framework-stress-position-v1:'+row['window_id']+':'+str(m['mmsi'])).encode()).digest();theta=2*np.pi*int.from_bytes(digest[:8],'big')/2**64;delta=600*np.array([np.cos(theta),np.sin(theta)]);combined['neighbor_pos'][:,q:q+2]+=delta
 data['combined_l100_m600']=combined
 for case in range(4):
  j=4*ri+case;assert d['source'][j]==row['window_id'] and d['case'][j]==case
  n=int(d['length'][j]);steps=sorted(set([0,n//4,n//2,3*n//4,n-1]))
  for view in views:
   o=data[view]
   for step in steps:
    valid=bool(d[view+'__field'][j,0,step,7]>0);x=calc(o,case,step,hh,hv,valid);stored=np.stack([f[view+'__F0'][j,step],f[view+'__F1'][j,step]]);err=float(np.max(np.abs(x-stored)));maxerr=max(maxerr,err);assert err<1e-5,(view,ri,case,step,err)
    # Mutate only future neighbor headings; observed-prefix outputs must be unchanged.
    future=o['neighbor_t'][case]>o['target_t'][case,step];hh2=hh.copy();hh2[future]=123.;xx=calc(o,case,step,hh2,hv,valid);future_err=float(np.max(np.abs(x-xx)));maxfuture=max(maxfuture,future_err);assert future_err==0
    altered={k:v.copy() for k,v in o.items()};altered['neighbor_pos'][case,future]+=100000.;altered['neighbor_sog_kn'][case,future]=90.;altered['neighbor_cog_deg'][case,future]=270.;xx=calc(altered,case,step,hh2,hv,valid);err_all=float(np.max(np.abs(x-xx)));maxfuture_all=max(maxfuture_all,err_all);assert err_all==0
    counter+=1;changed+=int(valid)
out={'sources':len(chosen),'comparisons':counter,'field_valid_comparisons':changed,'max_independent_abs_error':maxerr,'max_future_heading_mutation_error':maxfuture,'max_future_all_observation_mutation_error':maxfuture_all,'note':'fixed target window and fixed offline cohort; not a proof of full online causality'};(OUT/'heading_features_independent.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out))
