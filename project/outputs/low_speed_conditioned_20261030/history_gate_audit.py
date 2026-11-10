"""Independent training-only diagnostic; condition is causal, labels used for evaluation only."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
O=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_conditioned_20261030')
manifest=json.loads((R/'dataset_manifest_combined.json').read_text())
rows=sorted([r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='train'],key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())[:400]
data={k:[] for k in ('base','natural','suffix','combined')}
for nr,row in enumerate(rows):
 ds={}
 for view in ('base','natural','suffix'):
  with np.load(R/row['files'][view]['path'],allow_pickle=False) as z:ds[view]={k:z[k].copy() for k in ('neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets','cohort_mmsi')}
 members={int(x['member_index']):x for x in row['stress']['members']}
 for case in range(4):
  b=ds['base']
  for j,(lo,hi) in enumerate(zip(b['member_offsets'][:-1],b['member_offsets'][1:])):
   lo,hi=int(lo),int(hi); tt=b['neighbor_t'][case,lo:hi]; orig=b['neighbor_sog_kn'][case,lo:hi]; cog=b['neighbor_cog_deg'][case,lo:hi]; origpos=b['neighbor_pos'][case,lo:hi]
   if len(tt)<4:continue
   if j in members:
    d=hashlib.sha256(('framework-stress-position-v1:'+row['window_id']+':'+str(members[j]['mmsi'])).encode()).digest();theta=2*np.pi*int.from_bytes(d[:8],'big')/2**64;shift=600*np.array([np.cos(theta),np.sin(theta)]);hit=int(members[j]['indices']['suffix']['report_index'])
   for view in data:
    sv=ds['suffix' if view=='combined' else view]['neighbor_sog_kn'][case,lo:hi];pos=origpos.copy()
    if view=='combined' and j in members and hit+1<len(pos):pos[hit:hit+2]+=shift
    dt=np.diff(tt);angle=np.deg2rad(cog);v=sv[:,None]*.514444444444*np.column_stack([np.sin(angle),np.cos(angle)])
    idx=np.arange(3,len(tt)); valid=(dt[idx-1]>0)&(dt[idx-1]<=300)&np.isfinite(v[idx]).all(1)&np.isfinite(v[idx-1]).all(1)&np.isfinite(pos[idx]).all(1)&np.isfinite(pos[idx-1]).all(1)
    idx=idx[valid]
    if len(idx)==0:continue
    hist=np.median(np.stack([sv[idx-3],sv[idx-2],sv[idx-1]],axis=1),axis=1)
    posv=(pos[idx]-pos[idx-1])/dt[idx-1,None];mid=np.linalg.norm(posv-(v[idx]+v[idx-1])/2,axis=1);cur=np.linalg.norm(posv-v[idx],axis=1)
    ratio=np.abs(sv[idx]-hist)/(hist+.5);active=(hist<2)&(cur>=.5)&(ratio>1);w0=np.exp(-np.maximum(0,mid-2.5)**2/8);w1=w0*np.exp(-.5*np.where(active,np.maximum(0,ratio-1)**2,0))
    changed=np.abs(sv[idx]-orig[idx])>1e-9;prev=np.abs(sv[idx-1]-orig[idx-1])>1e-9;pp=np.any(pos[idx]!=origpos[idx],axis=1)|np.any(pos[idx-1]!=origpos[idx-1],axis=1)
    for k,i in enumerate(idx):data[view].append((str(row['mmsi']),str(b['cohort_mmsi'][j]),float(orig[i]),float(sv[i]),float(hist[k]),float(mid[k]),float(cur[k]),float(ratio[k]),bool(active[k]),float(w0[k]),float(w1[k]),bool(changed[k]),bool(prev[k]),bool(pp[k])))
 if nr%100==0:print('progress',nr,flush=True)
dtype=[('target','U32'),('member','U32'),('base_sog','f8'),('sog','f8'),('hist','f8'),('emid','f8'),('ecur','f8'),('ratio','f8'),('active','?'),('w0','f8'),('w1','f8'),('changed','?'),('prev','?'),('pos','?')]
result={'scope':'Training first 400 SHA sorted sources, four correlated cases; original speed and pressure labels only used for diagnosis. Experimental history gate inputs are observed-only, no future. Report-level counts are not independent subjects.','views':{}}
for view,rr in data.items():
 a=np.asarray(rr,dtype=dtype); result['views'][view]={}
 for key,sel in [('changed_true_low',a['changed']&(a['base_sog']<2)),('unchanged_true_low',(~a['changed'])&(~a['prev'])&(a['base_sog']<2)),('priorchanged_current_not',(~a['changed'])&a['prev']),('position_changed',a['pos']),('all',np.ones(len(a),bool))]:
  z=a[sel]
  if not len(z):continue
  result['views'][view][key]={'n':len(z),'targets':len(np.unique(z['target'])),'members':len(np.unique(z['member'])),'active':float(z['active'].mean()),'old_w_lt03':float((z['w0']<.3).mean()),'new_w_lt03':float((z['w1']<.3).mean()),'old_mean':float(z['w0'].mean()),'new_mean':float(z['w1'].mean())}
np.savez_compressed(O/'history_gate_records.npz',**{k:np.asarray(v,dtype=dtype) for k,v in data.items()})
(O/'history_gate_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
