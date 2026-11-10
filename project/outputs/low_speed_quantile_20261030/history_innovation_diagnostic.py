"""Exploratory training-only, causal speed-history consistency gate diagnostic.
Fixed thresholds: history median <2kn, ratio floor 0.5kn, ratio onset=1,
current position consistency residual threshold=.5m/s, exponent=.5.
These were chosen after prior training-set diagnostics; not a preregistered test.
"""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027'); O=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_quantile_20261030')
m=json.loads((R/'dataset_manifest_combined.json').read_text())
rows=sorted([r for r in m['samples'] if r['version']=='relaxed' and r['split']=='train'],key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())[:400]
rec={v:[] for v in ('suffix','natural','combined')}
def wfun(t,p,s,c):
 angle=np.deg2rad(c);v=s[:,None]*.514444444444*np.column_stack([np.sin(angle),np.cos(angle)]);dt=np.diff(t)
 idx=np.arange(3,len(t)); keep=(dt[idx-1]>0)&(dt[idx-1]<=300)&np.isfinite(p[idx-1:idx[0]+1]).all() if False else np.ones(len(idx),bool)
 keep=(dt[idx-1]>0)&(dt[idx-1]<=300)&np.isfinite(p[idx-1]).all(1)&np.isfinite(p[idx]).all(1)&np.isfinite(v[idx-1]).all(1)&np.isfinite(v[idx]).all(1)
 idx=idx[keep]; past=np.stack([s[idx-3],s[idx-2],s[idx-1]],axis=1); valid=np.isfinite(past).all(1);idx=idx[valid];past=past[valid]
 vp=(p[idx]-p[idx-1])/dt[idx-1,None]; mid=np.linalg.norm(vp-(v[idx]+v[idx-1])/2,axis=1); cur=np.linalg.norm(vp-v[idx],axis=1);h=np.median(past,axis=1)
 ratio=np.abs(s[idx]-h)/(h+.5); multiplier=np.where((h<2)&(cur>=.5),np.exp(-.5*np.maximum(0,ratio-1)**2),1.0)
 return idx,h,cur,ratio,np.exp(-np.maximum(0,mid-2.5)**2/8),multiplier
for no,row in enumerate(rows):
 d={}
 for name in ['base','natural','suffix']:
  with np.load(R/row['files'][name]['path'],allow_pickle=False) as z:d[name]={k:z[k].copy() for k in ('neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets','cohort_mmsi')}
 for case in range(4):
  b=d['base']; selected={int(x['member_index']):x for x in row['stress']['members']}
  for j,(lo,hi) in enumerate(zip(b['member_offsets'][:-1],b['member_offsets'][1:])):
   lo,hi=int(lo),int(hi);t=b['neighbor_t'][case,lo:hi];p=b['neighbor_pos'][case,lo:hi];s=b['neighbor_sog_kn'][case,lo:hi];c=b['neighbor_cog_deg'][case,lo:hi]
   bi,bh,bc,br,bw,bmult=wfun(t,p,s,c)
   for view in rec:
    source='suffix' if view=='combined' else view;sp=d[source]['neighbor_sog_kn'][case,lo:hi]; pp=p.copy()
    if view=='combined' and j in selected:
     marker=selected[j]; ix=int(marker['indices']['suffix']['report_index']); h=hashlib.sha256(('framework-stress-position-v1:'+row['window_id']+':'+marker['mmsi']).encode()).digest();th=2*np.pi*int.from_bytes(h[:8],'big')/2**64
     if ix+1<len(pp):pp[ix:ix+2]+=600*np.array([np.cos(th),np.sin(th)])
    ii,h,ec,rat,w,mult=wfun(t,pp,sp,c)
    assert np.array_equal(bi,ii)
    changed=np.abs(sp[ii]-s[ii])>1e-9;prev=np.abs(sp[ii-1]-s[ii-1])>1e-9; pos=np.any(pp[ii]!=p[ii],axis=1)|np.any(pp[ii-1]!=p[ii-1],axis=1)
    turn=np.abs((c[ii]-c[ii-1]+180)%360-180); natural=np.abs(s[ii]-s[ii-1])>=1
    for k,i in enumerate(ii):rec[view].append((str(row['mmsi']),str(b['cohort_mmsi'][j]),float(s[i]),float(h[k]),bool(changed[k]),bool(prev[k]),bool(pos[k]),float(turn[k]),bool(natural[k]),float(w[k]),float(w[k]*mult[k]),float(bw[k]),float(bw[k]*bmult[k]),float(ec[k]),float(rat[k])))
 if no%100==0:print('PROGRESS',no,flush=True)
dtype=[('target','U30'),('neighbor','U30'),('base_speed','f8'),('h','f8'),('changed','?'),('prev','?'),('pos','?'),('turn','f8'),('natural_accel','?'),('old','f8'),('new','f8'),('baseold','f8'),('basenew','f8'),('ecur','f8'),('ratio','f8')]
out={k:np.array(v,dtype=dtype) for k,v in rec.items()};np.savez_compressed(O/'history_innovation_records.npz',**out)
def stat(a,sel):
 q=a[sel];return {'n':len(q),'targets':len(np.unique(q['target'])),'members':len(np.unique(q['neighbor'])),'old_lt03':float(np.mean(q['old']<.3)) if len(q) else None,'new_lt03':float(np.mean(q['new']<.3)) if len(q) else None,'old_mean':float(np.mean(q['old'])) if len(q) else None,'new_mean':float(np.mean(q['new'])) if len(q) else None,'baseold_lt03':float(np.mean(q['baseold']<.3)) if len(q) else None,'basenew_lt03':float(np.mean(q['basenew']<.3)) if len(q) else None}
s={}
for view,a in out.items():
 lo=(a['h']<2);s[view]={'changed_truebase_low':stat(a,lo&a['changed']&(a['base_speed']<2)),'changed_all':stat(a,lo&a['changed']),'unmodified_low':stat(a,lo&~a['changed']&~a['prev']&~a['pos']&(a['base_speed']<2)),'unmodified_turn45':stat(a,lo&~a['changed']&~a['prev']&~a['pos']&(a['turn']>=45)),'unmodified_accel1kn':stat(a,lo&~a['changed']&~a['prev']&~a['pos']&a['natural_accel']),'changed_position':stat(a,lo&a['pos']),'previous_altered_only':stat(a,lo&a['prev']&~a['changed']&~a['pos'])}
result={'scope':'training only, 400 SHA-ordered train source IDs, four counterfactual cases, base is unverified; repeated member reports','candidate':'h=median of previous 3 reported SOG; if h<2kn, ecur>=0.5 m/s multiply old weight by exp(-0.5*max(0,abs(SOG_t-h)/(h+0.5kn)-1)^2); else return old. Unknown first3 reports fallback old; causal; labels used solely in diagnostics','views':s}
(O/'history_innovation_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
