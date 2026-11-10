from pathlib import Path
import sys,json,hashlib
import numpy as np
BASE=Path('/mnt/sda/WangZijia/AIS/outputs')
R=BASE/'balanced_global_v2/relation_dual_pressure_v2_20261027'
OUT=BASE/'gate_step_diagnostic_20261030'
sys.path.insert(0,str(BASE/'framework_validation_v1_20261028'))
from data_builder import absolute_weights
manifest=json.loads((R/'dataset_manifest_combined.json').read_text())
rows=sorted([r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='train'],key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())[:400]
byview={v:[] for v in ('suffix','natural','combined')}
N=0
def weight(t,p,s,c,minimum=0.8):
 a=np.deg2rad(c); v=s[:,None]*.514444444444*np.stack((np.sin(a),np.cos(a)),axis=-1)
 dt=np.diff(t); known=np.r_[False,(dt>0)&(dt<=300)&np.isfinite(p[:-1]).all(1)&np.isfinite(p[1:]).all(1)&np.isfinite(s[:-1])&np.isfinite(s[1:])&np.isfinite(c[:-1])&np.isfinite(c[1:])]
 ii=np.flatnonzero(known);vp=(p[ii]-p[ii-1])/dt[ii-1,None]; vm=(v[ii]+v[ii-1])/2; vc=v[ii]
 e0=np.linalg.norm(vp-vm,axis=1);ec=np.linalg.norm(vp-vc,axis=1);es=np.minimum(e0,ec)
 den=np.minimum(np.linalg.norm(vp,axis=1),np.linalg.norm(vm,axis=1))+.3
 rel=es/den
 wa=np.exp(-np.maximum(0,e0-2.5)**2/8)
 ws=np.exp(-np.maximum(0,es-2.5)**2/8)
 wr=np.exp(-np.maximum(0,rel-1.)**2/2)
 wh=np.where(es>=minimum,np.minimum(ws,wr),ws)
 return ii,wa,ws,wh,e0,ec,es,rel,dt[ii-1]
def direction(window,member):
 d=hashlib.sha256(('framework-stress-position-v1:'+window+':'+member).encode()).digest()
 th=2*np.pi*int.from_bytes(d[:8],'big')/2**64
 return np.array([np.cos(th),np.sin(th)])
for row in rows:
 ds={}
 for view in ('base','natural','suffix'):
  with np.load(R/row['files'][view]['path'],allow_pickle=False) as z:
   ds[view]={k:z[k].copy() for k in ('neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets','cohort_mmsi')}
 for case in range(4):
  o=ds['base']
  for i,(lo,hi) in enumerate(zip(o['member_offsets'][:-1],o['member_offsets'][1:])):
   lo,hi=int(lo),int(hi)
   t=o['neighbor_t'][case,lo:hi];p=o['neighbor_pos'][case,lo:hi];s=o['neighbor_sog_kn'][case,lo:hi];c=o['neighbor_cog_deg'][case,lo:hi]
   bi,bwa,bws,bwh,*_=weight(t,p,s,c)
   sel={int(x['member_index']):x for x in row['stress']['members']}
   for view in byview:
    src='suffix' if view=='combined' else view
    sv=ds[src]['neighbor_sog_kn'][case,lo:hi]; ps=p.copy()
    if view=='combined' and i in sel:
     ix=int(sel[i]['indices']['suffix']['report_index'])
     if ix+1<len(ps): ps[ix:ix+2]+=600*direction(row['window_id'],sel[i]['mmsi'])
    vi,wa,ws,wh,e0,ec,es,rel,dt=weight(t,ps,sv,c)
    if not np.array_equal(bi,vi): continue
    changed=np.abs(sv-s)>1e-9
    position_changed=np.any(np.abs(ps-p)>1e-9,axis=1)
    adjacent=np.r_[False,changed[:-1]]&~changed
    for k,j in enumerate(vi):
     kind=1 if changed[j] else (2 if adjacent[j] else 0)
     turning=abs((c[j]-c[j-1]+180)%360-180)
     byview[view].append((str(row['mmsi']),kind,float(s[j]),float(sv[j]),float(wa[k]),float(ws[k]),float(wh[k]),float(bwa[k]),float(bws[k]),float(bwh[k]),float(e0[k]),float(ec[k]),float(es[k]),float(rel[k]),float(dt[k]),float(turning),int(position_changed[j]),int(position_changed[j-1])))
 N+=1
 if N%100==0: print('PROGRESS',N,flush=True)
OUT.mkdir(exist_ok=True)
dtp=[('mmsi','U32'),('kind','i4'),('speed','f8'),('speed2','f8'),('wa','f8'),('ws','f8'),('wh','f8'),('bwa','f8'),('bws','f8'),('bwh','f8'),('e0','f8'),('ec','f8'),('es','f8'),('rel','f8'),('dt','f8'),('turn','f8'),('poschange','i4'),('prevposchange','i4')]
records={k:np.array(v,dtype=dtp) for k,v in byview.items()}
np.savez_compressed(OUT/'train_sample_records.npz',**records)
summary={'scope':'training sources only, fixed deterministic sha-sorted 400 of 2034, 4 counterfactual cases; repeated background reports are not independent','sources':len(rows),'speed_thresholds_kn':[2,5,10],'candidate':'e_safe=min(e_mid,e_current); w=min(w_abs_safe,w_rel) when e_safe>=0.8 m/s, else w_abs_safe; e_rel=e_safe/(min(norm(v_pos), norm(v_mid))+0.3); w_rel=exp(-max(0,e_rel-1)^2/2)','views':{}}
for view,a in records.items():
 summary['views'][view]={}
 for name,sel in [('changed_all',a['kind']==1),('changed_lt2',(a['kind']==1)&(a['speed']<2)),('changed_2to5',(a['kind']==1)&(a['speed']>=2)&(a['speed']<5)),('changed_ge10',(a['kind']==1)&(a['speed']>=10)),('adjacent',(a['kind']==2)),('unchanged',(a['kind']==0)),('unchanged_lt5',(a['kind']==0)&(a['speed']<5))]:
  q=a[sel]
  if not len(q):continue
  summary['views'][view][name]={'n':len(q),'mmsi':len(np.unique(q['mmsi'])),'baseline_abs_lt03':float(np.mean(q['bwa']<.3)),'stress_abs_lt03':float(np.mean(q['wa']<.3)),'stress_safe_lt03':float(np.mean(q['ws']<.3)),'stress_hybrid_lt03':float(np.mean(q['wh']<.3)),'baseline_safe_lt03':float(np.mean(q['bws']<.3)),'baseline_hybrid_lt03':float(np.mean(q['bwh']<.3)),'e_mid_median':float(np.median(q['e0'])),'e_cur_median':float(np.median(q['ec'])),'e_safe_median':float(np.median(q['es'])),'rel_median':float(np.median(q['rel'])),'delta_safe_median':float(np.median(q['ws']-q['wa'])),'delta_hybrid_median':float(np.median(q['wh']-q['wa'])),'dt_median_s':float(np.median(q['dt'])),'turn_p90_deg':float(np.quantile(q['turn'],.9)),'position_changed_fraction':float(np.mean((q['poschange']|q['prevposchange'])>0))}
(OUT/'train_sample_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2),flush=True)
