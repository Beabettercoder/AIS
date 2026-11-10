"""Training-only history-based low-speed candidate: unlabeled calibration and labeled diagnostic."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
O=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_quantile_20261030')
m=json.loads((R/'dataset_manifest_combined.json').read_text())
rows=sorted([r for r in m['samples'] if r['version']=='relaxed' and r['split']=='train'],key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())
seen=set(); calib=[]; agents=set(); targets=set(); counts={}; records=[]
for no,row in enumerate(rows):
 a={}
 for view in ['base','suffix']:
  with np.load(R/row['files'][view]['path'],allow_pickle=False) as z:
   a[view]={k:z[k].copy() for k in ('neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets','cohort_mmsi')}
 b=a['base']; s=a['suffix']; targets.add(str(row['mmsi']))
 for ci in range(4):
  for j,(lo,hi) in enumerate(zip(b['member_offsets'][:-1],b['member_offsets'][1:])):
   lo,hi=int(lo),int(hi); tt=b['neighbor_t'][ci,lo:hi]; pp=b['neighbor_pos'][ci,lo:hi]; ss=b['neighbor_sog_kn'][ci,lo:hi]; cs=b['neighbor_cog_deg'][ci,lo:hi]; sp=s['neighbor_sog_kn'][ci,lo:hi]; mid=str(b['cohort_mmsi'][j]); ang=np.deg2rad(cs); unit=np.column_stack([np.sin(ang),np.cos(ang)]); v=ss[:,None]*.514444444444*unit; vs=sp[:,None]*.514444444444*unit
   for i in range(1,len(tt)):
    dt=tt[i]-tt[i-1]
    if not (0<dt<=300 and np.isfinite(pp[i-1:i+1]).all() and np.isfinite(v[i-1:i+1]).all()):continue
    if i<3 or not np.isfinite(ss[i-3:i]).all():continue
    low=np.median(ss[i-3:i])<2
    if not low: continue
    key=(mid,float(tt[i-1]),float(tt[i]),pp[i-1:i+1].tobytes(),v[i-1:i+1].tobytes(),ss[i-3:i].tobytes())
    e=float(np.linalg.norm((pp[i]-pp[i-1])/dt-(v[i]+v[i-1])/2))
    if key not in seen:seen.add(key);calib.append(e);agents.add(mid)
    if not np.isfinite(sp[i-3:i+1]).all():continue
    low_s=np.median(sp[i-3:i])<2
    es=float(np.linalg.norm((pp[i]-pp[i-1])/dt-(vs[i]+vs[i-1])/2))
    changed=abs(ss[i]-sp[i])>1e-9
    previous=abs(ss[i-1]-sp[i-1])>1e-9
    records.append((row['mmsi'],mid,e,es,changed,previous,low_s,float(ss[i])))
 if no%200==0:print('PROGRESS',no,flush=True)
q95,q99=np.quantile(calib,[.95,.99]); assert len(calib)>=100 and q99>q95
r=np.array(records,dtype=[('target','U30'),('member','U30'),('base','f8'),('stress','f8'),('changed','?'),('prev','?'),('low','?'),('base_speed','f8')])
def table(sel):
 a=r[sel]; out={'n':int(len(a)),'n_target':int(len(np.unique(a['target']))),'n_member':int(len(np.unique(a['member'])))}
 for nm,mask in [('old_e_gt_q95',a['stress']>q95),('new_soft_03',(np.exp(-np.maximum(0,a['stress']-2.5)**2/8)*np.where(a['low'],np.exp(np.log(.3)*np.maximum(0,(a['stress']-q95)/(q99-q95))**2),1))<.3),('old_03',np.exp(-np.maximum(0,a['stress']-2.5)**2/8)<.3)]:out[nm]=float(mask.mean()) if len(a) else None
 return out
result={'scope':'all train sources; calibration uses only Base, no labels; diagnostics use suffix change flags','num_sources':len(rows),'targets':len(targets),'unique_base_calibration_records':len(calib),'neighbor_mmsi':len(agents),'q95':float(q95),'q99':float(q99),'group':'median of three prior reported SOG <2kn; i>=3; no future','counts':{'changed_priorlow':table(r['changed']&r['low']),'unmodified_priorlow':table(~r['changed']&r['low']&~r['prev']),'changed_true_base_speed_lt2_priorlow':table(r['changed']&r['low']&(r['base_speed']<2))}}
(O/'history_probe.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2),flush=True)
