"""Training-only posthoc diagnostic for causal-at-decision delayed confirmation.

The next report may be used only at target decision time >= its timestamp;
no injection markers or clean values enter the candidate weight calculation.
"""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
O=Path('/mnt/sda/WangZijia/AIS/outputs/identifiability_audit_20261030')
m=json.loads((R/'dataset_manifest_combined.json').read_text())
rows=sorted([r for r in m['samples'] if r['version']=='relaxed' and r['split']=='train'],key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())[:400]
records={v:[] for v in ('base','natural','suffix','combined')}
for rownum,row in enumerate(rows):
    data={}
    for name in ('base','natural','suffix'):
        with np.load(R/row['files'][name]['path'],allow_pickle=False) as z:
            data[name]={k:z[k].copy() for k in ('target_t','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets','cohort_mmsi')}
    for case in range(4):
        b=data['base']; tt=b['target_t'][case]; members={int(x['member_index']):x for x in row['stress']['members']}
        for j,(lo,hi) in enumerate(zip(b['member_offsets'][:-1],b['member_offsets'][1:])):
            lo,hi=int(lo),int(hi); t=b['neighbor_t'][case,lo:hi];p=b['neighbor_pos'][case,lo:hi];sb=b['neighbor_sog_kn'][case,lo:hi];c=b['neighbor_cog_deg'][case,lo:hi]
            for view in records:
                src=data['suffix' if view=='combined' else view];s=src['neighbor_sog_kn'][case,lo:hi];pp=p.copy()
                if view=='combined' and j in members:
                    marker=members[j]; ix=int(marker['indices']['suffix']['report_index']); hh=hashlib.sha256(('framework-stress-position-v1:'+row['window_id']+':'+marker['mmsi']).encode()).digest(); th=2*np.pi*int.from_bytes(hh[:8],'big')/2**64
                    if ix+1<len(pp):pp[ix:ix+2]+=600*np.array([np.cos(th),np.sin(th)])
                an=np.deg2rad(c);v=s[:,None]*.514444444444*np.column_stack([np.sin(an),np.cos(an)])
                for i in range(3,len(t)-1):
                    dt=t[i]-t[i-1]; dn=t[i+1]-t[i]
                    if not (0<dt<=300 and 0<dn<=300 and np.isfinite(pp[i-1:i+2]).all() and np.isfinite(v[i-1:i+2]).all() and np.isfinite(s[i-3:i]).all()):continue
                    h=float(np.median(s[i-3:i])); vp=(pp[i]-pp[i-1])/dt; vn=(pp[i+1]-pp[i])/dn;ec=float(np.linalg.norm(vp-v[i]));en=float(np.linalg.norm(vn-v[i]));em=float(np.linalg.norm(vp-(v[i]+v[i-1])*.5));w=float(np.exp(-max(0,em-2.5)**2/8)); mult=.7+.3*np.exp(-.5*max(0,2*ec-en-.5)**2) if h<2 else 1.
                    changed=abs(s[i]-sb[i])>1e-9;prev=abs(s[i-1]-sb[i-1])>1e-9;nextc=abs(s[i+1]-sb[i+1])>1e-9;pos=bool(np.any(pp[i-1:i+2]!=p[i-1:i+2]));turn=abs((c[i]-c[i-1]+180)%360-180); accel=abs(sb[i]-sb[i-1])>=1
                    # A historical report can be retroactively used only if at least
                    # one target decision after successor arrives still retains it
                    # in the same pre/post phase as the cumulative reference.
                    avail=0
                    for k,decision in enumerate(tt):
                        start=tt[0] if k<len(tt)//2 else tt[len(tt)//2]
                        q=np.searchsorted(t,decision,side='right')-1
                        if q>=i+1 and 0<=decision-t[q]<=300 and start<=t[i]<=decision:avail+=1
                    records[view].append((str(row['mmsi']),str(b['cohort_mmsi'][j]),float(sb[i]),h,bool(changed),bool(prev),bool(nextc),pos,float(turn),bool(accel),w,mult,float(ec),float(en),int(avail)))
    if (rownum+1)%100==0:print('progress',rownum+1,flush=True)
dtype=[('target','U30'),('neighbor','U30'),('base_speed','f8'),('h','f8'),('changed','?'),('prev','?'),('nextchanged','?'),('pos','?'),('turn','f8'),('accel','?'),('w','f8'),('mult','f8'),('ec','f8'),('en','f8'),('decision_steps','i4')]
ar={k:np.asarray(v,dtype=dtype) for k,v in records.items()};np.savez_compressed(O/'delayed_gate_records.npz',**ar)
def stat(a,sel):
    q=a[sel];dec=q['decision_steps']; extra=1-q['mult']; return {'n':len(q),'target_mmsi':len(np.unique(q['target'])),'neighbors':len(np.unique(q['neighbor'])),'mean_extra_suppression_assuming_successor':float(np.mean(extra)) if len(q) else None,'mean_extra_suppression_applied_any_target':float(np.mean(extra*(dec>0))) if len(q) else None,'mean_extra_suppression_per_retroactive_target_step':float(np.sum(extra*dec)/np.sum(dec)) if np.sum(dec)>0 else None,'fraction_extra_gt_01_assuming_successor':float(np.mean(q['mult']<.99)) if len(q) else None,'w_original_lt03':float(np.mean(q['w']<.3)) if len(q) else None,'w_candidate_lt03_assuming_successor':float(np.mean(q['w']*q['mult']<.3)) if len(q) else None,'confirmed_by_decision':float(np.mean(dec>0)) if len(q) else None,'mean_decision_steps':float(np.mean(dec)) if len(q) else None}
out={}
for name,a in ar.items():
    low=(a['h']<2); stable=(~a['changed']&~a['prev']&~a['nextchanged']&~a['pos']); targeted=low&a['changed']&(a['base_speed']<2)
    out[name]={'all_eligible':stat(a,np.ones(len(a),bool)), 'baseline_low_no_intervention':stat(a,low&stable&(a['base_speed']<2)), 'unmodified_sog_change':stat(a,low&stable&a['accel']), 'unmodified_turn45':stat(a,low&stable&(a['turn']>=45)), 'controlled_low_changed':stat(a,targeted), 'controlled_low_continuous':stat(a,targeted&a['nextchanged']), 'controlled_low_single':stat(a,targeted&~a['nextchanged']), 'position_affected':stat(a,low&a['pos'])}
doc={'scope':'400 SHA-ordered train sources; groups use intervention labels or original speed only for diagnosis, no injection marker used by candidate. Baseline has unknown natural errors and no confirmed maneuver labels; repeated observations across cases/sources. Arithmetic is at the report level conditional on successor existence; decision availability computed separately. For use at decision, never apply correction before successor arrival.','formula':'m=0.7+0.3*exp(-0.5*max(0,2e_cur-e_next-0.5)^2) if past3 median SOG<2kn else1; candidate w=w_old*m only when successor is available at the target decision; e in m/s; 0<both dt<=300s','views':out}
(O/'delayed_gate_summary.json').write_text(json.dumps(doc,indent=2),encoding='utf8');print(json.dumps(doc,indent=2))
