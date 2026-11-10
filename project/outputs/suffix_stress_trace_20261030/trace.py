from pathlib import Path
import sys, json
import numpy as np

BASE=Path('/mnt/sda/WangZijia/AIS/outputs')
R=BASE/'balanced_global_v2/relation_dual_pressure_v2_20261027'
OUT=BASE/'suffix_stress_trace_20261030'
sys.path.insert(0,str(BASE/'framework_validation_v1_20261028'))
from data_builder import absolute_weights
manifest=json.loads((R/'dataset_manifest_combined.json').read_text())
rows=sorted([r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='val'],key=lambda r:(r['region'],r['window_id']))
acc={v:{k:[] for k in ('changed','adjacent','unaffected_selected','unselected')} for v in ('natural','suffix')}
checks={'numeric_changed_disagreement':0,'target_in_cohort':0,'source_count':len(rows)}
for row in rows:
    checks['target_in_cohort'] += str(row['mmsi']) in [str(z) for z in row['cohort_mmsi']]
    ds={}
    for v in ('base','natural','suffix'):
        with np.load(R/row['files'][v]['path'],allow_pickle=False) as z:
            ds[v]={k:z[k].copy() for k in ('neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets','cohort_mmsi')}
    member_sel={int(m['member_index']):m for m in row['stress']['members']}
    for c in range(4):
        o=ds['base']; offsets=o['member_offsets']
        for i in range(len(offsets)-1):
            lo,hi=int(offsets[i]),int(offsets[i+1]); t=o['neighbor_t'][c,lo:hi]; p=o['neighbor_pos'][c,lo:hi];s=o['neighbor_sog_kn'][c,lo:hi];cog=o['neighbor_cog_deg'][c,lo:hi]
            base,known=absolute_weights(t,p,s,cog)
            for view in acc:
                sv=ds[view]['neighbor_sog_kn'][c,lo:hi]; w,valid=absolute_weights(t,p,sv,cog)
                changed=np.abs(sv-s)>1e-9
                expected=np.zeros(len(t),bool)
                if i in member_sel:
                    ix=int(member_sel[i]['indices'][view]['report_index']);expected[ix:ix+2]=True
                checks['numeric_changed_disagreement']+=int(np.any(changed & ~expected))
                neighbor=np.r_[False,changed[:-1]] & ~changed
                for j in np.flatnonzero(known & valid):
                    category='changed' if changed[j] else ('adjacent' if neighbor[j] else ('unaffected_selected' if i in member_sel else 'unselected'))
                    acc[view][category].append((float(base[j]),float(w[j]),float(s[j]),float(sv[j])))
def stats(v):
    a=np.asarray(v)
    if not len(a):return {'count':0}
    return {'count':int(len(a)),'base_w_mean':float(a[:,0].mean()),'stress_w_mean':float(a[:,1].mean()),'base_w_lt_03_fraction':float((a[:,0]<.3).mean()),'stress_w_lt_03_fraction':float((a[:,1]<.3).mean()),'stress_w_ge_09_fraction':float((a[:,1]>=.9).mean()),'mean_delta_w':float((a[:,1]-a[:,0]).mean()),'speed_base_kn_median':float(np.median(a[:,2])),'speed_stress_kn_median':float(np.median(a[:,3]))}
result={'scope':'272 existing val source x 4 constructed cases; repeated neighbor reports across source/cases, not independent. No model training.','checks':checks,'report_level':{view:{k:stats(x) for k,x in cats.items()} for view,cats in acc.items()}}
result['changed_by_baseline_speed_kn']={}
for view in acc:
    changed=np.asarray(acc[view]['changed'])
    result['changed_by_baseline_speed_kn'][view]={}
    for name,sel in [('lt2',changed[:,2]<2),('2to5',(changed[:,2]>=2)&(changed[:,2]<5)),('5to10',(changed[:,2]>=5)&(changed[:,2]<10)),('ge10',changed[:,2]>=10)]:
        result['changed_by_baseline_speed_kn'][view][name]=stats(changed[sel])
# Compare observed-view field perturbation relative to that same arm's base field; base is not clean truth.
with np.load(BASE/'relation_repair_20261030/val_relation_field.npz',allow_pickle=False) as z:
    validmask=np.arange(20)[None,:]<z['length'][:,None]
    result['field_view_shifts']={}
    for view in ('natural','suffix','combined_l100_m600'):
        a=z['base__field'];b=z[view+'__field']; valid=validmask&(a[:,0,:,7]>0)&(b[:,0,:,7]>0)
        per={}
        for arm,i in [('F0',0),('F1',1)]:
            delta=np.linalg.norm(b[:,i,:,:2]-a[:,i,:,:2],axis=-1)[valid]
            per[arm]={'count':int(len(delta)),'mean_mps':float(np.mean(delta)),'median_mps':float(np.median(delta)),'p90_mps':float(np.quantile(delta,.9)),'fraction_nonzero':float(np.mean(delta>1e-6))}
        d0=np.linalg.norm(b[:,0,:,:2]-a[:,0,:,:2],axis=-1)[valid];d1=np.linalg.norm(b[:,1,:,:2]-a[:,1,:,:2],axis=-1)[valid]
        affected=(d0>1e-6)|(d1>1e-6)
        per['paired_F1_shift_minus_F0_shift_mps']={'mean':float(np.mean(d1-d0)),'fraction_F1_smaller':float(np.mean(d1<d0-1e-6)),'fraction_equal':float(np.mean(np.abs(d1-d0)<=1e-6)),'affected_steps':int(affected.sum()),'fraction_F1_smaller_among_affected':float(np.mean(d1[affected]<d0[affected]-1e-6)),'mean_f0_shift_among_affected':float(np.mean(d0[affected])),'mean_f1_shift_among_affected':float(np.mean(d1[affected]))}
        result['field_view_shifts'][view]=per
OUT.mkdir(exist_ok=True)
(OUT/'stress_trace.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(result,indent=2))
