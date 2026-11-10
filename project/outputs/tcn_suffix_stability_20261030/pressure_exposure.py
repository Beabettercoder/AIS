from pathlib import Path
import json,numpy as np
base=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
out=Path('/mnt/sda/WangZijia/AIS/outputs/tcn_suffix_stability_20261030')
m=json.loads((base/'dataset_manifest_combined.json').read_text())
rows=[r for r in m['samples'] if r['version']=='relaxed' and r['split']=='val']
vals={v:[] for v in ('natural','suffix')};source_hit={v:set() for v in vals}
for row in rows:
 with np.load(base/row['files']['base']['path']) as z: ts=z['target_t'][0];nt=z['neighbor_t'][0];off=z['member_offsets']
 for m in row['stress']['members']:
  member=m['member_index']; lo,hi=int(off[member]),int(off[member+1]); times=nt[lo:hi];
  for view in vals:
   index=int(m['indices'][view]['report_index']); affected=times[index:index+2]
   for j,t in enumerate(ts):
    phase=ts[0] if j<len(ts)//2 else ts[len(ts)//2]
    q=np.searchsorted(times,t,side='right')-1
    if q<0 or t-times[q]>300: continue
    n=((times>=phase)&(times<=t)).sum()
    if not n: n=1
    k=((affected>=phase)&(affected<=t)).sum()
    vals[view].append(float(k/n))
    if k: source_hit[view].add(row['window_id'])
r={}
for v,ls in vals.items():
 a=np.asarray(ls);r[v]={'selected_member_aligned_steps':len(a),'sources_with_any_averaged_corruption':len(source_hit[v]),'steps_with_corruption_in_average':int((a>0).sum()),'steps_with_corruption_fraction':float((a>0).mean()),'mean_contaminated_report_fraction_all_aligned_steps':float(a.mean()),'mean_fraction_conditional_on_affected':float(a[a>0].mean())}
(out/'pressure_exposure.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
