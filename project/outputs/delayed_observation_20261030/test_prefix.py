"""No future observation leakage in delayed-neighbor field."""
import json,sys
from pathlib import Path
import numpy as np
import builder

root=builder.ROOT
manifest=json.loads((root/'dataset_manifest_combined.json').read_text())
rows=sorted([r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='train'],key=lambda r:(r['region'],r['window_id']))[:8]
keys=['target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','y']
nchecks=0
for row in rows:
 o=builder.field.read_npz(root/row['files']['base']['path'],keys)
 for case in range(4):
  full,_=builder.calc(o,case,row['mmsi'])
  T=o['target_t'].shape[1]
  for k in sorted(set([max(0,T//2),max(0,T-3),T-1])):
   cutoff=o['target_t'][case,k]
   keep=np.flatnonzero(o['neighbor_t'][case]<=cutoff)
   new={key:val.copy() for key,val in o.items()}
   for key in ('neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg'):
    new[key]=new[key][:,keep]
   new['neighbor_mmsi']=new['neighbor_mmsi'][keep]
   new['member_offsets']=np.r_[0,np.cumsum([np.sum((keep>=a)&(keep<b)) for a,b in zip(o['member_offsets'][:-1],o['member_offsets'][1:])])]
   truncated,_=builder.calc(new,case,row['mmsi'])
   np.testing.assert_allclose(full[:,:k+1],truncated[:,:k+1],atol=1e-6,rtol=0)
   nchecks+=1
# A single changed report is not reweighted until its successor arrives.
t=np.arange(5,dtype=float)*60; p=np.column_stack((np.zeros(5),t*.514444444444))
s=np.array([1,1,1,4,1.]); c=np.zeros(5); v=builder.velocity(s,c)
f=builder.delayed_factor(t,p,s,v); assert f[3]<1 and f[4]==1
obs={'target_t':np.tile(t,(4,1)),'target_pos':np.tile(p[None],(4,1,1)), 'target_sog_kn':np.tile(np.ones(5),(4,1)), 'target_cog_deg':np.zeros((4,5)), 'cohort_mmsi':np.array(['101']),'member_offsets':np.array([0,5]),'neighbor_t':np.tile(t,(4,1)),'neighbor_sog_kn':np.tile(s,(4,1)), 'neighbor_cog_deg':np.zeros((4,5)),'neighbor_pos':np.tile(p[None],(4,1,1))}
r,_=builder.calc(obs,0,'202'); assert np.isclose(r[0,3,1],r[1,3,1]); assert not np.isclose(r[0,4,1],r[1,4,1])
print(json.dumps({'prefix_truncation_tests':nchecks,'synthetic_factor':float(f[3]),'future_cutoff_checked':True,'future_not_model_input':True}))

