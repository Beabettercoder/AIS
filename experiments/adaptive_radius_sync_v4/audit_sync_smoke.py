from pathlib import Path
import json,hashlib,numpy as np
import build_relation_sync as b
root=Path('/home/ps/AIS_adaptive_sync_20261110');registry=json.loads((b.BAL/'release_20261026/source_registry.json').read_text())['regions'];manifest=json.loads((b.COHORT/'dataset_manifest_combined.json').read_text())['samples'];checked=0;maxerr=0.;selected=0
for region in ('houston','great_belt'):
 src=b.helper.Source(registry[region]);allowed=b.allowed_for('val');rows=sorted((r for r in manifest if r['version']=='relaxed' and r['split']=='val' and r['region']==region),key=lambda r:(r['region'],r['window_id']))[:2]
 for row in rows:
  path=b.COHORT/row['files']['base']['path']
  with np.load(path,allow_pickle=False) as z:original={k:z[k] for k in ('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','y')}
  cohort=set(map(str,original['cohort_mmsi']));views,proof=b.extend_and_stress(original,src,allowed,str(row['mmsi']),cohort,row['window_id']);base=views['base'];anchor=max(2,len(base['target_t'][0])//2-1);onset=base['target_t'][0,anchor]+1.
  for j,mid in enumerate(base['cohort_mmsi'].astype(str)):
   if mid in cohort:continue
   lo,hi=map(int,base['member_offsets'][j:j+2]);tt=base['neighbor_t'][0,lo:hi];s0=base['neighbor_sog_kn'][0,lo:hi];s1=base['neighbor_sog_kn'][1,lo:hi];u=np.clip((tt-onset)/300.,0,1);expected=s0*(1-.3*(3*u*u-2*u*u*u))
   if not np.allclose(s1,expected,rtol=1e-12,atol=1e-12):raise ValueError('unsynchronized '+mid)
   checked+=1
  for record in proof['records']:
   j=list(base['cohort_mmsi'].astype(str)).index(record['mmsi']);lo=int(base['member_offsets'][j]);selected+=1
   for view,k in [('natural',record['natural_index']),('suffix',record['suffix_index']),('combined_l100_m600',record['suffix_index'])]:
    want=np.minimum(base['neighbor_sog_kn'][:,lo+k:lo+k+2]*record['factor'],102.1)
    if not np.array_equal(views[view]['neighbor_sog_kn'][:,lo+k:lo+k+2],want):raise ValueError('stress mismatch')
  for view in views:
   for key in ('neighbor_pos','neighbor_sog_kn','neighbor_cog_deg'):
    assert np.array_equal(views[view][key][0],views[view][key][2]) and np.array_equal(views[view][key][1],views[view][key][3])
  maxerr=max(maxerr,proof['innovation_max_error_mps'])
out={'status':'passed','sources':4,'ambient_ships_checked':checked,'selected_ships_checked':selected,'max_innovation_error_mps':maxerr,'checks':'added b=1 speed envelope, quartet neighbor marginal, case-shared stress per selected ship'};(root/'smoke_sync_audit.json').write_text(json.dumps(out,indent=2));print('SYNC_SMOKE_AUDIT_PASSED',json.dumps(out),flush=True)
