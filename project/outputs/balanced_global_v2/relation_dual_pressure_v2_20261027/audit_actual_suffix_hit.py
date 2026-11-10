"""Post-construction audit: suffix-selected report values must actually change."""
from pathlib import Path
from collections import Counter,defaultdict
import json,numpy as np
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
manifest=json.loads((OUT/'dataset_manifest.json').read_text());counts=defaultdict(Counter);non_effective=[]
for item in manifest['samples']:
 k=item['version']+'_'+item['split'];counts[k]['sources']+=1
 with np.load(OUT/item['files']['base']['path'],allow_pickle=False) as z:base={q:z[q] for q in ('target_t','neighbor_t','neighbor_sog_kn','member_offsets')}
 with np.load(OUT/item['files']['suffix']['path'],allow_pickle=False) as z:changed=z['neighbor_sog_kn'].copy()
 t=base['target_t'][0];anchor=max(2,len(t)//2-1);suffix=t[np.arange(anchor+1,len(t))[-3:]];source_any=False
 for member in item['stress']['members']:
  counts[k]['selected_members']+=1;j=member['member_index'];lo=int(base['member_offsets'][j]);hi=int(base['member_offsets'][j+1]);tt=base['neighbor_t'][0,lo:hi];ix=np.searchsorted(tt,suffix,side='right')-1;valid=ix>=0;safe=np.maximum(ix,0);valid &= (suffix-tt[safe]<=300)
  touched=valid&(ix>=member['indices']['suffix']['report_index'])&(ix<member['indices']['suffix']['report_index']+2)
  assert touched.any();counts[k]['members_suffix_index_hit']+=1
  grid=lo+ix[touched];actual=bool(np.any(np.abs(changed[:,grid]-base['neighbor_sog_kn'][:,grid])>1e-12));counts[k]['members_suffix_value_changed']+=actual;source_any|=actual
  if not actual:non_effective.append({'version':item['version'],'split':item['split'],'window_id':item['window_id'],'mmsi':item['mmsi'],'member_mmsi':member['mmsi'],'reason':'suffix-selected actual reports have no numeric change in any quartet case'})
 counts[k]['sources_suffix_actual_value_changed']+=source_any
result={'status':'posthoc_quality_audit_all_samples_retained','scope':'at target suffix latest-neighbor aligned reports, actual SOG changed across any quartet case; not inferred from index alone','counts':{k:dict(v) for k,v in counts.items()},'members_without_actual_numeric_effect':non_effective}
with (OUT/'suffix_actual_value_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result['counts'],ensure_ascii=False),flush=True)
