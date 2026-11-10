"""Audit realized value changes at target's last three grid points for both stresses."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,numpy as np
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027');m=json.loads((OUT/'dataset_manifest_combined.json').read_text());counts=defaultdict(Counter);audit=[]
for item in m['samples']:
 k=item['version']+'_'+item['split'];counts[k]['sources']+=1
 with np.load(OUT/item['files']['base']['path'],allow_pickle=False) as z:b={a:z[a] for a in ('target_t','neighbor_t','neighbor_sog_kn','member_offsets')}
 obs={}
 for view in ['natural','suffix']:
  with np.load(OUT/item['files'][view]['path'],allow_pickle=False) as z:obs[view]=z['neighbor_sog_kn'].copy()
 t=b['target_t'][0];anchor=max(2,len(t)//2-1);suffix=t[np.arange(anchor+1,len(t))[-3:]]
 for view in ['natural','suffix']:
  hit_any=False;numeric_any=False
  for member in item['stress']['members']:
   j=member['member_index'];lo=int(b['member_offsets'][j]);hi=int(b['member_offsets'][j+1]);tt=b['neighbor_t'][0,lo:hi];ix=np.searchsorted(tt,suffix,side='right')-1;valid=ix>=0;safe=np.maximum(ix,0);valid &= (suffix-tt[safe]<=300)
   q=member['indices'][view]['report_index'];touched=valid&(ix>=q)&(ix<q+2);changed=bool(touched.any() and np.any(np.abs(obs[view][:,lo+ix[touched]]-b['neighbor_sog_kn'][:,lo+ix[touched]])>1e-12))
   counts[k][view+'_members_selected']+=1;counts[k][view+'_members_suffix_index_hit']+=int(touched.any());counts[k][view+'_members_suffix_numeric_change']+=int(changed);hit_any|=bool(touched.any());numeric_any|=changed
  counts[k][view+'_sources_index_hit']+=hit_any;counts[k][view+'_sources_numeric_hit']+=numeric_any
  audit.append({'version':item['version'],'split':item['split'],'region':item['region'],'window_id':item['window_id'],'view':view,'index_hit':hit_any,'numeric_hit':numeric_any})
out={'status':'posthoc_quality_audit','time_utc':datetime.now(timezone.utc).isoformat(),'note':'retain every source regardless of hit; based on observable SOG at target-aligned suffix reports','counts':{k:dict(v) for k,v in counts.items()},'per_source':audit}
with (OUT/'effective_pressure_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out['counts'],ensure_ascii=False),flush=True)
