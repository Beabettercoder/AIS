from pathlib import Path
import json,hashlib
import numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'formal_20261101/relation';old=R/'relation_repair_20261030';man=R/'balanced_global_v2/relation_dual_pressure_v2_20261027/dataset_manifest_combined.json'
assert (O/'heldout.lock.json').exists() and (O/'test_relation_field.npz').exists()
splits={};data={}
for sp in ('train','val','test'):
 path=(old if sp!='test' else O)/(sp+'_relation_field.npz')
 with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in ('mmsi','source','case','y','region','length')}
 data[sp]=d;src=np.unique(d['source'].astype(str));v=np.unique(d['mmsi'].astype(str));splits[sp]={'window_count':len(d['y']),'source_count':len(src),'target_mmsi_count':len(v),'positive':int(d['y'].sum()),'regional':{}}
 for reg in np.unique(d['region']):
  mask=d['region']==reg;splits[sp]['regional'][str(reg)]={'windows':int(mask.sum()),'sources':len(np.unique(d['source'][mask])),'target_mmsi':len(np.unique(d['mmsi'][mask]))}
for a,b in [('train','val'),('train','test'),('val','test')]:
 for key in ('mmsi','source'):
  overlap=set(data[a][key].astype(str))&set(data[b][key].astype(str));assert len(overlap)==0,(a,b,key,len(overlap))
m=json.loads(man.read_text());rows=[x for x in m['samples'] if x['version']=='relaxed'];background={sp:set() for sp in splits}
for r in rows:
 if r['split'] in background:background[r['split']].update(map(str,r['cohort_mmsi']))
bg={}
for a,b in [('train','val'),('train','test'),('val','test')]:
 key=a+'_'+b;bg[key]={'cohort_mmsi_intersection_recomputed':len(background[a]&background[b]),'preexisting_report_overlap_from_manifest':m['background_overlap']['relaxed'][key]}
 assert bg[key]['cohort_mmsi_intersection_recomputed']==bg[key]['preexisting_report_overlap_from_manifest']['shared_background_mmsi']
out={'splits':splits,'target_mmsi_and_source_overlap_all_zero':True,'background_overlap':bg,'background_note':'reports count is inherited from prior audited manifest, not recomputed in this script','target_labels':'4 controlled variants per source; not independent real events','test':'sources previously exposed'}
p=O/'dataset_audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
