from pathlib import Path
import json
import numpy as np

root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');out=root/'independent_audit_causal_v1_20261027';prior=root/'original_method_and_relation_v1_20261026/relation/all_sources_v1';manifest=json.loads((prior/'relation_manifest.json').read_text());prior_targets=set();prior_bg=set();prior_reports=set();prior_target_reports=set();prior_windows=set()
for item in manifest['samples']:
 with np.load(prior/item['file'],allow_pickle=False) as z:
  target=str(z['target_mmsi']);prior_targets.add(target);prior_windows.add(str(z['source_window_id']));prior_bg.update(map(str,z['cohort_mmsi']));prior_reports.update((str(a),float(b)) for a,b in zip(z['neighbor_mmsi'],z['source_neighbor_t']));prior_target_reports.update((target,float(t)) for t in z['source_target_t'])
results=[]
for name in ('relation_test_v1_20261027','relation_test_expanded_v1_20261027'):
 folder=root/name;m=json.loads((folder/'test_manifest.json').read_text());targets=set();bg=set();reports=set();target_reports=set();windows=set();allowed_id_violations=[]
 for item in m['samples']:
  with np.load(folder/'cases'/item['file'],allow_pickle=False) as z:
   target=str(z['target_mmsi']);targets.add(target);windows.add(str(z['source_window_id']));bg.update(map(str,z['cohort_mmsi']));reports.update((str(a),float(b)) for a,b in zip(z['neighbor_mmsi'],z['source_neighbor_t']));target_reports.update((target,float(t)) for t in z['source_target_t'])
 assert len(targets)==m['test_target_mmsi'] and len(bg)==m['background_mmsi_test']
 assert len(bg&prior_bg)==m['shared_background_mmsi'] and len(reports&prior_reports)==m['shared_background_source_reports']
 assert not (targets&prior_targets) and not (windows&prior_windows) and not (target_reports&prior_target_reports)
 assert not bg&prior_targets
 results.append({'directory':name,'target_mmsi':len(targets),'source_windows':len(windows),'target_mmsi_overlap_trainval':len(targets&prior_targets),'target_observation_overlap_trainval':len(target_reports&prior_target_reports),'source_window_overlap_trainval':len(windows&prior_windows),'test_background_is_prior_target_overlap':len(bg&prior_targets),'background_mmsi_overlap':len(bg&prior_bg),'background_observation_overlap':len(reports&prior_reports),'test_background_observations':len(reports)})
result={'status':'passed','prior_source_windows':len(prior_windows),'prior_target_mmsi':len(prior_targets),'results':results,'limitation':'overlapping background/scene reports remain; target observation separation does not establish statistically independent scenes'}
(out/'audit_relation_isolation.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
