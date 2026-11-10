"""Exact replay keys for the frozen v1 metadata assignment (no relabeling).

Physical corruption noise_seed is inherited correctly from the source plan.
The original auxiliary *_seed columns are unused hash namespaces, not the rank
keys used during v1 quota assignment. This sidecar makes the actual keys explicit
without modifying the frozen label ledger or any model input.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

OUT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/labeling/labeled_v1');SEED=20261026
def seed(*p):return int.from_bytes(hashlib.sha256(':'.join(map(str,p)).encode()).digest()[:8],'little')&((1<<63)-1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=OUT/'label_manifest.parquet';before=sha(source);g=pd.read_parquet(source)
replay=g[['window_id','region','split','y','noise_seed']].copy()
replay['noise_selection_rank_key']=[seed(SEED,'noise_selection',w) for w in replay.window_id]
replay['noise_kind_rank_key']=[seed(SEED,'noise_kind',w) for w in replay.window_id]
replay['noise_scope_rank_key']=[seed(SEED,'noise_scope',w) for w in replay.window_id]
replay['noise_kind_group_offset']=0;replay['noise_scope_group_offset']=0
replay['expected_observation_noise']=False;replay['expected_noise_kind']='none';replay['expected_noise_scope']='none'
byid=replay.set_index('window_id',drop=False)
for (region,split,y),s in g.groupby(['region','split','y'],sort=True):
 k=int(np.floor(len(s)/3+.5));ranked=sorted(s.window_id,key=lambda w:(seed(SEED,'noise_selection',w),w));noisy=ranked[:k]
 byid.loc[noisy,'expected_observation_noise']=True
 for factor,choices in [('noise_kind',['teleport','speed_spike','drift']),('noise_scope',['target_only','neighbors_only','both'])]:
  ranked=sorted(noisy,key=lambda w:(seed(SEED,factor,w),w));offset=int(seed(SEED,region,split,int(y),factor,'offset')%3)
  byid.loc[s.window_id,factor+'_group_offset']=offset
  for j,w in enumerate(ranked):byid.loc[w,'expected_'+factor]=choices[(j+offset)%3]
r=byid.loc[g.window_id].reset_index(drop=True)
assert np.array_equal(r.expected_observation_noise,g.observation_noise)
assert np.array_equal(r.expected_noise_kind,g.noise_kind)
assert np.array_equal(r.expected_noise_scope,g.noise_scope)
assert np.array_equal(r.noise_seed,g.noise_seed) and sha(source)==before
path=OUT/'noise_assignment_provenance.parquet';r.to_parquet(path,index=False)
report={'status':'passed','frozen_label_manifest_sha256':before,'rows_replayed':len(g),'observation_noise_flags_matched':True,'noise_kind_matched':True,'noise_scope_matched':True,'physical_corruption_noise_seed_unchanged':True,'frozen_label_manifest_changed':False,'actual_rank_key_columns':['noise_selection_rank_key','noise_kind_rank_key','noise_scope_rank_key'],'legacy_auxiliary_columns_not_used_for_v1_rank_replay':['noise_selection_seed','noise_kind_seed','noise_scope_seed'],'legacy_field_explanation':'The frozen v1 auxiliary seed columns hash (SEED,window_id,factor), whereas actual ranking hashes (SEED,factor,window_id). They were not consumed by any physical generator or feature builder. This sidecar records the actual ranking keys and group offsets. The physical noise_seed is unchanged and remains the corruption RNG seed.','namespace_equation':'int.from_bytes(SHA256(colon_join(parts))[:8],little) & ((1<<63)-1)','seed':SEED,'sidecar_sha256':sha(path),'replay_script_sha256':sha(Path(__file__))}
(OUT/'noise_assignment_replay_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
