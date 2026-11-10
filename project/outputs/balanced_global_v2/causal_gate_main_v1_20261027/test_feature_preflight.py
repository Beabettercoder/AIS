"""Replay frozen test-feature builder on validation and compare saved train arrays."""
import sys,json
from pathlib import Path
import numpy as np
P=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/causal_gate_main_v1_20261027')
sys.path.insert(0,str(P))
import train_controls as t
s=(P/'evaluate_test.py').read_text()
before="p=BASE/region/'test.npz';q=FOLDER/region/'test.npz'"
after="p=BASE/region/'val.npz';q=ROOT/'robust_background_v1_20261027'/region/'val.npz'"
assert before in s
ctx={'__name__':'preflight_validation'};exec(compile(s.replace(before,after),str(P/'evaluate_test.py'),'exec'),ctx)
v,_=ctx['prep']();d,_=t.build_data();assert np.array_equal(v['window_id'],d['val']['window_id'])
out={k:{'max_abs_difference':float(np.max(np.abs(v['features'][k]-d['val']['features'][k]))),
         'nonidentical_values':int(np.sum(v['features'][k]!=d['val']['features'][k]))} for k in t.arms}
assert all(x['nonidentical_values']==0 for x in out.values())
(P/'test_feature_preflight.json').write_text(json.dumps({'status':'passed','validated_windows':len(v['window_id']),'comparisons':out},indent=2),encoding='utf8')
print(json.dumps(out),flush=True)
