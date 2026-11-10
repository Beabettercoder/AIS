from pathlib import Path
import importlib.util,json,sys,hashlib
import numpy as np

root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');work=root/'relation_test_expanded_v1_20261027';out=root/'independent_audit_causal_v1_20261027';prior=root/'relation_robust_v1_20261027';sys.path.insert(0,str(prior));sys.path.insert(0,str(work))
import run_relation as rr
spec=importlib.util.spec_from_file_location('supplement_context',work/'report_context_common.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
manifest=json.loads((work/'test_manifest.json').read_text());lock=json.loads((work/'report_context_lock.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in lock['files'].items():assert sha(p)==h
rng=np.random.default_rng(52711);tested=0;maxerr=0.;casepairs=0
for item in manifest['samples']:
 for field in ('file','existing_file','high_file'):
  with np.load(work/'cases'/item[field],allow_pickle=False) as z:o=rr.observed(z)
  for case in range(4):
   n=len(o['target_t'][case]);cut=max(2,n//2);baseline=mod.extract(o,case);altered={k:v.copy() for k,v in o.items()};cuttime=altered['target_t'][case,cut-1]
   altered['target_sog_kn'][case,cut:]+=20;altered['target_cog_deg'][case,cut:]=(altered['target_cog_deg'][case,cut:]+45)%360
   f=altered['neighbor_t'][case]>cuttime;altered['neighbor_sog_kn'][case,f]+=20;altered['neighbor_cog_deg'][case,f]=(altered['neighbor_cog_deg'][case,f]+45)%360;altered['neighbor_pos'][case,f]+=rng.normal(0,10000,size=(int(f.sum()),2))
   changed=mod.extract(altered,case);maxerr=max(maxerr,float(np.max(np.abs(changed[:cut]-baseline[:cut]))));tested+=1
assert tested==len(manifest['samples'])*3*4 and maxerr==0
result={'status':'passed','sources':len(manifest['samples']),'view_quartet_cases':tested,'suffix_perturbation_prefix_max_error':maxerr,'lock_hashes_match':True,
 'scope':'future speed/course/position values changed with timestamps, total window length, and membership held fixed; anchor uses total length and source membership is selected using whole-window future availability; this is not proof of online eligibility or causal window selection',
 'chronology':'supplemental protocol and six models were frozen after initial test results had been viewed; supplemental test scoring occurred after this later freeze'}
(out/'audit_report_context_causality.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
