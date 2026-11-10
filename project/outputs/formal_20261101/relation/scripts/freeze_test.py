from pathlib import Path
import json,hashlib,datetime
import numpy as np
O=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/relation');names=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];out=O/'heldout.lock.json'
if out.exists():raise FileExistsError(out)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
deps=json.loads((O/'builder_dependencies.lock.json').read_text());assert all(sha(Path(k))==v for k,v in deps['files'].items())
status=json.loads((O/'status.json').read_text());assert len(status['jobs'])==24 and all(v['status']=='completed' for v in status['jobs'].values())
locks={}
for name in names:
 for seed in (0,1,2):
  d=O/'runs'/name/f'seed{seed}';pr=json.loads((d/'protocol.json').read_text()); epochs=75 if name=='itransformer' else 25;assert pr['epochs']==epochs
  for arm in ('C0','F0','F1'):
   f=d/arm;r=json.loads((f/'result.json').read_text());assert r['last_epoch']==epochs and sha(f/'final.pt')==r['weights_sha256'] and sha(f/'val_scores.npz')==r['prediction_sha256']
   with np.load(f/'val_scores.npz',allow_pickle=False) as z:
    neg=z['base_score'][z['y']==0];assert len(neg)>0;ordered=np.sort(neg)
    th={str(b):float(ordered[min(len(ordered)-1,max(0,int(np.ceil((1-b)*len(ordered)))-1))]) for b in (.01,.05)}
   locks[f'{name}/seed{seed}/{arm}']={'model_sha256':sha(f/'final.pt'),'val_sha256':sha(f/'val_scores.npz'),'thresholds':th,'negative_count':len(neg),'epoch':epochs}
assert len(locks)==72
obj={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rule':'validation Base negatives, ceil order statistic, strict >','status':'all 72 models frozen BEFORE loading/construction of test','models':locks,'protocol_sha256':sha(O/'protocol.lock.json'),'builder_dependencies_sha256':sha(O/'builder_dependencies.lock.json'),'test_builder_sha256':sha(O/'scripts/build_test.py'),'postfreeze_scripts_sha256':{p.name:sha(p) for p in [O/'scripts/freeze_test.py',O/'scripts/eval.py',O/'scripts/launch_eval.py',O/'scripts/summarize.py',O/'scripts/render.py',O/'scripts/render_figure.py',O/'scripts/dataset_audit.py']}}
out.write_text(json.dumps(obj,indent=2),encoding='utf8');print(str(out),len(locks),sha(out))
