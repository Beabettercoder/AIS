"""Freeze all final checkpoints and validation-calibrated thresholds before test."""
from pathlib import Path
import hashlib,json,math
import numpy as np
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');N=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];A=['raw','causal_bundle','f0_bundle','f1_bundle']
def sh(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
completed=json.loads((P/'launch_result.json').read_text());assert len(completed)==20
accelerated=json.loads((P/'cde2_accelerated_result.json').read_text());assert accelerated['exit_code']==0
graph_acc=json.loads((P/'graph_accelerated_result.json').read_text());assert len(graph_acc['results'])==3 and all(x['exit_code']==0 for x in graph_acc['results']) and {x['seed'] for x in graph_acc['results']}=={0,1,2}
protocol=json.loads((P/'frozen_train_protocol.json').read_text()); cde_launch=json.loads((P/'cde2_accelerated_launch.json').read_text()); graph_launch=json.loads((P/'graph_accelerated_launch.json').read_text()); expected_train=protocol['sha256'][str(P/'train_single.py')];assert sh(P/'train_single.py')==expected_train==cde_launch['script_sha256']==graph_launch['script_sha256']
for item in graph_acc['results']: assert sh(P/f"graph_seed{item['seed']}_accelerated.log")==item['training_log_sha256']
for x in completed:
 if (x['name']=='neuralcde' and x['seed']==2) or x['name']=='graph':
  assert x['exit_code']!=0 and 'FileExistsError' in Path(x['log']).read_text(), 'Expected safe duplicate refusal not verified'
 else: assert x['exit_code']==0,x

expected={(n,s) for n in N for s in (0,1,2)}-{(n,0) for n in N[:4]};assert {(x['name'],x['seed']) for x in completed}==expected
out={'scope':'fixed 25th epoch; thresholds fit on validation negatives; test previously exposed; no test scoring before freeze','train_protocol_sha256':sh(P/'frozen_train_protocol.json'),'acceleration':{'note':'single CDE seed2 started once on GPU0; later queued attempt safely refused existing directory','launch_sha256':sh(P/'cde2_accelerated_launch.json'),'result_sha256':sh(P/'cde2_accelerated_result.json'),'training_log_sha256':sh(P/'cde2_accelerated.log'),'graph_launch_sha256':sh(P/'graph_accelerated_launch.json'),'graph_result_sha256':sh(P/'graph_accelerated_result.json')},'arms':{}}
for n in N:
 for s in (0,1,2):
  keys=[]
  for a in A:
   p=P/'runs'/n/f'seed{s}'/a;j=json.loads((p/'result.json').read_text());assert len(j['history'])==25 and j['history'][-1]['epoch']==25 and j['seed']==s;keys.append(j['init'])
   with np.load(p/'final_val.npz',allow_pickle=False) as z:
    score=z['score'];y=z['y'];assert len(score)==1680 and np.isfinite(score).all();neg=np.sort(score[y==0]);assert len(neg)>10
    th={}
    for budget in (.001,.01,.05):
     i=min(len(neg)-1,max(0,int(math.ceil((1-budget)*len(neg)))-1));t=float(neg[i]);th[str(budget)]={'threshold':t,'val_negative_count':len(neg),'val_FAR':float((neg>t).mean())}
   out['arms'][f'{n}/{s}/{a}']={'checkpoint':str(p/'final.pt'),'checkpoint_sha256':sh(p/'final.pt'),'validation':str(p/'final_val.npz'),'validation_sha256':sh(p/'final_val.npz'),'train_result_sha256':sh(p/'result.json'),'init_sha256':j['init'],'thresholds':th}
  assert len(set(keys))==1,(n,s)
assert len(out['arms'])==96
out['evaluation_code_sha256']={str(p):sh(p) for p in [P/'freeze_checkpoints.py',P/'evaluate_fixed.py',P/'launch_evaluation.py']};out['test_input_sha256']={str(p):sh(p) for p in [Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028/data/individual/test.npz'),Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028/extra_data/field_test.npz'),Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030/rho_test.npz')]}
f=P/'frozen_checkpoint_thresholds.json';assert not f.exists();f.write_text(json.dumps(out,indent=2));print('FROZEN_CHECKPOINTS',len(out['arms']),sh(f))
