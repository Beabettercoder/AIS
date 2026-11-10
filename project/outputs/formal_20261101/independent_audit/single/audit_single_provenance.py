"""Independently verify frozen source inputs, target splits and all init states."""
from pathlib import Path
import json,hashlib,random,sys
import numpy as np,torch
R=Path('/mnt/sda/WangZijia/AIS/outputs');P=R/'formal_20261101/single';O=P.parent/'independent_audit/single';T=R/'framework_validation_v1_20261028/data/individual';sys.path.insert(0,str(P))
from eight_backbone_adapter import Model8

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((P/'frozen_checkpoint_thresholds.json').read_text());proto=json.loads((P/'frozen_train_protocol.json').read_text());assert sha(P/'frozen_train_protocol.json')==f['train_protocol_sha256']
source=[]
for section,values in [('train',proto['sha256']),('test_inputs',f['test_input_sha256']),('evaluation_code',f['evaluation_code_sha256'])]:
 for p,expected in values.items():
  got=sha(p);assert got==expected,(section,p);source.append({'section':section,'path':p,'sha256':got})
splits={}
for sp in ('train','val','test'):
 with np.load(T/f'{sp}.npz',allow_pickle=False) as z:splits[sp]={k:z[k].astype(str) for k in ('mmsi','source_id','base_window_id','window_id')}
for a,b in [('train','val'),('train','test'),('val','test')]:
 for k in ('mmsi','source_id','base_window_id','window_id'):
  overlap=set(splits[a][k])&set(splits[b][k]);assert not overlap,(a,b,k,len(overlap))
init=[]
for n in ['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph']:
 for seed in (0,1,2):
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Model8(n)
  actual=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest()
  exp={f['arms'][f'{n}/{seed}/{a}']['init_sha256'] for a in ['raw','causal_bundle','f0_bundle','f1_bundle']};assert len(exp)==1 and actual in exp,(n,seed,actual,exp)
  init.append({'name':n,'seed':seed,'sha256':actual})
report={'source_hashes_checked':len(source),'files':source,'paired_init_models':len(init),'init_hashes':init,'splits':{sp:{k:len(set(v)) for k,v in d.items()} for sp,d in splits.items()},'intersection':'all mmsi, source_id, base_window_id, window_id train/val/test pairwise 0','warning':'target/source disjoint does not imply background vessel/report independence; test previously exposed'}
p=O/'independent_single_provenance_audit.json';assert not p.exists();p.write_text(json.dumps(report,indent=2));print('SOURCE HASHES',len(source),'INIT',len(init),'SPLITS',json.dumps(report['splits']))
