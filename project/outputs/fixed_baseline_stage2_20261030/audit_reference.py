import hashlib,sys,json
from pathlib import Path
import numpy as np
B=Path('/mnt/sda/WangZijia/AIS/outputs');H=B/'heading_alignment_20261030';O=B/'relation_repair_20261030'
sys.path.insert(0,str(H));import train_heading as base
base.OLD=O;base.VARIANT='aligned_embed'
tr=base.read('train');mu,sd=base.fit(tr)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={
 'tcn0':H/'runs/aligned_embed/tcn/protocol.json',
 'gru0':H/'runs/aligned_embed/gru/protocol.json',
 'transformer0':H/'runs/aligned_embed/transformer/protocol.json',
 'itrans0_25':H/'runs/aligned_embed/itransformer/protocol.json',
 'itrans0_75':B/'itrans_stability_20261030/extended75/runs/seed0/aligned_embed/itransformer/protocol.json',
 'itrans1_75':B/'itrans_stability_20261030/extended75/runs/seed1/aligned_embed/itransformer/protocol.json',
 'itrans2_75':B/'itrans_stability_20261030/extended75/runs/seed2/aligned_embed/itransformer/protocol.json'}
results={}
for name,p in paths.items():
 d=json.loads(p.read_text(encoding='utf8')); mm=np.array(d['means']);ss=np.array(d['stds']);
 results[name]={'path':str(p),'file_sha256':sha(p),'mean_max_abs_diff':float(np.max(np.abs(mu-mm))),'std_max_abs_diff':float(np.max(np.abs(sd-ss))),'exact':bool(np.array_equal(mu,mm) and np.array_equal(sd,ss))}
 if not results[name]['exact']:raise AssertionError(name)
out={'scope':'train/Base only; no test, no training','reference_train':str(O/'train_relation_field.npz'),'reference_train_sha256':sha(O/'train_relation_field.npz'),'heading_source_sha256':sha(H/'train_heading.py'),'mean':mu.tolist(),'std':sd.tolist(),'checks':results}
p=Path('/mnt/sda/WangZijia/AIS/outputs/fixed_baseline_stage2_20261030/reference_normalizer.json');p.write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps({'pass':True,'path':str(p),'checks':{k:v['exact'] for k,v in results.items()}}))
