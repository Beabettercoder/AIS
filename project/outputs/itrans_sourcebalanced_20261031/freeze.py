import json, hashlib
from pathlib import Path
import numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'itrans_sourcebalanced_20261031';H=R/'heading_evidence_pilot_20261030'
paths=[O/'train.py',O/'summary.py',H/'train_heading_features.npz',H/'val_heading_features.npz',H/'feature_builder.py',R/'heading_alignment_20261030/train_heading.py',R/'framework_validation_v1_20261028/models.py',R/'relation_repair_20261030/train_relation_field.npz',R/'relation_repair_20261030/val_relation_field.npz']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
with np.load(H/'train_heading_features.npz',allow_pickle=False) as z:
 source=z['source'].astype(str);_,inv,counts=np.unique(source,return_inverse=True,return_counts=True); assert len(source)==8136 and len(counts)==2034 and np.all(counts==4)
 batches=[np.concatenate([np.flatnonzero(inv==j) for j in np.random.default_rng(300000).permutation(len(counts))[i:i+32]]) for i in range(0,len(counts),32)]
 assert len(batches)==64 and len(batches[-1])==72 and len(np.unique(np.concatenate(batches)))==len(source)
 frozen={'task':'itransformer optimization: source-block minibatch, not model change','status':'frozen before training','selection':'one training-only rationale, chosen before evaluating candidate validation','seed':[0,1,2],'architecture':'40D EmbeddedITransformer identical old','arms':['C0','F0_heading','F1_noheading','F1_heading'],'epochs':75,'minibatch':'32 sources x 4 cases, source ID only; independent random source order with seed 300000+epoch; last 18 sources=72 windows','steps_per_epoch':len(batches),'optimizer':'AdamW lr0.001 wd0.01, clip1, BCE, dropout model unchanged','input_normalization':'unchanged train/Base','heldout':'previously exposed development validation; no test','uncertainty':'paired target MMSI bootstrap fixed models 2000, 3seed average, no seed-population variance or multiplicity correction','hashes':{str(p):sha(p) for p in paths}}
 (O/'frozen_protocol.json').write_text(json.dumps(frozen,indent=2),encoding='utf8')
 print(json.dumps({'sources':len(counts),'batches':len(batches),'last_batch':len(batches[-1]),'hash_train':frozen['hashes'][str(O/'train.py')]}))
