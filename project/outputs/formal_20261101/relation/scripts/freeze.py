from pathlib import Path
import json,hashlib,datetime
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'formal_20261101/relation'
paths=[O/'scripts/train.py',R/'eight_backbone_preparation_20261030/relation8.py',R/'heading_alignment_20261030/train_heading.py',R/'framework_validation_v1_20261028/models.py',R/'relation_repair_20261030/builder.py',R/'relation_repair_20261030/train_relation_field.npz',R/'relation_repair_20261030/val_relation_field.npz']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p={'study':'user-authorized eight-backbone three-seed relation study','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'train/dev freeze; held-out test not constructed or read','source_sha256':{str(x):sha(x) for x in paths},'backbones':['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'],'seeds':[0,1,2],'arms':['C0','F0','F1'],'input':'35D (24 causal target/field +11 heading-aligned mathematical projections, no AIS Heading field)','training':'Base view only, BCE, 25 epochs except iTransformer 75; iTransformer and PatchTST source-balanced four-case groups, others window shuffled','test_policy':'previously exposed test, evaluation after all train/val checkpoints and thresholds frozen','notes':['F0 distance weighted field, F1 distance*kinematic weighting, plus their pre-aggregation and support differences','Graph is custom two-node approximation and model-specific adaptations are not official replicas','Offline cohort fixed from full window','CI clustered on target MMSI; shared background can remain']}
t=O/'protocol.lock.json'
if t.exists():raise FileExistsError(t)
t.write_text(json.dumps(p,indent=2),encoding='utf8')
print(t)
