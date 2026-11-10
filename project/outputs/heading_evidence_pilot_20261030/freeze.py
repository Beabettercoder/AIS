"""Freeze exploratory Heading side-evidence pilot before model training."""
import hashlib,json
from pathlib import Path
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'heading_evidence_pilot_20261030'
paths=[O/'feature_builder.py',O/'train_heading_evidence.py',O/'train_information_probe.py',O/'report_train_probe.py',O/'train_information_probe.json',O/'train_report_probe.json',O/'train_heading_features.npz',O/'val_heading_features.npz',R/'heading_coverage_20261030/heading_train.npz',R/'heading_coverage_20261030/heading_val.npz',R/'relation_repair_20261030/train_relation_field.npz',R/'relation_repair_20261030/val_relation_field.npz',R/'heading_alignment_20261030/train_heading.py',R/'framework_validation_v1_20261028/models.py']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
lock={'scope':'Exploratory train/val on previously exposed data; no test labels used; heading not verified as independent sensor.',
 'models':['tcn','gru','transformer','itransformer'],
 'seeds':{'tcn':[0,1,2],'gru':[0],'transformer':[0],'itransformer':[0,1,2]},
 'epochs':{'tcn':25,'gru':25,'transformer':25,'itransformer':75},
 'arms':['C0','F0_heading','F1_noheading','F1_heading'],
 'evidence':'neighbor Heading-COG circular sin/cos, heading availability, causal heading yaw-rate and its availability; no future heading, speed-based sensor ground truth, injection identity, or labels as features',
 'normalization':'old 35 target/context train-only; additional heading features use training F0/F1 valid steps; coverage remains raw, unavailable normalized slots zero',
 'fit':'fixed last epoch, BCE, AdamW lr .001 and weight decay .01, hidden64,batch128, gradient clip1; paired same init, batch and budget per backbone/seed',
 'primary_comparisons':['F1_heading minus F1_noheading','F1_heading minus F0_heading','F1_heading minus C0'],
 'uncertainty':'target-MMSI paired cluster bootstrap 2000, samples fixed trained model seeds, exploratory no multiplicity adjustment',
 'safety':'0<=heading<360; heading validity mask; yaw uses two valid observed headings with 0<dt<=300; fixed offline cohort and half-window boundaries',
 'pretraining_probe':'held-out target-MMSI logistic within train: AP original .61767 vs +Heading .61698; low target initial speed<2kn subset .55117 vs .54504. Simplified probe is not a forecast of full models.',
 'sha256':{str(p):sha(p) for p in paths}}
p=O/'frozen_protocol.json';assert not p.exists();p.write_text(json.dumps(lock,indent=2),encoding='utf8');print(p,sha(p))
