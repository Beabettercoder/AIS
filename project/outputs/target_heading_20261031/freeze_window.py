import json,hashlib
from pathlib import Path
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OUT=ROOT/'target_heading_20261031'
paths=[OUT/'train_target.py',OUT/'audit_input.py',OUT/'smoke.py',ROOT/'heading_evidence_pilot_20261030/train_heading_evidence.py',ROOT/'heading_evidence_pilot_20261030/train_heading_features.npz',ROOT/'heading_evidence_pilot_20261030/val_heading_features.npz',ROOT/'heading_coverage_20261030/heading_train.npz',ROOT/'heading_coverage_20261030/heading_val.npz',ROOT/'relation_repair_20261030/train_relation_field.npz',ROOT/'relation_repair_20261030/val_relation_field.npz']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for q in iter(lambda:f.read(1048576),b''):h.update(q)
 return h.hexdigest()
d={'task':'target_heading_additional_evidence','scope':'exploratory offline train/previously exposed val; no test labels','source':'target source heading with exact UTC and MMSI, one shared trajectory across four controlled cells, not proven independent or physically consistent sensor','features':'target Heading minus current controlled COG circular sin/cos, heading valid, previous observed heading yaw-rate deg/s, yaw valid; no source speed or position is added','arms':['C0_target','F0_target','F1_no_target','F1_target'],'arms_definition':'C0 target-only with target heading; F0 and F1 baseline neighbor heading; F1_no_target has target heading inputs zero but equal width; F1_target and F0_target include target heading','all_models':'equal width 45 channels, matched init/optim/seed/normalizer within backbone; train only base; masked missing','precommitted_plan':{'tcn':[0,1,2],'gru':[0],'transformer':[0]},'epochs':25,'batch':'original window-level permutation shared across arms','optimizer':'AdamW lr=.001 wd=.01 batch128 gradient clip1','itransformer':'separate plan will be declared before it starts after source-grouping assessment','uncertainty':'paired target-MMSI bootstrap 2000 for fixed models; exploratory','sha256':{str(p):sha(p) for p in paths}}
(OUT/'frozen_protocol_window.json').write_text(json.dumps(d,indent=2),encoding='utf8')
print(sha(OUT/'frozen_protocol_window.json'))
