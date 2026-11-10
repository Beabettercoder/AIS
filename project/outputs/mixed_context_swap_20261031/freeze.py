from pathlib import Path
import hashlib,json,datetime
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OUT=ROOT/'mixed_context_swap_20261031'
assert not (OUT/'summary.json').exists() and not (OUT/'summary_partial.json').exists()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
paths=[OUT/'run_swap.py', ROOT/'mixed_stress_probe_20261031'/'frozen_protocol.json',ROOT/'mixed_stress_probe_20261031'/'train_mixed.py',ROOT/'heading_evidence_pilot_20261030'/'train_heading_evidence.py',ROOT/'heading_alignment_20261030'/'train_heading.py',ROOT/'framework_validation_v1_20261028'/'models.py',ROOT/'relation_repair_20261030'/'train_relation_field.npz',ROOT/'relation_repair_20261030'/'val_relation_field.npz',ROOT/'heading_evidence_pilot_20261030'/'train_heading_features.npz',ROOT/'heading_evidence_pilot_20261030'/'val_heading_features.npz']
for name in ('tcn','gru','transformer','itransformer'):
 for seed in ((0,1,2) if name in ('tcn','itransformer') else (0,)):
  for arm in ('F0_heading','F1_heading'):
   p=ROOT/'mixed_stress_probe_20261031'/'runs'/name/f'seed{seed}'/arm
   paths.extend((p/'final.pt',p/'val_scores.npz',p/'result.json'))
proto={'version':'mixed_context_swap_v1','status':'frozen before cross-over scores','model_selection':'all fixed mixed-trained F0/F1 checkpoints, TCN/iTransformer seeds 0/1/2, GRU/Transformer seed0','train_data':'train/base pooled normalizer of heading_evidence pipeline; no retraining','data':'existing previously exposed val, all 272 sources and 4 variants; no test data','inputs':'entire ordinary or reliable 40-D context including heading, field residual and support, with shared target, timestamps, masks and train-only normalization', 'matrix':['F0 model + F0 input','F0 model + F1 input','F1 model + F0 input','F1 model + F1 input'], 'views':['base','natural','suffix','combined_l100_m600'],'metric':'pooled AP and paired 2000 target MMSI cluster bootstrap; fixed-seed model-average; also per seed', 'notes':['off-diagonal inputs are inference interventions possibly out of distribution','F0/F1 context differs in several features; no causal single-weight attribution','baseline reference is same mixed-trained models; native diagonal predictions must reproduce within 1e-5','development already exposed; no multiplicity correction and seed-population uncertainty not included'],'sha256':{str(p):sha(p) for p in paths}}
with (OUT/'frozen_protocol.json').open('x',encoding='utf8') as f:json.dump(proto,f,indent=2)
print('frozen',sha(OUT/'frozen_protocol.json'),flush=True)
