"""Explicit code/data fingerprints for recent late-bridge experiments."""
from pathlib import Path
import hashlib
import gzip,json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def source_files(method,include_test=False):
    vendor={'tft_gcn':'TFT-GCN','simad':'SimAD','dmrad':'DMRAD','carla':'CARLA'}[method]
    code=[HERE/'train_recent_late_bridge.py',HERE/'recent_late_bridge.py',HERE/'recent_late_provenance.py',
          HERE/'recent_late_bridge_preflight.py',HERE/'freeze_recent_late_bridge.py',HERE/'summarize_recent_late_bridge.py',
          HERE/'run_recent_late_training.sh',HERE/'run_recent_late_finalize.sh',HERE/'run_recent_late_launch.sh',HERE/'summarize_eight_matched.py',
          HERE/'adapters'/f'{method}.py',ROOT/'formal_20261101/single/train_single.py',
          ROOT/'formal_20261101/single/eight_backbone_adapter.py',ROOT/'framework_validation_v1_20261028/models.py',
          ROOT/'original_mainline_pilot_20261029/pilot_features.py',ROOT/'original_mainline_pilot_20261029/pilot_historical_loss.py']
    code.extend((HERE/'vendor'/vendor).rglob('*.py'))
    code=sorted(set(p for p in code if '__pycache__' not in p.parts),key=str)
    splits=('train','val','test') if include_test else ('train','val')
    data=[]
    for split in splits:
        data.extend([ROOT/'framework_validation_v1_20261028/data/individual'/f'{split}.npz',
                     ROOT/'framework_validation_v1_20261028/extra_data'/f'field_{split}.npz'])
        data.append((ROOT/'eight_backbone_preparation_20261030'/'rho_test.npz') if split=='test' else (ROOT/'original_mainline_pilot_20261029'/f'rho_{split}.npz'))
    if include_test:
        data.append(ROOT/'framework_validation_v1_20261028/extra_data/field_member_provenance.jsonl.gz')
        code.extend([ROOT/'formal_20261101/single/evaluate_fixed.py',ROOT/'eight_backbone_preparation_20261030/prepare_test_rho.py'])
        code=sorted(set(code),key=str)
    missing=[str(p) for p in [*code,*data] if not p.is_file()]
    if missing:raise FileNotFoundError(missing)
    return {'code':{str(p):sha(p) for p in code},'data':{str(p):sha(p) for p in data}}

def verify_test_rho():
    """Recompute test rho against source IDs/members, without reading labels."""
    root=ROOT/'framework_validation_v1_20261028'
    with np.load(root/'data/individual/test.npz',allow_pickle=False) as z:
        ids=z['window_id'].astype(str);mmsi=z['mmsi'].astype(str);mask=z['mask'];times=z['time_s']
    with np.load(root/'extra_data/field_test.npz',allow_pickle=False) as z:
        for key,expected in (('window_id',ids),('mmsi',mmsi),('mask',mask),('time_s',times)):
            if not np.array_equal(z[key],expected):raise ValueError('test core/field provenance mismatch: '+key)
        field_valid=z['field_common_valid']
    if len(set(ids))!=len(ids):raise ValueError('duplicate test window IDs')
    with np.load(ROOT/'eight_backbone_preparation_20261030/rho_test.npz',allow_pickle=False) as z:rho=z['rho']
    if rho.shape!=mask.shape or not np.array_equal(rho>0,field_valid & mask):raise ValueError('test rho validity mismatch')
    idx={wid:i for i,wid in enumerate(ids)};recomputed=np.zeros_like(rho);seen=set()
    path=root/'extra_data/field_member_provenance.jsonl.gz'
    with gzip.open(path,'rt',encoding='utf8') as f:
        for line in f:
            row=json.loads(line)
            if row['split']!='test' or not row['shared_field_valid']:continue
            i=idx[str(row['window_id'])];step=int(row['step'])
            if str(row['target_mmsi'])!=mmsi[i]:raise ValueError('test rho target MMSI mismatch')
            if step<0 or step>=mask.shape[1] or not mask[i,step]:raise ValueError('test rho invalid step')
            if (i,step) in seen:raise ValueError('duplicate test rho provenance')
            seen.add((i,step));weights=np.array([x['w_abs'] if x['kinematic_valid'] else 0. for x in row['members']],dtype=float)
            recomputed[i,step]=float(weights.mean()*min(len(weights)/4.,1.)) if len(weights) else 0.
    if not np.array_equal(recomputed,rho):raise ValueError('test rho does not match ID-mapped member provenance')
    return {'window_count':len(ids),'valid_steps':int((rho>0).sum()),'target_mmsi_count':len(set(mmsi)),
            'window_id_mmsi_mask_time_and_member_recomputation':'passed; no test labels read'}
