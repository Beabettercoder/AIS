"""Build held-out support from recorded members without reading test labels.

This script is preparatory only. It must not train, score, or inspect outcomes.
"""
import gzip
import json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028')
OUT = Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')

with np.load(ROOT / 'data/individual/test.npz', allow_pickle=False) as z:
    ids = z['window_id']
    rho = np.zeros(z['mask'].shape, np.float32)
    lookup = {str(x): i for i, x in enumerate(ids)}
    assert len(lookup) == len(ids)
for line in gzip.open(ROOT / 'extra_data/field_member_provenance.jsonl.gz', 'rt', encoding='utf8'):
    r = json.loads(line)
    if r['split'] != 'test' or not r['shared_field_valid']:
        continue
    weights = np.array([m['w_abs'] if m['kinematic_valid'] else 0. for m in r['members']], dtype=float)
    rho[lookup[r['window_id']], int(r['step'])] = float(weights.mean() * min(len(weights) / 4., 1.)) if len(weights) else 0.
with np.load(ROOT / 'extra_data/field_test.npz', allow_pickle=False) as z:
    assert np.array_equal(rho > 0, z['field_common_valid'])
dest = OUT / 'rho_test.npz'
if dest.exists():
    with np.load(dest, allow_pickle=False) as z:
        assert np.array_equal(z['rho'], rho)
else:
    np.savez_compressed(dest, rho=rho)
print(json.dumps({'test_target_count': len(ids), 'supported_steps': int((rho > 0).sum()),
                  'labels_read': False, 'path': str(dest)}))
