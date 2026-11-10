from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen_pilot'))
import train as T
import features as F
import build_data as B
import summarize as S
from forecast_models import ForecastModel,state_hash
CONFIG=json.loads((ROOT/'release_config.json').read_text(encoding='utf-8'))
DEFAULT_PILOT=Path('/mnt/sda/WangZijia/AIS/forecast_pilot_20261105')

def sha(path):return T.digest(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):T.atomic_json(path,obj)
def code_hashes():
    # Freeze the executable research pipeline, independently of manuscript or PDF assets.
    names=['common.py','pipeline.py','prepare.py','train_worker.py','freeze.py','build_test.py',
           'evaluate_test.py','report_results.py','audit_run.py','selftest.py','monitor.py',
           'reanchor.py','cv_and_build.py','release_config.json']
    paths=[ROOT/n for n in names]+sorted((ROOT/'frozen_pilot').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def load_npz(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def assert_frozen(work):
    freeze=read(Path(work)/'checkpoint_freeze.json')
    if freeze['code_hashes']!=code_hashes():raise RuntimeError('Code changed after freeze. Keep this release immutable.')
    for row in freeze['models']:
        if sha(row['checkpoint'])!=row['checkpoint_sha256']:raise RuntimeError('Frozen model checkpoint changed.')
    if sha(Path(work)/'data/normalizer.json')!=freeze['normalizer_sha256']:raise RuntimeError('Frozen normalizer changed.')
    return freeze
def validate_parent(pilot):
    manifest=read(pilot/'data/data_manifest.json')
    for relative,key in [('outputs/balanced_global_v2/source_registry.json','source_registry_sha256'),
                         ('outputs/balanced_global_v2/labeling/labeled_v1/label_manifest.parquet','identity_manifest_sha256'),
                         ('outputs/balanced_global_v2/release_20261026/global_all_mmsi_split.parquet','global_split_sha256')]:
        if sha(pilot.parent/relative)!=manifest['config'][key]:raise RuntimeError(f'Frozen source/split registry changed: {relative}')
    for n in ['features.py','build_data.py','train.py','forecast_models.py','vendor_encoders.py']:
        if sha(pilot/n)!=sha(ROOT/'frozen_pilot'/n):raise RuntimeError(f'Parent source version mismatch: {n}')
    for split in ['train','val']:
        if sha(pilot/f'data/{split}.npz')!=manifest['npz_sha256'][split]:raise RuntimeError(f'Parent {split} data changed.')
    return manifest
