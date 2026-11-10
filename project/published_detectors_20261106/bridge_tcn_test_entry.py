#!/usr/bin/env python3
"""Resolve formal test rho source after freeze; leave training code untouched."""
from pathlib import Path
import hashlib,runpy
import numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs')
missing=R/'original_mainline_pilot_20261029/rho_test.npz'
actual=R/'eight_backbone_preparation_20261030/rho_test.npz'
core=R/'framework_validation_v1_20261028/data/individual/test.npz'
field=R/'framework_validation_v1_20261028/extra_data/field_test.npz'
native=np.load
with native(actual,allow_pickle=False) as z:rho=z['rho']
with native(core,allow_pickle=False) as z:mask=z['mask'];ids=z['window_id'];mmsi=z['mmsi'];y=z['y'];region=z['region']
with native(field,allow_pickle=False) as z:valid=z['field_common_valid']
if rho.shape!=mask.shape or not np.array_equal(rho>0,valid&mask):raise ValueError('test rho does not match paired support')
formal=R/'formal_20261101/single/runs/tcn/seed0/test/raw_scores.npz'
with native(formal,allow_pickle=False) as z:
    for key,value in (('window_id',ids),('mmsi',mmsi),('y',y),('region',region)):
        if not np.array_equal(z[key],value):raise ValueError('test ID mismatch with original formal '+key)
print('TEST_INPUT_RESOLUTION',actual,'sha256',hashlib.sha256(actual.read_bytes()).hexdigest(),'entry_sha256',hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'formal_ids_equal',len(ids),flush=True)
def load(file,*args,**kwargs):
    try:path=Path(file)
    except TypeError:path=None
    return native(actual if path==missing else file,*args,**kwargs)
np.load=load
runpy.run_path(str(Path(__file__).with_name('bridge_tcn_20261107.py')),run_name='__main__')
