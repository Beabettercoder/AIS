"""Use the locked single-vessel detection windows and their original LOO features."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/sda/WangZijia/AIS/outputs')
sys.path.insert(0, str(ROOT/'original_mainline_pilot_20261029'))
from pilot_features import build_field_inputs

def load(split):
    base=ROOT/'framework_validation_v1_20261028'
    with np.load(base/'data/individual'/f'{split}.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
    with np.load(base/'extra_data'/f'field_{split}.npz',allow_pickle=False) as z: f={k:z[k] for k in z.files}
    a=build_field_inputs(d,f,'F0'); b=build_field_inputs(d,f,'F1')
    # No temporal interpolation, future lookahead or padded values in native losses.
    ix=np.flatnonzero(np.asarray(d['mask']).all(axis=1))
    result={k:d[k][ix] for k in ('y','mmsi','region','window_id','noise_type','observation_noise','time_s')}
    result.update(raw=d['base__C0'][ix,:,:7],target=a['self'][ix],f0=a['field'][ix],f1=b['field'][ix],valid=a['valid'][ix],rho=a['rho'][ix])
    return result

def fit(d):
    neg=d['y']==0
    target=np.concatenate([d['raw'][neg].reshape(-1,7), d['target'][neg].reshape(-1,7)])
    v=d['valid'] & neg[:,None]
    field=np.concatenate([d['f0'][v],d['f1'][v]])
    def st(x):
        m=x.mean(0);s=x.std(0);return m,np.where(s<1e-7,1,s)
    sm,ss=st(target);fm,fs=st(field)
    return dict(sm=sm,ss=ss,fm=fm,fs=fs)

def features(d, norm, arm):
    if arm not in ('raw','gate','ordinary','reliable'): raise ValueError(arm)
    key='raw' if arm=='raw' else 'target'
    target=(d[key]-norm['sm'])/norm['ss']
    field=np.zeros((*target.shape[:2],6),dtype=np.float32)
    if arm in ('ordinary','reliable'):
        key='f0' if arm=='ordinary' else 'f1'
        field[:,:,:4]=np.where(d['valid'][...,None],(d[key]-norm['fm'])/norm['fs'],0)
        field[:,:,4]=d['rho'];field[:,:,5]=d['valid']
    x=np.concatenate([target,field],axis=-1).astype(np.float32)
    if not np.isfinite(x).all(): raise ValueError('nonfinite AIS features')
    return x
