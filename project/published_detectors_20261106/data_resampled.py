"""Offline 20-point interpolation within each existing AIS window.

No extrapolation beyond the original window. Field values are interpolated
only inside an adjacent pair of valid observations: gaps stay unsupported.
Do not use this full-window interpolation as a causal streaming feature.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/sda/WangZijia/AIS/outputs')
sys.path.insert(0, str(ROOT/'original_mainline_pilot_20261029'))
from pilot_features import build_field_inputs

def _resample(t, values, grid):
    return np.stack([np.interp(grid,t,values[:,j]) for j in range(values.shape[1])],axis=-1)

def _field_on_grid(t, values, ok, grid):
    """Interpolate within consecutive valid steps only, retaining isolated knots."""
    out=np.zeros((len(grid),values.shape[1]), dtype=np.float32)
    support=np.zeros(len(grid),dtype=bool)
    pos=np.searchsorted(t,grid,side='right')-1
    for k, p in enumerate(pos):
        if p<0: continue
        if np.isclose(grid[k],t[p],rtol=0,atol=1e-6):
            if ok[p]:out[k]=values[p];support[k]=True
        elif p+1<len(t) and ok[p] and ok[p+1]:
            frac=(grid[k]-t[p])/(t[p+1]-t[p])
            out[k]=(1-frac)*values[p]+frac*values[p+1];support[k]=True
    return out,support

def load(split, seq_len=20):
    base=ROOT/'framework_validation_v1_20261028'
    with np.load(base/'data/individual'/f'{split}.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
    with np.load(base/'extra_data'/f'field_{split}.npz',allow_pickle=False) as z: f={k:z[k] for k in z.files}
    a=build_field_inputs(d,f,'F0');b=build_field_inputs(d,f,'F1')
    if not np.array_equal(a['valid'],b['valid']):raise ValueError('Unpaired field validity')
    N=len(d['y']);raw=np.zeros((N,seq_len,7),np.float32);target=raw.copy()
    f0=np.zeros((N,seq_len,4),np.float32);f1=f0.copy();rho=np.zeros((N,seq_len),np.float32)
    valid=np.zeros((N,seq_len),bool);times=np.zeros((N,seq_len),np.float64)
    lengths=d['mask'].sum(1)
    for i in range(N):
        mask=np.asarray(d['mask'][i],bool);t=d['time_s'][i,mask].astype(np.float64)
        if len(t)<2 or not np.all(np.diff(t)>0):raise ValueError(f'Invalid timestamps: {split} {i}')
        # Relative time avoids floating-point precision loss for Unix seconds.
        relative=t-t[0];g=np.linspace(relative[0],relative[-1],seq_len);times[i]=t[0]+g
        raw[i]=_resample(relative,d['base__C0'][i,mask,:7],g)
        target[i]=_resample(relative,a['self'][i,mask],g)
        v=a['valid'][i,mask]
        f0[i],valid[i]=_field_on_grid(relative,a['field'][i,mask],v,g)
        f1[i],v1=_field_on_grid(relative,b['field'][i,mask],v,g)
        rr,vr=_field_on_grid(relative,a['rho'][i,mask,None],v,g)
        if not(np.array_equal(valid[i],v1) and np.array_equal(valid[i],vr)):raise ValueError('Unpaired support')
        rho[i]=rr[:,0]
        if times[i,0]<t[0] or times[i,-1]>t[-1]:raise ValueError('Time extrapolation')
    out={k:d[k] for k in ('y','mmsi','region','window_id','noise_type','observation_noise')}
    out.update(raw=raw,target=target,f0=f0,f1=f1,valid=valid,rho=rho,time_s=times,original_length=lengths)
    return out

def fit(d,supervised=False):
    neg=np.ones(len(d['y']),dtype=bool) if supervised else d['y']==0
    target=np.concatenate([d['raw'][neg].reshape(-1,7),d['target'][neg].reshape(-1,7)])
    v=d['valid'] & neg[:,None]
    field=np.concatenate([d['f0'][v],d['f1'][v]])
    if len(field)==0:raise ValueError('No supported training observations')
    def stats(x):
        m=x.mean(0);s=x.std(0);return m,np.where(s<1e-7,1,s)
    sm,ss=stats(target);fm,fs=stats(field)
    return dict(sm=sm,ss=ss,fm=fm,fs=fs)

def features(d,norm,arm):
    if arm not in ('raw','gate','ordinary','reliable'):raise ValueError(arm)
    key='raw' if arm=='raw' else 'target';target=(d[key]-norm['sm'])/norm['ss']
    field=np.zeros((*target.shape[:2],6),np.float32)
    if arm in ('ordinary','reliable'):
        key='f0' if arm=='ordinary' else 'f1'
        field[:,:,:4]=np.where(d['valid'][...,None],(d[key]-norm['fm'])/norm['fs'],0)
        field[:,:,4]=d['rho'];field[:,:,5]=d['valid']
    x=np.concatenate([target,field],axis=-1).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite features')
    return x
