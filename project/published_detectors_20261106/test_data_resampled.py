"""Audit the resampled AIS windows; run with the project's Python environment."""
import numpy as np
import data_resampled as data

t=np.array([0.,1.,2.,3.]);values=np.arange(4.)[:,None];grid=np.arange(0.,3.1,.5)
z,mask=data._field_on_grid(t,values,np.array([1,1,0,1],bool),grid)
assert mask.tolist()==[True,True,True,False,False,False,True]
np.testing.assert_allclose(z[[0,1,2,6],0],[0,.5,1,3])
splits={s:data.load(s) for s in ('train','val','test')}
expected={'train':(7840,980),'val':(1680,210),'test':(1680,210)}
norm=data.fit(splits['train'])
for name,d in splits.items():
    n,per_region=expected[name]
    assert len(d['y'])==n and len(np.unique(d['region']))==8
    assert all((d['region']==r).sum()==per_region for r in np.unique(d['region']))
    assert np.all(np.diff(d['time_s'],axis=1)>0)
    assert not np.any(d['rho'][~d['valid']])
    for arm in ('raw','gate','ordinary','reliable'):
        x=data.features(d,norm,arm)
        assert x.shape==(n,20,13) and np.isfinite(x).all()
    print(name,n,'support fraction',float(d['valid'].mean()),'regions',len(np.unique(d['region'])))
for left,right in (('train','val'),('train','test'),('val','test')):
    assert not(set(splits[left]['mmsi']) & set(splits[right]['mmsi']))
print('MMSI separation, feature shape, gap preservation and time ordering: PASS')
