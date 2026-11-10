import numpy as np
from sklearn.metrics import average_precision_score
rng=np.random.default_rng(20261101)
maxerr=0
for trial in range(30):
 n=100;y=rng.integers(0,2,n);s=np.round(rng.random(n),1);g=rng.integers(0,12,n);draw=rng.multinomial(12,np.ones(12)/12,size=5)
 order=np.argsort(-s,kind='stable');v=s[order];ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),n-1]
 w=draw[:,g[order]].T.astype(float);tp=np.cumsum(w*y[order,None],axis=0)[ends];tot=np.cumsum(w,axis=0)[ends];inc=np.diff(np.vstack([np.zeros((1,5)),tp]),axis=0);precision=np.divide(tp,tot,out=np.zeros_like(tp),where=tot>0);got=np.divide((precision*inc).sum(0),tp[-1],out=np.full(5,np.nan),where=tp[-1]>0)
 exp=[average_precision_score(y,s,sample_weight=d[g]) for d in draw]
 maxerr=max(maxerr,float(np.max(np.abs(got-exp))))
assert maxerr<1e-12
print('Weighted cluster bootstrap AP test, 150 random cases including tied scores; max abs error:',maxerr)
