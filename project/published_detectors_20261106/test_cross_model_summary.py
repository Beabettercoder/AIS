import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from summarize_cross_model_matched import auc_ci
from summarize_eight_matched import paired_bootstrap

y=np.array([0,1,0,1,0,1,0,1]);m=np.array(['a','a','b','b','c','c','d','d'])
a=np.array([.2,.8,.2,.7,.5,.6,.3,.9]);b=np.array([.2,.4,.5,.5,.4,.6,.1,.3]);aa=[a,a,a];bb=[b,b,b]
n=200;seed=123;ap=paired_bootstrap(y,aa,bb,m,n,seed);auc,_=auc_ci(y,aa,bb,m,n,seed)
u,inv=np.unique(m,return_inverse=True);rng=np.random.default_rng(seed);da=[];du=[]
for _ in range(n):
 counts=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u));w=counts[inv]
 da.append(average_precision_score(y,a,sample_weight=w)-average_precision_score(y,b,sample_weight=w))
 du.append(roc_auc_score(y,a,sample_weight=w)-roc_auc_score(y,b,sample_weight=w))
assert np.allclose([ap['lower95'],ap['upper95']],np.quantile(da,[.025,.975]))
assert np.allclose(auc,np.quantile(du,[.025,.975]))
print('PASS paired AP/AUROC cluster bootstrap including tied scores')
