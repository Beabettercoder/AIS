import numpy as np
from radius_core import choose,NM
p=np.array([[0,500],[500,0],[-500,0],[0,-500],[500,500],[4*NM,0]],float)
v=np.array([[2,0],[2.1,0],[2,-.1],[1.9,0],[2,.1],[2.2,0]],float);w=np.ones(6);ids=np.array([str(i) for i in range(6)]);k=np.ones(6,bool)
a,_=choose(p,v,v,w,k,ids,'target')
assert a['tight_plain']['mode']==0
assert a['wide_plain']['mode']==2 and a['wide_plain']['count']==6
assert a['adaptive_plain_fallback']['mode']==2 and a['adaptive_plain_fallback']['radius_nm']==5
b,_=choose(p[:5],v[:5],v[:5],w[:5],k[:5],ids[:5],'target')
assert b['wide_plain']['mode']==0 and b['adaptive_plain_fallback']['mode']==1
c,_=choose(p,v,v,np.full(6,.01),np.zeros(6,bool),ids,'target')
for arm in ('tight_plain','wide_plain','adaptive_plain_nofallback','adaptive_plain_fallback'):
 assert a[arm]['mode']==c[arm]['mode'] and np.array_equal(a[arm]['bg'],c[arm]['bg']) and a[arm]['rho']==c[arm]['rho']
assert c['adaptive_credible_fallback']['mode']==0
print('PASS six-neighbor expansion, zero-order fallback, ordinary reliability invariance')
