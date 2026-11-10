import numpy as np
from radius_core import choose,NM
def check(p,w,vvalid=None):
 p=np.asarray(p,float).reshape(-1,2)*NM;n=len(p);v=np.tile([2.,1.],(n,1));valid=np.ones(n,bool) if vvalid is None else vvalid
 return choose(p,v,v,np.asarray(w,float),valid,[str(i) for i in range(n)],'target')
def main():
 p=[[1,0],[-1,0],[0,1],[0,-1],[4,0],[0,6]];a,d=check(p,[1,1,1,.1,1,1]);
 assert a['tight_plain']['mode']==2 and a['tight_plain']['count']==4
 assert a['adaptive_credible_fallback']['radius_nm']==5 and a['adaptive_credible_fallback']['count']==4
 assert a['adaptive_plain_fallback']['radius_nm']==3.5 and a['adaptive_plain_deg6']['mode']==1
 assert a['adaptive_plain_fallback']['rho']==1 and d['plain_count_8nm']==6 and d['credible_count_8nm']==5
 b,_=check(p,[.01]*6,np.zeros(6,bool));assert b['tight_plain']['mode']==2 and b['wide_plain']['mode']==2
 assert b['adaptive_credible_fallback']['mode']==0
 for arm in ('target_only','tight_plain','wide_plain','adaptive_plain_nofallback','adaptive_plain_fallback','adaptive_plain_deg6'):
  assert a[arm]['mode']==b[arm]['mode'] and a[arm]['radius_nm']==b[arm]['radius_nm'] and a[arm]['count']==b[arm]['count'] and a[arm]['rho']==b[arm]['rho']
  np.testing.assert_array_equal(a[arm]['bg'],b[arm]['bg']);np.testing.assert_array_equal(a[arm]['grad'],b[arm]['grad'])
 c,_=check([[1,0],[-1,0],[0,1]],[1,1,1]);assert c['adaptive_plain_nofallback']['mode']==0 and c['adaptive_plain_fallback']['mode']==1
 try:choose([[1,0]],[[1,0]],[[1,0]],[1],[True],['target'],'target')
 except ValueError:pass
 else:raise AssertionError('self-neighbor not excluded')
 print('PASS: ordinary confidence-invariance, credibility screen, degree6, fallback, self-exclusion')
if __name__=='__main__':main()
