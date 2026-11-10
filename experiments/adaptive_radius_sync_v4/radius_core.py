"""Causal candidate fits with truly unfiltered ordinary and credible modes."""
import numpy as np
NM=1852.;RADII=(3.5,5.,8.);COND=1e8;LAMBDA=.001
ARMS=('target_only','tight_plain','wide_plain','adaptive_plain_nofallback','adaptive_plain_fallback','adaptive_qualified_fallback','adaptive_credible_fallback')

def solve(pos,vel,weight,r):
    if len(pos)<6 or weight.sum()<=1e-12:return None
    w=weight/weight.sum();X=np.column_stack((np.ones(len(pos)),pos/(r*NM)));gram=X.T@(w[:,None]*X);cond=np.linalg.cond(gram)
    if not np.isfinite(cond) or cond>COND:return None
    try:B=np.linalg.solve(gram+LAMBDA*np.diag([0.,1.,1.]),X.T@(w[:,None]*vel))
    except np.linalg.LinAlgError:return None
    return B if np.isfinite(B).all() else None

def choose(positions,vel_plain,vel_cred,reliability,kinematic_valid,ids,target_id,coverage=None):
    p=np.asarray(positions,float).reshape(-1,2);v0=np.asarray(vel_plain,float).reshape(-1,2);v1=np.asarray(vel_cred,float).reshape(-1,2)
    w=np.asarray(reliability,float).reshape(-1);valid=np.asarray(kinematic_valid,bool).reshape(-1);ids=np.asarray(ids,str).reshape(-1)
    if len({len(p),len(v0),len(v1),len(w),len(valid),len(ids)})!=1 or len(set(ids))!=len(ids) or np.any(ids==str(target_id)):raise ValueError('invalid candidate identity')
    if not all(np.isfinite(x).all() for x in (p,v0,v1,w)) or np.any(w<0):raise ValueError('nonfinite observation')
    d=np.linalg.norm(p,axis=1);inside=d<=8*NM+1e-7;p,v0,v1,w,valid,ids,d=p[inside],v0[inside],v1[inside],w[inside],valid[inside],ids[inside],d[inside]
    coverage=coverage or (lambda _:True)
    def zero():return {'mode':0,'radius_nm':0.,'count':0,'rho':0.,'bg':np.zeros(2),'grad':np.zeros((2,2)),'ids':[]}
    cache={}
    def compute(qualified,weighted,velocity,r,degree=6,fallback=False):
        if not coverage(r):return zero()
        key=(qualified,weighted,velocity is v1,r)
        if key not in cache:
            m=(d<=r*NM+1e-7)&(valid&(w>=.3) if qualified else np.ones(len(d),bool))
            pp,vv,ww,ii,dd=p[m],velocity[m],w[m],ids[m],d[m];weight=np.exp(-.5*(dd/(r*NM/2))**2)
            if weighted:weight=weight*ww
            cache[key]=(vv,ww,ii,weight,solve(pp,vv,weight,r))
        vv,ww,ii,weight,B=cache[key];n=len(vv)
        rho=min(n/4.,1.)*(float(ww.mean()) if qualified and n else 1.)
        base={'radius_nm':r,'count':n,'ids':ii.tolist(),'rho':0.,'mode':0,'bg':np.zeros(2),'grad':np.zeros((2,2))}
        if B is not None and n>=degree:return base|{'mode':2,'rho':rho,'bg':B[0],'grad':B[1:,:].T/(r*NM)}
        if fallback and n and weight.sum()>1e-12:return base|{'mode':1,'rho':rho,'bg':np.average(vv,axis=0,weights=weight),'grad':np.zeros((2,2))}
        return base
    def adaptive(qualified,weighted,velocity,degree=6,fallback=False):
        avail=[r for r in RADII if coverage(r)]
        if not avail:return zero()
        # Find minimum supported affine radius based on the native eligibility
        # and weight policy of this arm, before applying the slope complexity rule.
        r=next((r for r in avail if compute(qualified,weighted,velocity,r,degree=degree)['mode']==2),None)
        return compute(qualified,weighted,velocity,r if r is not None else max(avail),degree,fallback)
    out={'target_only':zero(),'tight_plain':compute(False,False,v0,3.5), 'wide_plain':compute(False,False,v0,8.),
         'adaptive_plain_nofallback':adaptive(False,False,v0),'adaptive_plain_fallback':adaptive(False,False,v0,fallback=True),
         'adaptive_qualified_fallback':adaptive(True,False,v0,fallback=True),'adaptive_credible_fallback':adaptive(True,True,v1,fallback=True)}
    return out,{'plain_count_8nm':len(ids),'credible_count_8nm':int(np.sum(valid&(w>=.3)))}
