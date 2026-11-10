"""Single pre-specified causal position re-anchoring candidate; velocity unchanged."""
from __future__ import annotations
import numpy as np

def reanchor_positions(t, observed_p, old_gate_relative_p, w, valid, threshold=.7):
    """Recover when interval compatibility returns. Never accesses future/reference positions.

    A high weight is an operational reset condition, NOT verified absolute GPS truth.
    Low/medium-weight steps keep the original increment and its original branch logic.
    """
    n=len(t)
    assert observed_p.shape==(n,2) and old_gate_relative_p.shape==(n,2)
    candidate=np.zeros_like(old_gate_relative_p,dtype=float);reset=np.zeros(n,bool)
    for k in range(1,n):
        if bool(valid[k]) and float(w[k])>=threshold:
            candidate[k]=observed_p[k]-observed_p[0];reset[k]=True
        else:
            step=old_gate_relative_p[k]-old_gate_relative_p[k-1]
            candidate[k]=candidate[k-1]+step
    return candidate,reset

def observed_history(features,t,p,sog,cog,window_id,view):
    """Reproduce the frozen pilot's visible target history, without consulting its future."""
    p=p.copy();sog=sog.copy()
    rng=features.seeded(window_id+'/target');angle=rng.uniform(0,2*np.pi)
    direction=np.array([np.sin(angle),np.cos(angle)])
    if view=='target_jump':p[-1]+=1000*direction
    elif view=='target_drift':p[-4:]+=np.linspace(0,600,4)[:,None]*direction
    elif view=='target_speed':sog[-2:]=np.minimum(sog[-2:]*4,102.1)
    return p,features.vectors(sog,cog)
