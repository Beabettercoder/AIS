"""Mixed report-weighted, then median member context; observed arrays only."""
from pathlib import Path
import sys,numpy as np,torch
from torch import nn
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');PRIOR=ROOT/'relation_robust_v1_20261027';sys.path.insert(0,str(PRIOR));import run_relation as rr
class Model(nn.Module):
 def __init__(self,arch):
  super().__init__();self.arch=arch;self.body=nn.GRU(7,16,batch_first=True) if arch=='gru' else nn.Sequential(nn.Conv1d(7,16,3,padding=2),nn.ReLU(),nn.Conv1d(16,16,3,padding=2),nn.ReLU());self.out=nn.Linear(16,1)
 def forward(self,x,l):
  if self.arch=='gru':
   h,_=self.body(x);return self.out(h[torch.arange(len(l),device=l.device),l-1]).squeeze(-1)
  h=x.transpose(1,2);h=self.body[1](self.body[0](h)[:,:,:x.shape[1]]);h=self.body[3](self.body[2](h)[:,:,:x.shape[1]]);mask=(torch.arange(x.shape[1],device=x.device)[None,:]<l[:,None]).float();return self.out((h*mask[:,None,:]).sum(2)/l[:,None]).squeeze(-1)
def extract(o,case):
 t=o['target_t'][case];s=o['target_sog_kn'][case];c=o['target_cog_deg'][case];n=len(t);anchor=max(2,n//2-1);pre=np.arange(anchor+1);post=np.arange(anchor+1,n)[-3:];grids=[]
 for lo,hi in zip(o['member_offsets'][:-1],o['member_offsets'][1:]):
  lo=int(lo);hi=int(hi);nt=o['neighbor_t'][case,lo:hi];ns=o['neighbor_sog_kn'][case,lo:hi]
  _,_,wm,_,_=rr.weights(nt,o['neighbor_pos'][case,lo:hi],ns,o['neighbor_cog_deg'][case,lo:hi]);ix=np.searchsorted(nt,t,side='right')-1;safe=np.maximum(ix,0);present=(ix>=0)&(t>=nt[safe])&(t-nt[safe]<=300)
  if np.sum(present[pre])<2 or np.sum(present[post])<2:continue
  raw=np.zeros(n);weights=np.zeros(n);raw[present]=ns[safe[present]];weights[present]=wm[safe[present]];vals=np.full(n,np.nan)
  for k in range(n):
   if not present[k]:continue
   start=0 if k<=anchor else anchor+1;idx=np.arange(start,k+1);idx=idx[present[idx]];w=weights[idx]
   vals[k]=float(np.average(raw[idx],weights=w)) if w.sum()>1e-12 else float(raw[idx].mean())
  grids.append(vals)
 if not grids:raise RuntimeError('no common observed cohort')
 grids=np.asarray(grids);ctx=np.zeros((n,3),np.float32)
 for k in range(n):
  v=grids[:,k];v=v[np.isfinite(v)]
  if len(v):ctx[k,0]=np.median(v)/10;ctx[k,2]=len(v)/8
 ctx[1:,1]=np.diff(ctx[:,0]);a=np.deg2rad(c);target=np.column_stack((s/10,np.sin(a),np.cos(a),np.r_[0,np.diff(s)]/10)).astype(np.float32)
 return np.concatenate((target,ctx),axis=1)
