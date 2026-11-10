import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
import pandas as pd
import json
from scipy.optimize import milp,LinearConstraint,Bounds
from scipy.sparse import coo_matrix
root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
ds={p.stem:pd.read_parquet(p) for p in (root/'candidates').glob('*.parquet')}
counts={r:c[c.duration_s>=660].groupby('mmsi').size().clip(upper=20).sort_index() for r,c in ds.items()}
ids=np.array(sorted(set().union(*[set(v.index) for v in counts.values()])))
ix={int(m):i for i,m in enumerate(ids)}; n=len(ids); probs=np.array([.7,.15,.15])
row=[];col=[];dat=[];lb=[];ub=[]
def add(indices,values,low,high):
 k=len(lb);row.extend([k]*len(indices));col.extend(indices);dat.extend(values);lb.append(low);ub.append(high)
for i in range(n):add([i*3,i*3+1,i*3+2],[1,1,1],1,1)
ideal=n*probs;g=np.floor(ideal).astype(int)
for s in np.argsort(-(ideal-g),kind='stable')[:n-int(g.sum())]:g[s]+=1
for s in range(3):add([i*3+s for i in range(n)],[1]*n,g[s],g[s])
for r,v in counts.items():
 # Capacity rank strata; tie breaking independent of anomaly labels/predictions.
 stratum=pd.qcut(v.rank(method='first'),q=4,labels=False)
 for q in range(4):
  ms=v.index[stratum==q];k=len(ms)
  for s,p in enumerate(probs):
   add([ix[int(m)]*3+s for m in ms],[1]*k,np.floor(k*p),np.ceil(k*p))
 # Balance aggregate per-ship capped capacity, within one maximum ship contribution.
 for s,p in enumerate(probs):
  add([ix[int(m)]*3+s for m in v.index],v.to_list(),max(0,v.sum()*p-20),v.sum()*p+20)
A=coo_matrix((dat,(row,col)),shape=(len(lb),n*3)).tocsc()
r=milp(c=np.random.default_rng(20261025).uniform(0,1,n*3),integrality=np.ones(n*3),bounds=Bounds(0,1),constraints=LinearConstraint(A,lb,ub),options={'time_limit':25,'mip_rel_gap':.1,'presolve':True})
print('SOLVER',r.message,flush=True)
if r.x is None:raise SystemExit('No feasible capacity-stratified assignment within 25s')
Y=r.x.reshape(n,3);assert np.allclose(Y,np.rint(Y),atol=1e-5)
Y=np.rint(Y).astype(int);z=A@Y.ravel();assert np.all(z>=np.array(lb)-1e-6) and np.all(z<=np.array(ub)+1e-6)
report={'assignment':'metadata-only four capacity strata per region, global MMSI grouping, global exact70/15/15 rounded counts, seed20261025 random objective; no selected files changed','all_unique_ship_counts':Y.sum(0).tolist(),'regions':{}}
for r,v in counts.items():
 rec={}
 for k_total,shipqs in [(270,[189,40,41]),(300,[210,45,45])]:
  rr={}
  for s,(label,k,w) in enumerate(zip(['train','val','test'],shipqs,[1890,405,405])):
   vs=v[[Y[ix[int(m)],s]==1 for m in v.index]]
   available=int(vs.nlargest(k).sum())
   rr[label]={'candidate_ships':len(vs),'required_ships':k,'max_windows_for_quota_cap20':available,'required_windows':w,'feasible':len(vs)>=k and available>=w}
  rec[str(k_total)]={'all_splits_feasible':all(x['feasible'] for x in rr.values()),'splits':rr}
 report['regions'][r]=rec
print(json.dumps(report,indent=2),flush=True)
