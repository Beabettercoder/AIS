import sys,numpy as np,torch,json
from pathlib import Path
sys.path.insert(0,'/mnt/sda/WangZijia/AIS/outputs/target_heading_20261031')
import train_target as t
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/target_heading_20261031')
tr=t.load('train');va=t.load('val');mu,sd=t.fit(tr)
assert not set(tr['source'].astype(str))&set(va['source'].astype(str))
assert not set(tr['mmsi'].astype(str))&set(va['mmsi'].astype(str))
summary={}
for split,d in [('train',tr),('val',va)]:
 unique,inv,counts=np.unique(d['source'].astype(str),return_inverse=True,return_counts=True)
 assert np.all(counts==4)
 data=d['target_heading_feature']
 assert all(np.array_equal(data[inv==j],np.broadcast_to(data[inv==j][0],data[inv==j].shape)) for j in range(len(unique)))
 for arm,old in [('C0_target','C0'),('F0_target','F0_heading'),('F1_no_target','F1_heading'),('F1_target','F1_heading')]:
  for view in t.VIEWS:
   x,m,time=t.prep(d,view,arm,mu,sd,'cpu');a,b,c=t.h.prep(d,view,old,mu[:40],sd[:40],'cpu')
   assert torch.equal(x[...,:40],a) and torch.equal(m,b) and torch.equal(time,c)
   if arm=='F1_no_target':assert x[...,40:].abs().sum()==0
 source_mask=d['mask'];valid=d['target_heading_feature'][...,2].astype(bool)
 summary[split]={'sources':len(unique),'mmsi':len(np.unique(d['mmsi'])),'valid_heading_steps':int(valid.sum()),'total_steps':int(source_mask.sum()),'source_fourcase_heading_identical':True}
print(json.dumps(summary,indent=2));(OUT/'input_audit.json').write_text(json.dumps(summary,indent=2))
