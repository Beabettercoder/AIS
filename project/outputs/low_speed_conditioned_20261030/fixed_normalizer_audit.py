from pathlib import Path
import importlib.util,json
import numpy as np,torch
P=Path('/mnt/sda/WangZijia/AIS/outputs')
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=module(P/'heading_alignment_20261030/train_heading.py','original_heading')
b=module(P/'low_speed_history_floor_20261030/fixed_normalizer/train_fixed.py','candidate_heading')
train=a.read('train'); mu,sd=a.fit(train); data={}; device=torch.device('cpu')
for split in ['train','val']:
 old=a.read(split);new=b.read(split); data[split]={}
 for view in a.VIEWS:
  data[split][view]={}
  for arm in ['C0','F0']:
   x0,m0,t0=a.prep(old,view,arm,mu,sd,device);x1,m1,t1=b.prep(new,view,arm,mu,sd,device)
   equal=bool(torch.equal(x0,x1) and torch.equal(m0,m1) and torch.equal(t0,t1));assert equal
   data[split][view][arm]={'exactly_equal':equal,'shape':list(x0.shape)}
out={'scope':'fixed original train/base normalizer; C0/F0 paired input check, train/val only','all_checks':data}
(P/'low_speed_conditioned_20261030/fixed_normalizer_audit.json').write_text(json.dumps(out,indent=2));print('train/val x 4 views x 2 arms: exactly equal')
