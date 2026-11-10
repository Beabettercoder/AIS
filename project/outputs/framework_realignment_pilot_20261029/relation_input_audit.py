"""Independent checks on relationship input fields and controlled quartet labels."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
P=Path('/mnt/sda/WangZijia/AIS/outputs/framework_realignment_pilot_20261029')
report={}
for split in ('train','val'):
 with np.load(P/(split+'_relation_field.npz'),allow_pickle=False) as z: d={k:z[k] for k in z.files}
 y=d['y'];n=len(y); item={'rows':n,'target_causal_paired_equal':bool(np.array_equal(d['causal7'][::4],d['causal7'][1::4]) and np.array_equal(d['causal7'][2::4],d['causal7'][3::4])),'label_pattern_correct':bool(np.array_equal(y.reshape(-1,4),np.tile([0,1,1,0],(n//4,1)))),'views':{}}
 for view in ('base','natural','suffix','combined_l100_m600'):
  f=d[view+'__field'];rho=f[:,0,:,6];valid=rho>0;diff=f[:,0,:,:6]-f[:,1,:,:6]
  simple=np.zeros(n)
  for i in range(n):
   ix=np.flatnonzero(valid[i]);
   if len(ix)>=4:
    bg=f[i,1,:,:2];tar=d['causal7'][i,:,4]*1852/3600; k=min(3,len(ix)//2)
    a=ix[:k];b=ix[-k:];tchg=tar[b].mean()-tar[a].mean();bchg=np.linalg.norm(bg[b],axis=1).mean()-np.linalg.norm(bg[a],axis=1).mean();simple[i]=abs(tchg-bchg)
  item['views'][view]={'finite':bool(np.isfinite(f).all()),'common_rho_equal':bool(np.array_equal(f[:,0,:,6],f[:,1,:,6])),'invalid_field_zero':bool(np.all(f.transpose(0,2,1,3)[~valid]==0)),'supported_steps':int(valid.sum()),'total_steps':int(d['length'].sum()),'mean_abs_f0_minus_f1':float(np.abs(diff[valid]).mean()),'max_abs_f0_minus_f1':float(np.abs(diff[valid]).max()),'mean_support_fraction_by_label':[float(valid[y==c].sum()/d['length'][y==c].sum()) for c in (0,1)],'simple_delta_speed_difference_AP':float(average_precision_score(y,simple))}
 report[split]=item
out=P/'relation_input_audit.json'
with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
