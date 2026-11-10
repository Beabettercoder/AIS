import numpy as np,json
from pathlib import Path
p=Path('/mnt/sda/WangZijia/AIS/outputs/relation_repair_20261030/val_relation_field.npz')
with np.load(p) as z:d={k:z[k] for k in z.files}
mask=np.arange(20)[None,:]<d['length'][:,None]
out={}
for view in ('base','natural','suffix','combined_l100_m600'):
 f=d[view+'__field'];mode=f[:,0,:,7]; count=d[view+'__count'];valid=mode>0
 assert np.array_equal(mode,f[:,1,:,7]); assert np.all(mode[~mask]==0)
 result={'valid_steps':int(valid.sum()),'total_steps':int(mask.sum()),'no_neighbor_steps':int(((count==0)&mask).sum()),'has_neighbor_but_no_field_steps':int(((count>0)&mask&~valid).sum()),'fallback_steps':int((mode==1).sum()),'affine_steps':int((mode==2).sum()),'sources':int(len(d['source'])//4),'sources_with_both_half':0}
 sources=d['source'].reshape(-1,4);assert (sources==sources[:,:1]).all()
 halfok=0
 for i in range(0,len(d['y']),4):
  n=d['length'][i];v=valid[i,:n];halfok+=bool(v[:n//2].any() and v[n//2:].any())
 result['sources_with_both_half']=halfok
 out[view]=result
print(json.dumps(out,indent=2))
Path('support_audit.json').write_text(json.dumps(out,indent=2))
