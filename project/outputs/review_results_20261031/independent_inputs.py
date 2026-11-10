import numpy as np,torch,sys,json
from pathlib import Path
root=Path('/mnt/sda/WangZijia/AIS/outputs');sys.path.insert(0,str(root/'target_heading_20261031'));import train_target as t
out={}; data={split:t.load(split) for split in ('train','val')}
assert not set(data['train']['source'].astype(str))&set(data['val']['source'].astype(str));assert not set(data['train']['mmsi'].astype(str))&set(data['val']['mmsi'].astype(str));mu,sd=t.fit(data['train'])
for split,d in data.items():
 with np.load(root/'heading_coverage_20261030'/f'heading_{split}.npz',allow_pickle=False) as z:
  key=z['window_id'].astype(str);mapping={v:i for i,v in enumerate(key)};ix=np.array([mapping[str(s)] for s in d['source']]);np.testing.assert_array_equal(z['target_mmsi'][ix].astype(str),d['mmsi'].astype(str));np.testing.assert_array_equal(z['target_t'][ix],d['time_s']);heading=z['target_heading'][ix].astype(float);valid=z['target_valid'][ix]&d['mask']
 q=d['target_heading_feature']; active=d['target_heading_active'];h=np.radians(np.where(valid,heading,0));cog=np.arctan2(d['causal7'][...,5],d['causal7'][...,6]);delta=h-cog;expected_sin=np.where(valid,np.sin(delta),0);expected_cos=np.where(valid,np.cos(delta),0);np.testing.assert_allclose(q[...,0],expected_sin,atol=1e-6);np.testing.assert_allclose(q[...,1],expected_cos,atol=1e-6);np.testing.assert_array_equal(q[...,2].astype(bool),valid)
 time=d['time_s'];diff=time[:,1:]-time[:,:-1];rv=valid[:,1:]&valid[:,:-1]&(diff>0)&(diff<=300);angle=(heading[:,1:]-heading[:,:-1]+180)%360-180;rate=np.zeros_like(heading);np.divide(angle,diff,out=rate[:,1:],where=rv);np.testing.assert_allclose(q[...,3],rate,atol=1e-6);np.testing.assert_array_equal(q[:,1:,4].astype(bool),rv);assert np.all(q[~active]==0)
 ids=d['source'].astype(str);u,inv,count=np.unique(ids,return_inverse=True,return_counts=True);assert np.all(count==4)
 for i in range(len(u)):
  x=q[inv==i];np.testing.assert_array_equal(x,np.broadcast_to(x[:1],x.shape))
 for view in t.VIEWS:
  x,m,tm=t.prep(d,view,'F1_target',mu,sd,'cpu');base,bm,bt=t.h.prep(d,view,'F1_heading',mu[:40],sd[:40],'cpu');assert torch.equal(x[...,:40],base) and torch.equal(m,bm) and torch.equal(tm,bt)
  off,_,_=t.prep(d,view,'F1_no_target',mu,sd,'cpu');assert torch.equal(off[...,:40],base) and torch.count_nonzero(off[...,40:]).item()==0
 out[split]={'rows':len(ids),'sources':len(u),'mmsi':len(set(d['mmsi'].astype(str))),'legal_heading_steps':int(valid.sum()),'all_steps':int(d['mask'].sum())};print(split,out[split],flush=True)
(root/'review_results_20261031'/'independent_inputs.json').write_text(json.dumps(out,indent=2));print('TARGET INPUTS OK')

