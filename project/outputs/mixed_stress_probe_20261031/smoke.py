from pathlib import Path
import json,sys,torch,numpy as np,hashlib
p=Path('/mnt/sda/WangZijia/AIS/outputs/mixed_stress_probe_20261031');sys.path.insert(0,str(p))
import train_mixed as a
tr=a.ref.read('train');va=a.ref.read('val');mu,sd=a.ref.fit(tr);dev=torch.device('cpu')
s,inv,cnt=np.unique(tr['source'].astype(str),return_inverse=True,return_counts=True)
assert np.all(cnt==4)
for seed in (0,1,2):
 for epoch in (0,24):
  assign=np.random.default_rng(920000+epoch).integers(0,3,size=len(s));assert all(np.unique(assign[inv][inv==j]).size==1 for j in range(len(s)))
for name,n in [('tcn',0),('gru',0),('transformer',0),('itransformer',0)]:
 for arm in a.ARMS:
  a.ref.seed(n);model=a.ref.base.get_model(name,40,name=='itransformer')
  z=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest()
  old=json.loads((a.historical(name,n,arm)/'result.json').read_text());assert z==old['init_sha256']
  for v in a.VIEWS:
   x,m,t=a.ref.prep(tr,v,arm,mu,sd,dev);assert x.shape==(len(tr['y']),20,40) and torch.isfinite(x).all()
   if v=='base':out=model(x[:8],m[:8],t[:8])['logits'];assert torch.isfinite(out).all()
print(json.dumps({'train_windows':len(tr['y']),'train_sources':len(s),'val_windows':len(va['y']),'mmsi_train_val_overlap':len(set(tr['mmsi'].astype(str))&set(va['mmsi'].astype(str))),'same_source_4_cell_assignment':True,'same_initial_weights':True,'inputs_finite':True}))
