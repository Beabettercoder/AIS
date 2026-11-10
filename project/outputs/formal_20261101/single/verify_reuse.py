import sys,json,hashlib
from pathlib import Path
import numpy as np,torch
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single')
PREV=Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')
sys.path.insert(0,str(OUT));sys.path.insert(0,str(PREV))
from pilot_train import Model as Old
from eight_backbone_adapter import Model8
import train_single as ex
arms=['raw','causal_bundle','f0_bundle','f1_bundle']
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
res={};torch.set_num_threads(2)
tr=ex.load('train');va=ex.load('val');d=ex.convert(va,ex.fit(tr),torch.device('cuda:1'))
for name in ['tcn','gru','transformer','itransformer']:
 ex.seed(0);a=Old(name);ex.seed(0);b=Model8(name)
 for k,v in a.state_dict().items():assert k in b.state_dict() and torch.equal(v,b.state_dict()[k]),name+' '+k
 assert set(a.state_dict())==set(b.state_dict()); ent={}
 for arm in arms:
  p=PREV/'runs'/name/arm; model=Model8(name).to('cuda:1');model.load_state_dict(torch.load(p/'final.pt',map_location='cuda:1',weights_only=True));model.eval();key='raw' if arm=='raw' else 'self';f='f0' if arm=='f0_bundle' else 'f1';has=arm.startswith('f');scores=[]
  with torch.no_grad():
   for i in range(0,len(va['y']),256):
    z=slice(i,i+256);scores.extend(torch.sigmoid(model(d[key][z],d['mask'][z],d['times'][z],d[f][z],d['rho'][z],has)[0]).cpu().numpy())
  with np.load(p/'final_val.npz') as t:
   assert np.array_equal(t['y'],va['y']) and np.array_equal(t['mmsi'],va['mmsi']) and np.array_equal(t['window_id'],va['ids'])
   diff=float(np.max(np.abs(np.array(scores)-t['score'])))
  j=json.loads((p/'result.json').read_text());assert len(j['history'])==25 and j['seed']==0
  ent[arm]={'old_checkpoint_sha256':sh(p/'final.pt'),'old_predictions_sha256':sh(p/'final_val.npz'),'max_abs_replay_error':diff,'eligible_reuse':diff<1e-5}
  assert diff<1e-5,(name,arm,diff)
 res[name]=ent
 print(name,ent,flush=True)
(OUT/'seed0_reuse_audit.json').write_text(json.dumps(res,indent=2))
