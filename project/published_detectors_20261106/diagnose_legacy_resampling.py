"""Forward-only contrast: historical frozen weights on original vs resampled windows.

No model training or labels used to configure the transformation. Keeps old
normalizer and learned parameters, and interpolates historical support weights
only over consecutive supported original steps. Reports pooled AP.
"""
import sys,json
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028');PREP=Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')
sys.path.insert(0,str(P));import train_single as exp
sys.path.insert(0,'/mnt/sda/WangZijia/AIS/published_detectors_20261106');import data_resampled as dnew
torch.set_num_threads(2)
with np.load(ROOT/'data/individual/test.npz',allow_pickle=False) as z:orig={k:z[k] for k in z.files}
with np.load(PREP/'rho_test.npz',allow_pickle=False) as z:oldrho=z['rho']
new=dnew.load('test');rho=np.zeros_like(new['rho'])
for i in range(len(rho)):
 mask=orig['mask'][i];t=orig['time_s'][i,mask];rel=t-t[0];grid=new['time_s'][i]-t[0]
 v=oldrho[i,mask]>0; res,valid=dnew._field_on_grid(rel,oldrho[i,mask,None],v,grid)
 if not np.array_equal(valid,new['valid'][i]):raise ValueError('support mismatch')
 rho[i]=res[:,0]
times=(new['time_s']-new['time_s'][:,:1]).astype(np.float32)
payload=dict(raw=new['raw'],self=new['target'],f0=new['f0'],f1=new['f1'],rho=rho,mask=np.ones_like(new['valid']),valid=new['valid'],times=times,y=new['y'],yc=np.zeros(len(rho)),state0=np.zeros((*rho.shape,4)),state1=np.zeros((*rho.shape,4)))
norm=exp.fit(exp.load('train'));dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');t=exp.convert(payload,norm,dev);out=[]
for name in ['tcn','gru','transformer']:
 for seed in range(3):
  for arm in ['raw','causal_bundle','f1_bundle']:
   model=exp.Model(name).to(dev);path=P/'runs'/name/f'seed{seed}'/arm/'final.pt';model.load_state_dict(torch.load(path,map_location=dev,weights_only=True));model.eval();scores=[]
   with torch.no_grad():
    for j in range(0,len(rho),256):
     s=slice(j,j+256);log,*_=model(t['raw' if arm=='raw' else 'self'][s],t['mask'][s],t['times'][s],t['f1'][s],t['rho'][s],arm=='f1_bundle');scores.extend(torch.sigmoid(log).cpu().numpy())
   with np.load(P/'runs'/name/f'seed{seed}'/'test'/f'{arm}_scores.npz',allow_pickle=False) as z:
    assert np.array_equal(z['window_id'],orig['window_id']);old=average_precision_score(z['y'],z['score'])
   row={'model':name,'seed':seed,'arm':arm,'old_AP':float(old),'resampled_AP':float(average_precision_score(new['y'],scores))};out.append(row);print(json.dumps(row),flush=True)
Path('/home/ps/AIS_published_detectors_20261107/legacy_resampling_diagnostic.json').write_text(json.dumps(out,indent=2))
