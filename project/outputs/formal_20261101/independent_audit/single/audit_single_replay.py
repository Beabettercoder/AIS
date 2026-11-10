"""Independent GPU replay of fixed single-vessel test predictions."""
from pathlib import Path
import sys,json,hashlib,os
import numpy as np,torch
R=Path('/mnt/sda/WangZijia/AIS/outputs');P=R/'formal_20261101/single';O=P.parent/'independent_audit/single';root=R/'framework_validation_v1_20261028';prep=R/'eight_backbone_preparation_20261030';origin=R/'original_mainline_pilot_20261029';names=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];arms=['raw','causal_bundle','f0_bundle','f1_bundle']
sys.path[:0]=[str(P),str(origin),str(root)]
from eight_backbone_adapter import Model8
from pilot_features import build_field_inputs

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (P/'frozen_checkpoint_thresholds.json').exists() and (P/'single_formal_summary.json').exists();dev=torch.device('cuda:'+os.environ.get('SINGLE_AUDIT_GPU','7'));torch.set_num_threads(2)
with np.load(root/'data/individual/test.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
with np.load(root/'extra_data/field_test.npz',allow_pickle=False) as z:e={k:z[k] for k in z.files}
with np.load(prep/'rho_test.npz',allow_pickle=False) as z:rho=z['rho']
f0=build_field_inputs(d,e,'F0');f1=build_field_inputs(d,e,'F1');assert np.array_equal(f0['valid'],f1['valid']) and np.array_equal(rho>0,f0['valid']);norm=json.loads((P/'frozen_train_protocol.json').read_text())['normalizer_arrays'];raw={}
for key,value in [('raw',d['base__C0'][...,:7]),('self',f0['self'])]:
 x=((value-np.array(norm['sm']))/np.array(norm['ss'])).astype(np.float32);x[~d['mask']]=0;raw[key]=torch.tensor(x,device=dev)
for key,val in [('f0',f0['field']),('f1',f1['field'])]:
 x=((val-np.array(norm['fm']))/np.array(norm['fs'])).astype(np.float32);x[~f0['valid']]=0;raw[key]=torch.tensor(x,device=dev)
t=d['time_s'].copy();t-=t[:,:1];t[~d['mask']]=0
for k,v in [('times',t.astype(np.float32)),('mask',d['mask']),('rho',rho)]:raw[k]=torch.tensor(v,device=dev)
report={'device':str(dev)+', same architecture type as training/evaluation','models':{},'max_abs_diff':0.,'bitwise_equal_models':0}
for n in names:
 for seed in (0,1,2):
  folder=P/'runs'/n/f'seed{seed}'
  for arm in arms:
   model=Model8(n).to(dev);model.load_state_dict(torch.load(folder/arm/'final.pt',map_location=dev,weights_only=True));model.eval();key='raw' if arm=='raw' else 'self';fld='f0' if arm=='f0_bundle' else 'f1';has=arm.startswith('f');out=[]
   with torch.no_grad():
    for i in range(0,len(d['y']),256):
     z=slice(i,i+256);p=model(raw[key][z],raw['mask'][z],raw['times'][z],raw[fld][z],raw['rho'][z],has)[0];out.extend(torch.sigmoid(p).cpu().numpy())
   pred=np.asarray(out)
   with np.load(folder/'test'/f'{arm}_scores.npz',allow_pickle=False) as z:ref=z['score']
   err=float(np.max(np.abs(pred-ref)));same=bool(np.array_equal(pred,ref));report['max_abs_diff']=max(report['max_abs_diff'],err);report['bitwise_equal_models']+=int(same);report['models'][f'{n}/{seed}/{arm}']={'max_abs_diff':err,'bitwise_equal':same}
   if err>1e-5:raise ValueError('replay mismatch '+n+'/'+arm+' '+str(err))
  print('REPLAY',n,seed,flush=True)
assert len(report['models'])==96
p=O/'independent_single_replay_audit.json';assert not p.exists();p.write_text(json.dumps(report,indent=2));print(json.dumps({'checked':len(report['models']),'bitwise_equal':report['bitwise_equal_models'],'max_abs_diff':report['max_abs_diff']}))
