"""Same-family GPU independent replay of all frozen relation test predictions."""
from pathlib import Path
import sys,json,hashlib,time,numpy as np,torch
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'formal_20261101/relation';D=R/'formal_20261101/independent_audit';sys.path[:0]=[str(R/'eight_backbone_preparation_20261030'),str(R/'heading_alignment_20261030')]
from relation8 import model_for
import train_heading as base
assert (O/'heldout.lock.json').is_file() and (O/'summary.json').is_file();lock=json.loads((O/'heldout.lock.json').read_text());base.OLD=O;base.VARIANT='aligned_embed';te=base.read('test');device=torch.device('cuda:7');torch.set_num_threads(2)
views=['base','natural','suffix','combined_l100_m600'];arms=['C0','F0','F1'];names=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];r={};max_abs=0.;exact=0;ct=0
for name in names:
 for seed in range(3):
  d=O/'runs'/name/f'seed{seed}';spec=json.loads((d/'protocol.json').read_text());mu=np.asarray(spec['mean']);sd=np.asarray(spec['std'])
  for arm in arms:
   folder=d/arm;state=torch.load(folder/'final.pt',map_location=device,weights_only=True);m=model_for(name).to(device);m.load_state_dict(state);m.eval()
   with np.load(folder/'test_scores.npz',allow_pickle=False) as z: saved={k:z[k].copy() for k in z.files}
   for view in views:
    x,mask,t=base.prep(te,view,arm,mu,sd,device);p=[]
    with torch.no_grad():
     for i in range(0,len(x),256):p.extend(torch.sigmoid(m(x[i:i+256],mask[i:i+256],t[i:i+256])['logits']).cpu().numpy())
    p=np.asarray(p); ref=saved[view+'_score'];delta=float(np.max(np.abs(p-ref)));max_abs=max(max_abs,delta); exact+=int(np.array_equal(p,ref));ct+=1
    if delta>1e-5:raise ValueError(('model replay mismatch',name,seed,arm,view,delta))
   del m,state
 print('REPLAYED',name,flush=True)
result={'scope':'all 72 fixed models x four test views; device GPU7 of same host; evaluation input/weights as frozen','n_predictions':ct,'bitwise_exact_views':exact,'max_abs_probability_difference':max_abs,'tolerance':1e-5,'not_blind':'test previously exposed; replay does not alter it'}
p=D/'relation_replay_audit.json';assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
