from pathlib import Path
import sys,argparse,json,hashlib
import numpy as np
import torch
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');O=ROOT/'formal_20261101/relation';PRE=ROOT/'eight_backbone_preparation_20261030';H=ROOT/'heading_alignment_20261030'
sys.path[:0]=[str(PRE),str(H)]
from relation8 import model_for
import train_heading as base

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--gpu',required=True,type=int,choices=[3,5,6,7]);a=p.parse_args()
 lock=json.loads((O/'heldout.lock.json').read_text());assert lock['test_builder_sha256']==sha(O/'scripts/build_test.py');assert lock['postfreeze_scripts_sha256']['eval.py']==sha(Path(__file__));assert lock['builder_dependencies_sha256']==sha(O/'builder_dependencies.lock.json')
 train=base.read('train');val=base.read('val');base.OLD=O;te=base.read('test');base.VARIANT='aligned_embed'
 assert len(te['y'])==880
 assert not(set(te['mmsi'].astype(str)) & (set(train['mmsi'].astype(str))|set(val['mmsi'].astype(str))))
 dev=torch.device('cuda:'+str(a.gpu));torch.set_num_threads(2)
 d=O/'runs'/a.name/f'seed{a.seed}';spec=json.loads((d/'protocol.json').read_text());mu=np.array(spec['mean']);sd=np.array(spec['std'])
 for arm in ('C0','F0','F1'):
  f=d/arm;record=lock['models'][f'{a.name}/seed{a.seed}/{arm}'];assert sha(f/'final.pt')==record['model_sha256'] and sha(f/'val_scores.npz')==record['val_sha256']
  predfile=f/'test_scores.npz';resultfile=f/'test_result.json'
  if predfile.exists() or resultfile.exists():raise FileExistsError(predfile)
  model=model_for(a.name).to(dev);state=torch.load(f/'final.pt',map_location=dev,weights_only=True);model.load_state_dict(state);model.eval();results={};preds={}
  for view in base.VIEWS:
   x,m,t=base.prep(te,view,arm,mu,sd,dev);p=[]
   with torch.no_grad():
    for i in range(0,len(x),256):p.extend(torch.sigmoid(model(x[i:i+256],m[i:i+256],t[i:i+256])['logits']).cpu().numpy())
   p=np.array(p);res=base.metrics(te['y'],p,te['region']);y=te['y']>0
   for budget,threshold in record['thresholds'].items():
    pos=p>threshold;res['FAR_'+budget]={'threshold':threshold,'actual_FAR':float(pos[~y].mean()),'recall':float(pos[y].mean()),'n_neg':int((~y).sum()),'n_pos':int(y.sum())}
   results[view]=res;preds[view+'_score']=p
  np.savez_compressed(predfile,y=te['y'],mmsi=te['mmsi'],region=te['region'],source=te['source'],case=te['case'],observed_head_sog_kn=np.median(te['raw7'][:,:3,4],axis=1),**preds)
  r={'name':a.name,'seed':a.seed,'arm':arm,'test_note':'prior test sources exposed; backgrounds may overlap; not blind','test_data_sha256':sha(O/'test_relation_field.npz'),'heldout_lock_sha256':sha(O/'heldout.lock.json'),'weights_sha256':sha(f/'final.pt'),'prediction_sha256':sha(predfile),'views':results}
  resultfile.write_text(json.dumps(r,indent=2),encoding='utf8');print('DONE',a.name,a.seed,arm,results,flush=True)
if __name__=='__main__':main()
