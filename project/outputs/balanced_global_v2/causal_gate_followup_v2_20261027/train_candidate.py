"""Train the locked development candidate with the exact prior pure-BCE budget."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=ROOT/'causal_gate_followup_v2_20261027';BASE=ROOT/'causal_gate_main_v1_20261027'
sys.path.insert(0,str(BASE));import train_controls as old
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--backbone',choices=['tcn','gru'],required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 selection=json.loads((OUT/'development_selection.json').read_text(encoding='utf8'));method=selection['candidate'];assert selection['status']=='locked_before_test'
 d,proofs=old.build_data();parts={}
 for split in ('train','val'):
  xs=[];ids=[]
  for r in old.th.REGIONS:
   p=OUT/'inputs'/r/(split+'.npz')
   with np.load(p,allow_pickle=False) as z:xs.append(z[method].copy());ids.extend(z['window_id'].astype(str))
  assert np.array_equal(np.array(ids),d[split]['window_id'].astype(str));v=d[split]['features']['causal'].copy();v[...,:7]=np.concatenate(xs);d[split]['features']['candidate']=v
 old.hm.ARMS['candidate']=dict(key='candidate',nf=12,graph=False,recipe=False,masked=False)
 stats=old.th.fit_normalizer(d['train']['features']['candidate'],d['train']['mask'])
 worker=OUT/'runs'/args.backbone;worker.mkdir(parents=True,exist_ok=False)
 conf={'backbone':args.backbone,'method':method,'protocol_sha256':sha(OUT/'protocol.json'),'selection_sha256':sha(OUT/'development_selection.json'),'script_sha256':sha(Path(__file__)),'source_proofs':proofs,'seeds':[0,1,2],'loss':'BCE only'}
 (worker/'configuration.json').write_text(json.dumps(conf,ensure_ascii=False,indent=2),encoding='utf8');configsha=sha(worker/'configuration.json')
 training={'epochs':25,'batch_size':128,'learning_rate':.002,'weight_decay':.01,'policy':'all'};device=torch.device(args.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('no CUDA')
 for seed in (0,1,2):old.th.train_one(args.backbone,'candidate',seed,d,stats,training,device,worker/f'seed_{seed}'/'candidate',sha(OUT/'protocol.json'),configsha)
 print('TRAIN_DONE',args.backbone,flush=True)
if __name__=='__main__':main()
