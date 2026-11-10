"""Conditional 8-backbone relation expansion: 3 paired arms, 25 epochs.

The held-out set is never read during training. Gate record must be supplied
by the parent only if the user's pilot criteria are met.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')
HEADING = Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030')
sys.path.insert(0, '/mnt/sda/WangZijia/AIS/outputs/eight_backbone_independent_audit_20261030')
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HEADING))
import train_heading as base
base.OLD = Path('/mnt/sda/WangZijia/AIS/outputs/neighbor_gate_safe_20261030')
from relation8 import model_for
from guarded_scale_train import gate

def seed(n):
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True,choices=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'])
    p.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--epochs',type=int,choices=[25,75],required=True)
    a=p.parse_args()
    if a.gpu==4: raise SystemExit('GPU 4 is reserved by an existing service.')
    gate()
    from candidate_integrity import check_candidate,check_gate_candidate
    freeze=check_candidate(('train','val'))
    config=json.loads((HERE/'AUTHORIZED_STAGE2.json').read_text(encoding='utf8'))
    check_gate_candidate(config,freeze)
    registered=config.get('relation_epochs_by_backbone',{}).get(a.name,25)
    if a.epochs != registered or (a.epochs==75 and a.name!='itransformer'):
        raise SystemExit('Epoch budget is not registered for this relation backbone.')
    base.VARIANT='aligned_embed'
    torch.set_num_threads(2); dev=torch.device(f'cuda:{a.gpu}')
    tr,va=base.read('train'),base.read('val'); mu,sd=base.fit(tr)
    dest=Path('/mnt/sda/WangZijia/AIS/outputs/neighbor_gate_safe_20261030')/'formal_runs'/a.name/f'seed{a.seed}'
    from safe_single_launcher import reserve
    reserve(dest)
    spec={'backbone':a.name,'seed':a.seed,'epochs':a.epochs,'arms':['C0','F0','F1'],'loss':'BCE',
          'lr':0.001,'optimizer':'AdamW with no-decay SSM parameters', 'batch':128,'train_view':'base',
          'checkpoint':f'fixed epoch{a.epochs}', 'batch_order':'300000+epoch, same schedule across seeds','variant':'aligned_embed','normalization':'train/base only',
          'graph':'custom target/reference two-node graph; 8 target channels, support slot22; not original full-vessel GAT',
          'test':'not read or evaluated in training','means':mu.tolist(),'stds':sd.tolist(),
          'code_sha256':{str(x):sha(x) for x in [Path(__file__),HERE/'relation8.py',HEADING/'train_heading.py',base.OLD/'train.py',base.OLD/'frozen_protocol.json',base.OLD/'builder.py',base.PRE/'models.py',base.OLD/'train_relation_field.npz',base.OLD/'val_relation_field.npz']}}
    (dest/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    for arm in ('C0','F0','F1'):
        folder=dest/arm
        if (folder/'final.pt').exists(): raise FileExistsError(f'Refusing to overwrite existing {folder}')
        folder.mkdir(exist_ok=True)
        seed(a.seed); model=model_for(a.name).to(dev)
        initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest()
        nd=[p for p in model.parameters() if getattr(p,'_no_weight_decay',False)]
        dc=[p for p in model.parameters() if not getattr(p,'_no_weight_decay',False)]
        opt=torch.optim.AdamW([{'params':dc,'weight_decay':.01}]+([{'params':nd,'weight_decay':0.}] if nd else []),lr=.001)
        x,m,t=base.prep(tr,'base',arm,mu,sd,dev); y=torch.tensor(tr['y'],dtype=torch.float32,device=dev)
        history=[]
        for epoch in range(a.epochs):
            model.train(); order=np.random.default_rng(300000+epoch).permutation(len(y));losses=[]
            for i in range(0,len(y),128):
                ix=torch.as_tensor(order[i:i+128],device=dev); opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(model(x[ix],m[ix],t[ix])['logits'],y[ix])
                if not torch.isfinite(loss): raise ValueError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step();losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)))
            if epoch%5==0: print('EPOCH',a.name,arm,epoch+1,history[-1],flush=True)
        model.eval(); results={}; saved={}
        with torch.no_grad():
            for view in base.VIEWS:
                xv,mv,tv=base.prep(va,view,arm,mu,sd,dev); pred=[]
                for i in range(0,len(xv),256): pred.extend(torch.sigmoid(model(xv[i:i+256],mv[i:i+256],tv[i:i+256])['logits']).cpu().numpy())
                pred=np.asarray(pred);results[view]=base.metrics(va['y'],pred,va['region']);saved[view+'_score']=pred
        np.savez_compressed(folder/'val_scores.npz',y=va['y'],mmsi=va['mmsi'],region=va['region'],source=va['source'],case=va['case'],**saved)
        torch.save(model.state_dict(),folder/'final.pt')
        result={'arm':arm,'seed':a.seed,'init_sha256':initial,'last_epoch':a.epochs,'loss_history':history,'views':results,
                'prediction_sha256':sha(folder/'val_scores.npz'),'weights_sha256':sha(folder/'final.pt')}
        (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print('DONE',a.name,arm,results,flush=True)

if __name__=='__main__': main()


