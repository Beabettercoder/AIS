"""Kinematic-confidence + leave-one-out affine-field extension.

Uses the already fixed encoders, normalizer, initialization and training recipe.
Distinct results live in field_extension/, never replace the median study.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import torch
import train_study as ts
from models import make_model
PARENT=Path(__file__).resolve().parent
OUT=PARENT/'field_extension'
ARMS=['F0','F1','F2']
def setup():
    ts.ROOT=PARENT;parent_lock=ts.verify_lock();ts.verify_data(parent_lock,'individual',['train','val','test'])
    lock=json.loads((OUT/'code_lock.json').read_text())
    for path,h in lock['parent_files'].items():assert ts.sha(PARENT/path)==h,path
    for path,h in lock['field_files'].items():assert ts.sha(PARENT/path)==h,path
    return lock
def combined(split):
    core=ts.load_npz(PARENT/'data'/'individual'/(split+'.npz'))
    ext=ts.load_npz(PARENT/'extra_data'/('field_'+split+'.npz'))
    for key in ['window_id','mmsi','y','mask','time_s']:assert np.array_equal(ext[key],core[key]),key
    for arm in ARMS:core['base__'+arm]=ext['base__'+arm]
    return core
def main(a):
    lock=setup();proto=json.loads((PARENT/'protocol.json').read_text());train=ts.load_npz(PARENT/'data'/'individual'/'train.npz');norm=ts.fit_normalizer(train,'individual')
    ts.ROOT=OUT;device=torch.device(a.device)
    if a.mode=='train':
        tr=combined('train');va=combined('val')
        for seed in a.seeds:
          for arm in ARMS:
            if (PARENT/'STOP').exists():(OUT/'STOP').touch();return
            ts.train_one('individual',a.backbone,arm,seed,device,tr,va,norm,proto,lock)
    else:
        cl=json.loads((OUT/'checkpoint_lock.json').read_text());assert cl['count']==72
        te=combined('test')
        for seed in a.seeds:
          for arm in ARMS:
            folder=OUT/'runs'/'individual'/a.backbone/f'seed_{seed}'/arm;cp=folder/'final.pt';name=str(cp.relative_to(OUT)).replace('\\','/')
            assert cl['checkpoints'][name]==ts.sha(cp)
            if (folder/'test_results.json').exists():continue
            saved=torch.load(cp,map_location='cpu',weights_only=False);model=make_model(a.backbone,12,max_len=20,hidden=64).to(device);model.load_state_dict(saved['model'])
            td=ts.tensor_data(te,'base__'+arm,saved['normalization'],device);p=ts.predict(model,td);ts.save_predictions(folder/'test_base.npz',te,p)
            m=ts.metrics(te['y'],p,te['region'].astype(str));m['operating']=ts.operating(te['y'],p,saved['thresholds']);m['predictions_sha256']=ts.sha(folder/'test_base.npz')
            ts.write_json(folder/'test_results.json',{'backbone':a.backbone,'arm':arm,'seed':seed,'checkpoint_sha256':ts.sha(cp),'code_lock_sha256':ts.sha(OUT/'code_lock.json'),'status':'developmental_previously_exposed_test','views':{'base':m}})
            print('FIELD_TEST',a.backbone,seed,arm,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['train','test'],required=True);p.add_argument('--backbone',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    main(a)
