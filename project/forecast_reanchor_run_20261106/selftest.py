"""Fast synthetic checks, optional tiny CPU training smoke. Never opens real test data."""
import argparse,tempfile
from pathlib import Path
import numpy as np
import torch
from common import *
from reanchor import reanchor_positions

def synthetic(n=12,start_id=1000):
    t=np.arange(12)*60.;q=np.arange(1,9)*60.
    raw=np.zeros((n,12,9),'f4');raw[:,:,0]=5*t;raw[:,:,2]=5;raw[:,:,4]=5;raw[:,:,5]=1;raw[:,:,7]=t;raw[:,1:,8]=60
    anchor=np.tile([3300.,0.],(n,1));vel=np.tile([5.,0.],(n,1)).astype('f4')
    future=anchor[:,None]+vel[:,None]*q[None,:,None]
    data=dict(future_xy=future.astype('f8'),query_s=np.tile(q,(n,1)).astype('f4'),time_s=np.tile(t,(n,1)).astype('f4'),
        window_id=np.array([f'synthetic_{start_id+i}' for i in range(n)]),mmsi=np.arange(start_id,start_id+n),
        region=np.array(['synthetic']*n),maneuver=np.zeros(n,bool))
    for view in CONFIG['views']:
        for k,v in dict(raw=raw,gated=raw,f0=np.zeros((n,12,8),'f4'),f1=np.zeros((n,12,8),'f4'),
            f0_valid=np.zeros((n,12),bool),f1_valid=np.zeros((n,12),bool),anchor_raw=anchor,anchor_gate=anchor,vel_raw=vel,vel_gate=vel).items():data[view+'__'+k]=v.copy()
    return data

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke-train',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);tr=synthetic();va=synthetic(8,2000);norm=T.fit_normalizer(tr)
    pos=np.column_stack([np.arange(12)*300.,np.zeros(12)]);t=np.arange(12)*60.;w=np.ones(12);ok=np.ones(12,bool);ok[0]=False
    damaged=pos.copy();damaged[1:]-=[1000,0];w[1]=0
    cp,reset=reanchor_positions(t,pos,damaged,w,ok,.7)
    np.testing.assert_array_equal(cp[2:],pos[2:])
    for length in range(1,13):
        x,_=reanchor_positions(t[:length],pos[:length],damaged[:length],w[:length],ok[:length],.7)
        np.testing.assert_array_equal(x,cp[:length])
    for m in CONFIG['models']:
        T.seed_all(0);model=ForecastModel(m).eval();h=state_hash(model);T.seed_all(0);assert h==state_hash(ForecastModel(m))
        for arm in ['P0','P1','P2','P3']:
            tensors=T.prepare(va,'base',arm,norm);pred,error=T.evaluate(model,tensors,torch.device('cpu'),arm,8)
            np.testing.assert_array_equal(pred,va['future_xy'])
        print('PASS:',m,'initial CV, identical initialization, zero-support fallback',flush=True)
    # Test access cannot happen without checkpoint freeze.
    with tempfile.TemporaryDirectory() as tmp:
        try:assert_frozen(Path(tmp))
        except FileNotFoundError:pass
        else:raise AssertionError('Test barrier absent')
    if a.smoke_train:
        from argparse import Namespace
        with tempfile.TemporaryDirectory(prefix='reanchor_smoke_') as tmp:
            args=Namespace(out=Path(tmp),epochs=1,batch_size=8,lr=.002,threads=1,resume=False)
            for m in CONFIG['models']:
                T.one_run(args,tr,va,norm,{'synthetic_smoke_only':True},m,0,'P1',torch.device('cpu'))
            print('PASS: four one-epoch synthetic CPU training/checkpoint artifacts; no real test data opened')
        # Exercise the actual worker's epoch-weight, best-validation, and resume hooks.
        import copy,sys
        import train_worker
        original_config=copy.deepcopy(CONFIG);original_argv=sys.argv[:];old_save=T.save_checkpoint
        try:
            with tempfile.TemporaryDirectory(prefix='epoch_logging_smoke_') as tmp:
                work=Path(tmp);(work/'data').mkdir()
                for split,z in [('train',tr),('val',va)]:np.savez_compressed(work/f'data/{split}.npz',**z)
                save(work/'data/normalizer.json',norm)
                save(work/'preparation_complete.json',dict(data_sha256={s:sha(work/f'data/{s}.npz') for s in ['train','val']},normalizer_sha256=sha(work/'data/normalizer.json')))
                save(work/'release_lock.json',dict(code_hashes=code_hashes(),parent_manifest_sha256='synthetic-qa-only'))
                CONFIG['seeds']=[0];CONFIG['updated_arms']=['P1'];CONFIG['epochs']={'tcn':2}
                sys.argv=['train_worker.py','--work',str(work),'--model','tcn','--device','cpu']
                train_worker.main()
                folder=work/'runs/tcn/seed_0/P1'
                for name in ['weights/epoch_001.pt','weights/epoch_002.pt','checkpoint.pt','best_validation_weights.pt','history.csv']:
                    assert (folder/name).exists(),name
                before=sha(folder/'checkpoint.pt');train_worker.main();assert before==sha(folder/'checkpoint.pt')
                print('PASS: actual worker saves each epoch and best-validation weights; completed-run resume does not retrain')
        finally:
            CONFIG.clear();CONFIG.update(original_config);sys.argv=original_argv;T.save_checkpoint=old_save
    print('PASS: prefix causality, frozen input semantics and pre-freeze test barrier. No 36-run experiment started.',flush=True)

if __name__=='__main__':main()
