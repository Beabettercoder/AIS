"""Nine updated runs per backbone; exact frozen training recipe and normalization."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--model',choices=CONFIG['models'],required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();work=a.work
    preparation=read(work/'preparation_complete.json')
    assert read(work/'release_lock.json')['code_hashes']==code_hashes()
    for s in ['train','val']:assert sha(work/f'data/{s}.npz')==preparation['data_sha256'][s]
    norm=read(work/'data/normalizer.json');assert sha(work/'data/normalizer.json')==preparation['normalizer_sha256']
    tr=T.load_data(work/'data/train.npz',training=True);va=T.load_data(work/'data/val.npz')
    assert not set(tr['mmsi'])&set(va['mmsi'])
    device=torch.device(a.device);os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(CONFIG['cpu_threads_per_worker']);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    identity=dict(data_sha256=preparation['data_sha256'],source_sha256=code_hashes(),
        parent_manifest=read(work/'release_lock.json')['parent_manifest_sha256'],user_locked_reanchor=True,
        torch_version=torch.__version__,numpy_version=np.__version__)
    args=argparse.Namespace(out=work/'runs',epochs=CONFIG['epochs'][a.model],batch_size=CONFIG['batch_size'],lr=CONFIG['learning_rate'],
        threads=CONFIG['cpu_threads_per_worker'],resume=True)
    tasks=[(s,arm) for s in CONFIG['seeds'] for arm in CONFIG['updated_arms']]
    # Logging/serialization hook only: leave the frozen optimizer, losses and RNG sequence untouched.
    original_save=T.save_checkpoint
    def save_every_epoch(path,payload):
        original_save(path,payload)
        epoch=payload['completed_epoch'];row=payload['history'][-1]
        weights=path.parent/'weights';weights.mkdir(exist_ok=True)
        snapshot=dict(model=payload['model'],epoch=epoch,validation=row,
                      config_hash=payload['config_hash'],initial_model_hash=payload['initial_model_hash'])
        original_save(weights/f'epoch_{epoch:03d}.pt',snapshot)
        best_file=path.parent/'best_validation.json'
        is_best=not best_file.exists() or row['val_ADE_m']<read(best_file)['val_ADE_m']
        if is_best:
            original_save(path.parent/'best_validation_weights.pt',snapshot)
            save(best_file,dict(epoch=epoch,**{k:v for k,v in row.items() if k!='epoch'},
                criterion='minimum base-validation ADE; inspection only, official test uses frozen final epoch'))
        save(path.parent/'last_saved_weights.json',dict(epoch=epoch,path=str(weights/f'epoch_{epoch:03d}.pt'),best_updated=is_best))
    T.save_checkpoint=save_every_epoch
    for seed,arm in tqdm(tasks,desc=f'{a.model} NEW RUNS',unit='run'):
        T.one_run(args,tr,va,norm,identity,a.model,seed,arm,device)

if __name__=='__main__':main()
