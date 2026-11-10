"""Prepare locked inputs and reuse the twelve unchanged Raw models; no training."""
import argparse,shutil
import numpy as np
from pathlib import Path
from common import *
from cv_and_build import replay

def prepare(pilot,work):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    parent=validate_parent(pilot)
    identity=dict(config=CONFIG,code_hashes=code_hashes(),parent_manifest_sha256=sha(pilot/'data/data_manifest.json'))
    marker=work/'release_lock.json'
    if marker.exists():
        if read(marker)!=identity:raise RuntimeError('Code/protocol changed. Use a new work directory.')
    else:save(marker,identity)
    done=work/'preparation_complete.json'
    if done.exists():
        d=read(done)
        for s in ['train','val']:
            assert sha(work/f'data/{s}.npz')==d['data_sha256'][s]
        return d
    print('Preparing re-anchored histories; old CV failure record is preserved and user acceptance is recorded.',flush=True)
    old={};new={}
    for split in ['train','val']:
        old[split],new[split]=replay(pilot,work/'data',split,{'position_reset_threshold':.7},F)
        for key in old[split]:
            if key.endswith('__gated'):np.testing.assert_array_equal(old[split][key][...,2:],new[split][key][...,2:])
            elif not key.endswith('__anchor_gate'):np.testing.assert_array_equal(old[split][key],new[split][key])
    prior_norm=read(pilot/'parallel_runs/tcn/tcn/seed_0/P0/normalizer.json')
    assert T.fit_normalizer(T.load_data(pilot/'data/train.npz',training=True))==prior_norm
    save(work/'data/normalizer.json',prior_norm)
    reused=[]
    for m in CONFIG['models']:
        for seed in CONFIG['seeds']:
            src=pilot/f'parallel_runs/{m}/{m}/seed_{seed}/P0';dst=work/f'runs/{m}/seed_{seed}/P0'
            protocol=read(src/'protocol.json');res=read(src/'result.json')
            assert res['status']=='complete' and protocol['epochs']==CONFIG['epochs'][m]
            assert protocol['seed']==seed and protocol['arm']=='P0'
            assert protocol['batch_size']==CONFIG['batch_size'] and protocol['learning_rate']==CONFIG['learning_rate']
            assert read(src/'normalizer.json')==prior_norm
            for split in ['train','val']:
                for view in (['base'] if split=='train' else F.VIEWS):
                    aa=T.prepare(old[split],view,'P0',prior_norm);bb=T.prepare(new[split],view,'P0',prior_norm)
                    for k in aa:np.testing.assert_array_equal(aa[k].numpy(),bb[k].numpy())
            dst.mkdir(parents=True,exist_ok=True)
            for p in src.iterdir():
                if p.is_file() and p.name!='run.lock':shutil.copy2(p,dst/p.name)
            record=dict(model=m,seed=seed,source=str(src),checkpoint_sha256=sha(src/'checkpoint.pt'),
                        initial_hash=protocol['initial_model_hash'],reason='All P0 tensors and frozen normalization exactly unchanged')
            save(dst/'reuse_record.json',record);reused.append(record)
    d=dict(status='complete',new_training_runs=36,reused_runs=12,source_data=str(pilot/'data'),
        data_sha256={s:sha(work/f'data/{s}.npz') for s in ['train','val']},normalizer_sha256=sha(work/'data/normalizer.json'),
        parent_config=parent['config'],reused=reused,test_built=False)
    save(done,d);return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,default=DEFAULT_PILOT);p.add_argument('--work',type=Path,default=Path('work'))
    a=p.parse_args();prepare(a.pilot,a.work)
