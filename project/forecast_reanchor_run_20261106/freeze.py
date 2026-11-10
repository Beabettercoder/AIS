"""Freeze all 48 forecasting checkpoints BEFORE constructing test windows."""
import argparse
from pathlib import Path
from common import *

def freeze(work):
    work=Path(work);file=work/'checkpoint_freeze.json'
    if file.exists():return assert_frozen(work)
    if (work/'data/test.npz').exists():raise RuntimeError('Test data exists without a prior checkpoint freeze.')
    norm=read(work/'data/normalizer.json');models=[]
    for m in CONFIG['models']:
        for seed in CONFIG['seeds']:
            hashes=set();widths=set()
            for arm in ['P0','P1','P2','P3']:
                folder=work/f'runs/{m}/seed_{seed}/{arm}';r=read(folder/'result.json');p=read(folder/'protocol.json')
                assert r['status']=='complete' and r['completed_epochs']==CONFIG['epochs'][m]
                assert read(folder/'normalizer.json')==norm
                hashes.add(p['initial_model_hash']);widths.add(p['parameter_count'])
                models.append(dict(model=m,seed=seed,arm=arm,checkpoint=str((folder/'checkpoint.pt').resolve()),
                    checkpoint_sha256=sha(folder/'checkpoint.pt'),protocol_sha256=sha(folder/'protocol.json'),reused=arm=='P0'))
            assert len(hashes)==len(widths)==1
    data=dict(status='frozen',created_utc=T.utc(),checkpoint_rule='fixed final epoch',models=models,
        normalizer_sha256=sha(work/'data/normalizer.json'),code_hashes=code_hashes(),test_protocol=CONFIG,
        preparation_sha256=sha(work/'preparation_complete.json'),test_opened_by_this_stage=False)
    save(file,data);return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('work'));a=p.parse_args();freeze(a.work)
