"""Guarded single-vessel stage-2 launcher with non-overwrite and provenance.

It never creates a gate record and never reads test data. Preflight records input,
source and initial-model hashes before any gradient update. A failed run leaves its
preflight marker in place and must not be silently resumed or overwritten.
"""
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path

BASE=Path('/mnt/sda/WangZijia/AIS/outputs')
PREP=BASE/'eight_backbone_preparation_20261030'
ORIG=BASE/'original_mainline_pilot_20261029'
ROOT=BASE/'framework_validation_v1_20261028'
sys.path.insert(0,str(PREP))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def reserve(folder):
    """Accept absent or truly empty directory; atomically reserve before use."""
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError(f'Refusing existing nonempty output directory: {folder}')
    marker=folder/'RESERVED_BY_SAFE_LAUNCHER'
    fd=os.open(str(marker),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as out:out.write('One run only; do not overwrite or resume silently.\n')
    return marker

def run(a):
    if a.gpu==4: raise SystemExit('GPU 4 is reserved by an existing service.')
    from guarded_scale_train import gate
    gate()
    import scale_train as s
    out=PREP/'runs'/a.name/f'seed{a.seed}'
    reserve(out)
    deps=[Path(__file__),PREP/'guarded_scale_train.py',PREP/'scale_train.py',PREP/'eight_backbone_adapter.py',
      ORIG/'pilot_features.py',ORIG/'pilot_historical_loss.py',ROOT/'models.py',
      *[ROOT/'data/individual'/f'{split}.npz' for split in ('train','val')],
      *[ROOT/'extra_data'/f'field_{split}.npz' for split in ('train','val')],
      *[ORIG/f'rho_{split}.npz' for split in ('train','val')],PREP/'AUTHORIZED_STAGE2.json']
    s.seed(a.seed);model=s.Model(a.name)
    init=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))).hexdigest()
    meta={'backbone':a.name,'seed':a.seed,'gpu':a.gpu,'arms':['raw','causal_bundle','f0_bundle','f1_bundle'],
          'epoch':25,'optimizer':'AdamW lr .002 wd .01 with declared no-decay groups',
          'batch_size':128,'normalizer':'train only; pooled raw/causal and F0/F1',
          'initial_model_sha256':init,'dependencies_sha256':{str(p):sha(p) for p in deps},
          'test_access':'none before training','overwrite':'refuse existing nonempty directory'}
    (out/'experiment_preflight.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('SINGLE_PREFLIGHT',json.dumps(meta),flush=True)
    s.main(a.name,a.gpu,a.seed)
    artifacts={}
    for arm in meta['arms']:
        folder=out/arm
        result=json.loads((folder/'result.json').read_text(encoding='utf8'))
        assert result['history'][-1]['epoch']==25 and result['init']==init
        artifacts[arm]={str(p):sha(p) for p in (folder/'result.json',folder/'final.pt',folder/'final_val.npz')}
    (out/'experiment_complete.json').write_text(json.dumps({'preflight_sha256':sha(out/'experiment_preflight.json'),'artifacts_sha256':artifacts,'epochs':25},indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph']);p.add_argument('--seed',required=True,type=int,choices=[0,1,2]);p.add_argument('--gpu',required=True,type=int);run(p.parse_args())

