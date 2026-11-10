"""User-run validation pilot: four encoders, four paired arms, visible progress.

No test file is opened. Training reads only the base view. Future observation
timestamps are pre-specified query times; future positions are loss targets only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from forecast_models import ForecastModel, MODEL_NAMES, ARMS, state_hash

VIEWS = ('base', 'target_jump', 'target_drift', 'target_speed', 'neighbor_stress')
COMMON = ('future_xy', 'query_s', 'time_s', 'mmsi', 'region', 'window_id', 'maneuver')
VIEW_KEYS = ('raw', 'gated', 'f0', 'f1', 'f0_valid', 'f1_valid',
             'anchor_raw', 'anchor_gate', 'vel_raw', 'vel_gate')
ARM_NAMES = {'P0': 'Raw history', 'P1': 'Causal gate',
             'P2': 'Gate + distance LOO', 'P3': 'Gate + reliable LOO'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def jsonable(x):
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jsonable,
                              allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=jsonable,
                                    allow_nan=False).encode()).hexdigest()


def verify_data_manifest(data_dir, source_dir):
    path = data_dir/'data_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; finish build_data.py before training')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    config = manifest.get('config', {})
    for name, key in (('features.py','features_sha256'),('build_data.py','builder_sha256')):
        actual = digest(source_dir/name)
        if config.get(key) != actual:
            raise ValueError(f'Dataset was built with a different {name}; use its frozen source '
                             'or rebuild into a new data directory')
    hashes = {}
    for split in ('train','val'):
        actual = digest(data_dir/f'{split}.npz')
        if manifest.get('npz_sha256',{}).get(split) != actual:
            raise ValueError(f'{split}.npz does not match data_manifest.json; data may be incomplete or changed')
        hashes[f'{split}.npz'] = actual
    return manifest, hashes, digest(path)


def event(folder, kind, **details):
    with (Path(folder) / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'utc': utc(), 'event': kind, **details},
                           ensure_ascii=False, default=jsonable, allow_nan=False) + '\n')


def acquire_lock(folder):
    """One process per run; only remove locks from demonstrably dead local PIDs."""
    path = Path(folder) / 'run.lock'
    info = {'pid': os.getpid(), 'host': socket.gethostname(), 'utc': utc()}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        old = json.loads(path.read_text(encoding='utf-8'))
        alive = True
        if old.get('host') == info['host']:
            if os.name == 'nt':
                # os.kill(pid, 0) is not a safe Windows liveness probe.
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x1000,False,int(old['pid']))
                if handle:
                    code = wintypes.DWORD()
                    if kernel.GetExitCodeProcess(handle,ctypes.byref(code)): alive = code.value == 259
                    kernel.CloseHandle(handle)
                else:
                    alive = ctypes.get_last_error() != 87  # invalid PID
            else:
                try: os.kill(int(old['pid']), 0)
                except ProcessLookupError: alive = False
                except PermissionError: alive = True
                except OSError: alive = True
        if alive:
            raise RuntimeError(f'Run locked by {old}; do not run the same arm twice: {folder}')
        path.unlink()
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(info, f)
    return path


def load_data(path, training=False):
    wanted_views = ('base',) if training else VIEWS
    # Deliberately enumerate permitted keys; train stress arrays are never read.
    keys = list(COMMON) + [f'{v}__{k}' for v in wanted_views for k in VIEW_KEYS]
    with np.load(path, allow_pickle=False) as z:
        missing = set(keys) - set(z.files)
        if missing: raise ValueError(f'{path} lacks keys: {sorted(missing)}')
        data = {k: z[k] for k in keys}
    n = len(data['window_id'])
    if not n: raise ValueError(f'No usable samples in {path}')
    if len(np.unique(data['window_id'])) != n:
        raise ValueError(f'Duplicate window_id in {path}')
    for name in COMMON:
        if len(data[name]) != n: raise ValueError(f'{name} count mismatch')
    if data['future_xy'].shape != (n, 8, 2): raise ValueError('future_xy must be [N,8,2]')
    if data['future_xy'].dtype != np.float64: raise ValueError('future_xy must stay float64')
    if data['query_s'].shape != (n, 8): raise ValueError('query_s must be [N,8]')
    if data['time_s'].shape != (n, 12): raise ValueError('time_s must be [N,12]')
    if not np.isfinite(data['query_s']).all() or not np.isfinite(data['time_s']).all():
        raise ValueError('Non-finite query/history timestamps')
    if np.any(data['query_s'] <= 0) or np.any(np.diff(data['query_s'], axis=1) <= 0):
        raise ValueError('Future query times must be positive and strictly increasing')
    if np.any(np.diff(data['time_s'], axis=1) <= 0):
        raise ValueError('History timestamps must be strictly increasing')
    for v in wanted_views:
        expected = {'raw': (n,12,9), 'gated': (n,12,9),
                    'f0': (n,12,8), 'f1': (n,12,8),
                    'f0_valid': (n,12), 'f1_valid': (n,12),
                    'anchor_raw': (n,2), 'anchor_gate': (n,2),
                    'vel_raw': (n,2), 'vel_gate': (n,2)}
        for k, shape in expected.items():
            a = data[f'{v}__{k}']
            if a.shape != shape: raise ValueError(f'{v}__{k}: {a.shape} != {shape}')
            if not np.isfinite(a).all(): raise ValueError(f'Non-finite {v}__{k}')
            if k.startswith('anchor') and a.dtype != np.float64:
                raise ValueError(f'{v}__{k} must stay float64')
        for k in ('f0', 'f1'):
            rho = data[f'{v}__{k}'][...,6]
            if np.any((rho < 0) | (rho > 1)): raise ValueError('rho outside [0,1]')
    if not np.isfinite(data['future_xy']).all(): raise ValueError('Non-finite future position')
    return data


def fit_normalizer(train):
    # One train-only scaler shared by every arm. Pool both arms equally so
    # changed feature scaling is not an additional arm-specific intervention.
    x = np.concatenate([train['base__raw'], train['base__gated']], 0).astype(np.float64)
    xm, xs = x.mean((0,1)), x.std((0,1))
    xs = np.where(xs < 1e-8, 1., xs)
    fields = []
    for name in ('f0', 'f1'):
        fields.append(train[f'base__{name}'][train[f'base__{name}_valid'].astype(bool)])
    f = np.concatenate(fields, 0).astype(np.float64)
    fm, fs = (f.mean(0), f.std(0)) if len(f) else (np.zeros(8), np.ones(8))
    fs = np.where(fs < 1e-8, 1., fs)
    # Support and mode retain their literal meanings. Invalid field slots zeroed.
    fm[6:] = 0.; fs[6:] = 1.
    return {'self_mean': xm.tolist(), 'self_std': xs.tolist(),
            'field_mean': fm.tolist(), 'field_std': fs.tolist(),
            'fit': 'base training only; equal raw/gated rows and pooled valid f0/f1 rows',
            'field_unscaled_slots': [6,7]}


def prepare(data, view, arm, normalizer):
    raw = arm == 'P0'
    name = 'raw' if raw else 'gated'
    x = (data[f'{view}__{name}'] - np.asarray(normalizer['self_mean'])) / np.asarray(normalizer['self_std'])
    n = len(x)
    if arm in ('P2', 'P3'):
        fn = 'f0' if arm == 'P2' else 'f1'
        orig = data[f'{view}__{fn}']
        rho = orig[...,6].copy()
        valid = data[f'{view}__{fn}_valid'].astype(bool)
        field = (orig - np.asarray(normalizer['field_mean'])) / np.asarray(normalizer['field_std'])
        field[~valid] = 0.; rho[~valid] = 0.
    else:
        field, rho = np.zeros((n,12,8)), np.zeros((n,12))
    anchor = data[f'{view}__anchor_' + ('raw' if raw else 'gate')]
    velocity = data[f'{view}__vel_' + ('raw' if raw else 'gate')]
    return {'x': torch.from_numpy(x.astype(np.float32)),
            'field': torch.from_numpy(field.astype(np.float32)),
            'rho': torch.from_numpy(rho.astype(np.float32)),
            'query_s': torch.from_numpy(data['query_s'].astype(np.float32)),
            'times': torch.from_numpy(data['time_s'].astype(np.float32)),
            'anchor': torch.from_numpy(anchor.astype(np.float64)),
            'velocity': torch.from_numpy(velocity.astype(np.float32)),
            'target': torch.from_numpy(data['future_xy'].astype(np.float64))}


def batch_forward(model, tensors, idx, device, arm):
    b = {k: v[idx].to(device) for k,v in tensors.items()}
    pred = model(b['x'], b['field'], b['rho'], b['query_s'], b['anchor'],
                 b['velocity'], arm in ('P2','P3'), times=b['times'])
    return pred, b['target']


@torch.inference_mode()
def evaluate(model, tensors, device, arm, batch_size, label=None):
    model.eval()
    n = len(tensors['x'])
    batches = range(0, n, batch_size)
    if label: batches = tqdm(batches, desc=label, leave=False, unit='batch', dynamic_ncols=True)
    predictions = []
    for start in batches:
        pred, _ = batch_forward(model, tensors, slice(start, start+batch_size), device, arm)
        predictions.append(pred.cpu().numpy())
    pred = np.concatenate(predictions)
    error = np.linalg.norm(pred - tensors['target'].numpy(), axis=-1)
    if not np.isfinite(error).all(): raise FloatingPointError('Non-finite evaluation error')
    return pred, error


def metric_slice(errors, selected):
    n = int(selected.sum())
    return {'windows': n, 'ADE_m': float(errors[selected].mean()) if n else None,
            'FDE_m': float(errors[selected,-1].mean()) if n else None}


def metrics(errors, data):
    q = data['query_s'][:,-1]
    out = {'all': metric_slice(errors, np.ones(len(q),bool)),
           'observed_maneuver': metric_slice(errors, data['maneuver'].astype(bool)),
           'no_observed_maneuver': metric_slice(errors, ~data['maneuver'].astype(bool)),
           'horizon_definition': 'final requested future timestamp minus history cutoff',
           'horizon_le_300_s': metric_slice(errors, q <= 300),
           'horizon_300_to_600_s': metric_slice(errors, (q > 300) & (q <= 600)),
           'horizon_gt_600_s': metric_slice(errors, q > 600),
           'by_region': {str(r): metric_slice(errors,data['region']==r) for r in np.unique(data['region'])}}
    out['region_macro_ADE_m'] = float(np.mean([v['ADE_m'] for v in out['by_region'].values()]))
    out['region_macro_FDE_m'] = float(np.mean([v['FDE_m'] for v in out['by_region'].values()]))
    return out


def save_prediction(folder, view, pred, error, data, model, arm, seed):
    path = Path(folder) / f'predictions_{view}.npz'
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, pred_xy=pred.astype(np.float64), future_xy=data['future_xy'],
                            point_error=error, ADE=error.mean(1), FDE=error[:,-1],
                            query_s=data['query_s'], mmsi=data['mmsi'], region=data['region'],
                            window_id=data['window_id'], source_id=data['window_id'],
                            maneuver=data['maneuver'], split=np.array('val'), view=np.array(view),
                            model=np.array(model), arm=np.array(arm), seed=np.array(seed))
    os.replace(tmp,path)


def save_baselines(out, val, identity):
    for name in ('CV_raw','CV_gate'):
        folder = Path(out)/'baselines'/name
        folder.mkdir(parents=True,exist_ok=True)
        result_file = folder/'result.json'
        if result_file.exists():
            old = json.loads(result_file.read_text(encoding='utf-8'))
            if old.get('identity_hash') != identity: raise ValueError('Baseline data/source changed; use a new --out')
            if old.get('status') == 'complete': continue
        try: lock = acquire_lock(folder)
        except RuntimeError:
            tqdm.write(f'{name}: another process is creating the common baseline.')
            continue
        try:
            result = {'status':'complete','model':name,'arm':name,'seed':-1,'split':'val',
                      'identity_hash':identity,'completed_epochs':0,'expected_epochs':0,'views':{}}
            kind = 'raw' if name == 'CV_raw' else 'gate'
            base_prediction = None
            for view in VIEWS:
                pred = (val[f'{view}__anchor_{kind}'][:,None,:] +
                        val[f'{view}__vel_{kind}'].astype(np.float64)[:,None,:] *
                        val['query_s'].astype(np.float64)[...,None])
                if view == 'base': base_prediction = pred.copy()
                if view == 'neighbor_stress' and not np.array_equal(pred,base_prediction):
                    raise AssertionError(f'{name}: neighbor-only stress changed self-only predictions')
                err = np.linalg.norm(pred-val['future_xy'],axis=-1)
                save_prediction(folder,view,pred,err,val,name,name,-1)
                result['views'][view] = metrics(err,val)
            atomic_json(result_file,result)
        finally: lock.unlink(missing_ok=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(),device)


def save_checkpoint(path, payload):
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(payload,tmp)
    os.replace(tmp,path)


def one_run(args, train, val, normalizer, identity, model_name, seed, arm, device):
    epochs = args.epochs or (75 if model_name == 'itransformer' else 25)
    folder = args.out/model_name/f'seed_{seed}'/arm
    folder.mkdir(parents=True,exist_ok=True)
    config = {'model':model_name,'seed':seed,'arm':arm,'epochs':epochs,
              'batch_size':args.batch_size,'learning_rate':args.lr,'hidden':64,
              'device':str(device),'cpu_threads':args.threads,
              'history_steps':12,'future_steps':8,'train_view':'base',
              'loss':'SmoothL1(pred/1000,target/1000,beta=0.1)',
              'optimizer':'AdamW','weight_decay':0.0001,'gradient_clip':1.,
              'selection':'fixed final epoch; val base monitored only',
              'normalizer_hash':object_hash(normalizer),'identity':identity}
    config_hash = object_hash(config)
    protocol_path = folder/'protocol.json'
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding='utf-8'))
        if old.get('config_hash') != config_hash:
            raise ValueError(f'Configuration/data/source changed: {folder}. Use a new --out.')
        if not args.resume:
            raise FileExistsError(f'Existing run: {folder}. Supply --resume or a new --out.')
        if (folder/'result.json').exists():
            result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
            if result.get('status') == 'complete':
                tqdm.write(f'SKIP complete: {model_name} seed={seed} {arm}')
                return
    lock = acquire_lock(folder)
    start = time.monotonic()
    progress = {'model':model_name,'seed':seed,'arm':arm,'arm_name':ARM_NAMES[arm],
                'stage':'initializing','epoch':0,'completed_epochs':0,'total_epochs':epochs,
                'batch':0,'total_batches':math.ceil(len(train['window_id'])/args.batch_size),
                'train_loss':None,'val_ADE_m':None,'val_FDE_m':None,'pid':os.getpid()}
    def update(**kw):
        progress.update(kw); progress['updated_utc']=utc()
        progress['elapsed_seconds']=time.monotonic()-start
        atomic_json(folder/'progress.json',progress)
    try:
        seed_all(seed)
        model = ForecastModel(model_name)
        initial_hash = state_hash(model)
        protocol = {**config,'config_hash':config_hash,'initial_model_hash':initial_hash,
                    'parameter_count':sum(p.numel() for p in model.parameters()),
                    'torch_version':torch.__version__,'numpy_version':np.__version__}
        # Initial weights must match across all four arms, including unused modules.
        for peer in folder.parent.glob('*/protocol.json'):
            p = json.loads(peer.read_text(encoding='utf-8'))
            if p.get('initial_model_hash') != initial_hash:
                raise ValueError(f'Paired initialization mismatch: {peer}')
        atomic_json(protocol_path,protocol)
        atomic_json(folder/'normalizer.json',normalizer)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        completed = 0
        history = []
        checkpoint = folder/'checkpoint.pt'
        if args.resume and checkpoint.exists():
            ck = torch.load(checkpoint,map_location=device,weights_only=False)
            if ck['config_hash'] != config_hash or ck['initial_model_hash'] != initial_hash:
                raise ValueError('Checkpoint configuration/initialization does not match')
            model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
            completed=ck['completed_epoch']; history=ck['history']
            restore_rng(ck['rng'],device)
            tqdm.write(f'RESUME {model_name} seed={seed} {arm}: epoch {completed}/{epochs}')
        event(folder,'start',resume_epoch=completed,config_hash=config_hash,device=str(device))
        update(stage='training',completed_epochs=completed)
        tr = prepare(train,'base',arm,normalizer)
        va = prepare(val,'base',arm,normalizer)
        n = len(tr['x'])
        outer = tqdm(range(completed,epochs),desc=f'{model_name} s{seed} {arm} epochs',
                     initial=completed,total=epochs,unit='epoch',dynamic_ncols=True)
        for epoch in outer:
            model.train()
            order = np.random.default_rng(300000+seed*10000+epoch).permutation(n)
            losses=[]; counts=[]; last_write=0.
            batches=tqdm(range(0,n,args.batch_size),desc=f'  train {epoch+1}/{epochs}',
                         leave=False,unit='batch',dynamic_ncols=True)
            for batch,start_i in enumerate(batches,1):
                idx=order[start_i:start_i+args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                pred,target=batch_forward(model,tr,idx,device,arm)
                loss=F.smooth_l1_loss(pred/1000.,target/1000.,beta=.1)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach())); counts.append(len(idx))
                mean_loss=float(np.average(losses,weights=counts))
                batches.set_postfix(loss=f'{mean_loss:.6f}')
                if time.monotonic()-last_write >= 1 or start_i+args.batch_size>=n:
                    update(stage='training',epoch=epoch+1,batch=batch,train_loss=mean_loss)
                    last_write=time.monotonic()
            update(stage='validation',epoch=epoch+1)
            _,errors=evaluate(model,va,device,arm,args.batch_size,'  val base')
            ade,fde=float(errors.mean()),float(errors[:,-1].mean())
            row={'epoch':epoch+1,'train_loss':mean_loss,'val_ADE_m':ade,'val_FDE_m':fde}
            history.append(row)
            save_checkpoint(checkpoint,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                                        'completed_epoch':epoch+1,'rng':rng_state(device),
                                        'history':history,'config_hash':config_hash,
                                        'initial_model_hash':initial_hash})
            with (folder/'history.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            event(folder,'epoch_complete',**row)
            update(stage='epoch_complete',completed_epochs=epoch+1,val_ADE_m=ade,val_FDE_m=fde)
            outer.set_postfix(ADE=f'{ade:.1f}m',FDE=f'{fde:.1f}m')
            tqdm.write(f'{model_name:12s} seed {seed} {arm} | {epoch+1:02d}/{epochs} | '
                       f'loss {mean_loss:.6f} | val ADE {ade:.2f} m FDE {fde:.2f} m')
        result={'status':'complete','model':model_name,'arm':arm,'seed':seed,'split':'val',
                'completed_epochs':epochs,'expected_epochs':epochs,'config_hash':config_hash,
                'initial_model_hash':initial_hash,'views':{}}
        base_prediction = None
        for view in tqdm(VIEWS,desc=f'{model_name} s{seed} {arm} final views',unit='view',dynamic_ncols=True):
            update(stage='final_evaluation',view=view)
            tensors=va if view=='base' else prepare(val,view,arm,normalizer)
            pred,errors=evaluate(model,tensors,device,arm,args.batch_size)
            if view == 'base': base_prediction = pred.copy()
            if view == 'neighbor_stress' and arm in ('P0','P1') and not np.array_equal(pred,base_prediction):
                raise AssertionError(f'{arm}: neighbor-only stress changed self-only predictions')
            save_prediction(folder,view,pred,errors,val,model_name,arm,seed)
            result['views'][view]=metrics(errors,val)
        result['finished_utc']=utc();result['elapsed_seconds']=time.monotonic()-start
        atomic_json(folder/'result.json',result)
        event(folder,'complete',elapsed_seconds=result['elapsed_seconds'])
        update(stage='complete')
    except BaseException as exc:
        event(folder,'interrupted' if isinstance(exc,KeyboardInterrupt) else 'error',message=str(exc))
        update(stage='interrupted' if isinstance(exc,KeyboardInterrupt) else 'error',message=str(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',type=Path,required=True,help='Directory containing train.npz and val.npz')
    p.add_argument('--out',type=Path,required=True,help='A new result directory, or matching --resume directory')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--models',nargs='+',choices=MODEL_NAMES,default=list(MODEL_NAMES))
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--epochs',type=int,help='Override ALL epochs, intended for a separately named smoke output')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--resume',action='store_true',help='Resume only exact matching checkpoints; never overwrites configuration')
    p.add_argument('--threads',type=int,default=4,help='CPU torch worker threads per process')
    args=p.parse_args()
    if args.epochs is not None and args.epochs<1: p.error('--epochs must be >=1')
    if args.batch_size<1 or args.lr<=0 or args.threads<1: p.error('Positive batch-size/lr/threads required')
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate models/seeds are not allowed')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available in this Python environment; select a CUDA-enabled environment or --device cpu')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    print(f'Environment: torch={torch.__version__} numpy={np.__version__} device={device}',flush=True)
    print(f'Deterministic algorithms={torch.are_deterministic_algorithms_enabled()} | '
          f'cuDNN deterministic={torch.backends.cudnn.deterministic} | '
          f'cuBLAS workspace={os.environ["CUBLAS_WORKSPACE_CONFIG"]}',flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent
    manifest,data_hashes,manifest_hash=verify_data_manifest(args.data,here)
    print('Reading train/base and validation views only. test.npz is never opened.',flush=True)
    train=load_data(args.data/'train.npz',training=True)
    val=load_data(args.data/'val.npz',training=False)
    overlap=np.intersect1d(train['mmsi'],val['mmsi'])
    if len(overlap): raise ValueError(f'Train/validation target MMSI overlap: {len(overlap)}')
    normalizer=fit_normalizer(train)
    source_names=('train.py','forecast_models.py','vendor_encoders.py','features.py','build_data.py')
    identity={'data_sha256':data_hashes,'data_manifest_sha256':manifest_hash,
              'dataset_fingerprint':manifest.get('fingerprint'),
              'source_sha256':{name:digest(here/name) for name in source_names if (here/name).exists()},
              'train_windows':len(train['window_id']),'val_windows':len(val['window_id']),
              'torch_version':torch.__version__,'numpy_version':np.__version__}
    save_baselines(args.out,val,object_hash(identity))
    tasks=[(m,s,a) for m in args.models for s in args.seeds for a in ARMS]
    print(f'{len(tasks)} paired training runs. Four arms per model/seed share initialization and shuffle.',flush=True)
    print('Stop: Ctrl+C. Resume: repeat the same command with --resume. Incomplete epoch restarts.',flush=True)
    for model,seed,arm in tqdm(tasks,desc='ALL RUNS',unit='run',dynamic_ncols=True):
        one_run(args,train,val,normalizer,identity,model,seed,arm,device)
    print(f'Done. Prediction files and per-view validation metrics: {args.out.resolve()}',flush=True)


if __name__=='__main__':
    main()
