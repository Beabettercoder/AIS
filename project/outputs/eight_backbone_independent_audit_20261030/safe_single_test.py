"""Conditional single-vessel fixed-checkpoint held-out evaluator.

Refuses to open held-out data before a successful, parent-issued gate record.
Existing held-out sources were previously exposed and must be labelled as such.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0,'/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')
import scale_train as experiment
from eight_backbone_adapter import Model8

HERE = Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')
ORIGIN = Path('/mnt/sda/WangZijia/AIS/outputs/original_mainline_pilot_20261029')
ROOT = experiment.root
ARMS = ('raw', 'causal_bundle', 'f0_bundle', 'f1_bundle')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def load_test():
    # Access labels only after the stage-1 gate record is verified in main().
    with np.load(ROOT / 'data/individual/test.npz', allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(ROOT / 'extra_data/field_test.npz', allow_pickle=False) as z:
        e = {k: z[k] for k in z.files}
    with np.load(HERE / 'rho_test.npz', allow_pickle=False) as z:
        rho = z['rho']
    a = experiment.build_field_inputs(d, e, 'F0')
    b = experiment.build_field_inputs(d, e, 'F1')
    assert np.array_equal(a['valid'], b['valid'])
    assert np.array_equal(rho > 0, b['valid'])
    cat = {v: i for i, v in enumerate(['turn_back', 'deviation', 'stall', 'sinuous', 'speed_drop'], 1)}
    yc = np.array([cat.get(str(v), 0) for v in d['behavior_type']], np.int64)
    assert np.array_equal(yc > 0, d['y'] > 0)
    tm = d['time_s'].copy(); tm -= tm[:, :1]; tm[~d['mask']] = 0
    return dict(raw=d['base__C0'][..., :7], self=a['self'], mask=d['mask'], times=tm.astype(np.float32),
                y=d['y'].astype(np.float32), yc=yc, region=d['region'].astype(str),
                mmsi=d['mmsi'].astype(str), ids=d['window_id'].astype(str),
                f0=a['field'], f1=b['field'], state0=a['state'], state1=b['state'], rho=rho, valid=b['valid'])

def rates(score, labels, threshold):
    hit = score > threshold
    pos = labels > 0
    return {'actual_FAR': float(hit[~pos].mean()), 'recall': float(hit[pos].mean())}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, choices=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'])
    parser.add_argument('--seed', required=True, type=int, choices=[0,1,2])
    parser.add_argument('--gpu', required=True, type=int)
    args = parser.parse_args()
    if args.gpu == 4:
        raise SystemExit('GPU 4 is reserved by an existing service.')
    gate = HERE / 'AUTHORIZED_STAGE2.json'
    if not gate.exists():
        raise SystemExit('Stage 2 not authorized: no parent-issued qualifying pilot gate record.')
    approval = json.loads(gate.read_text(encoding='utf8'))
    if approval.get('relation_pilot_criteria_met') is not True:
        raise SystemExit('Stage 2 not authorized: pilot criteria did not pass.')
    # Fixed epoch 25 was specified before seeing results; never select from held-out.
    torch.set_num_threads(2)
    dev = torch.device(f'cuda:{args.gpu}')
    train = experiment.load('train')
    normalization = experiment.fit(train)
    task = HERE / 'runs' / args.name / f'seed{args.seed}'
    complete = json.loads((task/'experiment_complete.json').read_text(encoding='utf8'))
    if complete['epochs']!=25 or sha(task/'experiment_preflight.json')!=complete['preflight_sha256']: raise SystemExit('Invalid preflight or epoch budget.')
    for arm, artifacts in complete['artifacts_sha256'].items():
        for path, digest in artifacts.items():
            if sha(Path(path)) != digest: raise SystemExit('Stale or altered training artifact.')
        if (task/arm/'final_test.npz').exists() or (task/arm/'test_result.json').exists(): raise FileExistsError(f'Refusing overwrite: {task/arm}')
    if set(complete['artifacts_sha256'])!=set(ARMS): raise SystemExit('Incomplete training arms.')
    data = load_test()
    tensors = experiment.convert(data, normalization, dev)
    task = HERE / 'runs' / args.name / f'seed{args.seed}'
    for arm in ARMS:
        run = task / arm
        model = Model8(args.name).to(dev)
        try:
            state = torch.load(run / 'final.pt', map_location=dev, weights_only=True)
        except TypeError:
            state = torch.load(run / 'final.pt', map_location=dev)
        model.load_state_dict(state, strict=True); model.eval()
        target = 'raw' if arm == 'raw' else 'self'
        field = 'f0' if arm == 'f0_bundle' else 'f1'
        hasfield = arm in ('f0_bundle', 'f1_bundle')
        values = []
        with torch.no_grad():
            for start in range(0, len(data['y']), 256):
                s = slice(start, start + 256)
                logits = model(tensors[target][s], tensors['mask'][s], tensors['times'][s],
                               tensors[field][s], tensors['rho'][s], hasfield)[0]
                values.extend(torch.sigmoid(logits).cpu().numpy())
        scores = np.asarray(values)
        assert np.isfinite(scores).all()
        results = experiment.metrics(data['y'], scores, data['region'])
        with np.load(run / 'final_val.npz', allow_pickle=False) as z:
            val_scores, val_labels = z['score'], z['y']
            negatives = val_scores[val_labels == 0]
        for budget in (.01, .05):
            # Conservative order-statistic threshold; no held-out values enter it.
            ordered = np.sort(negatives)
            idx = min(len(ordered)-1, max(0, int(np.ceil((1-budget) * len(ordered))) - 1))
            threshold = float(ordered[idx])
            results[f'FAR_{budget:g}'] = {'threshold_from_validation': threshold, **rates(scores, data['y'], threshold)}
        out = run / 'final_test.npz'
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite {out}')
        np.savez_compressed(out, score=scores, y=data['y'], mmsi=data['mmsi'], region=data['region'], window_id=data['ids'])
        result = {'model': args.name, 'seed': args.seed, 'arm': arm, 'checkpoint': 'fixed epoch 25',
                  'caveat': 'previously exposed development test; not a fresh blind test',
                  'checkpoint_sha256': sha(run / 'final.pt'), 'test_predictions_sha256': sha(out), 'metrics': results}
        (run / 'test_result.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()

