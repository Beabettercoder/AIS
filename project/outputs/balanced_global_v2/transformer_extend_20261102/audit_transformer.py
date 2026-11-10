"""Independently reload all Transformer checkpoints and recompute paired validation metrics."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import train_transformer as exp

OUT = Path(__file__).resolve().parent
ARMS = exp.ARMS
PAIR = [('causal', 'raw'), ('raw_bundle', 'raw'), ('causal_bundle', 'causal'), ('causal_bundle', 'raw_bundle')]

def file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for part in iter(lambda: f.read(1048576), b''): h.update(part)
    return h.hexdigest()

def metrics(y, scores, regions):
    per = [average_precision_score(y[regions == r], scores[regions == r]) for r in sorted(set(regions))]
    return {'region_macro_ap': float(np.mean(per)), 'pooled_ap': float(average_precision_score(y, scores)),
            'pooled_auroc': float(roc_auc_score(y, scores))}

def main():
    torch.set_num_threads(4)
    exp.th.seed_all(0)
    device = torch.device('cuda:0')
    data, _ = exp.setup()
    dv = data['val']
    windows = dv['window_id'].astype(str)
    ys = dv['y'].astype(int)
    regions = dv['region'].astype(str)
    groups = dv['mmsi'].astype(str)
    worker = OUT/'runs'/'transformer'
    run_metrics = {}
    scores = {}
    max_diff = 0.
    for seed in (0, 1, 2):
        for arm in ARMS:
            p = worker/f'seed_{seed}'/arm
            result = json.loads((p/'result.json').read_text(encoding='utf-8'))
            completion = json.loads((p/'completion.json').read_text(encoding='utf-8'))
            assert file_sha(p/'result.json') == completion['result_sha256']
            for name, expected in result['output_hashes'].items():
                assert file_sha(p/name) == expected, (seed, arm, name)
            state = torch.load(p/'checkpoint_final.pt', map_location=device, weights_only=False)
            model = exp.TransformerModel('transformer', arm).to(device)
            model.load_state_dict(state['state_dict'])
            stats = np.load(p/'normalizer.npz', allow_pickle=False)
            x = exp.th.normalized(dv['features'][arm], dv['mask'], {'mean':stats['mean'], 'std':stats['std']}, False)
            tensors = exp.th.to_tensors(dv, x)
            pred, _ = exp.th.predict(model, tensors, device)
            with np.load(p/'epoch_validation_predictions.npz', allow_pickle=False) as z:
                assert np.array_equal(windows, z['window_id'].astype(str))
                prev = z['scores'][-1]
            diff = float(np.max(np.abs(pred-prev)))
            assert diff <= 5e-6, (seed, arm, diff)
            max_diff = max(diff, max_diff)
            rows = list(csv.DictReader((p/'val_predictions_final.csv').open(encoding='utf-8', newline='')))
            assert [row['window_id'] for row in rows] == windows.tolist()
            csv_scores = np.array([float(row['score']) for row in rows])
            assert np.max(np.abs(csv_scores-prev)) <= 1e-7
            m = metrics(ys, prev, regions)
            assert abs(m['region_macro_ap'] - result['final_validation']['macro']['AP']) < 1e-10
            assert abs(m['pooled_ap'] - result['final_validation']['pooled']['AP']) < 1e-10
            run_metrics[f'{arm}/seed_{seed}'] = m
            scores[(seed,arm)] = prev
    means = {}
    for arm in ARMS:
        means[arm] = {}
        for metric in ('region_macro_ap', 'pooled_ap', 'pooled_auroc'):
            a = np.array([run_metrics[f'{arm}/seed_{s}'][metric] for s in (0,1,2)])
            means[arm][metric] = {'mean':float(a.mean()), 'seed_sd':float(a.std(ddof=1)), 'per_seed':a.tolist()}
    pair_results = {}
    for after,before in PAIR:
        values = np.array([run_metrics[f'{after}/seed_{s}']['region_macro_ap'] - run_metrics[f'{before}/seed_{s}']['region_macro_ap'] for s in (0,1,2)])
        pair_results[f'{after} - {before}'] = {'macro_ap_delta_pp':float(values.mean()*100), 'per_seed_pp':(values*100).tolist()}
    report = {'status':'passed', 'scope':'development validation only, no test', 'model_runs':12, 'val_windows':len(ys),
              'val_target_mmsi':len(set(groups)), 'max_checkpoint_replay_probability_difference':max_diff,
              'checks':['artifact hashes and completion', 'all 12 checkpoints replayed', 'CSV vs stored per-epoch predictions',
                        'all window identities', 'independent region-macro AP and pooled AP recomputation'],
              'arms':means, 'paired_changes':pair_results, 'per_seed':run_metrics,
              'caveat':'Known development validation used repeatedly. Causal input changes gating and quality indicators; training bundle uses background state. Comparisons do not establish credible-context effects. The reused trainer inserts a generic Historical35 definition string in individual result.json files; this experiment uses the masked Transformer architecture specified in protocol.json and train_transformer.py, not the Historical35 backbone.'}
    path = OUT/'independent_audit.json'
    with path.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':report['status'],'replay_max_abs':max_diff,'arms':means,'pairs':pair_results},indent=2),flush=True)

if __name__ == '__main__': main()
