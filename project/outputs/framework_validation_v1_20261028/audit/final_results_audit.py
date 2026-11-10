"""Independent acceptance of locked final checkpoints, predictions and core summaries.

Read-only on all core artifacts; writes only audit/. No training, configuration
selection, model evaluation or test-source construction is performed.
"""
from pathlib import Path
import collections
import hashlib
import json
import math
import sys
import warnings
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import models
import summarize_results as summary

ARCH = ['tcn', 'gru', 'transformer', 'itransformer', 'patchtst', 'mamba', 'neuralcde', 'graph_temporal']
ARMS = ['C0', 'C1', 'C2', 'C3', 'C4']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def js(p): return json.loads(Path(p).read_text(encoding='utf8'))


def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def close(a, b, tol=1e-12):
    assert np.isclose(a, b, atol=tol, rtol=0, equal_nan=True), (a, b, abs(a-b))


def state_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def ap_oracle(y, p, weights):
    """Independent tie grouping: group weighted mass first, then cumulative precision."""
    order = np.argsort(p, kind='stable')[::-1]
    score, label = np.asarray(p)[order], np.asarray(y)[order]
    first = np.r_[0, np.flatnonzero(score[1:] != score[:-1]) + 1]
    w = np.asarray(weights, float)[:, order]
    pos = np.add.reduceat(w * label, first, axis=1)
    group_mass = np.add.reduceat(w, first, axis=1)
    cum_pos = np.cumsum(pos, axis=1); cum_mass = np.cumsum(group_mass, axis=1)
    precision = np.divide(cum_pos, cum_mass, out=np.zeros_like(cum_pos), where=cum_mass > 0)
    total_pos = pos.sum(1)
    return np.divide((pos * precision).sum(1), total_pos, out=np.full(len(w), np.nan), where=total_pos > 0)


def ap_tests():
    rng = np.random.default_rng(4927)
    cases = 0; max_error = 0.
    for n in [2, 3, 5, 17, 60, 200]:
        for trial in range(20):
            y = rng.integers(0, 2, n); y[0] = 0; y[-1] = 1
            p = rng.integers(0, 6, n) / 5  # many ties
            for continuous in [False, True]:
                w = rng.random((6, n)) if continuous else rng.integers(0, 5, (6, n)).astype(float)
                w[0] = 1
                got = summary.ap_matrix(y, p, w); independent = ap_oracle(y, p, w)
                np.testing.assert_allclose(got, independent, rtol=0, atol=1e-12, equal_nan=True)
                for row, estimate in zip(w, got):
                    if (row * y).sum() == 0:
                        assert np.isnan(estimate); continue
                    reference = average_precision_score(y, p, sample_weight=row)
                    error = abs(estimate - reference); max_error = max(error, max_error)
                    assert error < 1e-12
                    cases += 1
    return {'synthetic_weighted_tied_AP_comparisons_to_sklearn': cases, 'max_error': max_error,
            'zero_positive_mass_policy': 'NaN in ap_matrix, explicitly excluded from descriptive bootstrap'}


def independent_metrics(y, p, region):
    per = {}
    for r in np.unique(region):
        ix = region == r
        per[str(r)] = {'n': int(ix.sum()), 'positive': int(y[ix].sum()), 'AP': float(average_precision_score(y[ix], p[ix])) if y[ix].sum() else None}
    return {'pooled_AP': float(average_precision_score(y, p)), 'AUROC': float(roc_auc_score(y, p)),
            'macro_AP': float(np.mean([v['AP'] for v in per.values() if v['AP'] is not None])), 'per_region': per}


def verify_metrics(recomputed, stored):
    for key in ['pooled_AP', 'AUROC', 'macro_AP']: close(recomputed[key], stored[key])
    assert recomputed['per_region'].keys() == stored['per_region'].keys()
    for r, values in recomputed['per_region'].items():
        assert values['n'] == stored['per_region'][r]['n']
        assert values['positive'] == stored['per_region'][r]['positive']
        if values['AP'] is not None: close(values['AP'], stored['per_region'][r]['AP'])


def operating(y, p, threshold):
    negative = y == 0; positive = y == 1; decisions = p > threshold
    fp = int(decisions[negative].sum()); tp = int(decisions[positive].sum())
    return {'threshold': threshold, 'FAR': fp / negative.sum(), 'recall': tp / positive.sum(),
            'false_positive': fp, 'true_positive': tp, 'negative_count': int(negative.sum()), 'positive_count': int(positive.sum())}


def check_operating(y, p, thresholds, stored):
    assert set(thresholds) == set(stored)
    for key, th in thresholds.items():
        actual = operating(y, p, th['threshold'])
        for k, v in actual.items(): close(v, stored[key][k])


def check_metadata(pred, data):
    for key in ['y', 'mmsi', 'window_id', 'source_id', 'base_window_id', 'region', 'split', 'time_start_s', 'time_end_s']:
        np.testing.assert_array_equal(pred[key], data[key])
    for key in ['case', 'noise_type', 'noise_scope', 'behavior_type', 'target_noise_effective', 'neighbor_noise_effective']:
        if key in data: np.testing.assert_array_equal(pred[key], data[key])
    assert pred['score'].shape == data['y'].shape and np.isfinite(pred['score']).all()
    assert np.all((pred['score'] >= 0) & (pred['score'] <= 1))


def main():
    torch.set_num_threads(2)
    lock = js(ROOT / 'code_lock.json'); checkpoint_lock = js(ROOT / 'checkpoint_lock.json')
    assert checkpoint_lock['count'] == 240 and len(checkpoint_lock['checkpoints']) == 240
    assert checkpoint_lock['code_lock_sha256'] == sha(ROOT / 'code_lock.json')
    for name, expected in {**lock['files'], **lock['data_files']}.items(): assert sha(ROOT / name) == expected, name
    protocol = js(ROOT / 'protocol.json'); assert protocol['training']['epochs'] == 25
    assert 'fixed final epoch25' in protocol['training']['checkpoint_selection']
    datasets = {t: {s: load(ROOT / 'data' / t / f'{s}.npz') for s in ['val', 'test']} for t in ['individual', 'relation']}
    initial_registry = {}; rows = []; predictions = {}; best_not_final = 0; distinct_best_weights = 0
    for task in ['individual', 'relation']:
        for arch in ARCH:
            for seed in range(3):
                torch.manual_seed(10000 + seed)
                initialized = models.make_model(arch, 12 if task == 'individual' else 7, max_len=20, hidden=64)
                expected_initial = state_hash(initialized.state_dict()); expected_parameters = sum(p.numel() for p in initialized.parameters())
                group_norms = []
                for arm in ARMS:
                    folder = ROOT / 'runs' / task / arch / f'seed_{seed}' / arm
                    r = js(folder / 'result.json'); config = js(folder / 'configuration.json'); test = js(folder / 'test_results.json')
                    expected_identity = {'task': task, 'backbone': arch, 'seed': seed, 'arm': arm}
                    for key, value in expected_identity.items():
                        assert r[key] == config[key] == test[key] == value
                    for obj in [r, config, test]:
                        assert obj['code_lock_sha256'] == sha(ROOT / 'code_lock.json')
                        assert obj['data_manifest_sha256'] == sha(ROOT / 'data_manifest.json')
                    assert r['epochs'] == 25
                    assert r['initial_state_sha256'] == config['initial_state_sha256'] == expected_initial
                    assert r['parameters'] == config['parameters'] == expected_parameters
                    initial_registry[(task, arch, seed)] = expected_initial
                    assert sha(folder / 'configuration.json') == r['configuration_sha256']
                    assert sha(folder / 'validation_base.npz') == r['validation_predictions_sha256']
                    cp_rel = str((folder / 'final.pt').relative_to(ROOT)).replace('\\', '/')
                    assert sha(folder / 'final.pt') == checkpoint_lock['checkpoints'][cp_rel] == r['checkpoint_sha256'] == test['checkpoint_sha256']
                    final = torch.load(folder / 'final.pt', map_location='cpu', weights_only=False)
                    resume = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
                    best = torch.load(folder / 'best_validation.pt', map_location='cpu', weights_only=False)
                    history = js(folder / 'history.json')
                    assert final['epoch'] == resume['epoch'] == 25
                    assert final['config'] == config and final['normalization'] == r['normalization']
                    assert final['thresholds'] == r['thresholds']
                    assert state_hash(final['model']) == state_hash(resume['model'])
                    assert [x['epoch'] for x in history] == list(range(1, 26))
                    assert resume['history'] == history
                    key = 'macro_AP' if task == 'individual' else 'pooled_AP'
                    optimum = max(x['validation'][key] for x in history)
                    earliest = next(x['epoch'] for x in history if x['validation'][key] == optimum)
                    assert best['epoch'] == earliest
                    best_not_final += int(best['epoch'] != 25)
                    distinct_best_weights += int(state_hash(best['model']) != state_hash(final['model']))
                    group_norms.append(r['normalization'])
                    vp = load(folder / 'validation_base.npz'); check_metadata(vp, datasets[task]['val'])
                    vm = independent_metrics(vp['y'], vp['score'], vp['region'])
                    verify_metrics(vm, r['validation']); verify_metrics(vm, history[-1]['validation'])
                    negatives = np.sort(vp['score'][vp['y'] == 0]); thresholds = {}
                    for alpha in protocol['evaluation']['false_alarm_budgets']:
                        allowed = math.floor(alpha * len(negatives)); threshold = float(negatives[max(0, len(negatives) - allowed - 1)])
                        th = {'threshold': threshold, 'validation_negative_count': len(negatives), 'validation_false_positives': int((negatives > threshold).sum()), 'nominal_FAR': alpha}
                        assert th == r['thresholds'][str(alpha)]
                        thresholds[str(alpha)] = th
                    check_operating(vp['y'], vp['score'], thresholds, r['validation_operating'])
                    for view in protocol['views'][task]:
                        p = load(folder / f'test_{view}.npz'); check_metadata(p, datasets[task]['test'])
                        stored = test['views'][view]
                        assert sha(folder / f'test_{view}.npz') == stored['predictions_sha256']
                        actual = independent_metrics(p['y'], p['score'], p['region'])
                        verify_metrics(actual, stored); check_operating(p['y'], p['score'], thresholds, stored['operating'])
                        row = {**expected_identity, 'view': view, 'primary_AP': actual[key], 'pooled_AP': actual['pooled_AP'], 'macro_AP': actual['macro_AP'], 'AUROC': actual['AUROC'],
                               'validation_primary': vm[key], 'parameters': expected_parameters, 'train_seconds': r['train_elapsed_seconds']}
                        for budget, th in thresholds.items():
                            op = operating(p['y'], p['score'], th['threshold']); row['FAR_' + budget] = op['FAR']; row['Recall_' + budget] = op['recall']
                        rows.append(row); predictions[(task, arch, view, arm, seed)] = p['score']
                assert all(n == group_norms[0] for n in group_norms)
            print('RUNS_AUDITED', task, arch, flush=True)
    assert len(initial_registry) == 48 and len(rows) == 480
    per_seed = js(ROOT / 'analysis/per_seed_metrics.json'); aggregated = js(ROOT / 'analysis/aggregate_metrics.json'); paired = js(ROOT / 'analysis/paired_bootstrap.json')
    row_lookup = {(r['task'], r['backbone'], r['view'], r['arm'], r['seed']): r for r in rows}
    for r in per_seed:
        found = row_lookup[(r['task'], r['backbone'], r['view'], r['arm'], r['seed'])]
        for key, value in found.items():
            if isinstance(value, (int, float)): close(value, r[key])
            else: assert value == r[key]
    for a in aggregated:
        selected = [row_lookup[(a['task'], a['backbone'], a['view'], a['arm'], seed)] for seed in range(3)]
        for key, stats in a.items():
            if not isinstance(stats, dict): continue
            v = np.array([r[key] for r in selected], float)
            close(stats['mean'], v.mean()); close(stats['std'], v.std(ddof=1))
    weighted_tests = ap_tests(); actual_sk_cases = 0; maximum_sk_error = 0.; bootstrap_registry = {}
    paired_lookup = {(p['task'], p['backbone'], p['view'], p['comparison']): p for p in paired}
    for task in ['individual', 'relation']:
        data = datasets[task]['test']; y = data['y']; region = data['region']; unique, inverse = np.unique(data['mmsi'].astype(str), return_inverse=True)
        boot = load(ROOT / 'analysis' / f'bootstrap_{task}.npz'); counts = boot['cluster_counts']; np.testing.assert_array_equal(unique, boot['unique_mmsi'])
        assert np.all(counts.sum(1) == len(unique)); assert len(counts) == 2000
        rng = np.random.default_rng(772026 if task == 'individual' else 882026)
        expected_counts = np.asarray([np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique)) for _ in range(2000)], np.int16)
        np.testing.assert_array_equal(counts, expected_counts)
        sample_draws = np.random.default_rng(937).choice(2000, size=5, replace=False)
        bootstrap_registry[task] = {'clusters': len(unique), 'draws': len(counts), 'seeded_counts_exactly_reproduced': True}
        for arch in ARCH:
            for view in protocol['views'][task]:
                distributions = {}
                for arm in ARMS:
                    values = []
                    for seed in range(3):
                        score = predictions[(task, arch, view, arm, seed)]
                        small_w = counts[sample_draws][:, inverse].astype(float)
                        custom = summary.ap_matrix(y, score, small_w)
                        for j in range(len(small_w)):
                            reference = average_precision_score(y, score, sample_weight=small_w[j]); close(custom[j], reference)
                            maximum_sk_error = max(maximum_sk_error, abs(custom[j] - reference)); actual_sk_cases += 1
                        batches = []
                        for start in range(0, len(counts), 100):
                            w = counts[start:start + 100, inverse].astype(float)
                            if task == 'individual':
                                v = np.mean([ap_oracle(y[region == r], score[region == r], w[:, region == r]) for r in np.unique(region)], axis=0)
                            else: v = ap_oracle(y, score, w)
                            batches.extend(v.tolist())
                        values.append(np.asarray(batches))
                    distributions[arm] = np.mean(values, axis=0)
                for enhance, base in [('C3', 'C0'), ('C3', 'C2'), ('C4', 'C3'), ('C2', 'C1')]:
                    p = paired_lookup[(task, arch, view, enhance + '-' + base)]
                    d = distributions[enhance] - distributions[base]; d = d[np.isfinite(d)]
                    seed_deltas = np.array([row_lookup[(task, arch, view, enhance, seed)]['primary_AP'] - row_lookup[(task, arch, view, base, seed)]['primary_AP'] for seed in range(3)]) * 100
                    np.testing.assert_allclose(p['seed_deltas_pp'], seed_deltas, rtol=0, atol=1e-12)
                    close(p['mean_delta_pp'], seed_deltas.mean()); assert p['valid_bootstrap_draws'] == len(d)
                    np.testing.assert_allclose(p['ci95_pp'], np.quantile(d, [.025, .975]) * 100, rtol=0, atol=1e-10)
                print('BOOTSTRAP_AUDITED', task, arch, view, flush=True)
    pattern = {}
    for task in ['individual', 'relation']:
        for comparison in ['C3-C0', 'C3-C2', 'C4-C3']:
            rr = [p for p in paired if p['task'] == task and p['view'] == 'base' and p['comparison'] == comparison]
            pattern[task + '_base_' + comparison] = {'positive_backbones': sum(p['mean_delta_pp'] > 0 for p in rr), 'negative_backbones': sum(p['mean_delta_pp'] < 0 for p in rr),
                'backbones': {p['backbone']: {'delta_pp': p['mean_delta_pp'], 'ci95_pp': p['ci95_pp']} for p in rr}}
    result = {'status': 'passed', 'scope': 'Independent final-result audit; no retraining or model selection', 'runs': 240, 'initialization_groups': 48,
              'validation_prediction_files': 240, 'test_prediction_files': 480, 'aggregate_cells': len(aggregated), 'paired_cells': len(paired),
              'fixed_final_epoch': 25, 'runs_whose_best_validation_epoch_was_not_final': best_not_final, 'runs_with_distinct_best_and_final_weights': distinct_best_weights,
              'hashes_checked': 'All frozen source/data files, all 240 checkpoint-lock final weights, all configuration/validation/test prediction hashes',
              'metrics_recomputed': 'All validation and test AP, AUROC, region AP, validation-derived thresholds, actual FAR and recall',
              'weighted_ap_tests': {**weighted_tests, 'actual_cluster_weighted_comparisons_to_sklearn': actual_sk_cases, 'actual_max_error': maximum_sk_error},
              'bootstrap': bootstrap_registry, 'paired_intervals_independently_recomputed': True,
              'source_sha256': {'script': sha(__file__), 'summarizer': sha(ROOT / 'summarize_results.py'), 'code_lock': sha(ROOT / 'code_lock.json'), 'checkpoint_lock': sha(ROOT / 'checkpoint_lock.json')},
              'result_patterns': pattern}
    (ROOT / 'audit/final_results_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k != 'result_patterns'}, indent=2))


if __name__ == '__main__': main()
