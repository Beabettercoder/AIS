"""Independent saved-result audit of the frozen affine-field extension, CPU only."""
from pathlib import Path
import json
import math
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from final_results_audit import ROOT, ARCH, sha, js, load, close, state_hash, models, ap_oracle, independent_metrics, verify_metrics, check_metadata, operating, check_operating
OUT = ROOT / 'field_extension'
FARMS = ['F0', 'F1', 'F2']


def main():
    torch.set_num_threads(2)
    lock = js(OUT / 'code_lock.json'); cp_lock = js(OUT / 'checkpoint_lock.json'); parent_lock = js(ROOT / 'code_lock.json')
    assert cp_lock['count'] == len(cp_lock['checkpoints']) == 72
    assert cp_lock['field_code_lock_sha256'] == sha(OUT / 'code_lock.json')
    for name, expected in {**lock['parent_files'], **lock['field_files']}.items(): assert sha(ROOT / name) == expected, name
    for name, expected in {**parent_lock['files'], **parent_lock['data_files']}.items(): assert sha(ROOT / name) == expected, name
    data = {s: load(ROOT / 'data/individual' / (s + '.npz')) for s in ['val', 'test']}
    field_test = load(ROOT / 'extra_data/field_test.npz')
    stress_path = ROOT / 'extra_data/field_stress_test.npz'; stress = load(stress_path)
    stress_protocol = js(ROOT / 'field_stress_protocol.json'); views = [r['name'] for r in stress_protocol['view_specs']]
    assert len(views) == 15; stress_sha = sha(stress_path); stress_protocol_sha = sha(ROOT / 'field_stress_protocol.json')
    for key in ['window_id', 'source_id', 'mmsi', 'y', 'region', 'mask', 'time_s', 'length']:
        np.testing.assert_array_equal(stress[key], data['test'][key])
    for arm in FARMS:
        for view in ['speed_l000', 'position_m000', 'combined_l000_m000']:
            np.testing.assert_array_equal(stress[view + '__' + arm], field_test['base__' + arm])
        for view in views:
            np.testing.assert_array_equal(stress[view + '__' + arm][..., :7], field_test['base__' + arm][..., :7])
    records = []; predictions = {}; stress_rows = []; trained_runs = 0; best_not_final = 0; zero_comparisons = 0
    for arch in ARCH:
        for seed in range(3):
            torch.manual_seed(10000 + seed); initial = models.make_model(arch, 12, max_len=20, hidden=64)
            initial_hash = state_hash(initial.state_dict()); parameters = sum(p.numel() for p in initial.parameters())
            parent = js(ROOT / 'runs/individual' / arch / f'seed_{seed}/C0/result.json')
            assert parent['initial_state_sha256'] == initial_hash
            for arm in ['C0', 'C2', 'C3'] + FARMS:
                base = OUT if arm.startswith('F') else ROOT
                folder = base / 'runs/individual' / arch / f'seed_{seed}' / arm
                r = js(folder / 'result.json'); cfg = js(folder / 'configuration.json'); tested = js(folder / 'test_results.json')
                p = load(folder / 'test_base.npz'); vp = load(folder / 'validation_base.npz')
                check_metadata(p, data['test']); check_metadata(vp, data['val'])
                assert r['initial_state_sha256'] == cfg['initial_state_sha256'] == initial_hash
                assert r['parameters'] == parameters and r['normalization'] == parent['normalization']
                for name, expected in [('backbone', arch), ('seed', seed), ('arm', arm)]:
                    assert r[name] == cfg[name] == tested[name] == expected
                assert sha(folder / 'configuration.json') == r['configuration_sha256']
                assert sha(folder / 'validation_base.npz') == r['validation_predictions_sha256']
                cp = folder / 'final.pt'; assert sha(cp) == r['checkpoint_sha256'] == tested['checkpoint_sha256']
                final = torch.load(cp, map_location='cpu', weights_only=False)
                assert final['epoch'] == r['epochs'] == 25 and final['config'] == cfg and final['normalization'] == parent['normalization']
                if arm.startswith('F'):
                    trained_runs += 1
                    assert r['code_lock_sha256'] == cfg['code_lock_sha256'] == tested['code_lock_sha256'] == sha(OUT / 'code_lock.json')
                    assert r['data_manifest_sha256'] == cfg['data_manifest_sha256'] == sha(OUT / 'data_manifest.json')
                    cp_rel = str(cp.relative_to(OUT)).replace('\\', '/')
                    assert cp_lock['checkpoints'][cp_rel] == sha(cp)
                    resume = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
                    best = torch.load(folder / 'best_validation.pt', map_location='cpu', weights_only=False)
                    history = js(folder / 'history.json')
                    assert resume['epoch'] == 25 and state_hash(final['model']) == state_hash(resume['model'])
                    assert [x['epoch'] for x in history] == list(range(1, 26)) and resume['history'] == history
                    maximum = max(x['validation']['macro_AP'] for x in history)
                    assert best['epoch'] == next(x['epoch'] for x in history if x['validation']['macro_AP'] == maximum)
                    best_not_final += int(best['epoch'] != 25)
                validation = independent_metrics(vp['y'], vp['score'], vp['region']); actual = independent_metrics(p['y'], p['score'], p['region'])
                verify_metrics(validation, r['validation']); verify_metrics(actual, tested['views']['base'])
                if arm.startswith('F'): verify_metrics(validation, history[-1]['validation'])
                assert sha(folder / 'test_base.npz') == tested['views']['base']['predictions_sha256']
                negatives = np.sort(vp['score'][vp['y'] == 0]); thresholds = r['thresholds']; assert final['thresholds'] == thresholds
                for key, th in thresholds.items():
                    alpha = float(key); threshold = float(negatives[max(0, len(negatives) - math.floor(alpha * len(negatives)) - 1)])
                    assert th == {'threshold': threshold, 'validation_negative_count': len(negatives), 'validation_false_positives': int((negatives > threshold).sum()), 'nominal_FAR': alpha}
                check_operating(vp['y'], vp['score'], thresholds, r['validation_operating'])
                check_operating(p['y'], p['score'], thresholds, tested['views']['base']['operating'])
                row = {'backbone': arch, 'arm': arm, 'seed': seed, 'macro_AP': actual['macro_AP'], 'pooled_AP': actual['pooled_AP'], 'AUROC': actual['AUROC'],
                       'validation_macro_AP': validation['macro_AP'], 'initial_state_sha256': initial_hash, 'parameters': parameters, 'train_seconds': r['train_elapsed_seconds']}
                for key, th in thresholds.items():
                    op = operating(p['y'], p['score'], th['threshold']); row['FAR_' + key] = op['FAR']; row['Recall_' + key] = op['recall']
                records.append(row); predictions[(arch, arm, seed)] = p['score']
                if not arm.startswith('F'): continue
                stress_dir = ROOT / 'field_stress_results' / arch / f'seed_{seed}' / arm; sr = js(stress_dir / 'result.json')
                assert sr['checkpoint_sha256'] == sha(cp) and sr['data_sha256'] == stress_sha and sr['protocol_sha256'] == stress_protocol_sha
                assert (sr['backbone'], sr['arm'], sr['seed']) == (arch, arm, seed) and set(sr['views']) == set(views)
                for view in views:
                    sp = load(stress_dir / (view + '.npz')); check_metadata(sp, data['test'])
                    assert sha(stress_dir / (view + '.npz')) == sr['views'][view]['predictions_sha256']
                    sm = independent_metrics(sp['y'], sp['score'], sp['region']); verify_metrics(sm, sr['views'][view])
                    check_operating(sp['y'], sp['score'], thresholds, sr['views'][view]['operating'])
                    if view in ['speed_l000', 'position_m000', 'combined_l000_m000']:
                        np.testing.assert_array_equal(sp['score'], p['score']); zero_comparisons += 1
                    stress_rows.append({'backbone': arch, 'arm': arm, 'seed': seed, 'view': view, 'macro_AP': sm['macro_AP'], 'pooled_AP': sm['pooled_AP']})
        print('FIELD_RUNS_STRESS_AUDITED', arch, flush=True)
    assert trained_runs == 72 and len(stress_rows) == 1080 and zero_comparisons == 216
    lookup = {(r['backbone'], r['arm'], r['seed']): r for r in records}
    stored_rows = js(OUT / 'analysis/per_seed_metrics.json'); aggregates = js(OUT / 'analysis/aggregate_metrics.json'); paired = js(OUT / 'analysis/paired_bootstrap.json')
    assert len(stored_rows) == len(records) == 144 and len(aggregates) == 48 and len(paired) == 40
    for r in stored_rows:
        expected = lookup[(r['backbone'], r['arm'], r['seed'])]
        for key, value in expected.items():
            if isinstance(value, (int, float)): close(value, r[key])
            else: assert r[key] == value
    for a in aggregates:
        for key, value in a.items():
            if not isinstance(value, dict): continue
            v = np.array([lookup[(a['backbone'], a['arm'], seed)][key] for seed in range(3)])
            close(value['mean'], v.mean()); close(value['std'], v.std(ddof=1))
    test = data['test']; y = test['y']; regions = test['region']; unique, inverse = np.unique(test['mmsi'].astype(str), return_inverse=True)
    boot = load(ROOT / 'analysis/bootstrap_individual.npz'); np.testing.assert_array_equal(boot['unique_mmsi'], unique)
    counts = boot['cluster_counts']; assert len(counts) == 2000
    rng = np.random.default_rng(772026); expected_counts = np.asarray([np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique)) for _ in range(2000)], np.int16)
    np.testing.assert_array_equal(counts, expected_counts)
    pair_lookup = {(p['backbone'], p['comparison']): p for p in paired}
    for arch in ARCH:
        dist = {}
        for arm in ['C0', 'C2', 'C3'] + FARMS:
            seeds = []
            for seed in range(3):
                score = predictions[(arch, arm, seed)]; draws = []
                for start in range(0, 2000, 100):
                    w = counts[start:start+100, inverse].astype(float)
                    draws.extend(np.mean([ap_oracle(y[regions == region], score[regions == region], w[:, regions == region]) for region in np.unique(regions)], axis=0))
                seeds.append(draws)
            dist[arm] = np.mean(seeds, axis=0)
        for positive, baseline in [('F1','F0'),('F1','C0'),('F2','C0'),('F2','F1'),('F1','C3')]:
            stored = pair_lookup[(arch, positive + '-' + baseline)]; d = dist[positive] - dist[baseline]; d = d[np.isfinite(d)]
            point = np.array([lookup[(arch, positive, seed)]['macro_AP'] - lookup[(arch, baseline, seed)]['macro_AP'] for seed in range(3)]) * 100
            np.testing.assert_allclose(stored['seed_deltas_pp'], point, atol=1e-12, rtol=0); close(stored['mean_delta_pp'], point.mean())
            np.testing.assert_allclose(stored['ci95_pp'], np.quantile(d, [.025,.975])*100, atol=1e-10, rtol=0)
            assert stored['valid_draws'] == len(d) and stored['clusters'] == len(unique)
        print('FIELD_INTERVALS_AUDITED', arch, flush=True)
    patterns = {}
    for comparison in ['F1-F0','F1-C0','F2-C0','F2-F1','F1-C3']:
        chosen = [p for p in paired if p['comparison'] == comparison]
        patterns[comparison] = {'positive_backbones': sum(p['mean_delta_pp'] > 0 for p in chosen), 'negative_backbones': sum(p['mean_delta_pp'] < 0 for p in chosen),
                                'values': {p['backbone']: {'delta_pp': p['mean_delta_pp'], 'ci95_pp': p['ci95_pp']} for p in chosen}}
    stress_patterns = {}
    for view in ['speed_l100','position_m600','combined_l100_m600']:
        stress_patterns[view] = {}
        for arch in ARCH:
            means = {arm: float(np.mean([r['macro_AP'] for r in stress_rows if r['backbone'] == arch and r['view'] == view and r['arm'] == arm])) for arm in FARMS}
            stress_patterns[view][arch] = {**means, 'F1-F0_pp': 100*(means['F1']-means['F0']), 'F2-F1_pp': 100*(means['F2']-means['F1'])}
    output = {'status': 'passed', 'field_training_runs': 72, 'parent_C0_initialization_groups': 24, 'fixed_final_epoch': 25,
              'field_validation_files': 72, 'field_base_test_files': 72, 'field_stress_prediction_files': 1080,
              'zero_view_prediction_arrays_exact': 216, 'summary_seed_rows_including_parent_comparators': 144,
              'aggregate_cells': 48, 'paired_intervals': 40, 'shared_parent_bootstrap_draws': 2000, 'target_MMSI_clusters': len(unique),
              'earlier_best_validation_checkpoints_retained_not_used': best_not_final,
              'hashes': 'All frozen parent/field files, 72 field finals and configurations, all validation/base/stress predictions checked',
              'metrics': 'AP/AUROC/per-region AP, validation-derived thresholds and all actual FAR/recall independently recomputed',
              'counterfactual_pairing': 'All source/window/target/time/label metadata matched; original per-arm target input retained at all 15 neighbor-stress views; all three zero views exactly replay own unstressed scores',
              'input_review_scope': 'Existing independent input reviews were not repeated; this audit targets result authenticity and paired outputs',
              'source_sha256': {'audit':sha(__file__),'field_code_lock':sha(OUT/'code_lock.json'),'field_checkpoint_lock':sha(OUT/'checkpoint_lock.json'),'summarizer':sha(ROOT/'summarize_field.py'),'stress_data':stress_sha,'stress_protocol':stress_protocol_sha},
              'base_result_patterns': patterns, 'stress_end_level_patterns': stress_patterns}
    (HERE / 'FIELD_FINAL_AUDIT.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({k:v for k,v in output.items() if k not in ['base_result_patterns','stress_end_level_patterns']}, indent=2))


if __name__ == '__main__': main()
