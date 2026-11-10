"""Independent saved-output and endpoint audit of the 15-view stress extension."""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from final_results_audit import ROOT, ARCH, ARMS, sha, js, load, close, check_metadata, independent_metrics, verify_metrics, check_operating


def main():
    cl = js(ROOT / 'checkpoint_lock.json'); protocol = js(ROOT / 'stress_extension_protocol.json')
    data_path = ROOT / 'extra_data/relation_stress_test.npz'; data = load(data_path)
    base = load(ROOT / 'data/relation/test.npz'); data_sha = sha(data_path); protocol_sha = sha(ROOT / 'stress_extension_protocol.json')
    views = protocol['view_names']; assert len(views) == 15
    for key in ['window_id', 'source_id', 'mmsi', 'y', 'region', 'mask', 'time_s', 'length']:
        np.testing.assert_array_equal(data[key], base[key])
    for arm in ARMS:
        for view in ['speed_l000', 'position_m000', 'combined_l000_m000']:
            np.testing.assert_array_equal(data[view + '__' + arm], base['base__' + arm])
        np.testing.assert_array_equal(data['speed_l100__' + arm], base['suffix__' + arm])
        for view in views:
            np.testing.assert_array_equal(data[view + '__' + arm][..., :4], base['base__C0'][..., :4])
    for arm in ['C1', 'C2']:
        for view in [v for v in views if v.startswith('position')]:
            np.testing.assert_array_equal(data[view + '__' + arm], base['base__' + arm])
    rows = []; files = 0; endpoint_files = 0
    for arch in ARCH:
        for seed in range(3):
            for arm in ARMS:
                original = ROOT / 'runs/relation' / arch / f'seed_{seed}' / arm
                folder = ROOT / 'stress_results' / arch / f'seed_{seed}' / arm
                result = js(folder / 'result.json'); core = js(original / 'result.json')
                assert result['backbone'] == arch and result['seed'] == seed and result['arm'] == arm
                cp = original / 'final.pt'; rel = str(cp.relative_to(ROOT)).replace('\\', '/')
                assert sha(cp) == result['checkpoint_sha256'] == cl['checkpoints'][rel]
                assert result['stress_protocol_sha256'] == protocol_sha and result['stress_data_sha256'] == data_sha
                assert set(result['views']) == set(views)
                base_scores = load(original / 'test_base.npz')['score']; suffix_scores = load(original / 'test_suffix.npz')['score']
                for view in views:
                    p = load(folder / (view + '.npz')); check_metadata(p, base)
                    assert sha(folder / (view + '.npz')) == result['views'][view]['predictions_sha256']
                    actual = independent_metrics(p['y'], p['score'], p['region']); verify_metrics(actual, result['views'][view])
                    check_operating(p['y'], p['score'], core['thresholds'], result['views'][view]['operating'])
                    if view in ['speed_l000', 'position_m000', 'combined_l000_m000'] or arm == 'C0' or (arm in ['C1', 'C2'] and view.startswith('position')):
                        np.testing.assert_array_equal(p['score'], base_scores); endpoint_files += 1
                    if view == 'speed_l100':
                        np.testing.assert_array_equal(p['score'], suffix_scores); endpoint_files += 1
                    rows.append({'backbone': arch, 'seed': seed, 'arm': arm, 'view': view, 'pooled_AP': actual['pooled_AP'], 'AUROC': actual['AUROC']})
                    files += 1
        print('STRESS_AUDITED', arch, flush=True)
    assert files == 1800
    patterns = {}
    for view in ['speed_l100', 'position_m600', 'combined_l100_m600']:
        patterns[view] = {}
        for arch in ARCH:
            score = {arm: np.mean([r['pooled_AP'] for r in rows if r['backbone'] == arch and r['view'] == view and r['arm'] == arm]) for arm in ['C2', 'C3']}
            patterns[view][arch] = {'C2': float(score['C2']), 'C3': float(score['C3']), 'C3-C2_pp': float((score['C3'] - score['C2']) * 100)}
    output = {'status': 'passed', 'runs': 120, 'views': 15, 'distinct_parameter_conditions': 13, 'prediction_files': files,
              'endpoint_or_invariant_prediction_assertions': endpoint_files,
              'checks': 'All frozen checkpoint/protocol/data/prediction hashes, metadata, AP/AUROC/region AP, original validation thresholds and actual FAR/recall; exact zero/speed-endpoint and C0/position-only C1/C2 score invariants',
              'script_sha256': sha(__file__), 'stress_data_sha256': data_sha, 'stress_protocol_sha256': protocol_sha,
              'end_level_paired_mean_AP_patterns': patterns}
    (HERE / 'stress_results_audit.json').write_text(json.dumps(output, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in output.items() if k != 'end_level_paired_mean_AP_patterns'}, indent=2))


if __name__ == '__main__': main()
