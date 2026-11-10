"""Independent AP differences and paired intervals for every stored stress-figure cell."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from final_results_audit import ROOT, load, js, ap_oracle, close, sha


def main():
    data = load(ROOT / 'data/relation/test.npz'); y = data['y']
    unique, inverse = np.unique(data['mmsi'].astype(str), return_inverse=True)
    bootstrap = load(ROOT / 'analysis/bootstrap_relation.npz'); np.testing.assert_array_equal(unique, bootstrap['unique_mmsi'])
    counts = bootstrap['cluster_counts']; rows = js(ROOT / 'analysis/stress_curve_pairs.json')
    assert len(rows) == 45
    for row in rows:
        predictions = {arm: [load(ROOT / 'stress_results' / row['backbone'] / f'seed_{seed}' / arm / (row['view'] + '.npz'))['score'] for seed in range(3)] for arm in ['C2', 'C3']}
        means = {arm: np.mean([average_precision_score(y, p) for p in pp]) for arm, pp in predictions.items()}
        close(row['delta_pp'], (means['C3'] - means['C2']) * 100)
        diffs = []
        for start in range(0, len(counts), 100):
            w = counts[start:start + 100, inverse].astype(float)
            arm_AP = {arm: np.mean([ap_oracle(y, p, w) for p in pp], axis=0) for arm, pp in predictions.items()}
            diffs.extend((arm_AP['C3'] - arm_AP['C2']).tolist())
        np.testing.assert_allclose(row['CI95'], np.quantile(np.asarray(diffs) * 100, [.025, .975]), atol=1e-10, rtol=0)
    result = {'status': 'passed', 'figure_cells': len(rows), 'bootstrap_draws_per_cell': len(counts), 'independent_tie_grouped_AP_oracle': True,
              'summary_sha256': sha(ROOT / 'analysis/stress_curve_pairs.json'), 'script_sha256': sha(__file__)}
    (ROOT / 'audit/stress_curve_audit.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
