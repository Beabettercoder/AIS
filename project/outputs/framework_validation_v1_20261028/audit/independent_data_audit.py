"""Recompute deterministic source subsets independently; no training or score access."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

STAGE = Path(__file__).resolve().parents[1]
PRIOR = STAGE.parent / 'balanced_global_v2'
REGIONS = ('houston', 'ny', 'la', 'puget', 'great_belt', 'haugesund', 'kattegat', 'shanghai')
SPLITS = ('train', 'val', 'test')
KNOT = 1852 / 3600


def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def checksum(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def median_oracle(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    if v.ndim == 2:
        return np.array([median_oracle(v[:, c], w) for c in range(v.shape[1])])
    if w.sum() <= 1e-12: return float(np.median(v))
    positive = w > 0
    v, w = v[positive], w[positive]
    idx = np.argsort(v, kind='stable'); v, w = v[idx], w[idx]
    cumulative = np.cumsum(w) / w.sum()
    half = np.flatnonzero(np.abs(cumulative - .5) <= 1e-12)
    if len(half) and half[0] + 1 < len(v):
        j = half[0]
        return (v[j] + v[j + 1]) / 2
    return float(v[np.flatnonzero(cumulative >= .5)[0]])


def median_properties():
    import sys
    sys.path.insert(0, str(STAGE))
    from data_builder import weighted_median
    rng = np.random.default_rng(86421)
    for n in range(1, 65):
        v = rng.integers(-4, 5, (n, 2)).astype(float)
        np.testing.assert_array_equal(weighted_median(v, np.ones(n)), np.median(v, axis=0))
        for _ in range(5):
            w = rng.integers(0, 5, n).astype(float)
            expected = median_oracle(v, w)
            np.testing.assert_array_equal(weighted_median(v, w), expected)
            perm = rng.permutation(n)
            np.testing.assert_array_equal(weighted_median(v[perm], w[perm]), expected)
    for eps in (-1e-13, 0., 1e-13):
        assert weighted_median([0., 2.], [1. + eps, 1.]) == 1.


def structural(task):
    sets, records = {}, []
    target = 7 if task == 'individual' else 4
    views = ['base'] if task == 'individual' else ['base', 'natural', 'suffix']
    for split in SPLITS:
        p = STAGE / 'data' / task / f'{split}.npz'; d = load(p)
        n, t = d['mask'].shape
        assert np.all(d['split'] == split)
        assert np.array_equal(d['mask'].sum(1), d['length'])
        assert np.all(d['time_start_s'] == d['time_s'][:, 0])
        assert np.all(d['time_end_s'] == d['time_s'][np.arange(n), d['length'] - 1])
        for view in views:
            np.testing.assert_array_equal(d[view + '__C3'], d[view + '__C4'])
            assert not d[view + '__C0'][..., target:].any()
            for arm in ['C0', 'C1', 'C2', 'C3', 'C4']:
                x = d[view + '__' + arm]
                assert np.isfinite(x).all()
                assert not x[~d['mask']].any()
                np.testing.assert_array_equal(x[..., :target], d['base__C0'][..., :target])
            for arm in ['C2', 'C3', 'C4']:
                np.testing.assert_array_equal(d[view + '__C1'][..., -1], d[view + '__' + arm][..., -1])
        sets[split] = {key: set(d[key].astype(str)) for key in ['mmsi', 'source_id', 'window_id']}
        records.append({'split': split, 'windows': n, 'sources': len(sets[split]['source_id']),
                        'target_mmsi': len(sets[split]['mmsi']), 'sha256': checksum(p)})
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        for key in sets[a]: assert not sets[a][key] & sets[b][key], (task, a, b, key)
    return records


def individual_subset():
    records = []
    for split in SPLITS:
        new = load(STAGE / 'data/individual' / f'{split}.npz')
        lookup = {(str(r), str(w)): i for i, (r, w) in enumerate(zip(new['region'], new['window_id']))}
        for region in REGIONS:
            original_path = PRIOR / 'labeling/labeled_v1/features_scene' / region / f'{split}.npz'
            original = load(original_path)
            bg_root = PRIOR / ('causal_gate_main_v1_20261027/test_background' if split == 'test' else 'robust_background_v1_20261027')
            valid_windows = np.flatnonzero(original['context_valid'].any(1))[:3]
            selected = {str(original['window_id'][i]): i for i in valid_windows}
            checked = 0
            with gzip.open(bg_root / region / f'{split}_member_provenance.jsonl.gz', 'rt', encoding='utf8') as f:
                for line in f:
                    row = json.loads(line)
                    if row['window_id'] not in selected: continue
                    i, j = selected[row['window_id']], row['step']
                    q = lookup[(region, row['window_id'])]
                    members = row['members']; v = np.array([m['velocity_mps'] for m in members]); w = np.array([m['w_abs'] for m in members])
                    assert all(str(m['mmsi']) != str(original['mmsi'][i]) for m in members)
                    assert all(0 <= original['time_s'][i, j] - m['time_s'] <= 300 for m in members)
                    for arm, expected in [('C1', v.mean(0)), ('C2', np.median(v, axis=0)), ('C3', median_oracle(v, w))]:
                        np.testing.assert_array_equal(new['base__' + arm][q, j, 7:9], expected.astype('float32'))
                        target_speed = original['target_sog_kn'][i, j] * KNOT
                        angle = np.deg2rad(original['target_cog_deg'][i, j])
                        residual = target_speed * np.array([np.sin(angle), np.cos(angle)]) - expected.astype('float32')
                        np.testing.assert_allclose(new['base__' + arm][q, j, 9:11], residual, rtol=2e-6, atol=2e-6)
                    checked += 1
            for wid, i in selected.items():
                q = lookup[(region, wid)]; mask = original['mask'][i]
                assert np.array_equal(new['mask'][q], mask)
                assert np.array_equal(new['time_s'][q], original['time_s'][i])
                assert new['y'][q] == original['y'][i] and str(new['mmsi'][q]) == str(original['mmsi'][i])
                np.testing.assert_array_equal(new['base__C0'][q, mask, :2], (original['target_pos'][i] - original['target_pos'][i, 0])[mask].astype('float32'))
            records.append({'split': split, 'region': region, 'source_windows': len(selected), 'reference_steps_recomputed': checked, 'source_sha256': checksum(original_path)})
    return records


def relation_case_oracle(o, case):
    t = o['target_t'][case]; n = len(t); speeds = []; weights = []
    for lo, hi in zip(o['member_offsets'][:-1], o['member_offsets'][1:]):
        lo, hi = int(lo), int(hi)
        nt = o['neighbor_t'][case, lo:hi]; ns = o['neighbor_sog_kn'][case, lo:hi]
        nc = np.deg2rad(o['neighbor_cog_deg'][case, lo:hi]); pos = o['neighbor_pos'][case, lo:hi]
        v = KNOT * ns[:, None] * np.stack([np.sin(nc), np.cos(nc)], -1)
        w = np.zeros(len(nt))
        for k in range(1, len(nt)):
            dt = nt[k] - nt[k - 1]
            if not (0 < dt <= 300): continue
            if not np.isfinite(pos[k - 1:k + 1]).all() or not np.isfinite(v[k - 1:k + 1]).all(): continue
            err = np.linalg.norm((pos[k] - pos[k - 1]) / dt - (v[k] + v[k - 1]) / 2)
            w[k] = np.exp(-max(err - 2.5, 0) ** 2 / 8)
        gs = np.full(n, np.nan); gw = np.zeros(n)
        for j, now in enumerate(t):
            eligible = np.flatnonzero((nt <= now) & (now - nt <= 300))
            if len(eligible):
                k = eligible[-1]; gs[j] = ns[k]; gw[j] = w[k]
        anchor = max(2, n // 2 - 1)
        assert np.isfinite(gs[:anchor + 1]).sum() >= 2
        assert np.isfinite(gs[anchor + 1:][-3:]).sum() >= 2
        speeds.append(gs); weights.append(gw)
    speeds, weights = np.asarray(speeds), np.asarray(weights)
    contexts = {arm: np.zeros((n, 3), 'float32') for arm in ['C1', 'C2', 'C3']}
    for j in range(n):
        ok = np.isfinite(speeds[:, j]); v, w = speeds[ok, j], weights[ok, j]
        if not len(v): continue
        for arm, value in [('C1', v.mean()), ('C2', np.median(v)), ('C3', median_oracle(v, w))]:
            contexts[arm][j, 0] = value / 10
            contexts[arm][j, 2] = len(v) / 8
    for c in contexts.values(): c[1:, 1] = c[1:, 0] - c[:-1, 0]
    return contexts


def relation_subset():
    root = PRIOR / 'relation_dual_pressure_v2_20261027'
    manifest = json.loads((root / 'dataset_manifest_combined.json').read_text())
    records = []
    for split in SPLITS:
        new = load(STAGE / 'data/relation' / f'{split}.npz')
        lookup = {(str(w), int(c)): i for i, (w, c) in enumerate(zip(new['source_window_id'], new['case']))}
        rows = sorted([r for r in manifest['samples'] if r['version'] == 'relaxed' and r['split'] == split], key=lambda r: (r['region'], r['window_id']))
        selected = [r for region in REGIONS for r in [z for z in rows if z['region'] == region][:2]]
        for row in selected:
            for view in ['base', 'natural', 'suffix']:
                entry = row['files'][view]; p = root / entry['path']; assert checksum(p) == entry['sha256']; o = load(p)
                for case in range(4):
                    q = lookup[(row['window_id'], case)]; n = len(o['target_t'][case]); expected = relation_case_oracle(o, case)
                    assert new['y'][q] == o['y'][case]
                    np.testing.assert_array_equal(new['time_s'][q, :n], o['target_t'][case])
                    for arm, context in expected.items():
                        np.testing.assert_array_equal(new[view + '__' + arm][q, :n, 4:], context)
                records.append({'split': split, 'region': row['region'], 'source': row['window_id'], 'view': view, 'cases_recomputed': 4, 'source_sha256': checksum(p)})
    return records


def main():
    import sys
    if '--individual-only' in sys.argv:
        result = {'status': 'passed', 'structural': structural('individual'), 'reconstruction': individual_subset()}
        (STAGE / 'audit' / 'independent_individual_results.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        print(json.dumps({'status': 'passed', 'subsets': len(result['reconstruction']),
                          'steps': sum(r['reference_steps_recomputed'] for r in result['reconstruction'])}))
        return
    median_properties()
    result = {'status': 'passed', 'scope': 'All cached schema/mask/split invariants plus deterministic source subsets independently reconstructed; no model scores',
              'builder_sha256': checksum(STAGE / 'data_builder.py'), 'audit_sha256': checksum(__file__),
              'median_properties': 'equal weight, scalar/vector, zero weight, ties, permutation, bidirectional near-half tolerance',
              'structural': {t: structural(t) for t in ['individual', 'relation']},
              'individual_source_reconstruction': individual_subset(), 'relation_source_reconstruction': relation_subset()}
    (STAGE / 'audit' / 'independent_data_results.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({'status': result['status'], 'structural': result['structural'],
                      'individual_regions_splits': len(result['individual_source_reconstruction']),
                      'individual_steps_reconstructed': sum(r['reference_steps_recomputed'] for r in result['individual_source_reconstruction']),
                      'relation_cases_reconstructed': sum(r['cases_recomputed'] for r in result['relation_source_reconstruction'])}, indent=2))


if __name__ == '__main__': main()
