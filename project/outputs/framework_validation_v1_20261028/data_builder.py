"""Observed-only shared AIS input adapter and source audit; never trains or reads scores.

All writes stay under --out. Source NPZs, provenance and prior builders are read-only.
Predictors are explicitly named {view}__C[0-4]; every other field is metadata.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
import time
import numpy as np

REGIONS = ('houston', 'ny', 'la', 'puget', 'great_belt', 'haugesund', 'kattegat', 'shanghai')
SPLITS = ('train', 'val', 'test')
VIEWS = ('base', 'natural', 'suffix')
KNOT = 1852 / 3600


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def arrsha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def read_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if keys is None else keys)}


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def weighted_median(values, weights, midpoint=True):
    """Coordinatewise weighted median with ordinary-median equal-mass convention.

    At half-mass boundaries take the adjacent positive-weight values' midpoint.
    Relative 1e-12 tolerance only resolves floating-point half-mass equality.
    Unknown reliability has weight zero; total <=1e-12 falls back to np.median.
    """
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.float64)
    assert len(values) and len(values) == len(weights)
    assert np.isfinite(values).all() and np.isfinite(weights).all() and (weights >= 0).all()
    total = float(weights.sum())
    if total <= 1e-12:
        return np.median(values, axis=0)
    if values.ndim == 2:
        return np.asarray([weighted_median(values[:, j], weights, midpoint) for j in range(values.shape[1])])
    order = np.argsort(values, kind='stable')
    if midpoint:
        order = order[weights[order] > 0]
    v, w = values[order], weights[order]
    mass = np.cumsum(w)
    if midpoint:
        # Check both sides of the numerical insertion boundary, including a
        # cumulative value infinitesimally below half due to summation order.
        ties = np.flatnonzero(np.abs(mass[:-1] - total / 2) <= total * 1e-12)
        if len(ties):
            at = int(ties[0])
            return float((v[at] + v[at + 1]) / 2)
    at = min(int(np.searchsorted(mass, total / 2, side='left')), len(v) - 1)
    return float(v[at])


def legacy_individual_median(values, weights):
    if float(np.sum(weights)) <= 1e-12:
        return np.median(values, axis=0)
    result = []
    for j in range(2):
        order = np.argsort(values[:, j], kind='stable')
        at = np.searchsorted(np.cumsum(weights[order]), sum(weights) / 2, side='left')
        result.append(values[order[min(int(at), len(order) - 1)], j])
    return np.asarray(result)


def legacy_relation_median(values, weights):
    order = np.argsort(values, kind='stable')
    values, weights = values[order], weights[order]
    if weights.sum() <= 1e-12:
        return float(np.median(values))
    at = min(int(np.searchsorted(np.cumsum(weights), weights.sum() / 2, side='left')), len(values) - 1)
    return float(values[at])


def check_median_contract():
    rng = np.random.default_rng(20261028)
    for n in range(1, 33):
        values = rng.normal(size=(n, 2))
        np.testing.assert_array_equal(weighted_median(values, np.ones(n)), np.median(values, axis=0))
    assert weighted_median([0, 2, 100], [1, 1, 0]) == 1
    assert weighted_median([0, 2], [0, 0]) == 1


class Builder:
    def __init__(self, root, out):
        self.root = root.resolve()
        self.out = out.resolve()
        self.inputs = {}
        self.summary = {'scope': 'development and previously exposed test; not blind confirmation',
                        'no_test_scores_read': True, 'no_new_labels_created': True, 'tasks': {}}

    def source(self, path, expected=None):
        path = Path(path).resolve()
        assert path.is_relative_to(self.root), path
        h = sha(path)
        if expected is not None:
            assert h == expected, (str(path), h, expected)
        if str(path) in self.inputs:
            assert self.inputs[str(path)] == h, ('source_changed', path)
        self.inputs[str(path)] = h
        return path

    def npz(self, path, keys=None, expected=None):
        return read_npz(self.source(path, expected), keys)

    def save_split(self, task, split, arrays):
        arrays['source_mmsi'] = arrays['mmsi'].copy()
        target_dim = 7 if task == 'individual' else 4
        feature_dim = 12 if task == 'individual' else 7
        for view in (('base',) if task == 'individual' else VIEWS):
            support = arrays['common_valid'] if task == 'individual' else arrays[view + '__C1'][..., 6] > 0
            for config in ('C0', 'C1', 'C2', 'C3', 'C4'):
                feature_mask = np.zeros((*arrays['mask'].shape, feature_dim), bool)
                feature_mask[..., :target_dim] = arrays['mask'][..., None]
                if config != 'C0':
                    feature_mask[..., target_dim:-1] = support[..., None]
                    feature_mask[..., -1] = arrays['mask']
                arrays[view + '__feature_mask_' + config] = feature_mask
        directory = self.out / 'data' / task
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (split + '.npz')
        for key, a in arrays.items():
            if a.dtype.kind in 'fc':
                assert np.isfinite(a).all(), (task, split, key)
        if path.exists():
            prior = read_npz(path)
            assert set(prior) == set(arrays), ('existing_partial_schema_differs', path)
            for key in arrays:
                np.testing.assert_array_equal(prior[key], arrays[key], err_msg=str(path) + ':' + key)
        else:
            np.savez_compressed(path, **arrays)
        return {'path': str(path.relative_to(self.out)), 'sha256': sha(path),
                'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': arrsha(v)}
                           for k, v in arrays.items()}}

    def raw7(self, d):
        n, length = d['mask'].shape
        raw = np.zeros((n, length, 7), np.float32)
        raw[..., :2] = (d['target_pos'] - d['target_pos'][:, 0:1]).astype(np.float32)
        angle = np.deg2rad(d['target_cog_deg'])
        speed = d['target_sog_kn']
        raw[..., 2] = speed * KNOT * np.sin(angle)
        raw[..., 3] = speed * KNOT * np.cos(angle)
        raw[..., 4] = speed
        raw[..., 5] = np.sin(angle)
        raw[..., 6] = np.cos(angle)
        raw[~d['mask']] = 0
        return raw

    def individual(self):
        base = self.root / 'labeling/labeled_v1/features_scene'
        robust = self.root / 'robust_background_v1_20261027'
        testbg = self.root / 'causal_gate_main_v1_20261027/test_background'
        for p in [robust / 'protocol.json', robust / 'build_backgrounds.py',
                  self.root / 'causal_gate_main_v1_20261027/build_backgrounds_test.py',
                  self.root / 'gating_training_v2_20261027/build_inputs.py']:
            self.source(p)
        metadata = ['window_id', 'base_window_id', 'mmsi', 'region', 'y', 'mask', 'time_s',
                    'behavior_type', 'noise_type', 'noise_scope', 'observation_noise',
                    'observation_noise_effective', 'target_noise_effective', 'neighbor_noise_effective']
        obskeys = ['target_pos', 'target_sog_kn', 'target_cog_deg', 'context_valid']
        allsets, output = {}, {}
        for split in SPLITS:
            parts, region_audit = [], {}
            bgids, bgreports = set(), set()
            for region in REGIONS:
                source = base / region / (split + '.npz')
                d = self.npz(source, metadata + obskeys)
                bgdir = (testbg if split == 'test' else robust) / region
                report = json.loads(self.source(bgdir / 'build_report.json').read_text())['splits'][split]
                assert self.inputs[str(source.resolve())] == report['input_sha256']
                b = self.npz(bgdir / (split + '.npz'), expected=report['output_sha256'])
                proofpath = self.source(bgdir / (split + '_member_provenance.jsonl.gz'), report['member_provenance_sha256'])
                assert np.array_equal(d['window_id'], b['window_id'])
                mask = d['mask'].astype(bool)
                valid = b['common_valid'].astype(bool)
                assert np.array_equal(valid, d['context_valid']) and not (valid & ~mask).any()
                raw = self.raw7(d)
                if split != 'test':
                    legacy = self.npz(self.root / 'gating_training_v2_20261027/data' / region / (split + '.npz'),
                                      ['raw_legacy7', 'window_id', 'mask', 'time_s', 'y'])
                    for k in ['window_id', 'mask', 'time_s', 'y']:
                        np.testing.assert_array_equal(legacy[k], d[k])
                    np.testing.assert_array_equal(legacy['raw_legacy7'], raw)
                n, length = mask.shape
                byid = {str(w): i for i, w in enumerate(d['window_id'])}
                assert len(byid) == n and np.all(d['region'] == region)
                seen = np.zeros_like(mask)
                newbg = np.zeros((n, length, 2), np.float32)
                counts = Counter()
                with gzip.open(proofpath, 'rt', encoding='utf8') as f:
                    for line in f:
                        row = json.loads(line)
                        i, j = byid[row['window_id']], int(row['step'])
                        assert valid[i, j] and not seen[i, j]
                        seen[i, j] = True
                        assert row['target_mmsi'] == str(d['mmsi'][i]) and row['time_s'] == d['time_s'][i, j]
                        members = row['members']
                        ids = [m['mmsi'] for m in members]
                        assert len(ids) == len(set(ids)) and str(d['mmsi'][i]) not in ids
                        assert len(ids) == int(b['plain_count'][i, j]) and len(ids) >= 4
                        assert sum(bool(m['historical_affine_accepted']) for m in members) == b['trusted_count'][i, j]
                        assert all(0 <= row['time_s'] - m['time_s'] <= 300 for m in members)
                        velocities = np.asarray([m['velocity_mps'] for m in members], float)
                        weights = np.asarray([m['w_abs'] for m in members], float)
                        assert all(m['kinematic_valid'] or m['w_abs'] == 0 for m in members)
                        for name, value in [('mean', velocities.mean(0)), ('median', np.median(velocities, axis=0)),
                                            ('weighted_median', legacy_individual_median(velocities, weights))]:
                            np.testing.assert_array_equal(np.asarray(value, np.float32), b['background_' + name + '_mps'][i, j])
                        newbg[i, j] = weighted_median(velocities, weights)
                        counts['common_steps'] += 1
                        counts['member_observations_with_step_reuse'] += len(members)
                        counts['zero_weight_fallback_steps'] += int(weights.sum() <= 1e-12)
                        counts['unknown_reliability_members'] += sum(not m['kinematic_valid'] for m in members)
                        bgids.update(ids)
                        bgreports.update((m['mmsi'], float(m['time_s'])) for m in members)
                np.testing.assert_array_equal(seen, valid)
                arrays = {k: d[k] for k in metadata}
                arrays['mmsi'] = arrays['mmsi'].astype(str)
                arrays['source_id'] = np.asarray([region + ':' + str(w) for w in d['base_window_id']])
                arrays['split'] = np.full(n, split)
                arrays['length'] = mask.sum(1).astype(np.int16)
                arrays['common_valid'] = valid
                arrays['plain_member_count'] = b['plain_count']
                arrays['time_start_s'] = d['time_s'][:, 0]
                arrays['time_end_s'] = d['time_s'][np.arange(n), arrays['length'] - 1]
                assert np.array_equal(mask, np.arange(length)[None, :] < arrays['length'][:, None])
                assert np.all(np.diff(d['time_s'], axis=1)[mask[:, 1:]] > 0)
                arrays['base__C0'] = np.concatenate([raw, np.zeros((n, length, 5), np.float32)], -1)
                angle = np.deg2rad(d['target_cog_deg'])
                target_v = d['target_sog_kn'][..., None] * KNOT * np.stack([np.sin(angle), np.cos(angle)], -1)
                for config, bg in [('C1', b['background_mean_mps']), ('C2', b['background_median_mps']), ('C3', newbg)]:
                    residual = np.where(valid[..., None], target_v - bg, 0).astype(np.float32)
                    bg = np.where(valid[..., None], bg, 0).astype(np.float32)
                    arrays['base__' + config] = np.concatenate([raw, bg, residual, valid[..., None]], -1).astype(np.float32)
                arrays['base__C4'] = arrays['base__C3'].copy()
                for config in ('C1', 'C2', 'C3', 'C4'):
                    np.testing.assert_array_equal(arrays['base__' + config][..., :7], raw)
                    assert not arrays['base__' + config][~mask].any()
                np.testing.assert_array_equal(arrays['base__C2'][..., 11], arrays['base__C3'][..., 11])
                counts['weighted_median_values_changed_by_midpoint'] = int(np.sum(newbg != b['background_weighted_median_mps']))
                counts['weighted_median_steps_changed_by_midpoint'] = int(np.any(newbg != b['background_weighted_median_mps'], axis=-1).sum())
                region_audit[region] = {'windows': n, 'unique_mmsi': len(set(arrays['mmsi'])), 'positives': int(d['y'].sum()),
                                        'valid_steps': int(mask.sum()), 'support_fraction': float(valid.sum() / mask.sum()), **dict(counts)}
                parts.append(arrays)
                print('individual', split, region, n, 'support', round(valid.sum() / mask.sum(), 4), flush=True)
            data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            assert len(data['y']) == {'train': 7840, 'val': 1680, 'test': 1680}[split]
            allsets[split] = split_sets(data, bgids, bgreports)
            output[split] = {'counts': dataset_counts(data), 'regions': region_audit, 'artifact': self.save_split('individual', split, data)}
        output['isolation'] = isolation(allsets)
        self.summary['tasks']['individual'] = output

    def relation(self):
        root = self.root / 'relation_dual_pressure_v2_20261027'
        manifestpath = self.source(root / 'dataset_manifest_combined.json')
        manifest = json.loads(manifestpath.read_text())
        for p in [root / 'protocol.json', root / 'model_study/model_study.py',
                  self.root / 'relation_robust_v1_20261027/run_relation.py', root / 'quartet_builder_v2.py']:
            self.source(p)
        output, allsets = {}, {}
        observed = ['target_t', 'target_sog_kn', 'target_cog_deg', 'target_mask', 'neighbor_t', 'neighbor_pos',
                    'neighbor_sog_kn', 'neighbor_cog_deg', 'neighbor_mask', 'neighbor_mmsi', 'member_offsets', 'cohort_mmsi']
        for split in SPLITS:
            cachemanifest = json.loads(self.source(root / 'model_study' / ('features_' + split + '_manifest.json')).read_text())
            assert cachemanifest['dataset_manifest_sha256'] == sha(manifestpath)
            cache = self.npz(root / 'model_study' / ('features_' + split + '.npz'), expected=cachemanifest['feature_file_sha256'])
            rows = sorted([x for x in manifest['samples'] if x['version'] == 'relaxed' and x['split'] == split],
                          key=lambda x: (x['region'], x['window_id']))
            n, maxlen = len(cache['y']), 20
            assert n == len(rows) * 4
            data = {k: cache[k] for k in ['y', 'length', 'region', 'mmsi', 'case']}
            data['source_id'] = np.asarray([r['region'] + ':' + r['window_id'] for r in rows for _ in range(4)])
            data['window_id'] = np.asarray([r['window_id'] + ':case' + str(c) for r in rows for c in range(4)])
            data['base_window_id'] = np.repeat(np.asarray([r['base_window_id'] for r in rows]), 4)
            data['source_window_id'] = cache['source']
            data['split'] = np.full(n, split)
            data['mask'] = np.arange(maxlen)[None, :] < data['length'][:, None]
            data['time_s'] = np.zeros((n, maxlen), np.float64)
            data['behavior_type'] = np.full(n, 'quartet_group_consistency')
            tags = [int(hashlib.sha256(('quartet-noise:' + r['window_id']).encode()).hexdigest()[:8], 16) % 4 for r in rows]
            data['noise_type'] = np.repeat(np.asarray(['none' if tag == 0 else 'teleport' for tag in tags]), 4)
            data['noise_scope'] = np.repeat(np.asarray([['none', 'target_only', 'neighbor_only', 'both'][tag] for tag in tags]), 4)
            data['target_noise_effective'] = np.repeat(np.asarray([tag in (1, 3) for tag in tags]), 4)
            data['neighbor_noise_effective'] = np.repeat(np.asarray([tag in (2, 3) for tag in tags]), 4)
            data['observation_noise_effective'] = data['target_noise_effective'] | data['neighbor_noise_effective']
            for view in VIEWS:
                for config in ('C0', 'C1', 'C2', 'C3', 'C4'):
                    data[view + '__' + config] = np.zeros((n, maxlen, 7), np.float32)
                data[view + '__stress_numeric_changed'] = np.repeat(np.asarray([False if view == 'base' else any(m['indices'][view]['numeric_changed'] for m in r['stress']['members']) for r in rows]), 4)
                data[view + '__stress_suffix_grid_hits'] = np.repeat(np.asarray([0 if view == 'base' else sum(m['indices'][view]['suffix_grid_hits'] for m in r['stress']['members']) for r in rows], np.int32), 4)
            bgids, bgreports, counts = set(), set(), Counter()
            member_digest = {v: hashlib.sha256() for v in VIEWS}
            for rindex, row in enumerate(rows):
                lo = 4 * rindex
                assert np.all(cache['source'][lo:lo + 4] == row['window_id'])
                assert np.all(cache['region'][lo:lo + 4] == row['region'])
                assert np.all(cache['mmsi'][lo:lo + 4] == str(row['mmsi']))
                np.testing.assert_array_equal(cache['case'][lo:lo + 4], np.arange(4))
                for view in VIEWS:
                    entry = row['files'][view]
                    o = self.npz(root / entry['path'], observed + ['y', 'source_window_id', 'target_mmsi', 'split', 'region'], entry['sha256'])
                    assert str(o['source_window_id']) == row['window_id'] and str(o['target_mmsi']) == str(row['mmsi'])
                    assert str(o['split']) == split and str(o['region']) == row['region']
                    np.testing.assert_array_equal(o['y'], [0, 1, 1, 0])
                    np.testing.assert_array_equal(o['y'], cache['y'][lo:lo + 4])
                    assert list(o['cohort_mmsi'].astype(str)) == row['cohort_mmsi']
                    assert str(row['mmsi']) not in o['cohort_mmsi']
                    if view == 'base':
                        bgids.update(o['cohort_mmsi'].astype(str))
                        bgreports.update((str(m), float(t)) for m, t in zip(o['neighbor_mmsi'], o['neighbor_t'][0]))
                    for case in range(4):
                        i, length = lo + case, len(o['target_t'][case])
                        assert length == data['length'][i] and o['target_mask'][case].all() and o['neighbor_mask'][case].all()
                        assert np.all(np.diff(o['target_t'][case]) > 0)
                        if view == 'base':
                            data['time_s'][i, :length] = o['target_t'][case]
                        else:
                            np.testing.assert_array_equal(data['time_s'][i, :length], o['target_t'][case])
                        features, proof, previous, reliability_counts = relation_features(o, case)
                        for key, value in reliability_counts.items():
                            counts[view + '_' + key] += value
                        member_digest[view].update(json.dumps([row['window_id'], case, proof], separators=(',', ':')).encode())
                        for config, mode in [('C0', 'target'), ('C1', 'mean'), ('C2', 'median')]:
                            np.testing.assert_array_equal(features[config], cache[view + '__' + mode][i, :length])
                        np.testing.assert_array_equal(previous, cache[view + '__absolute_median'][i, :length])
                        counts[view + '_midpoint_changed_values'] += int(np.sum(features['C3'] != previous))
                        for config in features:
                            data[view + '__' + config][i, :length] = features[config]
                    if (rindex + 1) % 200 == 0 and view == 'suffix':
                        print('relation', split, rindex + 1, '/', len(rows), flush=True)
            data['time_start_s'] = data['time_s'][:, 0]
            data['time_end_s'] = data['time_s'][np.arange(n), data['length'] - 1]
            for view in VIEWS:
                data[view + '__C4'] = data[view + '__C3'].copy()
                for config in ('C0', 'C1', 'C2', 'C3', 'C4'):
                    np.testing.assert_array_equal(data[view + '__' + config][..., :4], data['base__C0'][..., :4])
                    assert not data[view + '__' + config][~data['mask']].any()
                np.testing.assert_array_equal(data[view + '__C2'][..., 6], data[view + '__C3'][..., 6])
                tar = data[view + '__C0']
                np.testing.assert_array_equal(tar[0::4], tar[1::4])
                np.testing.assert_array_equal(tar[2::4], tar[3::4])
                for config in ('C1', 'C2', 'C3', 'C4'):
                    ctx = data[view + '__' + config][..., 4:]
                    np.testing.assert_array_equal(ctx[0::4], ctx[2::4])
                    np.testing.assert_array_equal(ctx[1::4], ctx[3::4])
            allsets[split] = split_sets(data, bgids, bgreports)
            output[split] = {'counts': dataset_counts(data), 'midpoint_changes': dict(counts),
                             'paired_members_and_report_times_sha256': {v: h.hexdigest() for v, h in member_digest.items()},
                             'artifact': self.save_split('relation', split, data)}
            print('relation_complete', split, n, flush=True)
        output['isolation'] = isolation(allsets)
        self.summary['tasks']['relation'] = output


def absolute_weights(t, p, speed, course):
    angle = np.deg2rad(course)
    velocity = speed[:, None] * KNOT * np.column_stack((np.sin(angle), np.cos(angle)))
    dt = np.diff(t)
    valid = np.r_[False, (dt > 0) & (dt <= 300) & np.isfinite(p[:-1]).all(1) & np.isfinite(p[1:]).all(1)
                  & np.isfinite(speed[:-1]) & np.isfinite(speed[1:]) & np.isfinite(course[:-1]) & np.isfinite(course[1:])]
    w = np.zeros(len(t))
    ii = np.flatnonzero(valid)
    error = np.linalg.norm((p[ii] - p[ii - 1]) / dt[ii - 1, None] - (velocity[ii] + velocity[ii - 1]) / 2, axis=1)
    w[ii] = np.exp(-np.maximum(0, error - 2.5) ** 2 / 8)
    return w, valid


def relation_features(o, case):
    t, speed, course = (o[k][case] for k in ('target_t', 'target_sog_kn', 'target_cog_deg'))
    n = len(t)
    anchor = max(2, n // 2 - 1)
    pre, post = np.arange(anchor + 1), np.arange(anchor + 1, n)[-3:]
    grid_s, grid_w, proof = [], [], []
    counts = Counter()
    for member, (lo, hi) in zip(o['cohort_mmsi'], zip(o['member_offsets'][:-1], o['member_offsets'][1:])):
        lo, hi = int(lo), int(hi)
        nt, ns, nc = (o[k][case, lo:hi] for k in ('neighbor_t', 'neighbor_sog_kn', 'neighbor_cog_deg'))
        assert np.all(np.diff(nt) > 0)
        weights, kinematic_valid = absolute_weights(nt, o['neighbor_pos'][case, lo:hi], ns, nc)
        ix = np.searchsorted(nt, t, side='right') - 1
        safe = np.maximum(ix, 0)
        present = (ix >= 0) & (t >= nt[safe]) & (t - nt[safe] <= 300)
        if present[pre].sum() < 2 or present[post].sum() < 2:
            continue
        gs, gw = np.full(n, np.nan), np.zeros(n)
        gs[present], gw[present] = ns[safe[present]], weights[safe[present]]
        grid_s.append(gs)
        grid_w.append(gw)
        counts['aligned_member_observations_with_step_reuse'] += int(present.sum())
        counts['aligned_unknown_reliability_observations'] += int((~kinematic_valid[safe[present]]).sum())
        proof.append([str(member), [[int(j), float(nt[safe[j]])] for j in np.flatnonzero(present)]])
    assert len(grid_s) == len(o['cohort_mmsi']) >= 3
    grid_s, grid_w = np.asarray(grid_s), np.asarray(grid_w)
    context = {k: np.zeros((n, 3), np.float32) for k in ('C1', 'C2', 'C3', 'previous')}
    for j in range(n):
        ok = np.isfinite(grid_s[:, j])
        values, weights = grid_s[ok, j], grid_w[ok, j]
        if not len(values):
            continue
        counts['context_steps'] += 1
        counts['zero_weight_fallback_steps'] += int(weights.sum() <= 1e-12)
        for config, value in [('C1', values.mean()), ('C2', np.median(values)),
                              ('C3', weighted_median(values, weights)), ('previous', legacy_relation_median(values, weights))]:
            context[config][j, 0] = value / 10
            context[config][j, 2] = len(values) / 8
    for ctx in context.values():
        ctx[1:, 1] = np.diff(ctx[:, 0])
    angle = np.deg2rad(course)
    target = np.column_stack((speed / 10, np.sin(angle), np.cos(angle), np.r_[0, np.diff(speed)] / 10)).astype(np.float32)
    features = {'C0': np.concatenate([target, np.zeros((n, 3), np.float32)], -1)}
    features.update({k: np.concatenate([target, context[k]], -1) for k in ('C1', 'C2', 'C3')})
    return features, proof, np.concatenate([target, context['previous']], -1), dict(counts)


def split_sets(data, bgids, bgreports):
    return {'target_mmsi': set(data['mmsi'].astype(str)), 'source_id': set(data['source_id'].astype(str)),
            'window_id': set(data['window_id'].astype(str)), 'background_mmsi': bgids, 'background_reports': bgreports,
            'target_reports': {(str(mid), float(stamp)) for mid, times, mask in zip(data['mmsi'], data['time_s'], data['mask']) for stamp in times[mask]}}


def isolation(sets):
    result = {'counts': {s: {k: len(v) for k, v in entry.items()} for s, entry in sets.items()}, 'pairs': {}}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        row = {k: len(sets[a][k] & sets[b][k]) for k in sets[a]}
        for k in ['target_mmsi', 'source_id', 'window_id', 'target_reports']:
            assert row[k] == 0, (a, b, k, row[k])
        row[a + '_target_in_' + b + '_background'] = len(sets[a]['target_mmsi'] & sets[b]['background_mmsi'])
        row[b + '_target_in_' + a + '_background'] = len(sets[b]['target_mmsi'] & sets[a]['background_mmsi'])
        result['pairs'][a + '_' + b] = row
    return result


def dataset_counts(data):
    return {'windows': len(data['y']), 'sources': len(set(data['source_id'])), 'unique_target_mmsi': len(set(data['mmsi'])),
            'positives': int(data['y'].sum()), 'valid_steps': int(data['mask'].sum()),
            'regions': {str(r): {'windows': int((data['region'] == r).sum()),
                                'sources': len(set(data['source_id'][data['region'] == r])),
                                'target_mmsi': len(set(data['mmsi'][data['region'] == r]))} for r in np.unique(data['region'])}}


def schema():
    return {'version': 'framework_context_v1_20261028_midpoint',
            'predictor_key_pattern': '{view}__C0 through {view}__C4; other keys are metadata only',
            'configurations': {'C0': 'target; context slots zero', 'C1': 'ordinary mean context',
                               'C2': 'ordinary coordinatewise median', 'C3': 'absolute reliability weighted coordinatewise median',
                               'C4': 'byte-identical C3 inference input; training extension belongs to training protocol'},
            'weighted_median': {'half_mass': 'midpoint of adjacent positive-weight values, relative tolerance 1e-12',
                                'equal_weights': 'exact ordinary np.median including even counts',
                                'zero_weight': 'sum <=1e-12 falls back to ordinary np.median'},
            'individual': {'views': ['base'], 'feature_dim': 12, 'target_dim': 7, 'support_index': 11,
                           'channels': ['relative_position_e_m', 'relative_position_n_m', 'reported_velocity_e_mps',
                                        'reported_velocity_n_mps', 'reported_sog_kn', 'sin_cog', 'cos_cog',
                                        'background_velocity_e_mps', 'background_velocity_n_mps',
                                        'residual_velocity_e_mps', 'residual_velocity_n_mps', 'common_valid'],
                           'support': 'historical reliability-accepted affine-valid support; SAME C1/C2/C3 mask, not broad plain-neighborhood coverage',
                           'members': 'plain_accepted latest report per non-target MMSI, <=300s old, <=8nm, heading difference <=45deg, observable motion',
                           'target': 'raw observed cumulative displacement, reported speed/course; no position correction',
                           'time': 'actual original target time_s; mask marks valid times; padded predictors zero'},
            'relation': {'views': list(VIEWS), 'feature_dim': 7, 'target_dim': 4, 'support_index': 6,
                         'channels': ['reported_sog_kn_div10', 'sin_cog', 'cos_cog', 'delta_sog_kn_div10',
                                      'background_sog_kn_div10', 'delta_background_sog_kn_div10', 'member_count_div8'],
                         'time': 'actual observed target_t restored from each quartet NPZ; all views must match',
                         'offline_semantics': 'fixed midpoint cohort and full-window pre/suffix eligibility; latest report alignment is causal conditional on this offline cohort; not online causal membership',
                         'schema_difference': '7-channel speed context preserved; not identical to individual 12-channel velocity/residual interface',
                         'noise_metadata': 'base positional-noise identity rule copied from frozen generator; additional observed-speed stress flags copied from manifest; never predictive'},
            'normalization': 'none here; fit training-only normalization in training code, then re-mask padding and inactive C0 context',
            'feature_mask': 'per-view per-config boolean availability mask: target slots use time mask, context values use common support, support/count slot uses time mask; C0 context and support slots false; normalization must preserve support/count value',
            'excluded_predictors': ['labels', 'IDs', 'noise metadata', 'factor states', 'source and pre-noise hidden trajectories']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert args.out.resolve().name == 'framework_validation_v1_20261028'
    check_median_contract()
    started = time.time()
    builder = Builder(args.root, args.out)
    builder.individual()
    builder.relation()
    # Prove all registered source artifacts stayed unchanged during the build.
    for path, checksum in builder.inputs.items():
        assert sha(path) == checksum, ('source_modified_during_build', path)
    builder.summary.update(status='passed', elapsed_seconds=time.time() - started,
                           builder_sha256=sha(__file__), median_equal_weight_contract='passed lengths 1..32 plus zero weights',
                           input_files=len(builder.inputs))
    write_json(args.out / 'data_schema.json', schema())
    write_json(args.out / 'data_sources.json', {'source_root': str(builder.root), 'sha256': builder.inputs})
    write_json(args.out / 'data_audit.json', builder.summary)
    write_json(args.out / 'data_manifest.json', {
        'status': 'complete_and_audited', 'builder_sha256': sha(__file__),
        'schema_sha256': sha(args.out / 'data_schema.json'),
        'sources_sha256': sha(args.out / 'data_sources.json'), 'audit_sha256': sha(args.out / 'data_audit.json'),
        'tasks': {task: {split: {'path': value[split]['artifact']['path'],
                               'sha256': value[split]['artifact']['sha256'], 'counts': value[split]['counts']}
                         for split in SPLITS} for task, value in builder.summary['tasks'].items()}})
    print('DATA_BUILD_COMPLETE', builder.summary['input_files'], round(builder.summary['elapsed_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
