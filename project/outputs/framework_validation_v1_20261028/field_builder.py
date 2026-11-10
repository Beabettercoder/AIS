"""New fixed local-linear field interface, with no model/score access."""
from pathlib import Path
from collections import Counter
import argparse
import gc
import gzip
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
REGIONS = ('houston', 'ny', 'la', 'puget', 'great_belt', 'haugesund', 'kattegat', 'shanghai')
SPLITS = ('train', 'val', 'test')
R = 8 * 1852.
SIGMA = 4 * 1852.
RIDGE = .001
REASONS = {'outside_core_support': 0, 'valid': 1, 'fewer_than_four': 2, 'zero_weights': 3,
           'ill_conditioned': 4, 'solve_failed': 5, 'nonfinite': 6}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in (z.files if keys is None else keys)}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def write_json(path, data):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def fit_field(relative_positions, velocity, weight, member_ids, target_id):
    keep = np.asarray(member_ids, str) != str(target_id)
    p, v, w = np.asarray(relative_positions, float)[keep], np.asarray(velocity, float)[keep], np.asarray(weight, float)[keep]
    ids = np.asarray(member_ids, str)[keep]
    assert len(set(ids)) == len(ids)
    result = {'valid': False, 'reason': 'fewer_than_four', 'condition': 0., 'coefficient': np.zeros((3, 2)),
              'removed_target_rows': int((~keep).sum())}
    if len(ids) < 4:
        return result
    if not (np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(w).all()) or (w < 0).any():
        result['reason'] = 'nonfinite'
        return result
    if w.sum() <= 1e-12:
        result['reason'] = 'zero_weights'
        return result
    w = w / w.sum()
    x = np.column_stack([np.ones(len(p)), p / R])
    gram = x.T @ (w[:, None] * x)
    condition = float(np.linalg.cond(gram))
    result['condition'] = condition if np.isfinite(condition) else 0.
    if not np.isfinite(condition) or condition > 1e8:
        result['reason'] = 'ill_conditioned'
        return result
    try:
        b = np.linalg.solve(gram + RIDGE * np.diag([0., 1., 1.]), x.T @ (w[:, None] * v))
    except np.linalg.LinAlgError:
        result['reason'] = 'solve_failed'
        return result
    if not np.isfinite(b).all():
        result['reason'] = 'nonfinite'
        return result
    result.update(valid=True, reason='valid', coefficient=b)
    return result


def independently_fit(p, v, w):
    w = np.asarray(w, float) / np.sum(w)
    x = np.column_stack([np.ones(len(p)), np.asarray(p) / R])
    a = np.vstack([np.sqrt(w)[:, None] * x, np.sqrt(RIDGE) * np.array([[0, 1, 0], [0, 0, 1]])])
    y = np.vstack([np.sqrt(w)[:, None] * v, np.zeros((2, 2))])
    return np.linalg.lstsq(a, y, rcond=None)[0]


def unit_checks():
    rng = np.random.default_rng(20261028)
    p, v = rng.normal(size=(8, 2)) * 3000, rng.normal(size=(8, 2)) * 4
    ids = np.asarray(['n' + str(j) for j in range(8)])
    distance = np.exp(-.5 * (np.linalg.norm(p, axis=1) / SIGMA) ** 2)
    a = fit_field(p, v, distance, ids, 'target')
    b = fit_field(p, v, distance * np.ones(8), ids, 'target')
    assert a['valid'] and b['valid']
    np.testing.assert_array_equal(a['coefficient'], b['coefficient'])
    np.testing.assert_allclose(a['coefficient'], independently_fit(p, v, distance), rtol=1e-10, atol=1e-10)
    injected_ids = np.concatenate([ids, np.asarray(['target'])])
    injected = fit_field(np.vstack([p, [0, 0]]), np.vstack([v, [1e6, -1e6]]), np.r_[distance, 1.], injected_ids, 'target')
    np.testing.assert_array_equal(a['coefficient'], injected['coefficient'])
    assert injected['removed_target_rows'] == 1
    unexcluded = fit_field(np.vstack([p, [0, 0]]), np.vstack([v, [1e6, -1e6]]), np.r_[distance, 1.], injected_ids, 'not_present')
    assert np.linalg.norm(a['coefficient'][0] - unexcluded['coefficient'][0]) > 1
    removed = fit_field(p, v, distance, ids, ids[0])
    assert removed['valid'] and removed['removed_target_rows'] == 1
    assert np.linalg.norm(a['coefficient'][0] - removed['coefficient'][0]) > 1e-6
    permutation = rng.permutation(len(p))
    shuffled = fit_field(p[permutation], v[permutation], distance[permutation], ids[permutation], 'target')
    np.testing.assert_allclose(a['coefficient'], shuffled['coefficient'], rtol=1e-10, atol=1e-10)
    return {'equal_absolute_weights': True, 'augmented_lstsq_reference': True, 'target_injection_ignored': True,
            'target_injection_without_exclusion_changes_fit': True, 'actual_neighbor_removal_changes_fixture_fit': True,
            'permutation_invariance': True}


def causal_replay(raw, w, mask, valid, times, cap, gate):
    out, branches = raw.copy(), np.zeros(mask.shape, np.int8)
    for i in range(len(raw)):
        for j in range(1, raw.shape[1]):
            if w[i, j] < .2:
                out[i, j, 2:7] = out[i, j - 1, 2:7]
        candidate, branch = gate.causal(raw[i], w[i], mask[i], valid[i], times[i], cap)
        out[i, :, :2], branches[i] = candidate[:, :2], branch
    return out, branches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2'))
    args = parser.parse_args()
    stage, root = args.stage.resolve(), args.root.resolve()
    assert stage.name == 'framework_validation_v1_20261028'
    outdir = stage / 'extra_data'
    outdir.mkdir(exist_ok=True)
    assert not (stage / 'field_audit.json').exists()
    for split in SPLITS:
        assert not (outdir / ('field_' + split + '.npz')).exists()
    started = time.time()
    protocolpath = stage / 'field_protocol.json'
    protocol = json.loads(protocolpath.read_text())
    hashes = {str(protocolpath): sha(protocolpath), str(Path(__file__).resolve()): sha(__file__)}
    def register(path, expected=None):
        path = Path(path).resolve()
        checksum = sha(path)
        if expected is not None:
            assert checksum == expected, path
        hashes[str(path)] = checksum
        return path
    for name, field in [('data_builder.py', 'core_builder_sha256'), ('data_manifest.json', 'core_data_manifest_sha256')]:
        register(stage / name, protocol[field])
    lock = json.loads(register(stage / 'code_lock.json').read_text())
    for name, checksum in {**lock['files'], **lock['data_files']}.items():
        register(stage / name, checksum)
    core = load_module(stage / 'data_builder.py', 'frozen_core_field')
    helper = load_module(register(root / 'context_adapter_v1_20261026/neighborhood/audit_neighborhood.py'), 'source_observations_field')
    gate_root = root / 'causal_gate_main_v1_20261027'
    gate = load_module(register(gate_root / 'gate_builder.py'), 'prior_causal_gate_field')
    register(gate_root / 'evaluate_test.py')
    register(root / 'context/context_features.py')
    cap_info = json.loads(register(gate_root / 'gate_cap.json').read_text())
    cap = cap_info['cap_mps']
    assert cap_info['train_only'] and cap == protocol['target_gate']['cap_mps']
    gate_manifest = json.loads(register(gate_root / 'gate_input_manifest.json').read_text())
    gate_map = {(r['region'], r['split']): r for r in gate_manifest}
    registry = json.loads(register(root / 'release_20261026/source_registry.json').read_text())['regions']
    source_hashes = json.loads((stage / 'data_sources.json').read_text())['sha256']
    frozen = {s: read_npz(stage / 'data/individual' / (s + '.npz')) for s in SPLITS}
    data = {}
    for split in SPLITS:
        d = {k: v.copy() for k, v in frozen[split].items() if '__' not in k}
        n, t = d['mask'].shape
        d['field_common_valid'] = np.zeros((n, t), bool)
        d['gated7_legacy'] = np.zeros((n, t, 7), np.float32)
        d['gate_branches'] = np.zeros((n, t), np.int8)
        d['target_gate_weight'] = np.zeros((n, t), np.float32)
        d['target_kinematic_valid'] = np.zeros((n, t), bool)
        for config in ('F0', 'F1', 'F2'):
            d['base__' + config] = frozen[split]['base__C0'].copy()
            d[config + '__gradient_per_s'] = np.zeros((n, t, 2, 2), np.float64)
            d[config + '__divergence_per_s'] = np.zeros((n, t), np.float64)
            d[config + '__curl_per_s'] = np.zeros((n, t), np.float64)
        for config in ('F0', 'F1'):
            d[config + '__fit_valid'] = np.zeros((n, t), bool)
            d[config + '__fit_reason'] = np.zeros((n, t), np.int8)
            d[config + '__condition'] = np.zeros((n, t), np.float64)
        data[split] = d
    tests = unit_checks()
    totals = {s: Counter() for s in SPLITS}
    regions = {}
    cap_observations = []
    proofpath = outdir / 'field_member_provenance.jsonl.gz'
    assert not proofpath.exists()
    proof_digest = {s: hashlib.sha256() for s in SPLITS}
    with gzip.open(proofpath, 'xt', encoding='utf8') as proofout:
        for region in REGIONS:
            region_started = time.time()
            register(registry[region]['canonical_file'], registry[region]['sha256'])
            source = helper.Source(registry[region])
            regions[region] = {}
            for split in SPLITS:
                d, old = data[split], frozen[split]
                indices = np.flatnonzero(old['region'] == region)
                source_path = root / 'labeling/labeled_v1/features_scene' / region / (split + '.npz')
                register(source_path, source_hashes[str(source_path)])
                obs = read_npz(source_path, ['window_id', 'mmsi', 'mask', 'time_s', 'target_pos', 'target_sog_kn',
                    'target_cog_deg', 'self_gate_weight', 'kinematic_valid', 'context_valid'])
                for key in ('window_id', 'mask', 'time_s'):
                    np.testing.assert_array_equal(old[key][indices], obs[key])
                np.testing.assert_array_equal(old['mmsi'][indices], obs['mmsi'].astype(str))
                np.testing.assert_array_equal(old['common_valid'][indices], obs['context_valid'])
                raw = old['base__C0'][indices, :, :7]
                np.testing.assert_array_equal(raw, core.Builder.raw7(None, obs))
                w = obs['self_gate_weight'].astype(np.float32).copy()
                w[~obs['mask']] = 0
                gated, branches = causal_replay(raw, w, obs['mask'], obs['kinematic_valid'], obs['time_s'], cap, gate)
                if split in ('train', 'val'):
                    gm = gate_map[(region, split)]
                    gp = gate_root / 'gate_data' / region / (split + '.npz')
                    existing = read_npz(register(gp, gm['output_sha256']))
                    np.testing.assert_array_equal(existing['window_id'], obs['window_id'])
                    np.testing.assert_array_equal(existing['causal7'], gated)
                    np.testing.assert_array_equal(existing['branches'], branches)
                    assert sha(source_path) == gm['reference_sha256']
                    if split == 'train':
                        eligible = obs['mask'] & obs['kinematic_valid'] & (w >= .7)
                        cap_observations.append(np.linalg.norm(raw[..., 2:4][eligible], axis=1))
                d['gated7_legacy'][indices], d['gate_branches'][indices] = gated, branches
                d['target_gate_weight'][indices], d['target_kinematic_valid'][indices] = w, obs['kinematic_valid']
                gated_model = gated.copy()
                gated_model[~obs['mask']] = 0
                d['base__F2'][indices, :, :7] = gated_model
                bgroot = root / ('causal_gate_main_v1_20261027/test_background' if split == 'test' else 'robust_background_v1_20261027') / region
                report = json.loads(register(bgroot / 'build_report.json').read_text())['splits'][split]
                auditpath = source_path.with_name(split + '_context_audit.jsonl.gz')
                memberpath = bgroot / (split + '_member_provenance.jsonl.gz')
                register(auditpath, report['audit_sha256'])
                register(memberpath, report['member_provenance_sha256'])
                with gzip.open(memberpath, 'rt', encoding='utf8') as f:
                    saved = {(r['window_id'], int(r['step'])): r for r in map(json.loads, f)}
                seen, counts = set(), Counter()
                with gzip.open(auditpath, 'rt', encoding='utf8') as f:
                    for local_i, line in enumerate(f):
                        md = json.loads(line)
                        assert md['window_id'] == str(obs['window_id'][local_i]) and md['split'] == split
                        global_i = int(indices[local_i])
                        overlay = source.overlay(md)
                        for step in md['steps']:
                            j = int(step['step'])
                            if not old['common_valid'][global_i, j]:
                                continue
                            key = (md['window_id'], j)
                            assert key in saved and key not in seen
                            seen.add(key)
                            now = float(obs['time_s'][local_i, j])
                            target_id = str(obs['mmsi'][local_i])
                            pos = obs['target_pos'][local_i, j]
                            target_cog = float(obs['target_cog_deg'][local_i, j])
                            angle = np.deg2rad(target_cog)
                            target_v = obs['target_sog_kn'][local_i, j] * (1852 / 3600) * np.array([np.sin(angle), np.cos(angle)])
                            members = []
                            accepted = [e for e in step.get('neighbors', []) if e['plain_accepted']]
                            assert len(accepted) == len(saved[key]['members']) >= 4
                            for ent, prior in zip(accepted, saved[key]['members']):
                                mid, stamp = str(ent['mmsi']), float(ent['time_s'])
                                assert mid == prior['mmsi'] and stamp == prior['time_s'] and mid != target_id
                                assert 0 <= now - stamp <= 300
                                p, speed, course, valid, error, absolute = source.observation(mid, stamp, overlay)
                                distance = float(np.linalg.norm(p - pos))
                                heading = abs(float(helper.wrap(course - target_cog)))
                                assert abs(distance - ent['distance_m']) < 1e-6 and distance <= R
                                assert abs(heading - ent['heading_difference_deg']) < 1e-8 and heading <= 45
                                assert bool(valid) == ent['kinematic_valid'] == prior['kinematic_valid']
                                assert abs(error - ent['e_kin_mps']) < 1e-6
                                assert abs(absolute - ent['reliability']) < 1e-8
                                aw = float(absolute) if valid else 0.
                                assert aw == prior['w_abs']
                                a = np.deg2rad(course)
                                velocity = [float(speed * (1852 / 3600) * np.sin(a)), float(speed * (1852 / 3600) * np.cos(a))]
                                np.testing.assert_array_equal(velocity, prior['velocity_mps'])
                                members.append({'mmsi': mid, 'time_s': stamp, 'position_m': np.asarray(p).tolist(),
                                    'velocity_mps': velocity, 'distance_m': distance, 'e_kin_mps': float(error),
                                    'kinematic_valid': bool(valid), 'w_abs': aw})
                            mids = np.asarray([m['mmsi'] for m in members])
                            positions = np.asarray([m['position_m'] for m in members]) - pos
                            velocity = np.asarray([m['velocity_mps'] for m in members])
                            absolute = np.asarray([m['w_abs'] for m in members])
                            distance_weight = np.exp(-.5 * (np.linalg.norm(positions, axis=1) / SIGMA) ** 2)
                            arms = {'F0': fit_field(positions, velocity, distance_weight, mids, target_id),
                                    'F1': fit_field(positions, velocity, distance_weight * absolute, mids, target_id)}
                            for config, result in arms.items():
                                assert result['removed_target_rows'] == 0
                                d[config + '__fit_valid'][global_i, j] = result['valid']
                                d[config + '__fit_reason'][global_i, j] = REASONS[result['reason']]
                                d[config + '__condition'][global_i, j] = result['condition']
                                counts[config + '_' + result['reason']] += 1
                            shared = all(result['valid'] for result in arms.values())
                            d['field_common_valid'][global_i, j] = shared
                            counts['inherited_common_steps'] += 1
                            counts['member_observations_with_step_reuse'] += len(members)
                            counts['unknown_reliability_observations'] += sum(not m['kinematic_valid'] for m in members)
                            if shared:
                                counts['shared_field_steps'] += 1
                                if np.array_equal(absolute, np.ones(len(absolute))):
                                    np.testing.assert_array_equal(arms['F0']['coefficient'], arms['F1']['coefficient'])
                                    counts['actual_equal_weight_equivalence_checks'] += 1
                                for config, result in arms.items():
                                    coefficient = result['coefficient']
                                    background = coefficient[0].astype(np.float32)
                                    x = d['base__' + config]
                                    x[global_i, j, 7:9] = background
                                    x[global_i, j, 9:11] = (target_v - background).astype(np.float32)
                                    x[global_i, j, 11] = 1
                                    gradient = coefficient[1:, :].T / R
                                    d[config + '__gradient_per_s'][global_i, j] = gradient
                                    d[config + '__divergence_per_s'][global_i, j] = np.trace(gradient)
                                    d[config + '__curl_per_s'][global_i, j] = gradient[1, 0] - gradient[0, 1]
                                    if counts['shared_field_steps'] % 127 == 1:
                                        ww = distance_weight if config == 'F0' else distance_weight * absolute
                                        independent = independently_fit(positions, velocity, ww)
                                        np.testing.assert_allclose(coefficient, independent, rtol=1e-9, atol=1e-9)
                                        appended = fit_field(np.vstack([positions, [0, 0]]), np.vstack([velocity, [1e6, -1e6]]),
                                            np.r_[ww, 1.], np.concatenate([mids, np.asarray([target_id])]), target_id)
                                        np.testing.assert_array_equal(coefficient, appended['coefficient'])
                                        counts['independent_fit_and_target_injection_checks'] += 1
                                bg = d['base__F1'][global_i, j, 7:9]
                                d['base__F2'][global_i, j, 7:9] = bg
                                d['base__F2'][global_i, j, 9:11] = gated_model[local_i, j, 2:4] - bg
                                d['base__F2'][global_i, j, 11] = 1
                            proof = {'split': split, 'region': region, 'source_id': str(old['source_id'][global_i]),
                                'window_id': md['window_id'], 'target_mmsi': target_id, 'step': j, 'time_s': now,
                                'target_original_observed_position_m': pos.tolist(), 'members': members,
                                'shared_field_valid': shared, 'fits': {name: {'valid': value['valid'], 'reason': value['reason'],
                                    'condition': value['condition'], 'coefficient': value['coefficient'].tolist()} for name, value in arms.items()}}
                            encoded = json.dumps(proof, separators=(',', ':'))
                            proofout.write(encoded + '\n')
                            proof_digest[split].update(encoded.encode())
                    assert local_i + 1 == len(indices)
                assert seen == set(saved)
                counts['windows'] = len(indices)
                counts['F2_valid_nonposition_values_changed_from_raw'] = int(np.count_nonzero(gated_model[..., 2:7][obs['mask']] != raw[..., 2:7][obs['mask']]))
                counts['F2_legacy_padding_values_zeroed'] = int(np.count_nonzero(gated[~obs['mask']]))
                regions[region][split] = dict(counts)
                totals[split].update(counts)
                print('FIELD_REGION_SPLIT', region, split, counts['shared_field_steps'], '/', counts['inherited_common_steps'], flush=True)
                del saved, obs
            regions[region]['elapsed_seconds'] = time.time() - region_started
            del source
            gc.collect()
    assert sum(len(a) for a in cap_observations) == cap_info['eligible_steps']
    assert float(np.quantile(np.concatenate(cap_observations), cap_info['quantile'])) == cap
    outputs = {}
    for split in SPLITS:
        d, old = data[split], frozen[split]
        shared = d['field_common_valid']
        assert not (shared & ~old['common_valid']).any()
        for key in ('gradient_per_s', 'divergence_per_s', 'curl_per_s'):
            d['F2__' + key] = d['F1__' + key].copy()
        for config in ('F0', 'F1', 'F2'):
            x = d['base__' + config]
            np.testing.assert_array_equal(x[..., 11], shared.astype(np.float32))
            if config != 'F2':
                np.testing.assert_array_equal(x[..., :7], old['base__C0'][..., :7])
            else:
                np.testing.assert_array_equal(x[..., :7][old['mask']], d['gated7_legacy'][old['mask']])
                np.testing.assert_array_equal(x[..., 7:9], d['base__F1'][..., 7:9])
                np.testing.assert_array_equal(x[..., 9:11][shared], (x[..., 2:4] - x[..., 7:9])[shared])
            fm = np.zeros_like(x, bool)
            fm[..., :7] = old['mask'][..., None]
            fm[..., 7:11] = shared[..., None]
            fm[..., 11] = old['mask']
            d['base__feature_mask_' + config] = fm
            assert not x[~fm].any() and np.isfinite(x).all()
            gradient = d[config + '__gradient_per_s']
            np.testing.assert_array_equal(d[config + '__divergence_per_s'], gradient[..., 0, 0] + gradient[..., 1, 1])
            np.testing.assert_array_equal(d[config + '__curl_per_s'], gradient[..., 1, 0] - gradient[..., 0, 1])
        for key, value in d.items():
            if value.dtype.kind in 'fc':
                assert np.isfinite(value).all(), (split, key)
        path = outdir / ('field_' + split + '.npz')
        np.savez_compressed(path, **d)
        totals[split]['valid_target_steps'] = int(old['mask'].sum())
        totals[split]['common_field_support_percent'] = float(100 * shared.sum() / old['mask'].sum())
        outputs[split] = {'path': str(path.relative_to(stage)), 'sha256': sha(path), 'windows': len(d['y']),
                         'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': core.arrsha(v)} for k, v in d.items()}}
    for path, expected in hashes.items():
        assert sha(path) == expected, ('source_or_core_changed', path)
    audit = {'status': 'complete_fixed_math_no_score_access', 'protocol_sha256': sha(protocolpath), 'builder_sha256': sha(__file__),
        'outputs': outputs, 'source_sha256': hashes, 'regions': regions, 'counts': {k: dict(v) for k, v in totals.items()},
        'unit_checks': tests, 'member_provenance': {'path': str(proofpath.relative_to(stage)), 'sha256': sha(proofpath),
            'split_record_digest_sha256': {k: v.hexdigest() for k, v in proof_digest.items()}},
        'fit_reason_codes': REASONS, 'training_only_gate_cap_recomputed_exactly': cap,
        'target_gate_replay': {'train_val_saved_causal7_and_branches_all_exact': True, 'test_replay_same_code_and_saved_train_cap': True,
            'model_padding_zero': True, 'legacy_full_causal7_saved': True, 'F2_residual_uses_held_reported_velocity': True},
        'assertions': {'same_original_candidates_F0_F1': True, 'target_identity_excluded_all_steps': True,
            'same_shared_support_F0_F1_F2': True, 'original_labels_times_source_ids_splits_retained': True,
            'no_outcome_scores_read': True, 'no_parameter_selection': True, 'no_core_source_files_changed': True,
            'F2_same_original_observation_field_reference': True, 'derivatives_auxiliary_only': True},
        'elapsed_seconds': time.time() - started}
    write_json(stage / 'field_audit.json', audit)
    print('FIELD_BUILD_COMPLETE', round(audit['elapsed_seconds'], 2), json.dumps({k: v['sha256'] for k, v in outputs.items()}), flush=True)


if __name__ == '__main__':
    main()
