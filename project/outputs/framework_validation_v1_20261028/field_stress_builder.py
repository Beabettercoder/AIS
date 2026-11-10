"""Neighbor-observation stress for the fixed local-linear field family; no scores."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import gc
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
REGIONS = ('houston', 'ny', 'la', 'puget', 'great_belt', 'haugesund', 'kattegat', 'shanghai')
CONFIGS = ('F0', 'F1', 'F2')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if keys is None else keys)}


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def seed_unit(prefix, source_id, mmsi):
    digest = hashlib.sha256((prefix + source_id + ':' + mmsi).encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2 ** 64, digest.hex()


def observed_good(p, s, c):
    return bool(np.isfinite(p).all() and np.isfinite(s) and np.isfinite(c) and 0 <= s < 102.2 and 0 <= c < 360)


def plan_source(source, overlay, source_id, target_id, target_times, steps, common):
    onset = float(target_times[0] + 2 * (target_times[-1] - target_times[0]) / 3)
    end = float(target_times[-1])
    all_members = set()
    suffix_reports = defaultdict(set)
    for step in steps:
        j = int(step['step'])
        if not common[j]:
            continue
        for ent in step.get('neighbors', []):
            if ent['plain_accepted']:
                mid = str(ent['mmsi'])
                assert mid != target_id
                all_members.add(mid)
                if target_times[j] >= onset:
                    suffix_reports[mid].add(float(ent['time_s']))
    eligible = []
    for mid, used in sorted(suffix_reports.items()):
        ship = source.ship[mid]
        times = source.t[ship]
        lo, hi = np.searchsorted(times, [onset, end], side='left')
        hi = int(np.searchsorted(times, end, side='right'))
        choices = []
        for k in range(int(lo), hi - 1):
            a, b = int(ship[k]), int(ship[k + 1])
            ta, tb = float(source.t[a]), float(source.t[b])
            if ta not in used and tb not in used:
                continue
            oa, ob = source.observation(mid, ta, overlay), source.observation(mid, tb, overlay)
            if not observed_good(*oa[:3]) or not observed_good(*ob[:3]):
                continue
            tag = 'field-stress-report-pair-v1:' + json.dumps([source_id, mid, ta, tb], separators=(',', ':'))
            choices.append((hashlib.sha256(tag.encode()).hexdigest(), k, [a, b], [ta, tb]))
        if not choices:
            continue
        chosen = min(choices, key=lambda v: v[0])
        factor_u, factor_seed = seed_unit('field-stress-factor-v1:', source_id, mid)
        direction_u, direction_seed = seed_unit('field-stress-direction-v1:', source_id, mid)
        theta = 2 * np.pi * direction_u
        member_rank = hashlib.sha256(('field-stress-member-v1:' + source_id + ':' + mid).encode()).hexdigest()
        eligible.append({'mmsi': mid, 'rank': member_rank, 'canonical_indices': chosen[2], 'times_s': chosen[3],
            'ship_positions': [int(chosen[1]), int(chosen[1] + 1)], 'pair_rank': chosen[0],
            'factor': float(3 + 2 * factor_u), 'factor_seed_sha256': factor_seed,
            'direction': [float(np.cos(theta)), float(np.sin(theta))], 'direction_seed_sha256': direction_seed,
            'eligible_pair_count': len(choices)})
    eligible.sort(key=lambda e: e['rank'])
    selected = eligible[:math.ceil(len(eligible) / 2)]
    return {'source_id': source_id, 'target_mmsi': target_id, 'last_third_start_s': onset, 'end_time_s': end,
        'original_member_pool': len(all_members), 'suffix_query_member_pool': len(suffix_reports),
        'eligible_member_count': len(eligible), 'selected_member_count': len(selected), 'members': selected}


def modified_overlay(source, base, plan, lam, amplitude, kin):
    if lam == 0 and amplitude == 0:
        return base, {'speed_changed_reports': 0, 'position_changed_reports': 0,
                      'actual_changed_members': 0, 'kinematic_reports_recomputed': 0}
    result = dict(base)
    affected, changed_members = {}, set()
    speed_changed = position_changed = 0
    for member in plan['members']:
        mid = member['mmsi']
        assert mid != plan['target_mmsi']
        ship = source.ship[mid]
        for ix, k in zip(member['canonical_indices'], member['ship_positions']):
            ix, k = int(ix), int(k)
            assert int(ship[k]) == ix and plan['last_third_start_s'] <= source.t[ix] <= plan['end_time_s']
            pos, speed, course, valid, error, weight = source.observation(mid, source.t[ix], base)
            newpos = np.asarray(pos).copy()
            newspeed = float(speed)
            if amplitude:
                newpos += amplitude * np.asarray(member['direction'])
            if lam:
                endpoint = min(float(speed) * member['factor'], 102.1)
                newspeed = endpoint if lam == 1 else float(speed) + lam * (endpoint - float(speed))
            sp_changed, po_changed = newspeed != speed, bool(np.any(newpos != pos))
            speed_changed += int(sp_changed)
            position_changed += int(po_changed)
            if sp_changed or po_changed:
                changed_members.add(mid)
                result[ix] = (newpos, newspeed, course, valid, error, weight)
                affected[ix] = (mid, k)
                if k + 1 < len(ship) and source.t[ship[k + 1]] <= plan['end_time_s']:
                    affected[int(ship[k + 1])] = (mid, k + 1)
    for ix, (mid, k) in sorted(affected.items()):
        ship = source.ship[mid]
        current = source.observation(mid, source.t[ix], result)
        if k == 0:
            valid, error, weight = False, 0., 0.
        else:
            prior_ix = int(ship[k - 1])
            prior = source.observation(mid, source.t[prior_ix], result)
            _, valid_array, errors, weights = kin(np.asarray([source.t[prior_ix], source.t[ix]]),
                np.asarray([prior[0], current[0]]), np.asarray([prior[1], current[1]]), np.asarray([prior[2], current[2]]))
            valid, error, weight = bool(valid_array[1]), float(errors[1]), float(weights[1])
        result[ix] = (current[0], current[1], current[2], valid, error, weight)
    return result, {'speed_changed_reports': speed_changed, 'position_changed_reports': position_changed,
        'actual_changed_members': len(changed_members), 'kinematic_reports_recomputed': len(affected)}


def overlay_unit_test(kin):
    # A changed report affects its own and its next causal interval, never an
    # earlier report or an interval bridged across an invalid predecessor.
    t = np.asarray([0., 60., 120., 180., 240.])
    p = np.column_stack([np.arange(5) * 60, np.zeros(5)])
    s, c = np.full(5, 1 / (1852 / 3600)), np.full(5, 90.)
    good, valid, e, w = kin(t, p, s, c)
    class Fixture:
        def __init__(self):
            self.ship = {'n': np.arange(5)}
            self.t, self.p, self.s, self.c = t, p, s, c
        def observation(self, mid, stamp, overlay):
            i = int(np.flatnonzero(self.t == stamp)[0])
            return overlay.get(i, (self.p[i], self.s[i], self.c[i], valid[i], e[i], w[i]))
    src = Fixture()
    plan = {'target_mmsi': 'target', 'last_third_start_s': 120., 'end_time_s': 240.,
            'members': [{'mmsi': 'n', 'canonical_indices': [2, 3], 'ship_positions': [2, 3],
                         'factor': 4., 'direction': [1., 0.]}]}
    for lam, amplitude in [(0, 0), (.5, 50), (1, 600), (0, 150), (1, 0)]:
        out, _ = modified_overlay(src, {}, plan, lam, amplitude, kin)
        op, os, oc = p.copy(), s.copy(), c.copy()
        op[2:4] += amplitude * np.asarray([1., 0.])
        os[2:4] += lam * (np.minimum(os[2:4] * 4, 102.1) - os[2:4])
        _, vv, ee, ww = kin(t, op, os, oc)
        for j in range(5):
            r = src.observation('n', t[j], out)
            np.testing.assert_array_equal(r[0], op[j])
            assert r[1] == os[j] and r[3] == vv[j]
            np.testing.assert_allclose([r[4], r[5]], [ee[j], ww[j]], rtol=0, atol=1e-14)
    broken = {1: (p[1], np.nan, c[1], False, 0., 0.)}
    out, _ = modified_overlay(src, broken, plan, 1, 0, kin)
    assert not src.observation('n', t[2], out)[3]
    return {'modified_and_next_intervals_match_full_recomputation': True, 'invalid_predecessor_not_bridged': True,
            'unmodified_prefix_retained': True, 'all_zero_observations_exact': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2'))
    args = parser.parse_args()
    stage, root = args.stage.resolve(), args.root.resolve()
    assert stage.name == 'framework_validation_v1_20261028'
    dest = stage / 'extra_data/field_stress_test.npz'
    proofpath = stage / 'extra_data/field_stress_selection.jsonl.gz'
    auditpath = stage / 'field_stress_audit.json'
    assert not dest.exists() and not proofpath.exists() and not auditpath.exists()
    started = time.time()
    protocolpath = stage / 'field_stress_protocol.json'
    protocol = json.loads(protocolpath.read_text())
    hashes = {str(protocolpath): sha(protocolpath), str(Path(__file__).resolve()): sha(__file__)}
    def register(path, expected=None):
        path = Path(path).resolve()
        checksum = sha(path)
        if expected is not None:
            assert checksum == expected, path
        hashes[str(path)] = checksum
        return path
    for name, expected in [('field_builder.py', protocol['field_builder_sha256']),
                           ('field_protocol.json', protocol['field_protocol_sha256']),
                           ('field_audit.json', protocol['field_audit_sha256']),
                           ('extra_data/field_test.npz', protocol['field_test_sha256'])]:
        register(stage / name, expected)
    field = load_module(stage / 'field_builder.py', 'frozen_field_builder_stress')
    field_audit = json.loads((stage / 'field_audit.json').read_text())
    allowed_hashes = field_audit['source_sha256']
    helperpath = root / 'context_adapter_v1_20261026/neighborhood/audit_neighborhood.py'
    helper = load_module(register(helperpath, allowed_hashes[str(helperpath)]), 'source_observed_field_stress')
    core = load_module(register(stage / 'data_builder.py', allowed_hashes[str(stage / 'data_builder.py')]), 'core_hash_stress')
    tests = overlay_unit_test(helper.kin)
    registry_path = root / 'release_20261026/source_registry.json'
    registry = json.loads(register(registry_path, allowed_hashes[str(registry_path)]).read_text())['regions']
    original = read_npz(stage / 'extra_data/field_test.npz')
    assert len(original['y']) == 1680 and len(set(original['source_id'])) == 1680
    assert int(original['field_common_valid'].sum()) == 11866
    data = {k: v.copy() for k, v in original.items() if '__' not in k}
    views = protocol['view_specs']
    counts = {v['name']: Counter() for v in views}
    by_region, feasibility = {}, Counter()
    pair_digest = hashlib.sha256()
    for view in views:
        name = view['name']
        data[name + '__support'] = np.zeros_like(original['mask'])
        for config in CONFIGS:
            x = np.zeros_like(original['base__' + config])
            x[..., :7] = original['base__' + config][..., :7]
            data[name + '__' + config] = x
            data[name + '__' + config + '_divergence_per_s'] = np.zeros(original['mask'].shape, np.float64)
            data[name + '__' + config + '_curl_per_s'] = np.zeros(original['mask'].shape, np.float64)
    with gzip.open(proofpath, 'xt', encoding='utf8') as proofout:
        for region in REGIONS:
            region_started = time.time()
            source = helper.Source(registry[region])
            register(registry[region]['canonical_file'], registry[region]['sha256'])
            indices = np.flatnonzero(original['region'] == region)
            opath = root / 'labeling/labeled_v1/features_scene' / region / 'test.npz'
            apath = opath.with_name('test_context_audit.jsonl.gz')
            register(opath, allowed_hashes[str(opath)])
            register(apath, allowed_hashes[str(apath)])
            obs = read_npz(opath, ['window_id', 'mmsi', 'mask', 'time_s', 'target_pos', 'target_sog_kn', 'target_cog_deg'])
            for key in ('window_id', 'mask', 'time_s'):
                np.testing.assert_array_equal(obs[key], original[key][indices])
            np.testing.assert_array_equal(obs['mmsi'].astype(str), original['mmsi'][indices])
            local_counts = Counter()
            with gzip.open(apath, 'rt', encoding='utf8') as f:
                for local_i, line in enumerate(f):
                    md = json.loads(line)
                    assert md['split'] == 'test' and md['window_id'] == str(obs['window_id'][local_i])
                    i, n = int(indices[local_i]), int(original['length'][indices[local_i]])
                    target_id, source_id = str(original['mmsi'][i]), str(original['source_id'][i])
                    base_overlay = source.overlay(md)
                    plan = plan_source(source, base_overlay, source_id, target_id, obs['time_s'][local_i, :n],
                                       md['steps'], original['field_common_valid'][i])
                    plan.update(region=region, window_id=md['window_id'], global_row=i)
                    for key in ('original_member_pool', 'suffix_query_member_pool', 'eligible_member_count', 'selected_member_count'):
                        local_counts[key] += plan[key]
                    local_counts['sources'] += 1
                    local_counts['sources_with_eligible_members'] += int(plan['eligible_member_count'] > 0)
                    local_counts['sources_without_eligible_members'] += int(plan['eligible_member_count'] == 0)
                    proofout.write(json.dumps(plan, separators=(',', ':')) + '\n')
                    prepared = []
                    for step in md['steps']:
                        j = int(step['step'])
                        if not original['field_common_valid'][i, j]:
                            continue
                        members = [e for e in step.get('neighbors', []) if e['plain_accepted']]
                        assert len(members) >= 4
                        mids = np.asarray([str(e['mmsi']) for e in members])
                        times = np.asarray([float(e['time_s']) for e in members])
                        assert target_id not in mids and len(set(mids)) == len(mids)
                        assert np.all((times <= obs['time_s'][local_i, j]) & (obs['time_s'][local_i, j] - times <= 300))
                        pos = obs['target_pos'][local_i, j]
                        base_observed = [source.observation(m, t, base_overlay) for m, t in zip(mids, times)]
                        for ent, record in zip(members, base_observed):
                            assert abs(np.linalg.norm(record[0] - pos) - ent['distance_m']) < 1e-6
                            assert abs(record[4] - ent['e_kin_mps']) < 1e-6 and abs(record[5] - ent['reliability']) < 1e-8
                        angle = np.deg2rad(obs['target_cog_deg'][local_i, j])
                        target_v = obs['target_sog_kn'][local_i, j] * (1852 / 3600) * np.array([np.sin(angle), np.cos(angle)])
                        prepared.append((j, mids, times, pos, target_v, base_observed))
                        pair_digest.update(json.dumps([source_id, j, mids.tolist(), times.tolist()], separators=(',', ':')).encode())
                    for view in views:
                        name, lam, amplitude = view['name'], view['speed_lambda'], view['position_m']
                        overlay, mutations = modified_overlay(source, base_overlay, plan, lam, amplitude, helper.kin)
                        stat = counts[name]
                        stat['sources'] += 1
                        for key, value in mutations.items():
                            stat[key] += value
                        stat['sources_with_actual_observation_changes'] += int(mutations['actual_changed_members'] > 0)
                        stat['sources_with_no_actual_observation_changes'] += int(mutations['actual_changed_members'] == 0)
                        source_hit = source_fit_change = False
                        for j, mids, times, query, target_v, base_observed in prepared:
                            records = [source.observation(m, t, overlay) for m, t in zip(mids, times)]
                            positions = np.asarray([r[0] for r in records]) - query
                            speeds, courses = np.asarray([r[1] for r in records]), np.asarray([r[2] for r in records])
                            angles = np.deg2rad(courses)
                            velocity = speeds[:, None] * (1852 / 3600) * np.column_stack([np.sin(angles), np.cos(angles)])
                            absolute = np.asarray([float(r[5]) if r[3] else 0. for r in records])
                            distance = np.linalg.norm(positions, axis=1)
                            distance_weight = np.exp(-.5 * (distance / field.SIGMA) ** 2)
                            speed_hit = np.asarray([r[1] != b[1] for r, b in zip(records, base_observed)])
                            pos_hit = np.asarray([np.any(r[0] != b[0]) for r, b in zip(records, base_observed)])
                            weight_hit = np.asarray([r[3] != b[3] or r[4] != b[4] or r[5] != b[5] for r, b in zip(records, base_observed)])
                            stat['queried_member_observations'] += len(records)
                            stat['queried_speed_changed_observations'] += int(speed_hit.sum())
                            stat['queried_position_changed_observations'] += int(pos_hit.sum())
                            stat['queried_kinematics_changed_observations'] += int(weight_hit.sum())
                            hit = bool(np.any(speed_hit | pos_hit | weight_hit))
                            stat['query_steps_with_actual_observation_changes'] += int(hit)
                            source_hit |= hit
                            stat['queried_positions_beyond_original_8nm_after_stress'] += int((distance > field.R).sum())
                            fits = {'F0': field.fit_field(positions, velocity, distance_weight, mids, target_id),
                                    'F1': field.fit_field(positions, velocity, distance_weight * absolute, mids, target_id)}
                            shared = all(r['valid'] for r in fits.values())
                            data[name + '__support'][i, j] = shared
                            stat['inherited_field_steps'] += 1
                            stat['shared_field_steps'] += int(shared)
                            stat['support_lost_steps'] += int(not shared)
                            for config, fit in fits.items():
                                assert fit['removed_target_rows'] == 0
                                stat[config + '_' + fit['reason']] += 1
                            if shared:
                                for config, fit in fits.items():
                                    x = data[name + '__' + config]
                                    bg = fit['coefficient'][0].astype(np.float32)
                                    x[i, j, 7:9], x[i, j, 9:11], x[i, j, 11] = bg, (target_v - bg).astype(np.float32), 1
                                    gradient = fit['coefficient'][1:, :].T / field.R
                                    data[name + '__' + config + '_divergence_per_s'][i, j] = np.trace(gradient)
                                    data[name + '__' + config + '_curl_per_s'][i, j] = gradient[1, 0] - gradient[0, 1]
                                bg = data[name + '__F1'][i, j, 7:9]
                                data[name + '__F2'][i, j, 7:9] = bg
                                data[name + '__F2'][i, j, 9:11] = original['base__F2'][i, j, 2:4] - bg
                                data[name + '__F2'][i, j, 11] = 1
                            for config in ('F0', 'F1'):
                                source_fit_change |= bool(np.any(data[name + '__' + config][i, j, 7:] != original['base__' + config][i, j, 7:]))
                        stat['sources_with_query_observation_changes'] += int(source_hit)
                        stat['sources_with_field_input_changes'] += int(source_fit_change)
                    if (local_i + 1) % 100 == 0:
                        print('FIELD_STRESS_PROGRESS', region, local_i + 1, flush=True)
                assert local_i + 1 == 210
            by_region[region] = {**dict(local_counts), 'elapsed_seconds': time.time() - region_started}
            feasibility.update(local_counts)
            print('FIELD_STRESS_REGION', region, 'eligible_sources', local_counts['sources_with_eligible_members'], 'elapsed', round(time.time() - region_started, 2), flush=True)
            del source, obs
            gc.collect()
    for view in views:
        name = view['name']
        support = data[name + '__support']
        assert not np.any(support & ~original['field_common_valid'])
        for suffix in ('divergence_per_s', 'curl_per_s'):
            data[name + '__F2_' + suffix] = data[name + '__F1_' + suffix].copy()
        for config in CONFIGS:
            x = data[name + '__' + config]
            np.testing.assert_array_equal(x[..., :7], original['base__' + config][..., :7])
            np.testing.assert_array_equal(x[..., 11], support.astype(np.float32))
            if config == 'F2':
                np.testing.assert_array_equal(x[..., 7:9], data[name + '__F1'][..., 7:9])
                np.testing.assert_array_equal(x[..., 9:11][support], (x[..., 2:4] - x[..., 7:9])[support])
            fm = np.zeros_like(x, bool)
            fm[..., :7], fm[..., 7:11], fm[..., 11] = original['mask'][..., None], support[..., None], original['mask']
            data[name + '__feature_mask_' + config] = fm
            assert not x[~fm].any() and np.isfinite(x).all()
            if view['speed_lambda'] == 0 and view['position_m'] == 0:
                np.testing.assert_array_equal(x, original['base__' + config])
                for suffix in ('divergence_per_s', 'curl_per_s'):
                    np.testing.assert_array_equal(data[name + '__' + config + '_' + suffix], original[config + '__' + suffix])
            counts[name][config + '_changed_feature_values_from_base'] = int(np.count_nonzero(x != original['base__' + config]))
        counts[name]['common_support_percent'] = float(100 * support.sum() / original['mask'].sum())
        counts[name]['valid_target_steps'] = int(original['mask'].sum())
    assert feasibility['sources'] == 1680
    np.savez_compressed(dest, **data)
    for path, checksum in hashes.items():
        assert sha(path) == checksum, ('source_changed', path)
    audit = {'status': 'complete_field_family_observation_stress', 'no_scores_read': True, 'no_training': True,
        'no_outcome_selection': True, 'protocol_sha256': sha(protocolpath), 'builder_sha256': sha(__file__),
        'output_path': str(dest.relative_to(stage)), 'output_sha256': sha(dest), 'views': views,
        'artifact': {'path': str(dest.relative_to(stage)), 'sha256': sha(dest),
                    'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': core.arrsha(v)} for k, v in data.items()}},
        'feasibility': dict(feasibility), 'regions': by_region, 'counts': {k: dict(v) for k, v in counts.items()},
        'selection_provenance': {'path': str(proofpath.relative_to(stage)), 'sha256': sha(proofpath)},
        'fixed_candidate_identity_time_sha256': pair_digest.hexdigest(), 'source_sha256': hashes,
        'causal_overlay_unit_checks': tests,
        'assertions': {'all_zero_views_bitwise_match_field_test': True, 'all_1680_windows_retained': True,
            'target_inputs_and_gate_unchanged': True, 'fixed_original_query_positions_and_candidates': True,
            'new_positions_enter_design_and_distance_weights': True, 'actual_causal_kinematics_recomputed': True,
            'per_view_F0_F1_F2_support_shared': True, 'no_frozen_or_source_files_changed': True},
        'limitations': ['Added stress above an already corrupted/exposed base; not a pristine or fresh confirmation set',
            'Members without an observed usable suffix report pair cannot receive this operation; all such windows retained',
            'ceil50% refers to eligible members, not every original member; actual pool and observation fractions reported',
            'Membership is held fixed under position changes, including any moved reports outside the original8nm boundary'],
        'elapsed_seconds': time.time() - started}
    write_json(auditpath, audit)
    print('FIELD_STRESS_COMPLETE', sha(dest), round(audit['elapsed_seconds'], 2), json.dumps(dict(feasibility)), flush=True)


if __name__ == '__main__':
    main()
