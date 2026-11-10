"""Observation-level validation sensitivity; pure CPU and no result-score access."""
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
KNOT = 1852 / 3600


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if keys is None else keys)}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def weights(error, valid, e0, sigma):
    result = np.zeros(len(error), np.float64)
    result[valid] = np.exp(-np.maximum(0, error[valid] - e0) ** 2 / (2 * sigma ** 2))
    return result


def choose(spec, distances, headings, mids, times):
    index = np.arange(len(distances))
    family, value = spec['family'], spec['value']
    if family == 'topk' and len(index) > value:
        index = np.lexsort((times, mids, distances))[:int(value)]
    elif family == 'radius_nm':
        index = index[distances <= value * 1852]
    elif family == 'heading_deg':
        index = index[headings <= value]
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2'))
    args = parser.parse_args()
    stage, root = args.stage.resolve(), args.root.resolve()
    assert stage.name == 'framework_validation_v1_20261028'
    protocolpath = stage / 'sensitivity_protocol.json'
    protocol = json.loads(protocolpath.read_text(encoding='utf8'))
    outdir = stage / 'extra_data'
    outdir.mkdir(exist_ok=True)
    dest = outdir / 'individual_sensitivity_val.npz'
    proofpath = outdir / 'sensitivity_members_val.jsonl.gz'
    assert not dest.exists() and not proofpath.exists() and not (stage / 'sensitivity_audit.json').exists()
    started = time.time()
    hashes = {str(protocolpath): sha(protocolpath), str(Path(__file__).resolve()): sha(__file__)}
    for rel, field in [('data_builder.py', 'core_builder_sha256'), ('data_manifest.json', 'core_data_manifest_sha256'),
                       ('data/individual/val.npz', 'core_individual_val_sha256')]:
        p = stage / rel
        assert sha(p) == protocol[field]
        hashes[str(p)] = sha(p)
    code_lock = json.loads((stage / 'code_lock.json').read_text())
    hashes[str(stage / 'code_lock.json')] = sha(stage / 'code_lock.json')
    for rel, checksum in code_lock['files'].items():
        assert sha(stage / rel) == checksum
        hashes[str(stage / rel)] = checksum
    core = module(stage / 'data_builder.py', 'frozen_data_builder_sensitivity')
    helperpath = root / 'context_adapter_v1_20261026/neighborhood/audit_neighborhood.py'
    hashes[str(helperpath)] = sha(helperpath)
    helper = module(helperpath, 'frozen_source_observation_sensitivity')
    registry_path = root / 'release_20261026/source_registry.json'
    hashes[str(registry_path)] = sha(registry_path)
    registry = json.loads(registry_path.read_text())['regions']
    allowed_hashes = json.loads((stage / 'data_sources.json').read_text())['sha256']
    frozen = read_npz(stage / 'data/individual/val.npz')
    assert len(frozen['y']) == 1680 and frozen['common_valid'].sum() == 11032
    metadata = [k for k in frozen if '__' not in k]
    arrays = {key: frozen[key].copy() for key in metadata}
    views = protocol['view_specs']
    stats = {v['name']: Counter() for v in views}
    members_hash = {v['name']: hashlib.sha256() for v in views}
    for view in views:
        name = view['name']
        arrays[name + '__support'] = np.zeros_like(frozen['mask'])
        arrays[name + '__candidate_count'] = np.zeros(frozen['mask'].shape, np.int16)
        for config in ('C0', 'C1', 'C2', 'C3', 'C4'):
            arrays[name + '__' + config] = frozen['base__C0'].copy()
    regions_audit, all_counts = {}, Counter()
    with gzip.open(proofpath, 'xt', encoding='utf8') as proofout:
        for region in REGIONS:
            region_started = time.time()
            meta = registry[region]
            canonical = Path(meta['canonical_file'])
            assert sha(canonical) == meta['sha256']
            hashes[str(canonical)] = meta['sha256']
            source = helper.Source(meta)
            path = root / 'labeling/labeled_v1/features_scene' / region / 'val.npz'
            auditpath = path.with_name('val_context_audit.jsonl.gz')
            memberpath = root / 'robust_background_v1_20261027' / region / 'val_member_provenance.jsonl.gz'
            for p in (path, memberpath):
                assert sha(p) == allowed_hashes[str(p)]
                hashes[str(p)] = sha(p)
            build_report = json.loads((memberpath.parent / 'build_report.json').read_text())['splits']['val']
            assert sha(auditpath) == build_report['audit_sha256']
            hashes[str(auditpath)] = sha(auditpath)
            hashes[str(memberpath.parent / 'build_report.json')] = sha(memberpath.parent / 'build_report.json')
            obs = read_npz(path, ['window_id', 'mmsi', 'mask', 'time_s', 'target_pos', 'target_sog_kn', 'target_cog_deg', 'context_valid'])
            global_indices = np.flatnonzero(frozen['region'] == region)
            assert len(global_indices) == 210
            for key in ('window_id', 'mmsi', 'mask', 'time_s'):
                expected = obs[key].astype(str) if key == 'mmsi' else obs[key]
                np.testing.assert_array_equal(frozen[key][global_indices], expected)
            np.testing.assert_array_equal(frozen['common_valid'][global_indices], obs['context_valid'])
            with gzip.open(memberpath, 'rt', encoding='utf8') as f:
                saved = {(str(r['window_id']), int(r['step'])): r for r in map(json.loads, f)}
            seen = set()
            counts = Counter()
            with gzip.open(auditpath, 'rt', encoding='utf8') as f:
                for local_i, line in enumerate(f):
                    md = json.loads(line)
                    assert md['split'] == 'val' and md['window_id'] == str(obs['window_id'][local_i])
                    global_i = int(global_indices[local_i])
                    overlay = source.overlay(md)
                    for step in md['steps']:
                        j = int(step['step'])
                        if not frozen['common_valid'][global_i, j]:
                            continue
                        key = (md['window_id'], j)
                        assert key in saved and key not in seen
                        seen.add(key)
                        when = float(obs['time_s'][local_i, j])
                        pp = obs['target_pos'][local_i, j]
                        target_cog = float(obs['target_cog_deg'][local_i, j])
                        target_angle = np.deg2rad(target_cog)
                        target_v = obs['target_sog_kn'][local_i, j] * KNOT * np.array([np.sin(target_angle), np.cos(target_angle)])
                        eligible = [ent for ent in step.get('neighbors', []) if ent['plain_accepted']]
                        proofmembers = saved[key]['members']
                        assert len(eligible) == len(proofmembers) >= 4
                        details, velocities, errors, kvals, distances, headings, mids, times = [], [], [], [], [], [], [], []
                        for ent, old in zip(eligible, proofmembers):
                            mid, stamp = str(ent['mmsi']), float(ent['time_s'])
                            assert mid == old['mmsi'] and stamp == old['time_s']
                            assert mid != str(obs['mmsi'][local_i]) and 0 <= when - stamp <= 300
                            pos, speed, cog, valid, error, oldweight = source.observation(mid, stamp, overlay)
                            distance = float(np.linalg.norm(pos - pp))
                            heading = abs(float(helper.wrap(cog - target_cog)))
                            assert abs(distance - ent['distance_m']) < 1e-6 and distance <= 8 * 1852
                            assert ent['heading_difference_deg'] is not None and abs(heading - ent['heading_difference_deg']) < 1e-8 and heading <= 45
                            assert bool(valid) == bool(ent['kinematic_valid']) == bool(old['kinematic_valid'])
                            assert abs(float(error) - ent['e_kin_mps']) < 1e-6
                            assert abs(float(oldweight) - ent['reliability']) < 1e-8
                            expected_weight = float(oldweight) if valid else 0
                            assert abs(expected_weight - old['w_abs']) < 1e-8
                            angle = np.deg2rad(cog)
                            velocity = [float(speed * KNOT * np.sin(angle)), float(speed * KNOT * np.cos(angle))]
                            np.testing.assert_array_equal(velocity, old['velocity_mps'])
                            details.append({'mmsi': mid, 'time_s': stamp, 'observed_position_m': np.asarray(pos).tolist(),
                                            'distance_m': distance, 'heading_difference_deg': heading,
                                            'velocity_mps': velocity, 'kinematic_valid': bool(valid),
                                            'e_kin_mps': float(error), 'default_w_abs': expected_weight})
                            velocities.append(velocity); errors.append(error); kvals.append(valid)
                            distances.append(distance); headings.append(heading); mids.append(mid); times.append(stamp)
                        assert len(set(mids)) == len(mids)
                        velocities, errors, kvals = np.asarray(velocities), np.asarray(errors), np.asarray(kvals, bool)
                        distances, headings, mids, times = np.asarray(distances), np.asarray(headings), np.asarray(mids), np.asarray(times)
                        default_w = weights(errors, kvals, 2.5, 2)
                        np.testing.assert_allclose(default_w, [r['default_w_abs'] for r in details], rtol=0, atol=1e-15)
                        counts['inherited_common_steps'] += 1
                        counts['reconstructed_member_observations_with_step_reuse'] += len(velocities)
                        counts['unknown_reliability_observations'] += int((~kvals).sum())
                        selection_proof = {}
                        for view in views:
                            name, family, value = view['name'], view['family'], view['value']
                            chosen = choose(view, distances, headings, mids, times)
                            selection_proof[name] = chosen.tolist()
                            arrays[name + '__candidate_count'][global_i, j] = len(chosen)
                            stats[name]['inherited_common_steps'] += 1
                            stats[name]['candidate_observations'] += len(chosen)
                            members_hash[name].update(json.dumps([region, md['window_id'], j,
                                [[str(mids[k]), float(times[k])] for k in chosen]], separators=(',', ':')).encode())
                            if len(chosen) < 4:
                                stats[name]['support_lost_fewer_than_four'] += 1
                                continue
                            arrays[name + '__support'][global_i, j] = True
                            ww = default_w[chosen]
                            if family in ('e0_mps', 'sigma_mps'):
                                e0, sigma = (value, 2) if family == 'e0_mps' else (2.5, value)
                                ww = weights(errors[chosen], kvals[chosen], e0, sigma)
                            vv = velocities[chosen]
                            stats[name]['context_steps'] += 1
                            stats[name]['effective_member_observations'] += len(chosen)
                            stats[name]['zero_weight_fallback_steps'] += int(ww.sum() <= 1e-12)
                            for config, bg in [('C1', vv.mean(0)), ('C2', np.median(vv, axis=0)), ('C3', core.weighted_median(vv, ww))]:
                                background = np.asarray(bg, np.float32)
                                x = arrays[name + '__' + config]
                                x[global_i, j, 7:9] = background
                                x[global_i, j, 9:11] = (target_v - background).astype(np.float32)
                                x[global_i, j, 11] = 1
                        proofout.write(json.dumps({'region': region, 'window_id': md['window_id'], 'global_row': global_i,
                            'target_mmsi': str(obs['mmsi'][local_i]), 'step': j, 'time_s': when,
                            'target_observed_position_m': pp.tolist(), 'members': details,
                            'selected_member_indices': selection_proof}, separators=(',', ':')) + '\n')
                assert local_i + 1 == 210
            assert seen == set(saved)
            regions_audit[region] = {**dict(counts), 'elapsed_seconds': time.time() - region_started}
            all_counts.update(counts)
            print('SENSITIVITY_REGION_COMPLETE', region, counts['inherited_common_steps'], round(time.time() - region_started, 2), flush=True)
            del source, saved, obs
            gc.collect()
    defaults = ('baseline', 'radius_8p0', 'e0_2p5', 'sigma_2p0', 'heading_45')
    for view in views:
        name = view['name']
        support = arrays[name + '__support']
        assert not np.any(support & ~frozen['common_valid'])
        arrays[name + '__C4'] = arrays[name + '__C3'].copy()
        candidate = arrays[name + '__candidate_count']
        stats[name]['valid_target_steps'] = int(frozen['mask'].sum())
        stats[name]['common_context_percent'] = float(100 * support.sum() / frozen['mask'].sum())
        stats[name]['fraction_of_inherited_support_retained'] = float(support.sum() / frozen['common_valid'].sum())
        usable = candidate[support]
        stats[name]['actual_member_min'] = int(usable.min()) if len(usable) else 0
        stats[name]['actual_member_max'] = int(usable.max()) if len(usable) else 0
        stats[name]['actual_member_mean'] = float(usable.mean()) if len(usable) else None
        stats[name]['actual_member_median'] = float(np.median(usable)) if len(usable) else None
        for config in ('C0', 'C1', 'C2', 'C3', 'C4'):
            x = arrays[name + '__' + config]
            np.testing.assert_array_equal(x[..., :7], frozen['base__C0'][..., :7])
            fm = np.zeros_like(x, bool)
            fm[..., :7] = frozen['mask'][..., None]
            if config != 'C0':
                fm[..., 7:11] = support[..., None]
                fm[..., 11] = frozen['mask']
            arrays[name + '__feature_mask_' + config] = fm
            assert not x[~fm].any() and np.isfinite(x).all()
            if name in defaults:
                np.testing.assert_array_equal(x, frozen['base__' + config])
            if view['family'] in ('e0_mps', 'sigma_mps') and config in ('C1', 'C2'):
                np.testing.assert_array_equal(x, frozen['base__' + config])
            stats[name][config + '_values_changed_from_core'] = int(np.count_nonzero(x != frozen['base__' + config]))
    assert all_counts['inherited_common_steps'] == 11032
    assert all_counts['reconstructed_member_observations_with_step_reuse'] == 100037
    assert all_counts['unknown_reliability_observations'] == 1475
    np.savez_compressed(dest, **arrays)
    for path, checksum in hashes.items():
        assert sha(path) == checksum, ('source_or_core_changed', path)
    report = {'status': 'complete_validation_only_inference_sensitivity', 'protocol_sha256': sha(protocolpath),
              'builder_sha256': sha(__file__), 'core_builder_sha256': sha(stage / 'data_builder.py'),
              'output_path': str(dest.relative_to(stage)), 'output_sha256': sha(dest),
              'artifact': {'path': str(dest.relative_to(stage)), 'sha256': sha(dest),
                           'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': core.arrsha(value)} for key, value in arrays.items()}},
              'member_provenance': {'path': str(proofpath.relative_to(stage)), 'sha256': sha(proofpath)},
              'source_sha256': hashes, 'views': views, 'counts': {key: dict(value) for key, value in stats.items()},
              'regions': regions_audit, 'reconstruction_counts': dict(all_counts),
              'selected_members_and_times_sha256': {key: value.hexdigest() for key, value in members_hash.items()},
              'assertions': {'windows_1680_retained': True, 'all_default_views_exactly_match_core': list(defaults),
                  'raw_target_and_times_unchanged': True, 'labels_copied_as_metadata_only': True,
                  'actual_observed_e_kin_used_no_weight_inversion': True, 'source_members_distances_headings_weights_verified': True,
                  'causal_report_age_0_to_300_seconds': True, 'support_cannot_expand': True, 'no_training_or_test_window_file_read': True,
                  'no_model_score_read': True, 'no_parameter_selection': True, 'frozen_code_and_used_core_data_hashes_unchanged': True},
              'not_run': protocol['not_run'], 'elapsed_seconds': time.time() - started}
    write_json(stage / 'sensitivity_audit.json', report)
    print('SENSITIVITY_COMPLETE', sha(dest), round(report['elapsed_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
