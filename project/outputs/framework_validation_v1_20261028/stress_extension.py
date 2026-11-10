"""Evaluation-only observed-report stress extension; no training or score reads."""
from pathlib import Path
from collections import Counter
import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
OBSERVED = ('target_t', 'target_sog_kn', 'target_cog_deg', 'target_mask',
            'neighbor_t', 'neighbor_pos', 'neighbor_sog_kn', 'neighbor_cog_deg',
            'neighbor_mask', 'neighbor_mmsi', 'member_offsets', 'cohort_mmsi')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_npz(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if keys is None else keys)}


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def direction(window_id, member):
    digest = hashlib.sha256(('framework-stress-position-v1:' + window_id + ':' + member).encode()).digest()
    u = int.from_bytes(digest[:8], 'big') / 2 ** 64
    theta = 2 * np.pi * u
    return np.asarray([np.cos(theta), np.sin(theta)], np.float64), digest.hex()


def specs(protocol):
    speed = protocol['speed_path']['levels']
    position = protocol['position_path']['levels_m']
    rows = []
    for lam in speed:
        rows.append((f'speed_l{round(lam * 100):03d}', 'speed', lam, 0))
    for amplitude in position:
        rows.append((f'position_m{amplitude:03d}', 'position', 0, amplitude))
    for lam, amplitude in protocol['combined_path']['paired_levels']:
        rows.append((f'combined_l{round(lam * 100):03d}_m{amplitude:03d}', 'combined', lam, amplitude))
    assert [r[0] for r in rows] == protocol['view_names']
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027'))
    args = parser.parse_args()
    stage = args.stage.resolve()
    assert stage.name == 'framework_validation_v1_20261028'
    outdir = stage / 'extra_data'
    outdir.mkdir(exist_ok=True)
    dest = outdir / 'relation_stress_test.npz'
    assert not dest.exists() and not (stage / 'stress_extension_audit.json').exists()
    started = time.time()
    protocolpath = stage / 'stress_extension_protocol.json'
    protocol = json.loads(protocolpath.read_text(encoding='utf8'))
    filehashes = {str(protocolpath): sha(protocolpath), str(Path(__file__).resolve()): sha(__file__)}
    for name, field in [('data_builder.py', 'core_builder_sha256'), ('data_manifest.json', 'core_data_manifest_sha256'),
                        ('data/relation/test.npz', 'core_relation_test_sha256')]:
        p = stage / name
        assert sha(p) == protocol[field], name
        filehashes[str(p)] = sha(p)
    lockpath = stage / 'code_lock.json'
    lock = json.loads(lockpath.read_text())
    filehashes[str(lockpath)] = sha(lockpath)
    for name, expected in {**lock['files'], **lock['data_files']}.items():
        p = stage / name
        assert sha(p) == expected, name
        filehashes[str(p)] = expected
    spec = importlib.util.spec_from_file_location('frozen_context_builder_for_stress', stage / 'data_builder.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    view_specs = specs(protocol)
    core_data = read_npz(stage / 'data/relation/test.npz', None)
    manifestpath = args.source_root / 'dataset_manifest_combined.json'
    source_hashes = json.loads((stage / 'data_sources.json').read_text())['sha256']
    assert sha(manifestpath) == source_hashes[str(manifestpath.resolve())]
    filehashes[str(manifestpath.resolve())] = sha(manifestpath)
    manifest = json.loads(manifestpath.read_text())
    rows = sorted([r for r in manifest['samples'] if r['version'] == 'relaxed' and r['split'] == 'test'],
                  key=lambda r: (r['region'], r['window_id']))
    assert len(rows) == 220 and len(core_data['y']) == 880
    metadata = ['y', 'length', 'region', 'mmsi', 'source_mmsi', 'case', 'source_id', 'window_id',
                'base_window_id', 'source_window_id', 'split', 'mask', 'time_s', 'time_start_s', 'time_end_s',
                'behavior_type', 'noise_type', 'noise_scope', 'target_noise_effective',
                'neighbor_noise_effective', 'observation_noise_effective']
    arrays = {k: core_data[k].copy() for k in metadata}
    for view, _, _, _ in view_specs:
        for config in ('C0', 'C1', 'C2', 'C3', 'C4'):
            arrays[view + '__' + config] = np.zeros((880, 20, 7), np.float32)
            arrays[view + '__feature_mask_' + config] = core_data['base__feature_mask_' + config].copy()
    audits = {v: Counter() for v, _, _, _ in view_specs}
    pair_hashes = {v: hashlib.sha256() for v, _, _, _ in view_specs}
    source_ids = []
    provenance_path = outdir / 'stress_member_provenance.jsonl.gz'
    assert not provenance_path.exists()
    with gzip.open(provenance_path, 'xt', encoding='utf8') as proofout:
        for rowidx, row in enumerate(rows):
            start = rowidx * 4
            source_ids.append(row['region'] + ':' + row['window_id'])
            assert np.all(core_data['source_window_id'][start:start + 4] == row['window_id'])
            assert np.all(core_data['source_mmsi'][start:start + 4] == row['mmsi'])
            observations = {}
            for view in ('base', 'suffix'):
                entry = row['files'][view]
                path = (args.source_root / entry['path']).resolve()
                assert path.is_relative_to(args.source_root.resolve())
                assert sha(path) == entry['sha256'] == source_hashes[str(path)]
                filehashes[str(path)] = entry['sha256']
                observations[view] = read_npz(path, OBSERVED)
            base, suffix = observations['base'], observations['suffix']
            for key in OBSERVED:
                if key != 'neighbor_sog_kn':
                    np.testing.assert_array_equal(base[key], suffix[key])
            assert list(base['cohort_mmsi'].astype(str)) == row['cohort_mmsi']
            selected = np.zeros(base['neighbor_sog_kn'].shape[1], bool)
            offsets = np.zeros((len(selected), 2), np.float64)
            members = []
            for member in row['stress']['members']:
                j = int(member['member_index'])
                assert str(base['cohort_mmsi'][j]) == member['mmsi']
                lo, hi = map(int, base['member_offsets'][j:j + 2])
                local = int(member['indices']['suffix']['report_index'])
                indices = [lo + local, lo + local + 1]
                assert lo <= indices[0] < indices[1] < hi
                assert np.all(base['neighbor_t'][:, indices] > base['target_t'][:, row['stress']['anchor_index'], None])
                np.testing.assert_array_equal(base['neighbor_t'][0, indices], member['indices']['suffix']['times_s'])
                vector, direction_sha = direction(row['window_id'], member['mmsi'])
                offsets[indices] = vector
                selected[indices] = True
                members.append({'mmsi': member['mmsi'], 'member_index': j, 'flat_report_indices': indices,
                                'times_s': base['neighbor_t'][0, indices].tolist(), 'direction_xy': vector.tolist(),
                                'direction_seed_sha256': direction_sha,
                                'frozen_suffix_grid_hits': member['indices']['suffix']['suffix_grid_hits']})
            assert int(selected.sum()) == len(members) * 2
            assert not np.any(base['neighbor_sog_kn'][:, ~selected] != suffix['neighbor_sog_kn'][:, ~selected])
            provenance = {'source_id': source_ids[-1], 'source_window_id': row['window_id'], 'target_mmsi': row['mmsi'],
                          'region': row['region'], 'cohort_count': len(base['cohort_mmsi']), 'selected_count': len(members),
                          'base_sha256': row['files']['base']['sha256'], 'suffix_sha256': row['files']['suffix']['sha256'],
                          'members': members}
            proofout.write(json.dumps(provenance, separators=(',', ':')) + '\n')
            baseline_proof = None
            for view, family, lam, amplitude in view_specs:
                obs = {key: value.copy() for key, value in base.items()}
                if lam == 1:
                    obs['neighbor_sog_kn'] = suffix['neighbor_sog_kn'].copy()
                elif lam != 0:
                    obs['neighbor_sog_kn'] = base['neighbor_sog_kn'] + lam * (suffix['neighbor_sog_kn'] - base['neighbor_sog_kn'])
                if amplitude:
                    obs['neighbor_pos'] = base['neighbor_pos'] + amplitude * offsets[None]
                for key in OBSERVED:
                    if key not in ('neighbor_sog_kn', 'neighbor_pos'):
                        np.testing.assert_array_equal(obs[key], base[key])
                np.testing.assert_array_equal(obs['neighbor_sog_kn'][:, ~selected], base['neighbor_sog_kn'][:, ~selected])
                np.testing.assert_array_equal(obs['neighbor_pos'][:, ~selected], base['neighbor_pos'][:, ~selected])
                if amplitude:
                    np.testing.assert_allclose(np.linalg.norm(obs['neighbor_pos'][:, selected] - base['neighbor_pos'][:, selected], axis=-1), amplitude, rtol=0, atol=1e-7)
                audits[view]['sources'] += 1
                audits[view]['available_members'] += len(base['cohort_mmsi'])
                audits[view]['selected_members'] += len(members)
                audits[view]['neighbor_reports_across_siblings'] += int(base['neighbor_mask'].sum())
                audits[view]['speed_changed_reports_across_siblings'] += int(np.count_nonzero(obs['neighbor_sog_kn'] != base['neighbor_sog_kn']))
                audits[view]['position_changed_reports_across_siblings'] += int(np.any(obs['neighbor_pos'] != base['neighbor_pos'], axis=-1).sum())
                audits[view]['selected_suffix_grid_hits_per_sibling'] += sum(m['frozen_suffix_grid_hits'] for m in members)
                proofs = []
                for case in range(4):
                    i = start + case
                    n = int(core_data['length'][i])
                    np.testing.assert_array_equal(obs['target_t'][case], core_data['time_s'][i, :n])
                    features, paired, _, reliability = core.relation_features(obs, case)
                    proofs.append(paired)
                    pair_hashes[view].update(json.dumps([row['window_id'], case, paired], separators=(',', ':')).encode())
                    for key, value in reliability.items():
                        audits[view][key] += value
                    for config in ('C0', 'C1', 'C2', 'C3'):
                        arrays[view + '__' + config][i, :n] = features[config]
                if baseline_proof is None:
                    baseline_proof = proofs
                else:
                    assert proofs == baseline_proof, (view, row['window_id'], 'membership/time changed')
            if (rowidx + 1) % 40 == 0:
                print('STRESS_SOURCES', rowidx + 1, '/', len(rows), flush=True)
    assert len(set(source_ids)) == 220
    for view, family, lam, amplitude in view_specs:
        arrays[view + '__C4'] = arrays[view + '__C3'].copy()
        for config in ('C0', 'C1', 'C2', 'C3', 'C4'):
            x = arrays[view + '__' + config]
            assert x.shape == (880, 20, 7) and np.isfinite(x).all()
            np.testing.assert_array_equal(x[..., :4], core_data['base__C0'][..., :4])
            assert not x[~arrays[view + '__feature_mask_' + config]].any()
            if config != 'C0':
                np.testing.assert_array_equal(x[..., 6], core_data['base__C1'][..., 6])
                np.testing.assert_array_equal(x[0::4, :, 4:], x[2::4, :, 4:])
                np.testing.assert_array_equal(x[1::4, :, 4:], x[3::4, :, 4:])
            if lam == 0 and amplitude == 0:
                np.testing.assert_array_equal(x, core_data['base__' + config])
            if family == 'speed' and lam == 1:
                np.testing.assert_array_equal(x, core_data['suffix__' + config])
            if family == 'position' and config in ('C1', 'C2'):
                np.testing.assert_array_equal(x, core_data['base__' + config])
            audits[view][config + '_values_different_from_base'] = int(np.count_nonzero(x != core_data['base__' + config]))
    assert len({h.hexdigest() for h in pair_hashes.values()}) == 1
    np.savez_compressed(dest, **arrays)
    for path, checksum in filehashes.items():
        assert sha(path) == checksum, ('source_or_core_changed', path)
    report = {'status': 'complete_observation_level_evaluation_only', 'no_training': True, 'no_scores_read': True,
              'protocol_sha256': sha(protocolpath), 'extension_builder_sha256': sha(__file__),
              'core_code_lock_sha256': sha(lockpath), 'sources': 220, 'variants': 880, 'target_mmsi': len(set(arrays['mmsi'])),
              'view_specs': [{'view': view, 'family': family, 'speed_lambda': lam, 'position_amplitude_m': amplitude}
                             for view, family, lam, amplitude in view_specs],
              'source_file_hashes': filehashes, 'artifact': {'path': str(dest.relative_to(stage)), 'sha256': sha(dest),
                'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': core.arrsha(value)} for key, value in arrays.items()}},
              'member_provenance': {'path': str(provenance_path.relative_to(stage)), 'sha256': sha(provenance_path)},
              'paired_members_and_times_sha256': {v: h.hexdigest() for v, h in pair_hashes.items()},
              'counts': {key: dict(value) for key, value in audits.items()},
              'assertions': {'all_zero_endpoints_exactly_core_base': True, 'speed_one_exactly_core_suffix': True,
                             'position_only_C1_C2_exactly_core_base': True, 'all_target_inputs_unchanged': True,
                             'metadata_times_masks_and_labels_copied_exactly': True, 'all_cohorts_and_report_alignment_unchanged': True,
                             'four_sibling_marginal_pairing_unchanged': True, 'all_registered_frozen_files_unchanged': True,
                             'only_observed_arrays_used_for_features': list(OBSERVED)},
              'scope': protocol['base_meaning'], 'elapsed_seconds': time.time() - started}
    write_json(stage / 'stress_extension_audit.json', report)
    print('STRESS_EXTENSION_COMPLETE', sha(dest), round(report['elapsed_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
