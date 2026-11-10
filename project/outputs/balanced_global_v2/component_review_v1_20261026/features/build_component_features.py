"""Derive fixed train/validation component inputs from the frozen scene view.

This module never opens a test NPZ. It does not fit a new field, change labels,
or use model scores. Saved physical residuals recover ungated information even
when the gate is exactly zero. All additions are written to a new directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

REGIONS = ('houston', 'ny', 'la', 'puget', 'great_belt', 'haugesund', 'kattegat', 'shanghai')
SPLITS = ('train', 'val')
VARIANTS = ('raw16', 'raw_meta', 'gated_meta', 'mean_meta', 'affine_meta',
            'reliable_raw', 'reliable_gate_target', 'reliable_gate_residual',
            'legacy_full', 'reliable_raw_fallback')
VSCALE_MPS = 5 * 1852. / 3600.
REQUIRED = {'raw', 'gated', 'context', 'plain_context', 'plain_affine_context',
            'residual_mps', 'background_mean_mps', 'background_mean_valid',
            'context_valid', 'valid_neighbor_count', 'plain_neighbor_count',
            'boundary_context_valid', 'mask', 'self_gate_weight', 'self_reliability',
            'kinematic_valid', 'y', 'mmsi', 'window_id', 'base_window_id', 'region',
            'behavior_type', 'noise_type', 'noise_scope', 'target_noise_effective',
            'neighbor_noise_effective', 'observation_noise_effective', 'observation_noise',
            'target_sog_kn', 'target_cog_deg', 'pre_noise_sog_kn', 'pre_noise_cog_deg'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def validate_source(data):
    missing = REQUIRED - data.keys()
    if missing:
        raise ValueError(f'Missing source fields: {sorted(missing)}')
    mask = data['mask']
    if mask.dtype != bool or mask.ndim != 2 or mask.shape[1] != 20:
        raise ValueError('Expected boolean N x 20 mask')
    n, t = mask.shape
    if not np.all(mask.sum(1) > 0):
        raise ValueError('Empty windows are forbidden')
    if not np.array_equal(mask, np.arange(t)[None, :] < mask.sum(1)[:, None]):
        raise ValueError('Expected right-padded chronological source windows')
    for key, width in [('raw', 7), ('gated', 7), ('context', 16),
                       ('plain_context', 16), ('plain_affine_context', 16)]:
        x = data[key]
        if x.shape != (n, t, width) or x.dtype != np.float32:
            raise ValueError(f'Unexpected {key} shape or dtype')
        if not np.isfinite(x).all() or np.any(x[~mask] != 0):
            raise ValueError(f'Nonfinite or nonzero padded {key}')
    if data['y'].shape != (n,) or not np.isin(data['y'], [0, 1]).all():
        raise ValueError('Expected binary labels')
    for key in ['window_id', 'base_window_id']:
        if data[key].shape != (n,) or len(set(data[key].astype(str))) != n:
            raise ValueError(f'Nonunique or malformed {key}')
    for key in ['mmsi', 'region', 'behavior_type', 'noise_type', 'noise_scope',
                'target_noise_effective', 'neighbor_noise_effective',
                'observation_noise_effective', 'observation_noise']:
        if data[key].shape != (n,):
            raise ValueError(f'Unexpected metadata shape: {key}')
    for key in ['self_gate_weight', 'self_reliability', 'kinematic_valid', 'context_valid',
                'valid_neighbor_count', 'plain_neighbor_count', 'boundary_context_valid',
                'background_mean_valid', 'target_sog_kn', 'target_cog_deg',
                'pre_noise_sog_kn', 'pre_noise_cog_deg']:
        if data[key].shape != mask.shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Unexpected physical field: {key}')
    for key in ['residual_mps', 'background_mean_mps']:
        if data[key].shape != (n, t, 2) or not np.isfinite(data[key]).all():
            raise ValueError(f'Unexpected vector field: {key}')
    gate = data['self_gate_weight']
    if np.any((gate[mask] < 0) | (gate[mask] > 1)):
        raise ValueError('Gate outside [0,1]')
    expected_gated = data['raw'].copy()
    expected_gated[..., [0, 1, 2, 3, 5, 6]] *= gate[..., None]
    if not np.allclose(expected_gated, data['gated'], rtol=3e-6, atol=3e-7):
        raise ValueError('Source gate and target channels disagree')
    if not np.array_equal(data['context'][..., :7], data['gated']):
        raise ValueError('Existing context does not contain the saved gated channels')
    expected_residual = data['residual_mps'] / VSCALE_MPS * gate[..., None]
    if not np.allclose(expected_residual, data['context'][..., 7:9], rtol=3e-6, atol=3e-7):
        raise ValueError('Source physical residual and context residual disagree')
    if not np.array_equal(data['context'][..., 12].astype(bool), data['context_valid']):
        raise ValueError('Context-valid slot disagrees with physical flag')
    if np.any(data['residual_mps'][~data['context_valid']] != 0):
        raise ValueError('Invalid field contains a nonzero residual')
    return mask


def derive_variants(data):
    mask = validate_source(data)
    raw16 = np.zeros_like(data['context'])
    raw16[..., :7] = data['raw']
    raw_meta = raw16.copy()
    raw_meta[..., 9:12] = data['context'][..., 9:12]
    gated_meta = raw_meta.copy()
    gated_meta[..., :7] = data['gated']
    mean_meta = data['plain_context'].copy()
    mean_meta[..., 9:12] = data['context'][..., 9:12]
    affine_meta = data['plain_affine_context'].copy()
    affine_meta[..., 9:12] = data['context'][..., 9:12]
    reliable_raw = data['context'].copy()
    reliable_raw[..., :7] = data['raw']
    reliable_raw[..., 7:9] = data['residual_mps'] / VSCALE_MPS
    gate_target = reliable_raw.copy()
    gate_target[..., :7] = data['gated']
    gate_residual = reliable_raw.copy()
    gate_residual[..., 7:9] = data['context'][..., 7:9]
    fallback_active = (mask & ~data['context_valid'] & data['background_mean_valid']
                       & (data['valid_neighbor_count'] >= 4)
                       & data['boundary_context_valid'])
    fallback = reliable_raw.copy()
    mean_residual = data['raw'][..., 2:4] - data['background_mean_mps'] / VSCALE_MPS
    fallback[..., 7:9][fallback_active] = mean_residual[fallback_active]
    fallback[..., 12][fallback_active] = 1.
    fallback[..., 14:16][fallback_active] = 0.
    variants = dict(zip(VARIANTS, [raw16, raw_meta, gated_meta, mean_meta, affine_meta,
                                  reliable_raw, gate_target, gate_residual,
                                  data['context'].copy(), fallback]))
    for key, x in variants.items():
        if not np.isfinite(x).all() or np.any(x[~mask] != 0):
            raise ValueError(f'Invalid derived values or padding in {key}')
    return variants, fallback_active


def _mean(a):
    return float(np.mean(a)) if len(a) else None


def diagnostic_row(data, selected):
    mask = data['mask'][selected]
    valid = mask & data['kinematic_valid'][selected]
    gate = data['self_gate_weight'][selected]
    low = valid & (gate < .3)
    raw = data['raw'][selected]
    gated = data['gated'][selected]
    reported_speed = np.linalg.norm(raw[..., 2:4], axis=-1) * 5.
    gated_speed = np.linalg.norm(gated[..., 2:4], axis=-1) * 5.
    direction_original = np.linalg.norm(raw[..., 5:7], axis=-1)
    direction_gated = np.linalg.norm(gated[..., 5:7], axis=-1)
    unchanged_speed = np.isclose(data['target_sog_kn'][selected], data['pre_noise_sog_kn'][selected],
                                 rtol=0, atol=1e-6)
    course_delta = (data['target_cog_deg'][selected] - data['pre_noise_cog_deg'][selected] + 180) % 360 - 180
    unchanged_velocity = unchanged_speed & (np.abs(course_delta) < 1e-6)
    fallback = data['fallback_activated'][selected]
    valid_count = int(valid.sum())
    step_count = int(mask.sum())
    low_count = int(low.sum())
    field_valid = data['context_valid'][selected]
    supported = mask & (data['valid_neighbor_count'][selected] >= 4) & data['boundary_context_valid'][selected]
    return {
        'windows': int(np.sum(selected)), 'positive_windows': int(data['y'][selected].sum()),
        'valid_target_steps': step_count, 'kinematically_assessable_steps': valid_count,
        'unknown_kinematics_steps': int((mask & ~valid).sum()),
        'low_gate_steps_lt_0_3': low_count,
        'low_gate_fraction_of_assessable_steps': low_count / valid_count if valid_count else None,
        'low_gate_fraction_of_all_valid_steps': low_count / step_count if step_count else None,
        'attenuated_gate_steps_lt_0_95': int((valid & (gate < .95)).sum()),
        'low_gate_reported_speed_mean_kn': _mean(reported_speed[low]),
        'low_gate_reported_speed_ge_3kn_steps': int((low & (reported_speed >= 3)).sum()),
        'low_gate_reported_speed_attenuation_mean_kn': _mean((reported_speed - gated_speed)[low]),
        'low_gate_direction_norm_attenuation_mean': _mean((direction_original - direction_gated)[low]),
        'low_gate_reported_velocity_unchanged_by_controlled_corruption_steps': int((low & unchanged_velocity).sum()),
        'low_gate_reported_velocity_unchanged_by_controlled_corruption_fraction': _mean(unchanged_velocity[low]),
        'trusted_affine_valid_steps': int((mask & field_valid).sum()),
        'trusted_affine_valid_fraction': _mean(field_valid[mask]),
        'at_least_four_trusted_neighbors_and_complete_coverage_steps': int(supported.sum()),
        'trusted_constant_fallback_steps': int(fallback.sum()),
        'trusted_constant_fallback_fraction_of_valid_steps': _mean(fallback[mask]),
        'trusted_constant_fallback_fraction_of_supported_steps': _mean(fallback[supported]),
        'field_valid_after_fallback_fraction': _mean((field_valid | fallback)[mask]),
    }


def diagnostics(data):
    n = len(data['y'])
    effects = np.where(data['target_noise_effective'], 'target', 'no_target')
    effects = np.char.add(np.char.add(effects, '+'),
                          np.where(data['neighbor_noise_effective'], 'neighbors', 'no_neighbors'))
    categories = {'region': data['region'].astype(str), 'noise_scope': data['noise_scope'].astype(str),
                  'noise_type': data['noise_type'].astype(str), 'actual_effect': effects,
                  'class': data['y'].astype(str)}
    out = {'all': diagnostic_row(data, np.ones(n, bool)), 'strata': {}}
    for name, values in categories.items():
        out['strata'][name] = {str(v): diagnostic_row(data, values == v) for v in np.unique(values)}
    out['strata']['actual_effect_by_class'] = {}
    for effect in np.unique(effects):
        for y in [0, 1]:
            selection = (effects == effect) & (data['y'] == y)
            if selection.any():
                out['strata']['actual_effect_by_class'][f'{effect}|y={y}'] = diagnostic_row(data, selection)
    return out


def source_provenance(source, protocol):
    release = source.parent
    old_protocol_path = release / 'benchmark_protocol.json'
    old_protocol = json.loads(old_protocol_path.read_text(encoding='utf-8'))
    expected = protocol['source_label_manifest_sha256']
    if old_protocol['labelled_core']['labels_sha256'] != expected:
        raise ValueError('Frozen source label identity does not match the new protocol')
    # Label identity is verified through the frozen release protocol and file
    # manifest. The combined label table is not opened, avoiding test labels.
    manifest_path = release / 'stage_file_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))['files']
    if manifest['label_manifest.parquet']['sha256'] != expected:
        raise ValueError('Source label manifest identity mismatch')
    for rel, key in [
        ('reproducibility/build_labeled_features.py', 'source_feature_builder_sha256'),
        ('reproducibility/source_context/context_features.py', 'source_context_sha256')]:
        if sha(release / rel) != protocol[key]:
            raise ValueError(f'Source code hash mismatch: {rel}')
    if sha(old_protocol_path) != manifest['benchmark_protocol.json']['sha256']:
        raise ValueError('Frozen source protocol hash mismatch')
    return manifest, {'source_release_protocol_sha256': sha(old_protocol_path),
                      'source_release_file_manifest_sha256': sha(manifest_path),
                      'label_identity_verified_from_release_metadata': expected,
                      'combined_label_table_opened': False}


def build(source, out, protocol_path):
    source, out, protocol_path = Path(source), Path(out), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol['feature_names'] != list(VARIANTS) or protocol['splits_read'] != list(SPLITS):
        raise ValueError('Feature list or allowed split list differs from fixed protocol')
    if protocol['regions'] != list(REGIONS):
        raise ValueError('Region list differs from fixed protocol')
    if source.name != protocol['source_view']:
        raise ValueError('Unexpected source feature view')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite output: {out}')
    manifest, proof = source_provenance(source, protocol)
    out.mkdir(parents=True, exist_ok=False)
    input_proof, output_proof = [], []
    collected = {split: [] for split in SPLITS}
    ids = {split: {'mmsi': set(), 'window_id': set(), 'base_window_id': set()} for split in SPLITS}
    diagnostic_keys = {'mask', 'kinematic_valid', 'self_gate_weight', 'raw', 'gated', 'y', 'region',
                       'noise_scope', 'noise_type', 'target_noise_effective', 'neighbor_noise_effective',
                       'target_sog_kn', 'target_cog_deg', 'pre_noise_sog_kn', 'pre_noise_cog_deg',
                       'context_valid', 'valid_neighbor_count', 'boundary_context_valid', 'fallback_activated'}
    for region in REGIONS:
        (out / region).mkdir()
        for split in SPLITS:
            path = source / region / f'{split}.npz'
            before = sha(path)
            source_key = f'{source.name}/{region}/{split}.npz'
            if before != manifest[source_key]['sha256']:
                raise ValueError(f'Input does not match frozen release manifest: {path}')
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k].copy() for k in z.files}
            expected_n = protocol['samples'][f'{split}_per_region']
            if len(data['y']) != expected_n or not np.all(data['region'].astype(str) == region):
                raise ValueError(f'Source region or sample size mismatch: {path}')
            variants, fallback = derive_variants(data)
            if sha(path) != before:
                raise RuntimeError(f'Source changed while reading: {path}')
            for name in ids[split]:
                new = set(data[name].astype(str))
                if name != 'mmsi' and ids[split][name] & new:
                    raise ValueError(f'Duplicate {name} across regions in {split}')
                ids[split][name].update(new)
            payload = {**data, **variants, 'fallback_activated': fallback}
            dest = out / region / f'{split}.npz'
            with open(dest, 'xb') as handle:
                np.savez_compressed(handle, **payload)
            input_proof.append({'region': region, 'split': split, 'path': str(path), 'sha256': before,
                                'windows': len(data['y']), 'positive': int(data['y'].sum())})
            output_proof.append({'region': region, 'split': split, 'path': str(dest), 'sha256': sha(dest),
                                 'windows': len(data['y']), 'fallback_steps': int(fallback.sum())})
            collected[split].append({k: payload[k] for k in diagnostic_keys})
            print(f'DERIVED {region}/{split} n={len(data["y"])} fallback_steps={int(fallback.sum())}', flush=True)
    isolation = {key: len(ids['train'][key] & ids['val'][key]) for key in ids['train']}
    if any(isolation.values()):
        raise ValueError(f'Train/validation identity leakage: {isolation}')
    split_stats = {}
    for split, parts in collected.items():
        combined = {key: np.concatenate([p[key] for p in parts]) for key in diagnostic_keys}
        split_stats[split] = diagnostics(combined)
    source_check = [{'path': entry['path'], 'unchanged': sha(entry['path']) == entry['sha256']} for entry in input_proof]
    if not all(x['unchanged'] for x in source_check):
        raise RuntimeError('Source mutation detected during feature derivation')
    report = {'status': 'passed', 'protocol_sha256': sha(protocol_path), 'builder_sha256': sha(__file__),
              'variants': list(VARIANTS), 'velocity_scale_mps': VSCALE_MPS, 'input_provenance': proof,
              'source_files': input_proof, 'output_files': output_proof, 'source_preservation': source_check,
              'train_val_intersections': isolation, 'splits_opened': list(SPLITS), 'test_files_opened': 0,
              'counts': {s: split_stats[s]['all']['windows'] for s in SPLITS},
              'fallback_rule': 'invalid affine, saved trusted mean valid, >=4 accepted neighbors, complete known/available coverage flag, valid target step',
              'fallback_interpretation': 'Non-sparse affine failure; saved NPZ flags do not resolve geometry vs conditioning vs numerical-solve failure.',
              'label_audit_scope': 'Labels and row identities reused unchanged from hash-verified source NPZ; combined label table and all test NPZ remain unopened.',
              'retained_source_fields': 'All original NPZ arrays are preserved alongside the ten derived feature arrays.'}
    write_json(out / 'feature_manifest.json', report)
    write_json(out / 'feature_diagnostics.json', {
        'status': 'completed', 'protocol_sha256': sha(protocol_path), 'splits': split_stats,
        'interpretation': [
            'These are deterministic training/validation feature diagnostics, not causal evidence of performance changes.',
            'Kinematic residual establishes incompatibility; it does not identify which sensor channel is incorrect.',
            'Reported velocity unchanged by controlled corruption does not establish that the original velocity is physically error-free.',
            'Gate attenuation is measured on numerical input channels; the network may also use explicit reliability metadata.',
            'Fallback rates count target steps and are not independent sample counts.']})
    print(json.dumps({'status': 'passed', 'out': str(out), 'counts': report['counts'], 'intersections': isolation}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.out, args.protocol)


if __name__ == '__main__':
    main()
