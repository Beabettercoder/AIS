"""Read-only train/validation field-availability decomposition.

Writes one new JSON report. No features, model inputs or experiment settings are
changed. Counts are target-time steps, with windows reported separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

REGIONS = ('houston', 'ny', 'la', 'puget', 'great_belt', 'haugesund', 'kattegat', 'shanghai')
SPLITS = ('train', 'val')
FIELDS = ('mask', 'coverage_known', 'boundary_context_valid', 'context_valid',
          'valid_neighbor_count', 'plain_neighbor_count', 'background_mean_valid',
          'fallback_activated', 'target_sog_kn')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def summarise(data):
    m=data['mask'];b=data['boundary_context_valid'];k=data['coverage_known']
    f=data['context_valid'];t=data['valid_neighbor_count'];p=data['plain_neighbor_count']
    if any(data[name].shape != m.shape for name in FIELDS):
        raise ValueError('Per-step shapes differ')
    if np.any(m & (t > p)):
        raise ValueError('Trusted pool cannot exceed the comparable plain pool')
    if np.any(m & f & (~b | (t < 4))):
        raise ValueError('A valid affine field lacks boundary or neighborhood support')
    if np.any(m & ~b & ~k):
        raise ValueError('Unknown coverage is unexpectedly rejected by the source policy')
    cats={
        'coverage_boundary_insufficient':m & ~b,
        'boundary_usable_but_trusted_neighbors_below_four':m & b & (t < 4),
        'boundary_usable_four_or_more_neighbors_but_affine_invalid':m & b & (t >= 4) & ~f,
        'trusted_affine_valid':m & f,
    }
    membership=sum(x.astype(np.int8) for x in cats.values())
    if not np.array_equal(membership,m.astype(np.int8)):
        raise ValueError('Availability categories are not exhaustive and mutually exclusive')
    fallback_expected=cats['boundary_usable_four_or_more_neighbors_but_affine_invalid'] & data['background_mean_valid']
    if not np.array_equal(fallback_expected,data['fallback_activated']):
        raise ValueError('Recorded constant-mean fallback does not follow the fixed rule')
    valid_steps=int(m.sum());boundary_steps=int((m&b).sum())
    cat_counts={name:int(x.sum()) for name,x in cats.items()}
    fractions={name:count/valid_steps if valid_steps else None for name,count in cat_counts.items()}
    low_speed=m & (data['target_sog_kn'] < .5)
    ordinary_short=m & b & (p < 4)
    screening_short=m & b & (p >= 4) & (t < 4)
    retained=m & b & (p >= 4) & (t >= 4)
    if int((ordinary_short | screening_short | retained).sum()) != boundary_steps:
        raise ValueError('Plain/trusted support decomposition is not exhaustive')
    sparse_total=cat_counts['boundary_usable_but_trusted_neighbors_below_four']
    plain_enough=int((m & b & (p >= 4)).sum())
    trusted_enough=int((m & b & (t >= 4)).sum())
    return {
        'windows':int(m.shape[0]), 'valid_target_steps':valid_steps,
        'category_denominator':'valid_target_steps; one category per mask=True target step',
        'category_counts':cat_counts,'category_fractions_of_valid_steps':fractions,
        'coverage':{
            'known_roi_steps':int((m&k).sum()),
            'unknown_roi_steps':int((m&~k).sum()),
            'known_roi_complete_radius_steps':int((m&k&b).sum()),
            'known_roi_incomplete_radius_steps':int((m&k&~b).sum()),
            'unknown_roi_allowed_by_available_observations_policy_steps':int((m&~k&b).sum()),
            'valid_affine_with_unknown_roi_steps':int((m&f&~k).sum()),
            'boundary_policy_usable_steps':boundary_steps,
        },
        'plain_vs_trusted_within_boundary_usable_steps':{
            'denominator_steps':boundary_steps,
            'plain_neighbors_ge_four_steps':plain_enough,
            'trusted_neighbors_ge_four_steps':trusted_enough,
            'support_loss_after_reliability_screening_steps':int(screening_short.sum()),
            'support_loss_fraction_of_plain_ge_four_steps':int(screening_short.sum())/plain_enough if plain_enough else None,
            'plain_neighbors_below_four_steps':int(ordinary_short.sum()),
            'plain_below_four_target_sog_lt_half_knot_steps':int((ordinary_short&low_speed).sum()),
            'plain_below_four_target_sog_ge_half_knot_steps':int((ordinary_short&~low_speed).sum()),
            'trusted_below_four_total_steps':sparse_total,
            'ordinary_shortage_fraction_of_trusted_shortage':int(ordinary_short.sum())/sparse_total if sparse_total else None,
            'screening_loss_fraction_of_trusted_shortage':int(screening_short.sum())/sparse_total if sparse_total else None,
        },
        'plain_vs_trusted_all_valid_steps':{
            'denominator_steps':valid_steps,
            'plain_neighbors_ge_four_steps':int((m&(p>=4)).sum()),
            'trusted_neighbors_ge_four_steps':int((m&(t>=4)).sum()),
            'plain_ge_four_trusted_below_four_steps':int((m&(p>=4)&(t<4)).sum()),
        },
        'supported_affine_failure':{
            'affine_invalid_steps':cat_counts['boundary_usable_four_or_more_neighbors_but_affine_invalid'],
            'saved_trusted_mean_available_steps':int(fallback_expected.sum()),
            'saved_trusted_mean_unavailable_steps':int((cats['boundary_usable_four_or_more_neighbors_but_affine_invalid']&~data['background_mean_valid']).sum()),
            'recorded_fallback_steps':int(data['fallback_activated'].sum()),
        },
    }


def run(data_root,out):
    data_root=Path(data_root);out=Path(out)
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    manifest_path=data_root/'feature_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    expected={(r['region'],r['split']):r['sha256'] for r in manifest['output_files']}
    read_proof=[];result={}
    for split in SPLITS:
        chunks=[];per_region={}
        for region in REGIONS:
            p=data_root/region/f'{split}.npz'
            before=sha(p)
            if before != expected[(region,split)]:raise ValueError(f'Input hash mismatch: {p}')
            with np.load(p,allow_pickle=False) as z:
                d={name:z[name] for name in FIELDS}
            per_region[region]=summarise(d);chunks.append(d)
            if sha(p) != before:raise RuntimeError(f'Input changed: {p}')
            read_proof.append({'region':region,'split':split,'path':str(p),'sha256':before})
        merged={name:np.concatenate([x[name] for x in chunks]) for name in FIELDS}
        result[split]={'all_regions':summarise(merged),'per_region':per_region}
    report={
        'status':'passed','code_sha256':sha(__file__),
        'protocol_sha256':manifest['protocol_sha256'],'feature_manifest_sha256':sha(manifest_path),
        'splits_read':list(SPLITS),'test_files_opened':0,'inputs_unchanged':True,
        'input_files':read_proof,'results':result,
        'definitions':{
            'unit':'Target-time step with mask=True. Steps from a window are dependent; counts are not independent samples.',
            'coverage_known':'The source feature builder had a bounding box for the applicable source ROI and window. This flag does not itself establish complete neighborhood coverage.',
            'boundary_context_valid':'For known ROI the full 8 nautical mile search circle fits inside its bounding box. For unknown ROI the source policy permits available observations, so this flag is True without certifying complete coverage.',
            'plain_neighbor_count':'Unique non-target latest causal observations meeting spatial, heading, reported-speed and data-validity rules, before kinematic-reliability screening. It is not the total number of physical vessels nearby.',
            'valid_neighbor_count':'Subset passing kinematic availability and reliability screening; this count is computed before the coverage-boundary mask.',
            'low_speed_target':'Source context construction returns no neighborhood when target reported SOG <0.5 kn. Such steps are separately counted within the plain-neighbor shortage category.',
            'screening_support_loss':'Plain neighbor count >=4 but trusted neighbor count <4 at the same step. This identifies support lost after reliability screening, not necessarily harmful or incorrect screening.',
            'supported_affine_failure':'Coverage-policy usable and >=4 trusted neighbors but affine field invalid. Saved NPZ flags do not distinguish geometric degeneracy, condition-number rejection or numerical failure.',
            'causal_scope':'These deterministic data diagnostics locate availability bottlenecks. They do not identify the cause of model-performance differences.',
        },
    }
    with open(out,'x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({s:result[s]['all_regions'] for s in SPLITS}),flush=True)
    return report


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args();run(args.data,args.out)
