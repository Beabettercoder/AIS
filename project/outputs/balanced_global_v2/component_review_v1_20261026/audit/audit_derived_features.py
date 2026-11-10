"""Independent direct equality checks for all train/validation component inputs."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def insist(condition,message):
    if not condition:raise AssertionError(message)


def expected_variants(source):
    raw=source['raw'].astype(np.float32); gated=source['gated'].astype(np.float32)
    ctx=source['context'].astype(np.float32); mask=source['mask'].astype(bool)
    out={};out['raw16']=np.zeros(ctx.shape,np.float32);out['raw16'][...,:7]=raw
    out['raw_meta']=out['raw16'].copy();out['raw_meta'][...,9:12]=ctx[...,9:12]
    out['gated_meta']=out['raw_meta'].copy();out['gated_meta'][...,:7]=gated
    for name,key in [('mean_meta','plain_context'),('affine_meta','plain_affine_context')]:
        out[name]=source[key].astype(np.float32).copy();out[name][...,9:12]=ctx[...,9:12]
    out['reliable_raw']=ctx.copy();out['reliable_raw'][...,:7]=raw
    out['reliable_raw'][...,7:9]=source['residual_mps']/(5*1852/3600)
    out['reliable_gate_target']=out['reliable_raw'].copy();out['reliable_gate_target'][...,:7]=gated
    out['reliable_gate_residual']=out['reliable_raw'].copy();out['reliable_gate_residual'][...,7:9]=ctx[...,7:9]
    out['legacy_full']=ctx.copy()
    fallback=(mask & ~source['context_valid'].astype(bool) & source['background_mean_valid'].astype(bool)
              & (source['valid_neighbor_count']>=4) & source['boundary_context_valid'].astype(bool))
    out['reliable_raw_fallback']=out['reliable_raw'].copy()
    residual=raw[...,2:4]-source['background_mean_mps']/(5*1852/3600)
    out['reliable_raw_fallback'][...,7:9][fallback]=residual[fallback]
    out['reliable_raw_fallback'][...,12][fallback]=1
    out['reliable_raw_fallback'][...,14:16][fallback]=0
    for x in out.values():x[~mask]=0
    return out,fallback


def count_diagnostics(source,fallback):
    mask=source['mask'].astype(bool);assessable=mask & source['kinematic_valid'].astype(bool)
    low=assessable & (source['self_gate_weight']<.3)
    supported=mask & (source['valid_neighbor_count']>=4) & source['boundary_context_valid'].astype(bool)
    return dict(windows=len(mask),positive_windows=int(source['y'].sum()),valid_target_steps=int(mask.sum()),
                kinematically_assessable_steps=int(assessable.sum()),unknown_kinematics_steps=int((mask & ~assessable).sum()),
                low_gate_steps_lt_0_3=int(low.sum()),attenuated_gate_steps_lt_0_95=int((assessable & (source['self_gate_weight']<.95)).sum()),
                trusted_affine_valid_steps=int((mask & source['context_valid'].astype(bool)).sum()),
                at_least_four_trusted_neighbors_and_complete_coverage_steps=int(supported.sum()),
                trusted_constant_fallback_steps=int(fallback.sum()))


def verify_diagnostic_counts(counts,reported,label):
    for key,value in counts.items():
        insist(value==reported[key],f'{label}: diagnostic count {key}')
    ratios={
        'low_gate_fraction_of_assessable_steps':('low_gate_steps_lt_0_3','kinematically_assessable_steps'),
        'low_gate_fraction_of_all_valid_steps':('low_gate_steps_lt_0_3','valid_target_steps'),
        'trusted_affine_valid_fraction':('trusted_affine_valid_steps','valid_target_steps'),
        'trusted_constant_fallback_fraction_of_valid_steps':('trusted_constant_fallback_steps','valid_target_steps'),
        'trusted_constant_fallback_fraction_of_supported_steps':('trusted_constant_fallback_steps','at_least_four_trusted_neighbors_and_complete_coverage_steps')}
    for key,(num,den) in ratios.items():
        expected=counts[num]/counts[den] if counts[den] else None
        actual=reported[key]
        insist(expected==actual or (expected is not None and actual is not None and abs(expected-actual)<1e-12),f'{label}: denominator {key}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    insist(not args.out.exists(),'Refusing to overwrite audit report')
    protocol=json.loads(args.protocol.read_text());reports=[]
    diagnostics=json.loads((args.data/'feature_diagnostics.json').read_text())['splits']
    total_counts={'train':{},'val':{}}
    for region in protocol['regions']:
        for split in ('train','val'):
            src=args.source/region/(split+'.npz');dst=args.data/region/(split+'.npz')
            with np.load(src,allow_pickle=False) as source,np.load(dst,allow_pickle=False) as target:
                expected,fallback=expected_variants(source)
                errors={}
                for name in protocol['feature_names']:
                    x=target[name];e=expected[name]
                    insist(x.shape==e.shape and x.dtype==np.float32,f'{region}/{split}/{name}: shape or dtype')
                    error=float(np.max(np.abs(x-e)));errors[name]=error
                    insist(np.allclose(x,e,rtol=1e-6,atol=1e-7),f'{region}/{split}/{name}: feature mismatch {error}')
                for name in ('raw16','raw_meta','mean_meta','affine_meta','reliable_raw','reliable_gate_residual','reliable_raw_fallback'):
                    insist(np.array_equal(target[name][...,:7],source['raw']),f'{region}/{split}/{name}: raw trajectory not preserved')
                insist(np.array_equal(target['legacy_full'],source['context']),f'{region}/{split}: legacy not exact')
                insist(np.array_equal(target['fallback_activated'],fallback),f'{region}/{split}: fallback mask')
                for key in ('y','mask','mmsi','window_id','base_window_id','region','behavior_type','noise_type','noise_scope',
                            'target_noise_effective','neighbor_noise_effective','observation_noise_effective'):
                    insist(np.array_equal(target[key],source[key]),f'{region}/{split}: metadata changed {key}')
                mask=source['mask'];counts=int(mask.sum());num_fallback=int(fallback.sum())
                check_counts=count_diagnostics(source,fallback)
                verify_diagnostic_counts(check_counts,diagnostics[split]['strata']['region'][region],f'{region}/{split}')
                for key,value in check_counts.items():total_counts[split][key]=total_counts[split].get(key,0)+value
                reports.append(dict(region=region,split=split,source=str(src),source_sha256=digest(src),
                                    derived=str(dst),derived_sha256=digest(dst),windows=len(mask),valid_steps=counts,
                                    fallback_steps=num_fallback,maximum_absolute_errors=errors,diagnostic_counts=check_counts))
    for split in ('train','val'):
        verify_diagnostic_counts(total_counts[split],diagnostics[split]['all'],f'all/{split}')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='passed',scope='All derived train and validation features, source equality and exact specified changes; no test files opened.',
                protocol_sha256=digest(args.protocol),files=len(reports),windows=sum(r['windows'] for r in reports),
                valid_steps=sum(r['valid_steps'] for r in reports),fallback_steps=sum(r['fallback_steps'] for r in reports),
                diagnostic_count_and_denominator_check='passed for each region/split and all-region totals; padding excluded; unknown-kinematics steps excluded from assessable denominator',
                reports=reports)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='reports'}))


if __name__=='__main__':main()
