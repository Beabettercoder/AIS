"""Independent invariants for the single causal re-anchoring candidate.

No training or test-set access. Accept an optional validation/train cache produced
by the candidate builder to enforce that only position inputs changed.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import sys
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def fixture():
    t=np.arange(12,dtype=float)*60.
    p=np.column_stack([300.+5*t,np.full(12,700.)])
    v=np.column_stack([np.full(12,5.),np.zeros(12)])
    return t,p,v


def assert_close(a,b,atol=1e-8):
    np.testing.assert_allclose(a,b,rtol=0,atol=atol)


def analytic_reference(p,old,w,valid):
    """Independent last-reset formula, without recursive integration."""
    out=np.zeros_like(old);reset_at=0
    for k in range(1,len(p)):
        if valid[k] and w[k]>=.7:reset_at=k
        out[k]=(p[reset_at]-p[0])+(old[k]-old[reset_at])
    return out


def run_synthetic(F,R):
    checks=[]
    def passed(name,detail):
        checks.append(dict(check=name,status='pass',detail=detail));print(f'PASS {name}: {detail}',flush=True)
    t,p,v=fixture();w,valid,_=F.compatibility(t,p,v)
    old,gv,_=F.causal_gate(t,p,v,w,valid,20.)
    new,reset=R.reanchor_positions(t,p,old,w,valid)
    assert_close(new,p-p[0]);assert_close(gv,v)
    assert reset[1:].all() and not reset[0]
    passed('straight_motion','Compatible constant-speed history remains unchanged.')

    # Deliberately nontrivial earlier residual, later compatibility and low-weight suffix.
    p2=p.copy();p2[3]+=np.array([830.,-210.]);v2=v.copy();v2[7:9]*=4
    w2,ok2,_=F.compatibility(t,p2,v2);old2,gv2,_=F.causal_gate(t,p2,v2,w2,ok2,20.)
    out2,reset2=R.reanchor_positions(t,p2,old2,w2,ok2)
    assert_close(out2,analytic_reference(p2,old2,w2,ok2))
    for k in range(1,len(t)):
        if reset2[k]:assert_close(out2[k],p2[k]-p2[0])
        else:assert_close(out2[k]-out2[k-1],old2[k]-old2[k-1])
    passed('recurrence_and_last_reset','At reset use the actually observed coordinate; elsewhere retain exactly the old increment.')

    for n in range(1,len(t)+1):
        ww,vv,_=F.compatibility(t[:n],p2[:n],v2[:n])
        oo,_,_=F.causal_gate(t[:n],p2[:n],v2[:n],ww,vv,20.)
        prefix,mask=R.reanchor_positions(t[:n],p2[:n],oo,ww,vv)
        np.testing.assert_array_equal(prefix,out2[:n]);np.testing.assert_array_equal(mask,reset2[:n])
    passed('prefix_causality','Recomputing every truncated prefix exactly reproduces that prefix of the full candidate.')

    extended_t=np.r_[t,t[-1]+60*np.arange(1,9)]
    extended_p=np.vstack([p2,np.full((8,2),1e9)])
    extended_v=np.vstack([v2,np.full((8,2),1e7)])
    ew,ev,_=F.compatibility(extended_t,extended_p,extended_v)
    eo,_,_=F.causal_gate(extended_t,extended_p,extended_v,ew,ev,20.)
    en,er=R.reanchor_positions(extended_t,extended_p,eo,ew,ev)
    np.testing.assert_array_equal(en[:12],out2);np.testing.assert_array_equal(er[:12],reset2)
    passed('future_mutation','Arbitrary future positions and velocities cannot change any historical output.')

    # Entire observed track displaced: compatibility is perfect, but absolute position is wrong.
    offset=np.array([1000.,600.]);shifted=p+offset
    sw,sv,_=F.compatibility(t,shifted,v);so,_,_=F.causal_gate(t,shifted,v,sw,sv,20.)
    sn,sr=R.reanchor_positions(t,shifted,so,sw,sv)
    assert_close(sw,np.ones(12));assert_close(shifted[0]+sn[-1]-p[-1],offset)
    passed('constant_offset_counterexample','High compatibility preserves a constant absolute offset; the candidate does not claim verified GPS truth.')

    # Unknown intervals must not reset even when their neutral weight equals one.
    fake_old=np.column_stack([np.arange(12)*222.,np.arange(12)*11.])
    no_valid=np.zeros(12,bool)
    un,ur=R.reanchor_positions(t,p,fake_old,np.ones(12),no_valid)
    assert not ur.any();assert_close(un,fake_old-fake_old[0])
    passed('invalid_weight_is_not_evidence','A neutral weight=1 on an invalid interval does not permit a reset.')

    # Threshold is fixed and inclusive at 0.7, with lower weights following existing increments.
    tw=np.zeros(12);tv=np.ones(12,bool);tw[4]=.7;tw[7]=np.nextafter(.7,0)
    tn,tr=R.reanchor_positions(t,p,fake_old,tw,tv)
    assert tr[4] and not tr[7];assert_close(tn,analytic_reference(p,fake_old,tw,tv))
    passed('fixed_threshold','Exactly 0.7 resets; its immediately smaller float does not; there is no threshold search.')

    # Stress mechanism changes visible historical observations only and keeps caller buffers unchanged.
    sog=np.linalg.norm(v,axis=1)/F.KNOT;cog=np.full(12,90.)
    pristine=[x.copy() for x in [t,p,sog,cog]]
    for view in F.VIEWS:
        pp,vv=R.observed_history(F,t,p,sog,cog,'independent_window',view)
        ww,ok,_=F.compatibility(t,pp,vv);oo,_,_=F.causal_gate(t,pp,vv,ww,ok,20.)
        nn,rr=R.reanchor_positions(t,pp,oo,ww,ok)
        assert_close(nn,analytic_reference(pp,oo,ww,ok))
        if view=='target_jump':assert abs(np.linalg.norm(pp[-1]-p[-1])-1000)<1e-7
        if view=='target_speed':assert_close(pp,p)
    for x,y in zip([t,p,sog,cog],pristine):np.testing.assert_array_equal(x,y)
    passed('visible_stress_and_input_immutability','Stress acts on visible history without modifying caller arrays or accessing future references.')
    return checks


def check_cache(pilot,candidate_data):
    checks=[];details=[]
    for split in ['train','val']:
        new_path=candidate_data/f'{split}.npz'
        if not new_path.exists():continue
        old_path=pilot/'data'/f'{split}.npz'
        with np.load(old_path,allow_pickle=False) as a,np.load(new_path,allow_pickle=False) as b:
            absent=set(a.files)-set(b.files);assert not absent,absent
            allowed_new={v+'__reanchor_reset' for v in ['base','target_jump','target_drift','target_speed','neighbor_stress']}
            assert not(set(b.files)-set(a.files)-allowed_new)
            changes=[]
            for key in a.files:
                if key.endswith('__gated'):
                    np.testing.assert_array_equal(a[key][...,2:],b[key][...,2:],err_msg=key+' non-position columns')
                    assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape
                    if not np.array_equal(a[key][...,:2],b[key][...,:2]):changes.append(key)
                elif key.endswith('__anchor_gate'):
                    assert a[key].dtype==b[key].dtype==np.float64 and a[key].shape==b[key].shape
                    if not np.array_equal(a[key],b[key]):changes.append(key)
                else:np.testing.assert_array_equal(a[key],b[key],err_msg=f'{split}/{key}')
            for view in ['base','target_jump','target_drift','target_speed','neighbor_stress']:
                reset_key=view+'__reanchor_reset'
                if reset_key not in b:continue
                reset=b[reset_key].astype(bool)
                expected=(a[view+'__target_weight']>=.7)&a[view+'__target_weight_valid'].astype(bool)
                expected[:,0]=False
                np.testing.assert_array_equal(reset,expected,err_msg=reset_key)
            if split=='val':
                np.testing.assert_array_equal(b['base__gated'],b['neighbor_stress__gated'])
                np.testing.assert_array_equal(b['base__anchor_gate'],b['neighbor_stress__anchor_gate'])
            details.append(dict(split=split,windows=len(a['window_id']),changed_keys=changes,
                old_sha256=sha(old_path),new_sha256=sha(new_path)))
    assert details,'No candidate train.npz or val.npz exists'
    checks.append(dict(check='cache_single_factor',status='pass',detail=details))
    return checks


def check_frozen_scalers(pilot,new_runs):
    paths=sorted(new_runs.rglob('normalizer.json'));assert paths,'No candidate normalizers found'
    old=next((pilot/'parallel_runs').rglob('normalizer.json'))
    frozen=json.loads(old.read_text())
    for path in paths:assert json.loads(path.read_text())==frozen,str(path)
    return dict(check='frozen_train_normalizer',status='pass',count=len(paths),old_reference=str(old),old_sha256=sha(old))


def check_cv_decision(pilot,here,candidate_data):
    frozen=json.loads((here/'frozen_before_scoring.json').read_text())
    assert frozen['protocol_sha256']==sha(here/'protocol.json')
    assert frozen['reanchor_sha256']==sha(here/'reanchor.py')
    assert frozen['builder_sha256']==sha(here/'cv_and_build.py')
    decision=json.loads((here/'cv_decision.json').read_text())
    assert decision['protocol_sha256']==frozen['protocol_sha256']
    assert decision['candidate_val_sha256']==sha(candidate_data/'val.npz')
    report=json.loads((here/'cv_results.json').read_text())
    assert decision['criteria']==report['criteria']
    assert decision['passed']==report['passed']==all(x['passed'] for x in report['criteria'])
    protocol=json.loads((here/'protocol.json').read_text())
    with np.load(pilot/'data/val.npz',allow_pickle=False) as a,np.load(candidate_data/'val.npz',allow_pickle=False) as b:
        for view in ['base','target_jump','target_drift','target_speed','neighbor_stress']:
            errors={}
            for name,z,kind in [('CV_raw',a,'raw'),('CV_original_gate',a,'gate'),('CV_reanchor',b,'gate')]:
                pred=z[view+'__anchor_'+kind][:,None,:]+z[view+'__vel_'+kind].astype(float)[:,None,:]*z['query_s'].astype(float)[...,None]
                ee=np.linalg.norm(pred-z['future_xy'],axis=-1)
                errors[name]={'ADE':ee.mean(1),'FDE':ee[:,-1]}
            for subset in ['all','observed_maneuver']:
                mask=np.ones(len(a['window_id']),bool) if subset=='all' else a['maneuver'].astype(bool)
                identities,cluster=np.unique(a['mmsi'][mask].astype(str),return_inverse=True)
                sizes=np.bincount(cluster);m=len(identities)
                draw=np.random.default_rng(protocol['bootstrap_seed']).integers(0,m,size=(protocol['bootstrap_draws'],m))
                weights=np.stack([np.bincount(x,minlength=m) for x in draw])
                for metric in ['ADE','FDE']:
                    means={n:float(e[metric][mask].mean()) for n,e in errors.items()}
                    for name,value in means.items():
                        stated=next(r for r in report['results'] if (r['view'],r['subset'],r['method'],r['metric'])==(view,subset,name,metric))
                        assert_close(value,stated['mean_m'],atol=1e-8)
                    delta=errors['CV_reanchor'][metric][mask]-errors['CV_original_gate'][metric][mask]
                    totals=np.bincount(cluster,weights=delta,minlength=m)
                    values=(weights@totals)/(weights@sizes)
                    ci=np.quantile(values,[.025,.975])
                    stated=next(r for r in report['intervals'] if (r['view'],r['subset'],r['aggregation'],r['metric'])==(view,subset,'pooled',metric+'_m'))
                    assert_close([float(delta.mean()),*ci],[stated['delta'],stated['ci95_low'],stated['ci95_high']],atol=1e-8)
                    crit=[r for r in report['criteria'] if (r['view'],r['subset'],r['metric'])==(view,subset,metric)]
                    if crit:
                        r=crit[0]
                        if view=='base':expected=bool(ci[1]<0)
                        else:
                            expected=bool(ci[1]<=.01*means['CV_original_gate'])
                            if view in ['target_jump','target_speed']:
                                retained=(means['CV_raw']-means['CV_reanchor'])/(means['CV_raw']-means['CV_original_gate'])
                                expected=expected and retained>=.95
                        assert r['passed']==expected
    if not decision['passed']:
        assert not(candidate_data/'train.npz').exists(),'Train cache should not be prepared after failed continuation gate'
    failed=[{k:r[k] for k in ['view','subset','metric','ci_upper_m','tolerance_m']} for r in report['criteria'] if not r['passed']]
    return dict(check='cv_values_intervals_and_frozen_decision',status='pass',cv_continue_training=decision['passed'],
                failed_conditions=failed,detail='All CV means and pooled target-MMSI paired intervals recomputed independently; criterion hashes and stopping decision match.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,default=Path('/mnt/sda/WangZijia/AIS/forecast_pilot_20261105'))
    parser.add_argument('--candidate-data',type=Path)
    parser.add_argument('--candidate-runs',type=Path)
    parser.add_argument('--out',type=Path,default=Path('independent_reanchor_checks.json'))
    args=parser.parse_args();here=Path(__file__).resolve().parent
    F=load_module('original_pilot_features',args.pilot/'features.py')
    R=load_module('candidate_reanchor',here/'reanchor.py')
    checks=run_synthetic(F,R)
    if args.candidate_data:checks+=check_cache(args.pilot,args.candidate_data)
    if args.candidate_data and (here/'cv_decision.json').exists():
        checks.append(check_cv_decision(args.pilot,here,args.candidate_data))
    if args.candidate_runs:checks.append(check_frozen_scalers(args.pilot,args.candidate_runs))
    protocol=json.loads((here/'protocol.json').read_text())
    assert protocol['candidate_count']==1 and protocol['threshold_search'] is False
    assert protocol['position_reset_threshold']==.7 and protocol['test_evaluation'] is False
    checks.append(dict(check='protocol_fixed',status='pass',detail='One fixed 0.7 candidate, no test evaluation or threshold search.'))
    report=dict(status='pass',checks=checks,training_started=False,test_data_opened=False,
        source_sha256={'original_features':sha(args.pilot/'features.py'),'candidate':sha(here/'reanchor.py'),
                       'protocol':sha(here/'protocol.json'),'independent_checks':sha(__file__)})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS {len(checks)} checks: {args.out.resolve()}',flush=True)


if __name__=='__main__':main()
