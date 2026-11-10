"""Independent CPU invariants for the forecasting pilot; never train a model.

python independent_checks.py --out checks.json
python independent_checks.py --out checks.json --runs ./runs
"""
from __future__ import annotations
import argparse, hashlib, json, tempfile, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import features as F
from build_data import candidates

BBOX = [-1., -1., 1., 1.]
CFG = dict(radius_nm=8, sigma_nm=4, heading_deg=45, top_k=32,
           min_neighbor_kn=2, background_age_s=300, max_gap_s=300,
           max_history_s=1800, max_future_s=1200, sample_min_s=60)


def assert_close(a,b,atol=1e-5):
    np.testing.assert_allclose(a,b,rtol=0,atol=atol)


def fixture():
    t=np.arange(20,dtype=float)*60
    p=np.column_stack([t*5, np.zeros(20)])
    sog=np.full(20,5/F.KNOT); cog=np.full(20,90.)
    rows=[]
    for mid,dx,dy,vel in [(1001,500,300,5),(1002,-400,-200,6),
                          (1003,200,-500,4),(1004,-200,700,5.5),
                          (1005,600,800,4.5),(999,0,0,5)]:
        for k,qt in enumerate(t):
            rows.append(dict(mmsi=mid,timestamp=qt,lon=(qt*vel+dx)/111195,
                             lat=dy/111195,sog=vel/F.KNOT,cog=90.))
    return t,p,sog,cog,pd.DataFrame(rows)


def view(bg,which='base',p_override=None):
    t,p,sog,cog,_=fixture()
    if p_override is not None:p=p_override
    return F.make_view(t[:12],p[:12],sog[:12],cog[:12],999,bg,set(),
                       'independent_fixture',which,20.,CFG)[0]


def straight_gate():
    t,p,sog,cog,table=fixture()
    z=view(F.Background(table,BBOX))
    assert_close(z['raw'],z['gated'])
    assert_close(z['target_weight'],np.ones(12))
    assert_close(z['anchor_raw'],p[11]);assert_close(z['anchor_gate'],p[11])
    return 'Straight constant-speed history stays unchanged.'


def absolute_anchor():
    t,p,sog,cog,table=fixture();z=view(F.Background(table,BBOX),'target_jump')
    query=t[12:]-t[11];truth=p[12:]
    raw=z['anchor_raw']+query[:,None]*z['vel_raw']
    gate=z['anchor_gate']+query[:,None]*z['vel_gate']
    err_raw=np.linalg.norm(raw-truth,axis=-1);err_gate=np.linalg.norm(gate-truth,axis=-1)
    assert_close(err_raw,np.full(8,1000),atol=.001)
    assert err_gate.max()<.001
    assert abs(np.linalg.norm(z['anchor_raw']-p[11])-1000)<1e-6
    return f'1km last-point jump remains {err_raw.mean():.6f}m Raw-CV absolute error; gate {err_gate.mean():.6f}m.'


def past_only():
    t,p,sog,cog,table=fixture();bg=F.Background(table,BBOX)
    for k,qt in enumerate(t[:12]):
        dp,nv,dist,w,mids,nt=bg.lookup(qt,p[k],90,999,{1002},None,CFG)
        assert (nt<=qt).all();assert 999 not in mids;assert 1002 not in mids
        assert len(np.unique(mids))==len(mids)
    changed=table.copy();isfuture=changed.timestamp>t[11]
    changed.loc[isfuture,['lon','lat','sog','cog']]=[120.,70.,99.,270.]
    p_changed=p.copy();p_changed[12:]+=1e7
    for name in F.VIEWS:
        a=view(bg,name);b=view(F.Background(changed,BBOX),name,p_changed)
        for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=f'{name}/{key}')
    return 'Future target/neighbor changes leave every history-view input exactly unchanged; latest reports never exceed each step.'


def ordinary_independent_of_reliability():
    _,_,_,_,table=fixture()
    class WZero(F.Background):
        def lookup(self,*args,**kwargs):
            result=list(super().lookup(*args,**kwargs));result[3]=np.zeros_like(result[3]);return tuple(result)
    a=view(F.Background(table,BBOX));b=view(WZero(table,BBOX))
    np.testing.assert_array_equal(a['f0'],b['f0'])
    np.testing.assert_array_equal(a['f0_valid'],b['f0_valid'])
    assert not np.array_equal(a['f1'],b['f1'])
    assert not b['f1_valid'].any()
    return 'F0 receives the same members but ignores reliability; F1 changes when reliability alone changes.'


def sparse_fallback():
    dp=np.array([[200.,100.],[-100.,100.],[0.,100.]])
    vel=np.array([[3.,0.],[5.,0.],[7.,0.]])
    dist=np.linalg.norm(dp,axis=1);ww=np.array([1.,.2,.8])
    f,ok=F.fit_field(dp,vel,dist,ww,np.array([6.,0.]))
    alpha=np.exp(-dist**2/(2*(4*1852.)**2))*ww;alpha/=alpha.sum()
    assert ok and f[7]==1 and f[6]==.75
    assert_close(f[:2],alpha@vel);assert_close(f[4:6],np.zeros(2))
    f,ok=F.fit_field(np.zeros((0,2)),np.zeros((0,2)),np.zeros(0),np.zeros(0),np.array([6.,0.]))
    assert not ok and not f.any()
    return 'Three/collinear neighbors retain zero-order weighted speed; no neighbors yield zero support.'


def neighbor_self_invariance():
    _,_,_,_,table=fixture();bg=F.Background(table,BBOX)
    a=view(bg);b=view(bg,'neighbor_stress')
    for key in ('raw','gated','anchor_raw','anchor_gate','vel_raw','vel_gate',
                'target_weight','target_weight_valid','gate_branch'):
        np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    return 'Neighbor-only corruption does not alter any target-only input, anchor or velocity.'


def invalid_background():
    t,p,_,_,table=fixture()
    invalid=[]
    for mid,sog,cog,lon,lat in [(2001,5,360,.01,0),(2002,102.2,90,.01,0),
                               (2003,5,90,181,0),(2004,5,90,.01,91)]:
        for qt in [t[10],t[11]]:invalid.append(dict(mmsi=mid,timestamp=qt,sog=sog,cog=cog,lon=lon,lat=lat))
    bg=F.Background(pd.concat([table,pd.DataFrame(invalid)],ignore_index=True),BBOX)
    result=bg.lookup(t[11],p[11],90,999,set(),None,CFG)
    assert not(set(result[4])&{2001,2002,2003,2004})
    return 'AIS invalid SOG/COG codes and invalid coordinates never enter neighbor fit.'


def raw_future_qc():
    t,p,sog,cog,_=fixture()
    table=pd.DataFrame(dict(mmsi=np.full(20,999),timestamp=t,
                           lon=p[:,0]/111195,lat=p[:,1]/111195,sog=sog,cog=cog))
    base,_=candidates(table,{999},BBOX,CFG);assert len(base[999])==1
    shifted=table.copy();shifted.loc[12:,'lon']+=1.
    after,counts=candidates(shifted,{999},BBOX,CFG)
    assert not after and counts.get('extreme_future_position_excluded',0)==1
    return 'Future reference QC catches a large jump at the history/future boundary.'


def summary_numerics():
    import summarize as S
    n=8;truth=np.zeros((n,8,2))+np.array([100000.,-200000.])
    query=np.tile(np.arange(1,9)*60,(n,1));ids=np.array([f'w{i}' for i in range(n)])
    mmsi=np.repeat([11,22,33,44],2);reg=np.repeat(['a','a','b','b'],2);maneuver=np.arange(n)%2==0
    with tempfile.TemporaryDirectory(prefix='forecast_checks_') as tmp:
        tmp=Path(tmp);runs=tmp/'runs'
        for arm,error in [('P0',2.),('P1',1.)]:
            for seed in (0,1,2):
                folder=runs/'tcn'/f'seed_{seed}'/arm;folder.mkdir(parents=True)
                (folder/'result.json').write_text(json.dumps(dict(status='complete',completed_epochs=25,expected_epochs=25)))
                for name in F.VIEWS:
                    pred=truth+np.array([error,0.]);err=np.linalg.norm(pred-truth,axis=-1)
                    np.savez_compressed(folder/f'predictions_{name}.npz',pred_xy=pred,future_xy=truth,
                        query_s=query,mmsi=mmsi,region=reg,window_id=ids,maneuver=maneuver,
                        split='val',model='tcn',arm=arm,seed=seed,view=name,point_error=err,ADE=err.mean(1),FDE=err[:,-1])
        cand=[S.load_prediction(runs/'tcn'/f'seed_{s}'/'P1'/'predictions_base.npz') for s in (0,1,2)]
        base=[S.load_prediction(runs/'tcn'/f'seed_{s}'/'P0'/'predictions_base.npz') for s in (0,1,2)]
        pair=S.paired_bootstrap(cand,base,'all',128,20261105)
        assert len(pair)==4
        for item in pair:
            assert_close([item['delta'],item['ci95_low'],item['ci95_high']],[-1,-1,-1])
        cmd=[sys.executable,str(Path(S.__file__)), '--runs',str(runs),'--out',str(tmp/'summary'),'--bootstrap','32']
        done=subprocess.run(cmd,text=True,capture_output=True)
        assert done.returncode==0,done.stdout+done.stderr
        report=json.loads((tmp/'summary'/'summary.json').read_text())
        assert report['status']=='partial' and report['completed_training_runs']==6
        # Refuse mixed future references rather than averaging incompatible protocols.
        broken=dict(base[0]);broken['future_xy']=base[0]['future_xy']+10
        try:S.aligned(cand[0],broken)
        except ValueError:pass
        else:raise AssertionError('Mismatched future references accepted')
    return 'Absolute ADE/FDE recomputation and MMSI bootstrap recover -1m and [-1,-1]; 6/48 stays partial; mismatches rejected.'


def encoder_checks():
    import torch
    from forecast_models import ForecastModel,state_hash,MODEL_NAMES
    from train import fit_normalizer,prepare
    torch.set_num_threads(1)
    _,_,_,_,table=fixture();z=view(F.Background(table,BBOX));n=2
    data={f'base__{k}':np.stack([v,v]) for k,v in z.items()}
    data.update(query_s=np.tile(np.arange(1,9)*60.,(n,1)),time_s=np.tile(np.arange(12)*60.,(n,1)),
                future_xy=np.zeros((n,8,2)))
    norm=fit_normalizer(data)
    changed={**data,'future_xy':data['future_xy']+1e8}
    assert fit_normalizer(changed)==norm
    # Invalid fields stay exactly zero after applying nonzero training means.
    altered={k:v.copy() for k,v in data.items()}
    altered['base__f1_valid'][:]=False
    prepared=prepare(altered,'base','P3',norm)
    assert not prepared['field'].any() and not prepared['rho'].any()
    init={}
    for name in MODEL_NAMES:
        torch.manual_seed(7);m=ForecastModel(name).eval();h1=state_hash(m)
        torch.manual_seed(7);other=ForecastModel(name);assert h1==state_hash(other)
        init[name]=h1
        x=torch.randn(2,12,9);field=torch.randn(2,12,8);rho=torch.zeros(2,12)
        q=torch.arange(1,9).float().repeat(2,1)*60
        a=torch.tensor([[100000.,200000.],[300000.,400000.]],dtype=torch.float64)
        vel=torch.tensor([[5.,0.],[0.,4.]])
        with torch.no_grad():
            nofield=m(x,field,rho,q,a,vel,False)
            fallback=m(x,field,rho,q,a,vel,True)
        torch.testing.assert_close(nofield,fallback,rtol=0,atol=0)
        torch.testing.assert_close(nofield,a[:,None,:]+vel.double()[:,None,:]*q.double()[...,None],rtol=0,atol=0)
    return 'Four encoder initializations reproducible; zero-support branch exactly falls back; scalers ignore future coordinates; invalid fields stay zero.'


def trained_self_invariance(runs):
    checked=0
    for base in Path(runs).rglob('predictions_base.npz'):
        other=base.with_name('predictions_neighbor_stress.npz')
        if not other.exists():continue
        with np.load(base,allow_pickle=False) as a,np.load(other,allow_pickle=False) as b:
            assert str(a['split'].item())=='val' and str(b['split'].item())=='val'
            if str(a['arm'].item()) in ('P0','P1','CV_raw','CV_gate'):
                np.testing.assert_array_equal(a['window_id'],b['window_id'])
                np.testing.assert_array_equal(a['pred_xy'],b['pred_xy'],err_msg=str(base));checked+=1
    return f'Checked {checked} saved target-only/CV runs: neighbor-only stress leaves predictions exactly unchanged.'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('independent_checks.json'))
    ap.add_argument('--runs',type=Path)
    args=ap.parse_args()
    checks=[straight_gate,absolute_anchor,past_only,ordinary_independent_of_reliability,
            sparse_fallback,neighbor_self_invariance,invalid_background,raw_future_qc,
            summary_numerics,encoder_checks]
    reports=[]
    for fn in tqdm(checks,desc='Independent invariants',unit='check'):
        note=fn();reports.append(dict(check=fn.__name__,status='pass',detail=note));tqdm.write(f'PASS {fn.__name__}: {note}')
    if args.runs:
        reports.append(dict(check='trained_self_invariance',status='pass',detail=trained_self_invariance(args.runs)))
    root=Path(__file__).resolve().parent
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}
    payload=dict(status='pass',type='synthetic invariant checks, not experimental results',
                 checks=reports,source_sha256=hashes,training_performed=False,test_data_opened=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS {len(reports)} checks: {args.out.resolve()}',flush=True)

if __name__=='__main__':main()
