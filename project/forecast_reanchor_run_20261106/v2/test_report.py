"""Synthetic reporting tests; never reads project AIS or runs a model."""
import json,tempfile
from pathlib import Path
import numpy as np
import report as R

def fixture(root,name,policy,hidden,quota,error):
    exp=root/name;runs=exp/'test_runs';data=exp/'data';data.mkdir(parents=True)
    n=8;future=np.zeros((n,8,2),float);q=np.tile(np.arange(1,9)*60.,(n,1))
    meta=dict(future_xy=future,query_s=q,window_id=np.array([f'w{i}' for i in range(n)]),mmsi=np.repeat([1,2,3,4],2),
        region=np.repeat(['a','b'],4),maneuver=np.arange(n)%2==0)
    cache=dict(meta)
    for view in R.VIEWS:
        for kind in ['raw','gate']:
            cache[view+'__anchor_'+kind]=np.ones((n,2))*[4.,0.]
            cache[view+'__vel_'+kind]=np.zeros((n,2))
    np.savez(data/'test.npz',**cache)
    R.save(exp/'experiment.json',dict(name=name,policy=policy,hidden=hidden,train_per_region=quota,
        history_steps=12,future_steps=8,batch_size=128,lr=.002,norm_sha256='same-norm',data_dir=str(data)))
    for arm in ['P0','P1']:
        for seed in R.SEEDS:
            folder=runs/'tcn'/f'seed_{seed}'/arm;folder.mkdir(parents=True)
            settings=dict(model='tcn',arm=arm,seed=seed,hidden=hidden,policy=policy,history_steps=12,future_steps=8,
                batch_size=128,epochs=25,lr=.002,norm_sha256='same-norm')
            R.save(folder/'result.json',dict(status='complete',settings=settings,selected_epoch=3,
                best_val={'epoch':3,'ADE_m':100.,'FDE_m':200.},last_epoch={'epoch':25,'ADE_m':300.,'FDE_m':400.}))
            e=error-(.2 if arm=='P1' else 0.)
            for view in R.VIEWS:
                pred=future+[e,0.];err=np.linalg.norm(pred-future,axis=-1)
                np.savez(folder/f'predictions_{view}.npz',**meta,pred_xy=pred,ADE=err.mean(1),FDE=err[:,-1],
                    model='tcn',arm=arm,seed=seed,view=view,split='test',hidden=hidden,policy=policy,selected_epoch=3)
    return exp

def main():
    with tempfile.TemporaryDirectory(prefix='v2_reporting_test_') as tmp:
        root=Path(tmp)
        const=fixture(root,'small_constant','constant_last',64,700,3.)
        cos=fixture(root,'small_cosine','cosine_best',64,700,2.)
        big=fixture(root,'expanded64','cosine_best',64,2000,1.5)
        cap=fixture(root,'expanded128','cosine_best',128,2000,1.)
        summary=R.summarize(big,root/'report','test',big/'data',draws=32)
        assert summary['status']=='partial' and summary['completed_models']==6
        row=next(x for x in summary['summary'] if (x['model'],x['arm'],x['view'],x['subset'])==('tcn','P0','base','all'))
        assert row['ADE_m_mean']==1.5 and row['best_val_ADE_m_mean']==100. and row['last_epoch_ADE_m_mean']==300.
        cv=next(x for x in summary['summary'] if x['model']=='CV_raw' and x['view']=='base' and x['subset']=='all')
        assert cv['ADE_m_mean']==4. and cv['best_val_ADE_m_mean'] is None
        pair=next(x for x in summary['paired'] if x['model']=='tcn' and x['view']=='base' and x['subset']=='all' and x['aggregation']=='pooled' and x['metric']=='ADE_m')
        np.testing.assert_allclose([pair['delta'],pair['ci95_low'],pair['ci95_high']],[-.2,-.2,-.2],atol=1e-9)
        comparisons=R.compare([const,cos,big,cap],root/'compare','test',draws=32)
        assert {x['changed_factor'] for x in comparisons}=={'policy','quota','hidden'}
        assert len(comparisons)==3
        a=R.load_prediction(big/'test_runs/tcn/seed_0/P0/predictions_base.npz','test')
        broken=dict(a);broken['future_xy']=a['future_xy']+1
        try:R.S.aligned(a,broken)
        except ValueError:pass
        else:raise AssertionError('Mismatched future reference accepted')
        config=R.read(cap/'experiment.json');config['norm_sha256']='different';R.save(cap/'experiment.json',config)
        assert R.compare([big,cap],root/'badnorm','test',draws=8)==[]
        assert R.read(root/'badnorm/factor_comparisons.json')['skipped'][0]['reason']=='normalizer missing or differs'
    print('PASS: selected test metrics differ from best/last val diagnostics; CV and cluster differences exact; one-factor guards hold; partial matrix not called complete.')

if __name__=='__main__':main()
