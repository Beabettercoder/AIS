"""Recompute val/test metrics and paired MMSI intervals from absolute predictions."""
import argparse,csv,hashlib
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from common import *

def prediction(path,split):
    z=load_npz(path);assert str(z['split'])==split
    e=np.linalg.norm(z['pred_xy']-z['future_xy'],axis=-1)
    np.testing.assert_allclose(z['ADE'],e.mean(1),atol=1e-9,rtol=0);np.testing.assert_allclose(z['FDE'],e[:,-1],atol=1e-9,rtol=0)
    return dict(model=str(z['model']),arm=str(z['arm']),seed=int(z['seed']),view=str(z['view']),
        window_id=z['window_id'].astype(str),mmsi=z['mmsi'].astype(str),region=z['region'].astype(str),future_xy=z['future_xy'],
        query_s=z['query_s'],maneuver=z['maneuver'].astype(bool),ADE=e.mean(1),FDE=e[:,-1])

def report(work,split):
    work=Path(work);root=work/('runs' if split=='val' else 'test_runs');out=work/f'reports/{split}';out.mkdir(parents=True,exist_ok=True)
    if split=='val':T.save_baselines(root,T.load_data(work/'data/val.npz'),sha(work/'preparation_complete.json'))
    recs={};per_seed=[]
    for path in tqdm(sorted(root.rglob('predictions_*.npz')),desc=f'{split} prediction audit',unit='file'):
        r=prediction(path,split);key=(r['model'],r['arm'],r['seed'],r['view']);assert key not in recs
        done=read(path.with_name('result.json'));assert done['status']=='complete'
        if not r['model'].startswith('CV'):assert done['completed_epochs']==CONFIG['epochs'][r['model']]
        recs[key]=r
        if len(recs)>1:S.aligned(next(iter(recs.values())),r)
        for sub in ['all','observed_maneuver']:
            mask=S.subset_mask(r,sub);per_seed.append(dict(split=split,model=r['model'],arm=r['arm'],seed=r['seed'],view=r['view'],subset=sub,**S.metrics(r,mask)))
    expected={(m,a,s,v) for m in CONFIG['models'] for a in ['P0','P1','P2','P3'] for s in CONFIG['seeds'] for v in CONFIG['views']}
    assert expected.issubset(recs),'Incomplete prediction matrix; refuse final report'
    summaries=[]
    for model,arm in [(m,a) for m in CONFIG['models'] for a in ['P0','P1','P2','P3']]+[('CV_raw','CV_raw'),('CV_gate','CV_gate')]:
        seeds=CONFIG['seeds'] if not model.startswith('CV') else [-1]
        for view in CONFIG['views']:
            for sub in ['all','observed_maneuver']:
                rr=[next(x for x in per_seed if (x['model'],x['arm'],x['seed'],x['view'],x['subset'])==(model,arm,s,view,sub)) for s in seeds]
                row=dict(split=split,model=model,arm=arm,view=view,subset=sub,n_seeds=len(seeds),windows=rr[0]['windows'],target_mmsi=rr[0]['target_mmsi'])
                for met in ['ADE_m','FDE_m','region_macro_ADE_m','region_macro_FDE_m']:
                    vals=[x[met] for x in rr if x[met] is not None];row[met+'_mean']=float(np.mean(vals)) if vals else None
                    row[met+'_seed_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                summaries.append(row)
    pairs=[]
    tasks=[(m,v,a,b,sub) for m in CONFIG['models'] for v in CONFIG['views'] for a,b in [('P1','P0'),('P2','P1'),('P3','P2'),('P3','P0')]
           for sub in ['all','observed_maneuver']]
    for m,v,a,b,sub in tqdm(tasks,desc=f'{split} MMSI bootstrap',unit='contrast'):
        aa=[recs[m,a,s,v] for s in CONFIG['seeds']];bb=[recs[m,b,s,v] for s in CONFIG['seeds']]
        for x in S.paired_bootstrap(aa,bb,sub,CONFIG['bootstrap_draws'],CONFIG['bootstrap_seed']):pairs.append(dict(split=split,model=m,view=v,contrast=a+'-'+b,subset=sub,**x))
    S.write_csv(out/'summary.csv',summaries);S.write_csv(out/'per_seed.csv',per_seed);S.write_csv(out/'paired_intervals.csv',pairs)
    save(out/'summary.json',dict(status='complete',split=split,completed_models=48,summary=summaries,paired=pairs,
        prediction_files=len(recs),prediction_hashes={str(p.relative_to(root)):sha(p) for p in root.rglob('predictions_*.npz')},
        notes='Errors in metres; paired intervals condition on the three fixed trained seeds; no multiple-comparison correction.'))
    lines=['# '+('验证集' if split=='val' else '冻结后测试集')+'预测结果','',
        '单位为米，越低越好；各配置是三个种子的均值。CV为不训练的恒速参照。', '',
        '|骨干|配置|观测条件|分层|ADE|FDE|区域等权ADE|区域等权FDE|','|---|---|---|---|---:|---:|---:|---:|']
    for r in summaries:
        cells=[r['model'],S.ARM_ZH.get(r['arm'],r['arm']),S.VIEW_ZH[r['view']],r['subset']]
        cells += ['—' if r[k] is None else f'{r[k]:.2f}' for k in ['ADE_m_mean','FDE_m_mean','region_macro_ADE_m_mean','region_macro_FDE_m_mean']]
        lines.append('|'+ '|'.join(cells)+'|')
    (out/'结果表.md').write_text('\n'.join(lines),encoding='utf-8')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('work'));p.add_argument('--split',choices=['val','test'],required=True)
    a=p.parse_args();report(a.work,a.split)
