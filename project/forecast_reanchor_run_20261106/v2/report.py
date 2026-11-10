"""Auditable v2 forecasting summaries. Never train, select on test, or write papers/PDFs."""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'frozen_pilot'))
import summarize as S

MODELS=('tcn','gru','transformer','itransformer')
ARMS=('P0','P1','P2','P3');SEEDS=(0,1,2)
VIEWS=('base','target_jump','target_drift','target_speed','neighbor_stress')
PAIRS=(('P1','P0'),('P2','P1'),('P3','P2'),('P3','P0'))
METRICS=('ADE_m','FDE_m','region_macro_ADE_m','region_macro_FDE_m')
REUSE_NOTE='目标身份按预测划分留出；本测试来源此前已用于预测或检测评估，不称全新盲测或外部数据。'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def csv_save(path,rows):S.write_csv(Path(path),clean(rows))
def fmt(x):return '—' if x is None or not np.isfinite(x) else f'{x:.2f}'
def scalar(z,key,default=None):return np.asarray(z[key]).item() if key in z else default

def load_prediction(path,expected_split):
    path=Path(path)
    with np.load(path,allow_pickle=False) as z:
        split=str(scalar(z,'split'))
        if split!=expected_split:raise ValueError(f'Wrong evaluation split {split}: {path}')
        n=len(z['window_id']);order=np.argsort(z['window_id'].astype(str),kind='stable')
        if len(np.unique(z['window_id']))!=n or n==0:raise ValueError(f'Duplicate/empty window IDs: {path}')
        r={k:z[k][order] for k in ('window_id','mmsi','region','future_xy','query_s','maneuver','pred_xy')}
        for k in ('window_id','mmsi','region'):r[k]=r[k].astype(str)
        r['maneuver']=r['maneuver'].astype(bool)
        if r['pred_xy'].shape!=r['future_xy'].shape or r['pred_xy'].ndim!=3 or r['pred_xy'].shape[-1]!=2:raise ValueError('Invalid predicted/observed position shape')
        if r['query_s'].shape!=r['pred_xy'].shape[:2]:raise ValueError('Query times do not match positions')
        if not all(np.isfinite(r[k]).all() for k in ('pred_xy','future_xy','query_s')):raise ValueError('Nonfinite coordinates/times')
        if not np.all(r['query_s']>0) or not np.all(np.diff(r['query_s'],axis=1)>0):raise ValueError('Query times are not increasing positive offsets')
        error=np.linalg.norm(r['pred_xy'].astype(float)-r['future_xy'].astype(float),axis=-1)
        for key,value in [('ADE',error.mean(1)),('FDE',error[:,-1])]:
            if key in z:np.testing.assert_allclose(z[key][order],value,rtol=0,atol=1e-8,err_msg=str(path))
            r[key]=value
        r.update(path=str(path),split=split,view=str(scalar(z,'view')),model=str(scalar(z,'model')),
                 arm=str(scalar(z,'arm')),seed=int(scalar(z,'seed')),hidden=scalar(z,'hidden'),
                 policy=scalar(z,'policy'),selected_epoch=scalar(z,'selected_epoch'))
    meta=path.with_name('result.json');result=read(meta) if meta.exists() else {}
    r['result']=result;r['complete']=result.get('status')=='complete';r['settings']=result.get('settings',result.get('config',{}))
    return r

def baseline_records(data_dir,split):
    path=Path(data_dir)/f'{split}.npz'
    with np.load(path,allow_pickle=False) as z:
        order=np.argsort(z['window_id'].astype(str),kind='stable')
        meta={k:z[k][order] for k in ('window_id','mmsi','region','future_xy','query_s','maneuver')}
        for k in ('window_id','mmsi','region'):meta[k]=meta[k].astype(str)
        records=[]
        for kind in ('raw','gate'):
            for view in VIEWS:
                pred=z[f'{view}__anchor_{kind}'].astype(float)[order,None,:]+z[f'{view}__vel_{kind}'].astype(float)[order,None,:]*meta['query_s'].astype(float)[...,None]
                error=np.linalg.norm(pred-meta['future_xy'],axis=-1)
                records.append({**meta,'path':str(path),'split':split,'view':view,'model':'CV_'+kind,'arm':'CV_'+kind,'seed':-1,
                    'ADE':error.mean(1),'FDE':error[:,-1],'pred_xy':pred,'settings':{},'result':{},'complete':True,
                    'hidden':None,'policy':'deterministic','selected_epoch':None})
    return records

def discover(root,split,data_dir=None):
    root=Path(root);run_root=root/('runs' if split=='val' else 'test_runs')
    if not run_root.exists():run_root=root
    records=[]
    for path in tqdm(sorted(run_root.rglob('predictions_*.npz')),desc=f'{split} audit predictions',unit='file'):
        with np.load(path,allow_pickle=False) as z:
            if str(scalar(z,'split'))!=split:continue
        records.append(load_prediction(path,split))
    if not records:raise RuntimeError(f'No {split} predictions at {run_root}')
    if data_dir is None and (root/'experiment.json').exists():data_dir=read(root/'experiment.json').get('data_dir')
    if data_dir is not None and not any(r['model'].startswith('CV') for r in records):records+=baseline_records(data_dir,split)
    indexed={}
    for r in records:
        key=(r['model'],r['arm'],r['seed'],r['view'])
        if key in indexed:raise ValueError(f'Duplicate model/seed/view: {key}')
        S.aligned(records[0],r);indexed[key]=r
    return records,indexed

def selection_record(r):
    result=r['result'];best=result.get('best_val',{});last=result.get('last_epoch',{})
    def metric(item,key):
        return item.get(key,item.get('val_'+key)) if isinstance(item,dict) else None
    return dict(selection_reference='validation/base (not test)',selected_epoch=result.get('selected_epoch',r.get('selected_epoch')),
        best_val_epoch=best.get('epoch') if isinstance(best,dict) else None,
        best_val_ADE_m=metric(best,'ADE_m'),best_val_FDE_m=metric(best,'FDE_m'),
        last_epoch=last.get('epoch') if isinstance(last,dict) else None,
        last_epoch_ADE_m=metric(last,'ADE_m'),last_epoch_FDE_m=metric(last,'FDE_m'))

def summarize(experiment_root,out,split='val',data_dir=None,draws=2000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);root=Path(experiment_root)
    records,index=discover(root,split,data_dir);seed_rows=[]
    for r in records:
        for subset in ('all','observed_maneuver'):
            seed_rows.append(dict(model=r['model'],arm=r['arm'],seed=r['seed'],view=r['view'],subset=subset,split=split,
                hidden=r.get('hidden'),policy=r.get('policy'),**S.metrics(r,S.subset_mask(r,subset)),**selection_record(r)))
    groups=collections.defaultdict(list)
    for r in seed_rows:groups[(r['model'],r['arm'],r['view'],r['subset'])].append(r)
    summaries=[]
    for key,rows in sorted(groups.items()):
        x=dict(zip(('model','arm','view','subset'),key));x.update(split=split,n_seeds=len(rows),windows=rows[0]['windows'],target_mmsi=rows[0]['target_mmsi'],
            hidden=rows[0]['hidden'],policy=rows[0]['policy'],selection_reference='validation/base (best_val and last_epoch columns only)')
        for metric in METRICS+('best_val_ADE_m','best_val_FDE_m','last_epoch_ADE_m','last_epoch_FDE_m'):
            vals=[r[metric] for r in rows if r.get(metric) is not None]
            x[metric+'_mean']=float(np.mean(vals)) if vals else None
            x[metric+'_seed_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        x['selected_epochs']=','.join(str(r['selected_epoch']) for r in sorted(rows,key=lambda z:z['seed']))
        summaries.append(x)
    pairs=[]
    for model,view,(cand,base),subset in tqdm([(m,v,p,s) for m in MODELS for v in VIEWS for p in PAIRS for s in ('all','observed_maneuver')],desc=f'{split} paired MMSI bootstrap'):
        seeds=[s for s in SEEDS if (model,cand,s,view) in index and (model,base,s,view) in index]
        if not seeds:continue
        values=S.paired_bootstrap([index[model,cand,s,view] for s in seeds],[index[model,base,s,view] for s in seeds],subset,draws,20261106)
        for value in values:pairs.append(dict(split=split,model=model,view=view,contrast=cand+'-'+base,subset=subset,n_seeds=len(seeds),**value))
    completed={(m,a,s) for m in MODELS for a in ARMS for s in SEEDS if all((m,a,s,v) in index and index[m,a,s,v]['complete'] for v in VIEWS)}
    report=dict(status='complete' if len(completed)==48 else 'partial',split=split,completed_models=len(completed),expected_models=48,
        summary=summaries,paired=pairs,source_root=str(root.resolve()),data_dir=None if data_dir is None else str(data_dir),
        notes=[REUSE_NOTE,'best_val/last_epoch均为基础验证集记录；test行的ADE/FDE才是选中模型的测试误差。',
               '区间按目标MMSI聚类，以已训练种子为条件；逐项区间未经多重比较校正。'],
        prediction_hashes={r['path']:digest(r['path']) for r in records if r['path'].endswith('.npz')})
    csv_save(out/'per_seed.csv',seed_rows);csv_save(out/'summary.csv',summaries);csv_save(out/'paired_intervals.csv',pairs);save(out/'summary.json',report)
    lines=['# '+('验证' if split=='val' else '测试')+'结果汇总','',f"完成 {len(completed)}/48 组；单位为米，越低越好。",'',
        '“best_val”和“last_epoch”列均指基础验证集ADE，供检查选型；它们不是测试误差。','',
        '|模型|配置|条件|分层|所评估ADE|所评估FDE|区域等权ADE|best_val ADE|last_epoch ADE|','|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        cells=[r['model'],S.ARM_ZH.get(r['arm'],r['arm']),S.VIEW_ZH.get(r['view'],r['view']),r['subset']]
        cells +=[fmt(r[k]) for k in ('ADE_m_mean','FDE_m_mean','region_macro_ADE_m_mean','best_val_ADE_m_mean','last_epoch_ADE_m_mean')]
        lines.append('|'+ '|'.join(cells)+'|')
    lines+=['',REUSE_NOTE,'','全部逐种子结果和配对区间见同目录CSV。']
    (out/'结果汇总.md').write_text('\n'.join(lines),encoding='utf-8')
    return report

def _settings(root,records):
    root=Path(root);meta=read(root/'experiment.json') if (root/'experiment.json').exists() else {}
    first=next(r for r in records if not r['model'].startswith('CV'));s=first.get('settings',{})
    def pick(*keys):
        for k in keys:
            if meta.get(k) is not None:return meta[k]
            if s.get(k) is not None:return s[k]
        return None
    norm=pick('norm_sha256','normalizer_sha256')
    if norm is None:
        p=Path(first['path']).with_name('normalizer.json')
        if p.exists():norm=digest(p)
    return dict(name=pick('name') or root.name,root=str(root),hidden=pick('hidden','hidden_dim'),policy=pick('policy'),
        quota=pick('train_per_region','quota'),history=pick('history_steps','hist'),future=pick('future_steps','future'),
        batch=pick('batch_size','batch'),epochs=pick('epochs'),lr=pick('lr','learning_rate'),norm_sha256=norm)

def compare(roots,out,split='test',draws=2000):
    """One-factor comparisons only; never silently intersect mismatched held-out samples."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);items=[]
    for root in roots:
        records,index=discover(root,split);items.append((_settings(root,records),index))
    comparisons=[];skipped=[];all_rows=[]
    for i,(a,ai) in enumerate(items):
        for b,bi in items[i+1:]:
            factors=[k for k in ('hidden','policy','quota') if a[k]!=b[k]]
            if len(factors)!=1:continue
            reason=None
            for key in ('history','future','batch','lr'):
                if a[key]!=b[key]:reason=f'{key} differs'
            if not a['norm_sha256'] or a['norm_sha256']!=b['norm_sha256']:reason='normalizer missing or differs'
            if reason:skipped.append(dict(a=a['name'],b=b['name'],reason=reason));continue
            factor=factors[0]
            if factor=='policy':
                if {a['policy'],b['policy']}!={'constant_last','cosine_best'}:continue
                cand,ci,base,bs=(a,ai,b,bi) if a['policy']=='cosine_best' else (b,bi,a,ai)
            else:
                cand,ci,base,bs=(a,ai,b,bi) if float(a[factor])>float(b[factor]) else (b,bi,a,ai)
            comparison=f"{cand['name']} - {base['name']}"
            done=0
            cells=[(m,a,v) for m in MODELS for a in ARMS for v in VIEWS]
            for model,arm,view in tqdm(cells,desc=f'Compare {factor}: {cand["name"]}',unit='cell'):
                seeds=[s for s in SEEDS if (model,arm,s,view) in ci and (model,arm,s,view) in bs]
                if not seeds:continue
                aa=[ci[model,arm,s,view] for s in seeds];bb=[bs[model,arm,s,view] for s in seeds]
                # Epoch budgets can vary by backbone, but must match each paired intervention.
                for ar,br in zip(aa,bb):
                    ae=ar['settings'].get('epochs');be=br['settings'].get('epochs')
                    if ae!=be:raise ValueError(f'Unequal epoch budgets in {comparison}/{model}')
                for subset in ('all','observed_maneuver'):
                    for r in S.paired_bootstrap(aa,bb,subset,draws,20261106):
                        all_rows.append(dict(comparison=comparison,changed_factor=factor,model=model,arm=arm,view=view,subset=subset,n_seeds=len(seeds),split=split,**r))
                done+=1
            comparisons.append(dict(candidate=cand,baseline=base,changed_factor=factor,paired_cells=done))
    csv_save(out/'factor_comparisons.csv',all_rows);save(out/'factor_comparisons.json',dict(split=split,comparisons=comparisons,skipped=skipped,rows=all_rows,
        note='训练机制对照同时改变学习率调度与验证选型；不能单独归因余弦调度。数据/容量对照仅改变对应因素。'+REUSE_NOTE))
    lines=['# 逐因素配对比较','', '差值＝候选－对照，负值表示预测误差下降。训练机制对照是“余弦调度＋验证选型”整体改变，不能归因于调度器单项。','',
        '|比较|变化因素|模型|配置|条件|分层|指标|差值|95%下限|95%上限|','|---|---|---|---|---|---|---|---:|---:|---:|']
    for r in all_rows:
        if r['aggregation']!='pooled' or r['subset']!='all':continue
        lines.append('|'+ '|'.join([r['comparison'],r['changed_factor'],r['model'],r['arm'],r['view'],r['subset'],r['metric'],fmt(r['delta']),fmt(r['ci95_low']),fmt(r['ci95_high'])])+'|')
    (out/'逐因素比较.md').write_text('\n'.join(lines),encoding='utf-8')
    return comparisons

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('summarize');s.add_argument('--root',type=Path,required=True);s.add_argument('--out',type=Path,required=True);s.add_argument('--split',choices=['val','test'],default='val');s.add_argument('--data',type=Path);s.add_argument('--draws',type=int,default=2000)
    c=sub.add_parser('compare');c.add_argument('--roots',nargs='+',type=Path,required=True);c.add_argument('--out',type=Path,required=True);c.add_argument('--split',choices=['val','test'],default='test');c.add_argument('--draws',type=int,default=2000)
    a=p.parse_args()
    if a.command=='summarize':summarize(a.root,a.out,a.split,a.data,a.draws)
    else:compare(a.roots,a.out,a.split,a.draws)
