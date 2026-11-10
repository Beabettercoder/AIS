"""Exact raw-history replay, a fixed causal reset candidate, CV continuation gate."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from reanchor import reanchor_positions,observed_history

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def replay(pilot,out,split,protocol,F):
    manifest=json.loads((pilot/'data/data_manifest.json').read_text())
    old=load(pilot/f'data/{split}.npz');new={k:v.copy() for k,v in old.items()}
    index={str(x):i for i,x in enumerate(old['window_id'])}
    rows=[r for r in json.loads((pilot/'data/window_provenance.json').read_text()) if r['split']==split]
    views=['base'] if split=='train' else F.VIEWS
    for view in views:new[view+'__reanchor_reset']=np.zeros((len(rows),12),bool)
    max_replay_error=0.;records=[]
    for region in tqdm(manifest['config']['regions'],desc=f'{split} raw-history replay',unit='region'):
        rr=[r for r in rows if r['region']==region]
        source=Path(rr[0]['source_file']);assert sha(source)==rr[0]['source_sha256']
        table=pd.read_parquet(source,columns=['mmsi','timestamp','lon','lat','sog','cog'])
        registry=json.loads((source.parents[1]/'source_registry.json').read_text())['regions']
        bbox=registry[region]['core_bbox']
        for row in rr:
            i=index[row['window_id']];h=table.iloc[row['row_indices'][:12]]
            t=h.timestamp.to_numpy(float);p=F.project(h.lon.to_numpy(),h.lat.to_numpy(),bbox)
            sog=h.sog.to_numpy(float);cog=h.cog.to_numpy(float)
            for view in views:
                visible,v=observed_history(F,t,p,sog,cog,row['window_id'],view)
                w,valid,e=F.compatibility(t,visible,v)
                gp,gv,branches=F.causal_gate(t,visible,v,w,valid,manifest['speed_cap_mps'])
                exact_anchor=visible[0]+gp[-1]
                err=float(np.max(np.abs(exact_anchor-old[view+'__anchor_gate'][i])))
                max_replay_error=max(max_replay_error,err)
                np.testing.assert_allclose(exact_anchor,old[view+'__anchor_gate'][i],rtol=0,atol=1e-8)
                np.testing.assert_array_equal(F.self_features(t,gp,gv,cog),old[view+'__gated'][i])
                cp,reset=reanchor_positions(t,visible,gp,w,valid,protocol['position_reset_threshold'])
                new[view+'__gated'][i,:,:2]=cp.astype('f4')
                new[view+'__anchor_gate'][i]=visible[0]+cp[-1]
                new[view+'__reanchor_reset'][i]=reset
                records.append(dict(window_id=row['window_id'],view=view,mmsi=row['mmsi'],
                    region=region,reset_steps=np.flatnonzero(reset).tolist(),
                    original_anchor_shift_from_observed_m=float(np.linalg.norm(exact_anchor-visible[-1])),
                    candidate_anchor_shift_from_observed_m=float(np.linalg.norm(visible[0]+cp[-1]-visible[-1]))))
    for key in old:
        if key.endswith('__gated'):
            np.testing.assert_array_equal(old[key][:,:,2:],new[key][:,:,2:])
        elif not key.endswith('__anchor_gate'):np.testing.assert_array_equal(old[key],new[key])
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/f'{split}.npz',**new)
    save(out/f'{split}_replay_audit.json',dict(windows=len(rows),max_original_anchor_replay_difference_m=max_replay_error,
         source_npz_sha256=sha(pilot/f'data/{split}.npz'),candidate_npz_sha256=sha(out/f'{split}.npz'),
         changed_slots=['gated relative position x/y','absolute anchor_gate'],
         unchanged='raw, velocity, time, future observation reference, fields, source identifiers and masks',records=records))
    return old,new

def record(z,view,pred):
    err=np.linalg.norm(pred-z['future_xy'],axis=-1)
    return dict(window_id=z['window_id'].astype(str),mmsi=z['mmsi'].astype(str),region=z['region'].astype(str),
        future_xy=z['future_xy'],query_s=z['query_s'],maneuver=z['maneuver'].astype(bool),
        ADE=err.mean(1),FDE=err[:,-1])

def cv(z,view,kind):
    return z[view+'__anchor_'+kind][:,None,:]+z[view+'__vel_'+kind].astype(float)[:,None,:]*z['query_s'].astype(float)[...,None]

def score(old,new,protocol,S,out):
    results=[];details=[];intervals=[];criteria=[]
    for view in ['base','target_jump','target_drift','target_speed','neighbor_stress']:
        records={name:record(old,view,pred) for name,pred in [('CV_raw',cv(old,view,'raw')),
            ('CV_original_gate',cv(old,view,'gate')),('CV_reanchor',cv(new,view,'gate'))]}
        for subset in ['all','observed_maneuver']:
            mask=np.ones(len(old['window_id']),bool) if subset=='all' else old['maneuver'].astype(bool)
            for name,r in records.items():
                for metric in ['ADE','FDE']:
                    results.append(dict(view=view,subset=subset,method=name,metric=metric,
                        windows=int(mask.sum()),mean_m=float(r[metric][mask].mean())))
            boot=S.paired_bootstrap([records['CV_reanchor']],[records['CV_original_gate']],subset,
                protocol['bootstrap_draws'],protocol['bootstrap_seed'])
            for b in boot:intervals.append(dict(view=view,subset=subset,**b))
            for metric in ['ADE','FDE']:
                r=next(x for x in boot if x['aggregation']=='pooled' and x['metric']==metric+'_m')
                a=float(records['CV_raw'][metric][mask].mean());b=float(records['CV_original_gate'][metric][mask].mean());c=float(records['CV_reanchor'][metric][mask].mean())
                retained=(a-c)/(a-b) if a>b else None
                if view=='base':
                    criteria.append(dict(view=view,subset=subset,metric=metric,rule='paired CI upper < 0',
                        passed=bool(r['ci95_high']<0),ci_upper_m=r['ci95_high'],tolerance_m=0.))
                elif view in ['target_jump','target_speed','target_drift'] and subset=='all':
                    tolerance=.01*b;ok=bool(r['ci95_high']<=tolerance)
                    if view in ['target_jump','target_speed']:ok=ok and retained is not None and retained>=.95
                    criteria.append(dict(view=view,subset=subset,metric=metric,rule='paired CI upper <= 1% original CV; jump/speed retain >=95% old gain',
                        passed=bool(ok),ci_upper_m=r['ci95_high'],tolerance_m=tolerance,
                        raw_mean_m=a,old_mean_m=b,candidate_mean_m=c,retained_raw_gain_fraction=retained))
        for i in range(len(old['window_id'])):
            details.append(dict(window_id=str(old['window_id'][i]),mmsi=int(old['mmsi'][i]),region=str(old['region'][i]),
                maneuver=bool(old['maneuver'][i]),view=view,
                **{name+'_'+met+'_m':float(r[met][i]) for name,r in records.items() for met in ['ADE','FDE']}))
    passed=all(c['passed'] for c in criteria)
    save(out/'cv_results.json',dict(results=results,intervals=intervals,criteria=criteria,passed=passed))
    save(out/'cv_decision.json',dict(passed=passed,criteria=criteria,protocol_sha256=sha(Path(__file__).with_name('protocol.json')),
         candidate_val_sha256=sha(out/'data/val.npz'),decision_scope='development validation only; fixed single candidate; no test used'))
    for path,rows in [('cv_means.csv',results),('cv_paired_intervals.csv',intervals),('cv_window_errors.csv',details)]:
        with (out/path).open('w',newline='',encoding='utf-8-sig') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    lines=['# 因果重锚定：恒速验证','',f'预先冻结条件：**{"通过" if passed else "未全部通过"}**。阈值固定0.7，未搜索阈值。','',
        '只改变历史位置重建及预测起点；速度、流场、未来参考和查询时间均保持原样。下表单位为米，越低越好。',
        '','|观测条件|原始恒速ADE/FDE|原门控恒速ADE/FDE|重锚恒速ADE/FDE|','|---|---:|---:|---:|']
    names={'base':'基础观测','target_jump':'末点位置跳变','target_drift':'后段位置漂移','target_speed':'后段速度尖峰','neighbor_stress':'邻船污染'}
    for view,title in names.items():
        cells=[]
        for method in ['CV_raw','CV_original_gate','CV_reanchor']:
            vals=[next(r['mean_m'] for r in results if (r['view'],r['subset'],r['method'],r['metric'])==(view,'all',method,m)) for m in ['ADE','FDE']]
            cells.append(f'{vals[0]:.2f} / {vals[1]:.2f}')
        lines.append('|'+title+'|'+'|'.join(cells)+'|')
    lines.extend(['','## 继续训练判据','',
        '基础全集与自然机动子集的ADE/FDE配对95%区间上限须小于零；污染视图对应区间上限须不超过原门控CV误差的1%，位置跳变和速度尖峰还须保留至少95%原有改善。1%是本轮开发继续阈值，不是领域统一标准。',
        '','|条件|分层|指标|区间上限（米）|允许上限（米）|是否通过|','|---|---|---|---:|---:|---|'])
    for c in criteria:lines.append(f"|{names[c['view']]}|{c['subset']}|{c['metric']}|{c['ci_upper_m']:.2f}|{c['tolerance_m']:.2f}|{'通过' if c['passed'] else '未通过'}|")
    (out/'恒速验证报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=passed,criteria=criteria,results=[r for r in results if r['subset']=='all']),ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,default=Path('/mnt/sda/WangZijia/AIS/forecast_pilot_20261105'))
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--stage',choices=['cv','train'],default='cv');args=p.parse_args()
    root=args.out;root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.pilot));import features as F;import summarize as S
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text())
    dm=json.loads((args.pilot/'data/data_manifest.json').read_text())
    assert sha(args.pilot/'features.py')==dm['config']['features_sha256']
    freeze=dict(protocol_sha256=sha(Path(__file__).with_name('protocol.json')),reanchor_sha256=sha(Path(__file__).with_name('reanchor.py')),
        builder_sha256=sha(__file__),source_data_manifest_sha256=sha(args.pilot/'data/data_manifest.json'))
    marker=root/'frozen_before_scoring.json'
    if marker.exists():assert json.loads(marker.read_text())==freeze,'Frozen code/protocol changed. Use a new stage directory.'
    else:save(marker,freeze)
    if args.stage=='cv':
        if (root/'cv_results.json').exists():raise RuntimeError('CV result exists; do not overwrite a frozen candidate evaluation.')
        old,new=replay(args.pilot,root/'data','val',protocol,F);score(old,new,protocol,S,root)
    else:
        decision=json.loads((root/'cv_decision.json').read_text());assert decision['passed'],'CV criteria failed; no train cache/training authorized by this protocol'
        assert sha(root/'data/val.npz')==decision['candidate_val_sha256']
        replay(args.pilot,root/'data','train',protocol,F)
        save(root/'data/data_manifest.json',dict(source_pilot=str(args.pilot),parent_data_manifest_sha256=freeze['source_data_manifest_sha256'],
            source_npz_sha256=dm['npz_sha256'],npz_sha256={s:sha(root/f'data/{s}.npz') for s in ['train','val']},
            frozen=freeze,changed_inputs='gated position slots and anchor_gate only',frozen_original_normalizer=True,test_data_constructed=False))

if __name__=='__main__':main()
