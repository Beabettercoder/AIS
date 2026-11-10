"""Read-only audit of validation windows whose baseline gated anchor changes."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(x):
    x=np.asarray(x,float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,
        median=float(np.median(x)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None,
        maximum=float(x.max()) if len(x) else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original',type=Path,default=Path('/mnt/sda/WangZijia/AIS/forecast_pilot_20261105'))
    ap.add_argument('--out',type=Path,default=Path('.'));args=ap.parse_args();root=args.original;out=args.out
    out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root))
    import features as F
    manifest=json.loads((root/'data/data_manifest.json').read_text())
    assert digest(root/'features.py')==manifest['config']['features_sha256']
    assert digest(root/'data/val.npz')==manifest['npz_sha256']['val']
    with np.load(root/'data/val.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    provenance=[x for x in json.loads((root/'data/window_provenance.json').read_text()) if x['split']=='val']
    lookup={str(w):i for i,w in enumerate(data['window_id'])}
    registry=json.loads(Path(provenance[0]['source_file']).parents[1].joinpath('source_registry.json').read_text())['regions']
    shift=np.linalg.norm(data['base__anchor_gate']-data['base__anchor_raw'],axis=1)
    changed=shift>.001
    truth=data['future_xy'];query=data['query_s'].astype(float)[...,None]
    cv={}
    for pa in ['raw','gate']:
        for va in ['raw','gate']:
            pred=data['base__anchor_'+pa][:,None,:]+data['base__vel_'+va].astype(float)[:,None,:]*query
            err=np.linalg.norm(pred-truth,axis=-1)
            cv[pa+'_position_'+va+'_velocity']={'ADE':err.mean(1),'FDE':err[:,-1]}
    windows=[];steps=[];source_hashes={};max_anchor_replay_error=0.
    for region in tqdm(manifest['config']['regions'],desc='Audit raw history / regions'):
        docs=[d for d in provenance if d['region']==region and changed[lookup[d['window_id']]]]
        if not docs:continue
        path=Path(docs[0]['source_file']);source_hashes[region]=digest(path)
        assert source_hashes[region]==registry[region]['sha256']
        table=pd.read_parquet(path,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
        for doc in docs:
            i=lookup[doc['window_id']];raw=table.iloc[doc['row_indices'][:12]]
            assert (raw.mmsi.to_numpy()==doc['mmsi']).all()
            t=raw.timestamp.to_numpy(float);p=F.project(raw.lon.to_numpy(),raw.lat.to_numpy(),registry[region]['core_bbox'])
            v=F.vectors(raw.sog.to_numpy(),raw.cog.to_numpy());w,valid,e=F.compatibility(t,p,v)
            gp,gv,branch=F.causal_gate(t,p,v,w,valid,manifest['speed_cap_mps'])
            anchor=p[0]+gp[-1]
            replay=float(np.max(np.abs(anchor-data['base__anchor_gate'][i])))
            max_anchor_replay_error=max(max_anchor_replay_error,replay)
            np.testing.assert_allclose(anchor,data['base__anchor_gate'][i],rtol=0,atol=1e-9)
            np.testing.assert_allclose(w,data['base__target_weight'][i],rtol=1e-6,atol=1e-7)
            np.testing.assert_array_equal(branch,data['base__gate_branch'][i])
            dp=np.diff(p,axis=0);gdp=np.diff(gp,axis=0);correction=gdp-dp
            cnorm=np.linalg.norm(correction,axis=1);hits=(np.flatnonzero(cnorm>.001)+1).tolist()
            early=any(k<11 for k in hits);last=11 in hits
            category='both_early_and_final' if early and last else 'early_only' if early else 'final_only' if last else 'sub_mm_accumulation'
            row=dict(window_id=doc['window_id'],region=region,mmsi=int(doc['mmsi']),natural_maneuver=bool(data['maneuver'][i]),
                anchor_shift_m=float(shift[i]),anchor_shift_x_m=float(anchor[0]-p[-1,0]),anchor_shift_y_m=float(anchor[1]-p[-1,1]),
                correction_category=category,corrected_steps=hits,last_corrected_step=max(hits) if hits else None,
                final_weight=float(w[-1]),final_valid=bool(valid[-1]),final_high_weight=bool(valid[-1] and w[-1]>=.7),
                final_weight_exact_one=bool(w[-1]==1.),final_branch=int(branch[-1]),
                final_increment_change_m=float(cnorm[-1]),pre_final_accumulated_shift_m=float(np.linalg.norm(gp[-2]-(p[-2]-p[0]))),
                last_velocity_shift_mps=float(np.linalg.norm(gv[-1]-v[-1])))
            for name,vals in cv.items():
                row[name+'_ADE_m']=float(vals['ADE'][i]);row[name+'_FDE_m']=float(vals['FDE'][i])
            row['anchor_only_ADE_change_m']=row['gate_position_raw_velocity_ADE_m']-row['raw_position_raw_velocity_ADE_m']
            row['gate_both_ADE_change_m']=row['gate_position_gate_velocity_ADE_m']-row['raw_position_raw_velocity_ADE_m']
            windows.append(row)
            for k in range(1,12):
                cp=gp[k]-(p[k]-p[0]);j=k-1
                steps.append(dict(window_id=doc['window_id'],region=region,mmsi=int(doc['mmsi']),step=k,
                    timestamp=float(t[k]),dt_s=float(t[k]-t[k-1]),weight=float(w[k]),valid=bool(valid[k]),kinematic_residual_mps=float(e[k]),
                    branch=int(branch[k]),raw_dx_m=float(dp[j,0]),raw_dy_m=float(dp[j,1]),
                    gate_dx_m=float(gdp[j,0]),gate_dy_m=float(gdp[j,1]),correction_x_m=float(correction[j,0]),
                    correction_y_m=float(correction[j,1]),increment_change_m=float(cnorm[j]),
                    corrected_gt_1mm=bool(cnorm[j]>.001),cumulative_offset_x_m=float(cp[0]),cumulative_offset_y_m=float(cp[1]),
                    cumulative_offset_m=float(np.linalg.norm(cp))))
    windows.sort(key=lambda x:x['anchor_shift_m'],reverse=True)
    subsets={}
    for name,mask in [('all',np.ones(len(shift),bool)),('natural_maneuver',data['maneuver'].astype(bool)),
        ('not_natural_maneuver',~data['maneuver'].astype(bool)),('changed_anchors',changed)]:
        ids={str(x) for x in data['window_id'][mask]};ww=[x for x in windows if x['window_id'] in ids]
        subsets[name]=dict(windows=int(mask.sum()),changed_windows=int((mask&changed).sum()),
            anchor_shift_stats_m=stats(shift[mask]),changed_anchor_shift_stats_m=stats(shift[mask&changed]),
            correction_categories=dict(Counter(x['correction_category'] for x in ww)),
            final_high_weight_changed_windows=sum(x['final_high_weight'] for x in ww),
            final_weight_exact_one_changed_windows=sum(x['final_weight_exact_one'] for x in ww),
            last_correction_step_counts=dict(Counter(x['last_corrected_step'] for x in ww)),
            CV={key:{metric+'_m':float(vals[metric][mask].mean()) for metric in ['ADE','FDE']} for key,vals in cv.items()})
    total_anchor_delta=float(cv['gate_position_raw_velocity']['ADE'].mean()-cv['raw_position_raw_velocity']['ADE'].mean())
    contribution_by_category={key:sum(x['anchor_only_ADE_change_m'] for x in windows if x['correction_category']==key)/len(shift)
        for key in sorted({x['correction_category'] for x in windows})}
    cold_start_ids={x['window_id'] for x in steps if x['corrected_gt_1mm'] and x['branch']==5}
    top_contributors=[]
    for row in sorted(windows,key=lambda x:x['anchor_only_ADE_change_m'],reverse=True)[:10]:
        contributor={k:row[k] for k in ['window_id','region','mmsi','anchor_shift_m','corrected_steps','anchor_only_ADE_change_m']}
        contributor['overall_mean_ADE_contribution_m']=row['anchor_only_ADE_change_m']/len(shift)
        contributor['share_of_net_overall_anchor_ADE_delta']=contributor['overall_mean_ADE_contribution_m']/total_anchor_delta
        contributor['has_cold_start_branch']=row['window_id'] in cold_start_ids
        top_contributors.append(contributor)
    summary=dict(status='pass',original_dataset=str(root/'data/val.npz'),source_feature_sha256=digest(root/'features.py'),
        val_npz_sha256=digest(root/'data/val.npz'),source_hashes=source_hashes,script_sha256=digest(__file__),
        changed_anchor_threshold_m=.001,validation_windows=len(shift),changed_anchor_windows=int(changed.sum()),
        max_original_gate_replay_error_m=max_anchor_replay_error,step_rows=len(steps),
        corrected_step_branch_counts=dict(Counter(x['branch'] for x in steps if x['corrected_gt_1mm'])),
        cold_start_affected_windows=len(cold_start_ids),
        cold_start_affected_windows_contribution_to_mean_ADE_delta_m=sum(x['anchor_only_ADE_change_m'] for x in windows if x['window_id'] in cold_start_ids)/len(shift),
        net_anchor_only_ADE_delta_m=total_anchor_delta,mean_ADE_contribution_by_correction_category=contribution_by_category,
        top_anchor_error_contributors=top_contributors,
        subsets=subsets,notes=['Anchor shift is relative to original observed final position, not physical true position.',
            'ADE changes are forecast-error changes, not average anchor displacements.',
            'Counterfactual CV combinations hold time queries and future observation references fixed.',
            'No model trained; no dataset or baseline code modified.'])
    save_json(out/'audit_92_summary.json',summary);save_json(out/'audit_92_windows.json',windows);save_json(out/'audit_92_steps.json',steps)
    save_csv(out/'audit_92_windows.csv',windows);save_csv(out/'audit_92_steps.csv',steps)
    top=top_contributors[0]
    top_steps=[x for x in steps if x['window_id']==top['window_id'] and x['corrected_gt_1mm']]
    lines=['# 92个起点变化窗口独立核查','',f"原始验证集{len(shift)}窗，门控起点相对原始观测末点变化超过1毫米的有{int(changed.sum())}窗；逐窗重新读取原始报文并重放原门控，最大缓存复算差为{max_anchor_replay_error:.12g}米。",'',
        '## 最重要的逐窗证据','',
        f"最大贡献来自{top['region']}的{top['window_id']}（MMSI {top['mmsi']}），起点累计偏移{top['anchor_shift_m']:.2f}米。该例对全部1200窗的起点-only平均ADE差贡献{top['overall_mean_ADE_contribution_m']:.4f}米，占净差{total_anchor_delta:.4f}米的{top['share_of_net_overall_anchor_ADE_delta']*100:.2f}%。",'',
        f"该例修正触发步详情：{top_steps}。首次步间位移与报告速度严重不一致，低权重且可信历史不足进入分支5；后续步骤恢复相容，但旧偏移仍被累计保留。",'',
        '因此基础视图只能称未额外注入污染的原始观测视图，不能当作已证明无测量问题的干净轨迹。不能把这92窗一概解释为真实自然转弯或变速误伤。','',
        f"存在历史不足分支修正的窗口共{len(cold_start_ids)}个，对全部窗口起点-only平均ADE差的净贡献为{summary['cold_start_affected_windows_contribution_to_mean_ADE_delta_m']:.4f}米。",'']
    for name,s in subsets.items():
        lines.extend([f'## {name}','',f"窗口{s['windows']}，起点变化{s['changed_windows']}；前中段/末段分类：{s['correction_categories']}；末步有效且w≥0.7的变化窗口{s['final_high_weight_changed_windows']}；末步w=1的变化窗口{s['final_weight_exact_one_changed_windows']}。",'',
            '|起点与速度组合|ADE（米）|FDE（米）|','|---|---:|---:|'])
        for method,m in s['CV'].items():lines.append(f"|{method}|{m['ADE_m']:.4f}|{m['FDE_m']:.4f}|")
        lines.append('')
    lines.extend(['平均预测误差差值不是平均坐标偏移。坐标偏移是相对于原始观测末点的变化，不是测量真值误差。',
        '逐窗清单见audit_92_windows.csv/json；92窗全部1012个历史递推步见audit_92_steps.csv/json。'])
    (out/'audit_92_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_hashes','notes']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
