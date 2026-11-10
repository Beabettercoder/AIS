"""Read-only report-level sensitivity diagnostics for the fixed speed stress.

Use base-view versus stress-view observed arrays only. No source-truth arrays,
labels, predictive-score changes, threshold changes or model fitting.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def kin(t,p,s,c):
    theta=np.deg2rad(c);v=s[:,None]*(1852/3600)*np.column_stack((np.sin(theta),np.cos(theta)))
    good=np.isfinite(t)&np.isfinite(p).all(1)&np.isfinite(s)&np.isfinite(c)&(s>=0)&(s<102.2)&(c>=0)&(c<360)
    dt=np.diff(t);valid=np.r_[False,good[:-1]&good[1:]&(dt>0)&(dt<=300)]
    e=np.zeros(len(t));ii=np.flatnonzero(valid);e[ii]=np.linalg.norm((p[ii]-p[ii-1])/(t[ii]-t[ii-1])[:,None]-(v[ii]+v[ii-1])/2,axis=1)
    w=np.zeros(len(t));w[ii]=np.exp(-np.maximum(0,e[ii]-2.5)**2/8)
    return valid,e,w
def distribution(values):
    a=np.asarray(values,float)
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,
            'p10':float(np.percentile(a,10)) if len(a) else None,'p90':float(np.percentile(a,90)) if len(a) else None,
            'min':float(a.min()) if len(a) else None,'max':float(a.max()) if len(a) else None}
def summarise(rows):
    valid=[x for x in rows if x['new_kinematic_valid']]
    n=len(valid)
    def rate(key,pred):
        count=sum(pred(x[key]) for x in valid);return {'count':count,'denominator_valid_reports':n,'fraction':count/n if n else None}
    return {'observed_state_reports':len(rows),'source_windows':len({x['source_window_id'] for x in rows}),
            'unique_source_member_reports':len({(x['source_window_id'],x['neighbor_mmsi'],x['time_s']) for x in rows}),
            'kinematically_assessable_reports':n,'unknown_kinematics_reports':len(rows)-n,
            'original_observed_sog_kn':distribution([x['base_sog_kn'] for x in rows]),
            'new_e_kin_mps':distribution([x['new_e_kin_mps'] for x in valid]),
            'new_w':distribution([x['new_w'] for x in valid]),
            'base_e_kin_mps':distribution([x['base_e_kin_mps'] for x in valid]),
            'e_kin_change_mps':distribution([x['new_e_kin_mps']-x['base_e_kin_mps'] for x in valid]),
            'w_ge_0_9':rate('new_w',lambda v:v>=.9),'w_ge_0_3':rate('new_w',lambda v:v>=.3),
            'e_inside_deadzone_le_2_5':rate('new_e_kin_mps',lambda v:v<=2.5),
            'suffix_used_unique_state_reports':sum(x['used_in_scorer_suffix'] for x in rows),
            'suffix_used_unique_source_reports':len({(x['source_window_id'],x['neighbor_mmsi'],x['time_s']) for x in rows if x['used_in_scorer_suffix']}),
            'suffix_grid_hit_count_before_dedup':sum(x['suffix_grid_hits'] for x in rows),
            'low_report_w_but_member_w_ge_0_3':sum(x['used_in_scorer_suffix'] and x['new_kinematic_valid'] and x['new_w']<.3 and x['new_score_member_weight']>=.3 for x in rows)}
def csv_write(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty']);w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root;a.out.mkdir(exist_ok=False)
    bp=r/'all_sources_v1/relation_manifest.json';sp=r/'neighbor_speed_stress_v1/stress_manifest.json'
    bm=json.loads(bp.read_text());sm=json.loads(sp.read_text());base_items={x['window_id']:x for x in bm['samples']}
    changed=[];affected=[];source_proof=[]
    for item in sm['samples']:
        wid=item['window_id'];bitem=base_items[wid];bf=r/'all_sources_v1'/bitem['file'];sf=r/'neighbor_speed_stress_v1'/item['file']
        assert sha(bf)==bitem['sha256'] and sha(sf)==item['sha256']
        source_proof.append({'base_file':str(bf),'base_sha256':bitem['sha256'],'stress_file':str(sf),'stress_sha256':item['sha256']})
        with np.load(bf,allow_pickle=False) as bz,np.load(sf,allow_pickle=False) as sz:
            keys=['target_t','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets','cohort_mmsi']
            b={k:bz[k] for k in keys};s={k:sz[k] for k in keys}
        for x in (b,s):
            for k in ['neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg']:
                assert np.array_equal(x[k][0],x[k][2]) and np.array_equal(x[k][1],x[k][3])
        for k in ['target_t','neighbor_t','neighbor_pos','neighbor_cog_deg','member_offsets','cohort_mmsi']:assert np.array_equal(b[k],s[k])
        t=b['target_t'][0];anchor=max(2,len(t)//2-1);prefix=np.arange(anchor+1);suffix=np.arange(anchor+1,len(t))[-3:]
        operator_positions={flat:(rank,m['factor']) for m in item['speed_stress']['members'] if m['applied'] for rank,flat in enumerate(m['flat_indices'])}
        for state in [0,1]: # a duplicates are removed; b states have different observed speeds.
            for j,(lo,hi) in enumerate(zip(b['member_offsets'][:-1],b['member_offsets'][1:])):
                lo,hi=int(lo),int(hi);tt=b['neighbor_t'][state,lo:hi];pp=b['neighbor_pos'][state,lo:hi];sb=b['neighbor_sog_kn'][state,lo:hi];ss=s['neighbor_sog_kn'][state,lo:hi];cc=b['neighbor_cog_deg'][state,lo:hi]
                vb,eb,wb=kin(tt,pp,sb,cc);vs,es,ws=kin(tt,pp,ss,cc);assert np.array_equal(vb,vs)
                latest=np.searchsorted(tt,t,side='right')-1;safe=np.maximum(latest,0);ok=(latest>=0)&(t>=tt[safe])&(t-tt[safe]<=300)
                pre=prefix[ok[prefix]];post=suffix[ok[suffix]];usable=len(pre)>=2 and len(post)>=2
                post_ids=latest[post] if usable else np.array([],int)
                wm0=float(np.sqrt(wb[latest[pre]].mean()*wb[latest[post]].mean())) if usable else 0.
                wm1=float(np.sqrt(ws[latest[pre]].mean()*ws[latest[post]].mean())) if usable else 0.
                changed_ids=np.flatnonzero(sb!=ss)
                affected_ids=np.flatnonzero((np.abs(es-eb)>1e-10)|(np.abs(ws-wb)>1e-12))
                for index in sorted(set(changed_ids.tolist())|set(affected_ids.tolist())):
                    numeric_changed=bool(sb[index]!=ss[index]);flat=lo+int(index);op=operator_positions.get(flat)
                    if numeric_changed:assert op is not None
                    row={'source_window_id':wid,'region':item['region'],'split':item['split'],'neighbor_observed_state':state,
                         'representative_case_index':state,'neighbor_mmsi':str(b['cohort_mmsi'][j]),'member_index':j,'member_report_index':int(index),'time_s':float(tt[index]),
                         'base_sog_kn':float(sb[index]),'stress_sog_kn':float(ss[index]),'numeric_speed_changed':numeric_changed,
                         'speed_band':'lt5' if sb[index]<5 else '5to10' if sb[index]<10 else 'ge10',
                         'burst_report_position':int(op[0]+1) if op else 0,'scheduled_factor':float(op[1]) if op else None,
                         'base_kinematic_valid':bool(vb[index]),'new_kinematic_valid':bool(vs[index]),
                         'base_e_kin_mps':float(eb[index]),'new_e_kin_mps':float(es[index]),'base_w':float(wb[index]),'new_w':float(ws[index]),
                         'member_usable_for_statistics':bool(usable),'unique_prefix_reports':len(set(latest[pre].tolist())),'unique_suffix_reports':len(set(latest[post].tolist())),
                         'suffix_grid_hits':int(np.sum(post_ids==index)),'used_in_scorer_suffix':bool(index in set(post_ids.tolist())),
                         'base_score_member_weight':wm0,'new_score_member_weight':wm1}
                    if numeric_changed:changed.append(row)
                    if index in set(affected_ids.tolist()):affected.append(row)
    key=lambda x:(x['source_window_id'],x['neighbor_observed_state'],x['neighbor_mmsi'],x['time_s'])
    assert len({key(x) for x in changed})==len(changed)
    summary={'status':'passed','scope':'Read-only known-operation diagnostics. No predictive scores, models, thresholds, labels or existing protocols modified.',
             'unit':'Source window + neighbor observed counterfactual state(b=0/1) + MMSI + timestamp. Target-factor duplicate siblings are removed. Same report reused by multiple suffix target-grid queries is counted once in unique-report counts.',
             'original_speed_definition':'Final visible base-view SOG before the additional speed stress, including its existing cohort modulation and position-only noise; not hidden original/source-truth velocity.',
             'thresholds':{'w_ge_0_9_equivalent_e_max_mps':float(2.5+np.sqrt(-8*np.log(.9))),'w_ge_0_3_equivalent_e_max_mps':float(2.5+np.sqrt(-8*np.log(.3)))},
             'base_manifest_sha256':sha(bp),'stress_manifest_sha256':sha(sp),'code_sha256':sha(Path(__file__)),'source_files':source_proof,'splits':{}}
    for split in ['train','val']:
        rr=[x for x in changed if x['split']==split];aa=[x for x in affected if x['split']==split]
        summary['splits'][split]={'changed_reports':summarise(rr),'by_original_speed':{band:summarise([x for x in rr if x['speed_band']==band]) for band in ['lt5','5to10','ge10']},
                                 'by_burst_position':{str(i):summarise([x for x in rr if x['burst_report_position']==i]) for i in [1,2]},
                                 'by_observed_neighbor_state':{str(i):summarise([x for x in rr if x['neighbor_observed_state']==i]) for i in [0,1]},
                                 'only_previously_high_confidence':summarise([x for x in rr if x['base_kinematic_valid'] and x['base_w']>=.9]),
                                 'changed_and_suffix_used':summarise([x for x in rr if x['used_in_scorer_suffix']]),
                                 'suffix_used_by_speed':{band:summarise([x for x in rr if x['speed_band']==band and x['used_in_scorer_suffix']]) for band in ['lt5','5to10','ge10']},
                                 'all_affected_kinematic_intervals':summarise(aa),
                                 'unchanged_speed_successor_intervals':summarise([x for x in aa if not x['numeric_speed_changed']])}
    csv_write(a.out/'changed_neighbor_reports.csv',changed);csv_write(a.out/'affected_kinematic_intervals.csv',affected)
    (a.out/'mechanism_diagnostics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 速度污染与运动学门控：只读机制诊断附录','',
           '本附录只对照已冻结base和speed-stress可见观测，定位实际改变的邻船报告，不修改分数、模型、阈值或任何既有协议。','',
           '## 计数单位与分母','',
           '- 四格中同一b状态的两个a副本只计一次；b=0与b=1具有不同观测速度，分别保留。',
           '- 报告主键为来源窗口、b状态、邻船MMSI、报告时刻。不同来源窗口的同一物理时刻不合并，因为干预计划可能不同。',
           '- 多个suffix target-grid时刻重复使用同一latest报告时，在报告数量中只计一次，另列重复前的grid命中数。',
           '- “原SOG”指追加速度污染前base视图的可见SOG，包含原本群体变化，不是隐藏未扰动真值。',
           '- w比例的分母仅包括可计算相容误差的报告；未知运动学单列，不把未知当作成功识别污染。','',
           '## 实际改变的报告：按base可见SOG分层','',
           '| 划分 | 原SOG kn | 改变报告状态数 | 可评估数 | e_kin中位数m/s | w中位数 | w≥0.9 | w≥0.3 | suffix实际使用报告数 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def num(v):return 'NA' if v is None else f'{v:.4f}'
    for split in ['train','val']:
        for band,lab in [('lt5','<5'),('5to10','5≤SOG<10'),('ge10','≥10')]:
            d=summary['splits'][split]['by_original_speed'][band];q9=d['w_ge_0_9']['fraction'];q3=d['w_ge_0_3']['fraction']
            lines.append(f"| {split} | {lab} | {d['observed_state_reports']} | {d['kinematically_assessable_reports']} | {num(d['new_e_kin_mps']['median'])} | {num(d['new_w']['median'])} | {num(100*q9) if q9 is not None else 'NA'}% | {num(100*q3) if q3 is not None else 'NA'}% | {d['suffix_used_unique_state_reports']} |")
    lines+=['','## 报告是否进入固定后段统计','']
    for split in ['train','val']:
        d=summary['splits'][split]['changed_reports'];s=summary['splits'][split]['unchanged_speed_successor_intervals']
        lines.append(f"- {split}：{d['observed_state_reports']}个改变报告状态，去除b状态后对应{d['unique_source_member_reports']}个来源内物理报告；固定suffix实际使用{d['suffix_used_unique_state_reports']}个状态／{d['suffix_used_unique_source_reports']}个物理报告，重复latest展开前为{d['suffix_grid_hit_count_before_dedup']}个grid命中。")
        lines.append(f"  另有{s['observed_state_reports']}个SOG未改但相容误差受前一报告影响的后继区间，其中{s['suffix_used_unique_state_reports']}个被suffix使用。")
    lines+=['','## 解释范围','',
           f"- 当前w≥0.9对应e≤{summary['thresholds']['w_ge_0_9_equivalent_e_max_mps']:.4f}m/s，w≥0.3对应e≤{summary['thresholds']['w_ge_0_3_equivalent_e_max_mps']:.4f}m/s。倍增速度报错能否越过这些绝对门槛，取决于原速度、因子、首个/连续报错位置及原观测误差。", 
           '- 首个错误速度报告只通过区间两端平均速度的一端进入残差；连续第二个错误报告通常两端都已受污染。因此相同倍数不能直接推断相同w。',
           '- 统计方法最终使用的是成员级sqrt(mean(prefix_w)×mean(suffix_w))；报告级w较低不等于整个成员被删除。此附录同时保存了报告w与最终成员权重，不能把所有性能差异只归因于2.5m/s死区。',
           '- 该诊断只解释既定人工速度污染，不用于重新选择评分规则或支持未经验证的新门控阈值。','']
    (a.out/'速度污染门控机制诊断附录.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'passed','changed_report_states':len(changed),'affected_intervals':len(affected),'val':summary['splits']['val']['changed_reports']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
