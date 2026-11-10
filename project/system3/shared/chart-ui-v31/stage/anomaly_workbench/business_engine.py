"""Operational eligibility comes before neural inference or priority ranking."""
from __future__ import annotations
import math,time
import numpy as np
from .maritime_rules import assess_motion,score_priority

POLICY_VERSION=3


def _take(data,indices):
    n=len(data['raw'])
    return {k:v[indices] if isinstance(v,np.ndarray) and v.ndim and len(v)==n else v for k,v in data.items()}


def _local_context(data,i,meta):
    reliable=data['mask'][i]&(data['w'][i]>=.3)
    frames=meta.get('neighbor_frames',[])
    supported=np.array([bool(f.get('field_valid')) and len(f.get('neighbors',[]))>=3 and
                        all(n.get('distance_nm',99)<=3.5 for n in f.get('neighbors',[])) for f in frames],bool)
    supported &= reliable[:len(frames)]
    indices=np.flatnonzero(supported)
    # A single accidental snapshot does not establish sustained local support.
    available=(len(indices)>=3 and bool(supported[-1])) if len(supported) else False
    residual=float(np.mean(data['rf'][i,indices])) if available else None
    return available,residual,indices


def _relationship(meta,motion,local_residual,indices):
    if not motion.get('is_underway') or local_residual is None or len(indices)<3:return None
    frames=meta['neighbor_frames'];first,last=frames[int(indices[0])],frames[int(indices[-1])]
    if last['time_epoch']-first['time_epoch']<120:return None
    early={n['mmsi']:n for n in first['neighbors']};late={n['mmsi']:n for n in last['neighbors']}
    same=sorted(set(early)&set(late))
    if len(same)<3:return None
    nb_change=float(np.mean([late[k]['sog_kn']-early[k]['sog_kn'] for k in same]))
    own=meta['sog_kn'];own_change=float(own[int(indices[-1])]-own[int(indices[0])])
    if abs(nb_change)>=2 and abs(own_change-nb_change)>=2 and local_residual>=.6:
        strength=min(1.,.5*abs(own_change-nb_change)/6+.5*local_residual/4)
        return {'type':'group_inconsistency','tier':'Tier-III','strength':strength,
                'explanation':f'同一批 {len(same)} 艘 3.5 nm 内同向邻船均速变化 {nb_change:+.2f} kn，目标变化 {own_change:+.2f} kn，可靠局部残差 {local_residual:.2f} m/s。'}
    increases=[(late[k]['distance_nm']-early[k]['distance_nm'])*1852 for k in same]
    if np.median(increases)>300 and local_residual>=1.5:
        strength=min(1.,.5*float(np.median(increases))/1500+.5*local_residual/5)
        return {'type':'formation_departure','tier':'Tier-II','strength':strength,
                'explanation':f'与相同 {len(same)} 艘近距离同向伴行船的间距中位增幅 {float(np.median(increases)):.1f} m，可靠局部残差 {local_residual:.2f} m/s；编队成员身份尚未确认。'}
    return None


def evaluate_business_window(window,far=.01,exclude_stationary=True):
    from . import engine
    from .sqlite_stream import unproject
    started=time.perf_counter();data=engine._prepare(window['arrays']);n=len(data['raw'])
    motions=[assess_motion(m,exclude_stationary) for m in window['metadata']]
    local=[_local_context(data,i,m) for i,m in enumerate(window['metadata'])]
    relationships=[_relationship(m,motions[i],local[i][1],local[i][2]) if local[i][0] else None for i,m in enumerate(window['metadata'])]
    # Stationary and evidence-free vessels never reach the model. With sparse
    # neighbours, even a true event is scored entirely from its own dynamics.
    inference=[i for i,m in enumerate(motions) if not m['is_stationary'] and local[i][0] and (m['eligible'] or relationships[i])]
    raw_scores=np.full(n,np.nan);classes=np.full((n,5),.2)
    if inference:
        probabilities,cls,_=engine._predict(_take(data,np.array(inference)))
        raw_scores[inference]=probabilities;classes[inference]=cls
    records=[];counts={x['id']:0 for x in engine.TYPES}
    for i,meta in enumerate(window['metadata']):
        m=motions[i];available,residual,field_indices=local[i]
        relation=relationships[i]
        if relation and not m['is_stationary']:
            m={**m,'eligible':True,'event_type':relation['type'],'kinematic_strength':relation['strength'],
               'reasons':m['reasons']+[relation['explanation']]}
        raw=float(raw_scores[i]) if np.isfinite(raw_scores[i]) else None
        priority=score_priority(m,raw,available,residual,engine._normal_scores())
        endpoint_neighbors=len(meta['neighbor_frames'][-1]['neighbors']) if meta['neighbor_frames'] else 0
        if not available and endpoint_neighbors>=3 and m['eligible']:
            priority['priority_notice']=f'终点有 {endpoint_neighbors} 艘邻船，但局部场有效性或连续支持不足；优先级完全依据自身动态，未使用模型分或邻船速差。'
        placeholder=np.nan_to_num(raw_scores,nan=0)
        r=engine._record(data,i,meta,placeholder,classes,window['dataset'])
        event=m.get('event_type');kind='stationary' if m['is_stationary'] else (event or 'background')
        label=engine.TYPE_LOOKUP.get(event,{})
        type_label='符合靠泊/锚泊静止特征' if m['is_stationary'] else '疑似'+label['label'] if event else '无充分动态异常证据'
        tier=label.get('tier','背景')
        policy={**m,'stationary':m['is_stationary'],'neighbor_supported':available,
                'valid_local_steps':len(field_indices),'score_source':priority['score_basis'],
                'source_sog_max_kn':m['max_sog_kn'],'source_sog_median_kn':m['median_sog_kn'],
                'priority_notice':priority['priority_notice'],'score_components':priority['score_components'],
                'policy_version':POLICY_VERSION,'neighbor_radius_nm':3.5,'gaussian_sigma_nm':1.5,'scoring_reference_utc':meta['end_utc'],
                'scoring_neighbor_count':endpoint_neighbors}
        valid=data['mask'][i];ek=data['e_kin'][i,valid];finite=ek[np.isfinite(ek)]
        quality=np.isfinite(ek)&(data['w'][i,valid]<.3)
        r.update(type=kind,type_label=type_label,tier=tier,score=priority['score'],priority_score=priority['score'],
                 model_raw_score=raw,model_inference_skipped=i not in inference,
                 is_candidate=priority['is_behavior_candidate'],maritime_policy=policy,policy_version=POLICY_VERSION,
                 score_source=priority['score_basis'],score_components=priority['score_components'],
                 field_residual_mps=residual,quality_steps=np.flatnonzero(quality).tolist(),
                 has_quality_issue=bool(quality.any()),e_kin_mps=float(finite.max()) if len(finite) else None,
                 e_kin_series=ek.tolist(),sog_kn=meta['sog_kn'],cog_deg=meta['cog_deg'],heading_deg=meta['heading_deg'],
                 neighbor_mean_sog=meta['neighbor_mean_sog'],neighbor_sog_low=meta['neighbor_sog_low'],neighbor_sog_high=meta['neighbor_sog_high'],
                 neighbor_tracks=meta['neighbor_tracks'],neighbor_frames=meta['neighbor_frames'],neighbor_track_catalog=meta['neighbor_track_catalog'],
                 neighbor_context_version=2,neighbor_revision=meta['neighbor_revision'],neighbor_rules=meta['neighbor_rules'],
                 field_vectors=meta['field_vectors'],residual_vectors=[v.tolist() if ok else None for v,ok in zip(data['r_vec'][i,valid],data['field_valid'][i,valid])],
                 window_start_utc=meta['window_start_utc'],window_end_utc=meta['window_end_utc'],
                 repaired_latlon=unproject(data['gated'][i,valid,:2]+np.array(meta['origin_position']),meta['projection_origin']).tolist(),
                 motion_observations=meta['motion_observations'],motion_state=m['movement_state'],
                 classification_method='原始全窗运动状态硬门控 + 真实动态事件条件；稀疏场完全退化为自身运动学；局部场仅用于近距离确认',
                 in_training_motion_domain=m.get('is_underway',False),physical_support=m['eligible'],physical_evidence=m['evidence'])
        r['quality_only']=r['has_quality_issue'] and not r['is_candidate']
        r['selection_role']='candidate' if r['is_candidate'] else 'quality' if r['quality_only'] else 'context'
        why=f"MMSI {r['vessel_id']}：原始最大航速 {m['max_sog_kn']:.2f} kn，中位 {m['median_sog_kn']:.2f} kn，可信累计位移 {m['path_length_m']:.1f} m。" if m['max_sog_kn'] is not None else f"MMSI {r['vessel_id']}：原始报告航速不足，未赋予行为候选资格。"
        why+=' '.join(m['reasons'])+' '+priority['priority_notice']
        if not meta['report_fields_complete']:why+='存在缺少有效报告速度或航向的记录；有效原始SOG保持不变，派生字段不用于认定动态事件。'
        if available:why+=f"有效局部流场覆盖 {len(field_indices)} 个实际时间步，确认残差 {residual:.2f} m/s。"
        else:why+='未使用邻船速度差或神经模型输出来提高业务优先级。'
        r['why_flagged']=why
        if r['is_candidate']:counts[event]+=1
        if r['has_quality_issue']:counts['observation_glitch']+=1
        records.append(r)
    good=data['mask']&data['field_valid'];coverage=float(good.sum()/max(1,data['mask'].sum()))
    div=float(data['e_phys'][:,:,2][good].mean()) if good.any() else None
    curl=float(data['e_phys'][:,:,3][good].mean()) if good.any() else None
    congestion=coverage>=.5 and div is not None and div < -2e-5 and window['mean_sog_kn'] is not None and window['mean_sog_kn']<7
    if congestion:counts['congestion']=1
    summary={'vessels_in_window':window['vessels'],'moving_vessels':window['moving_vessels'],'point_count':window['point_count'],
       'mean_sog_kn':window['mean_sog_kn'],'source_sog_missing_count':window.get('source_sog_missing_count',0),'divergence_mean':div,'curl_mean':curl,'field_valid_ratio':coverage,'fallback_rate':1-coverage,
       'candidate_count':sum(r['is_candidate'] for r in records),'quality_count':sum(r['has_quality_issue'] for r in records),
       'stationary_excluded_count':sum(m['is_stationary'] for m in motions),'model_inferred_windows':len(inference),
       'macro_label':'拥堵积压迹象' if congestion else '可信邻船覆盖不足' if coverage<.5 else '流场可核查',
       'inference_ms':(time.perf_counter()-started)*1000}
    cal=engine.calibration(far);cal.update(priority_method='物理事件门控后的平滑核查优先级，非模型原始概率',
       guarantee='FAR仅保留原模型验证参考；业务优先级不是现场误报率保证')
    return engine._json_safe({'records':records,'summary':summary,'type_counts':counts,'calibration':cal,'policy_version':POLICY_VERSION})
