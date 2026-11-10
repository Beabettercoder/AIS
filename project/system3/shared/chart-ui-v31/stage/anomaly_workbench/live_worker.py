"""Persistent-per-job CPU worker: SQLite -> physical fields -> PG-DRGN -> JSON."""
from __future__ import annotations
from pathlib import Path
import gzip
import json
import sys
import time
import numpy as np
from . import sqlite_stream as stream
from . import engine


def write(path,value):
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(engine._json_safe(value),f,ensure_ascii=False,allow_nan=False)


def emit(value):print(json.dumps(engine._json_safe(value),ensure_ascii=False,allow_nan=False),flush=True)


def run(request):
    params=request['params'];entry=request.get('dataset') or stream.resolve(params['dataset_id']);folder=Path(request['folder'])
    folder.mkdir(parents=True,exist_ok=True)
    start=stream.epoch(params.get('start_utc') or entry['default_start_utc'])
    end=stream.epoch(params.get('end_utc') or entry['default_end_utc'])
    if params.get('full_period'):start=stream.epoch(entry['start_utc']);end=stream.epoch(entry['end_utc'])+1
    start=max(start,stream.epoch(entry['start_utc']));end=min(end,stream.epoch(entry['end_utc'])+1)
    if end<=start:raise ValueError('该时间范围不与数据集实际 UTC 范围相交')
    minutes=int(params.get('window_minutes',20));step=minutes*60
    bins=int(np.ceil((end-start)/step));timeline=[];scatter=[];all_candidates=[];quality=[];background=[];index={}
    unique=set();quality_vessels=set();stationary_vessels=set();stationary_windows=0;counts={t['id']:0 for t in engine.TYPES};points=0;tick=time.perf_counter()
    for number in range(bins):
        a=start+number*step;b=min(end,a+step)
        emit({'event':'progress','completed':number,'total':bins,'window_start':stream.utc(a),'message':'读取 SQLite 轨迹并计算门控与邻船流场'})
        window=stream.build_window(entry,a,b,neighbor_revision=int(params.get('neighbor_revision',1)))
        row={'index':number,'start_utc':stream.utc(a),'end_utc':stream.utc(b),'vessels':0,'moving_vessels':0,'mean_sog_kn':None,'divergence_mean':None,'curl_mean':None,'candidate_count':0,'quality_count':0,'point_count':0}
        if not window or window.get('empty'):
            if window:
                row.update(point_count=window['point_count'],vessels=window['vessels']);points+=row['point_count'];unique.update(window.get('vessel_ids',[]))
            timeline.append(row);continue
        unique.update(window.get('vessel_ids',[]))
        evaluated=engine.evaluate_window(window,params.get('far_budget',.01),params.get('exclude_stationary',True));records=evaluated['records']
        row.update(evaluated['summary']);row['vessels']=window['vessels'];timeline.append(row);points+=row['point_count']
        for key,value in evaluated['type_counts'].items():counts[key]+=value
        for r in records:
            r['dataset_id']=entry['id'];r['window_index']=number;index[r['id']]=number;unique.add(r['vessel_id'])
            scatter.append({k:r.get(k) for k in ['id','vessel_id','score','model_raw_score','maritime_policy','type','type_label','tier','field_residual_mps','e_kin_mps','mean_gate_weight','is_candidate','has_quality_issue','window_index','start_utc','end_utc']})
            if r['maritime_policy']['stationary']:stationary_vessels.add(r['vessel_id']);stationary_windows+=1
            if r['is_candidate']:all_candidates.append(r)
            if r['has_quality_issue']:quality.append(r);quality_vessels.add(r['vessel_id'])
            if not r['is_candidate'] and len(background)<12:background.append(r)
        write(folder/f'window-{number}.json.gz',{'records':records,'summary':row})
    # Merge actual overlapping/adjacent windows, keeping representative window
    # IDs and the full set of source windows for replay and audit.
    grouped={}
    for r in sorted(all_candidates,key=lambda r:r['start_utc']):
        key=(r['vessel_id'],r['type']);events=grouped.setdefault(key,[])
        previous=events[-1] if events else None
        if previous and stream.epoch(r['start_utc'])-stream.epoch(previous['event_end_utc'])<=120:
            previous['event_end_utc']=max(previous['event_end_utc'],r['end_utc']);previous['event_window_ids'].append(r['id'])
            if r['score']>previous['score']:
                bounds={k:previous[k] for k in ['event_start_utc','event_end_utc','event_window_ids']};previous.update(r);previous.update(bounds)
        else:events.append({**r,'event_start_utc':r['start_utc'],'event_end_utc':r['end_utc'],'event_window_ids':[r['id']]})
    candidates=sorted([e for group in grouped.values() for e in group],key=lambda r:r['score'],reverse=True)
    # Top-K is a vessel review queue. Distinct voyages remain distinct events,
    # while one frequently repeating ferry cannot occupy every queue position.
    vessel_queue={};event_index=[]
    for event in candidates:
        brief={k:event.get(k) for k in ['id','vessel_id','type','type_label','score','event_start_utc','event_end_utc','event_window_ids']}
        event_index.append(brief)
        vessel_queue.setdefault(event['vessel_id'],{**event,'related_events':[]})['related_events'].append(brief)
    ranked_vessels=sorted(vessel_queue.values(),key=lambda r:r['score'],reverse=True)
    for r in ranked_vessels:r['related_event_count']=len(r['related_events'])
    topk=params.get('top_k',20);quality.sort(key=lambda r:r['e_kin_mps'] or 0,reverse=True)
    div=[r['divergence_mean'] for r in timeline if r['divergence_mean'] is not None]
    valid=[r for r in timeline if r['point_count']]
    summary={'unique_mmsi':len(unique),'vessels_in_window':len(unique),'window_count':bins,'point_count':points,
             'candidate_count':len(candidates),'candidate_windows':len(all_candidates),'quality_count':len(quality_vessels),
             'candidate_vessel_count':len(ranked_vessels),
             'stationary_excluded_count':stationary_windows,'stationary_vessels_count':len(stationary_vessels),
             'source_sog_missing_count':sum(r.get('source_sog_missing_count',0) for r in timeline),
             'mean_sog_kn':float(np.average([r['mean_sog_kn'] for r in valid if r['mean_sog_kn'] is not None],weights=[r['point_count'] for r in valid if r['mean_sog_kn'] is not None])) if any(r['mean_sog_kn'] is not None for r in valid) else None,
             'divergence_mean':float(np.mean(div)) if div else None,'field_valid_ratio':float(np.mean([r.get('field_valid_ratio',0) for r in valid])) if valid else 0,
             'inference_ms':(time.perf_counter()-tick)*1000}
    summary['fallback_rate']=1-summary['field_valid_ratio']
    cal=engine.calibration(params.get('far_budget',.01));cal.update(priority_method='全窗原始运动硬门控后的平滑业务优先级',guarantee='预算仅保留原模型验证参考；业务排序分数不是现场误报率保证')
    emit({'event':'progress','completed':bins,'total':bins,'message':'复核真实运动过程、比对既有模式并合并同船连续线索'})
    from .review_batch import refine_saved_windows
    refined=refine_saved_windows(folder,timeline,topk)
    summary.update(refined.pop('summary'))
    result={'params':params,'dataset':entry,'summary':summary,'calibration':cal,
       'policy_version':5,'maritime_policy_version':3,'latest_policy_version':5,'needs_rescan':False,
       'records':ranked_vessels[:topk],'candidate_events':event_index,'quality_records':quality[:max(topk,30)],'context_records':background,
       'timeline':timeline,'scatter':scatter,'type_counts':counts,'record_index':index,
       'notices':['全部轨迹来自当前 system3 SQLite，只读查询；没有离线演示数据。',
          '观测质量疑点单列；分数仅作核查排序。实际数据误报率需要当地正常样本验证。',
          '静止窗口已前置退出行为候选；终点3.5nm内邻船不足3艘时，优先级完全依据自身动态。',
          '模型类别与关系物理判据均为辅助线索，不作违法或编队身份确认。',
          '普通运动变化不直接作为重点核查；优先核查需持续真实动作或独立近邻佐证，历史重复模式单列。']}
    result.update(refined)
    write(folder/'result.json.gz',result);emit({'event':'complete','completed':bins,'total':bins,'summary':summary})


def detail(request):
    params=request['params'];entry=request.get('dataset') or stream.resolve(params['dataset_id'])
    window=stream.build_window(entry,stream.epoch(request['start_utc']),stream.epoch(request['end_utc']),neighbor_revision=int(params.get('neighbor_revision',1)))
    if not window or window.get('empty'):raise ValueError('数据库中已无对应窗口，请重新扫描')
    result=engine.diagnose(window,request['record_id'])
    meta=next(m for m in window['metadata'] if m['window_id']==request['record_id'])
    geometry={key:meta[key] for key in ['neighbor_frames','neighbor_track_catalog','neighbor_context_version','neighbor_revision']}
    from .evidence_snapshot import evidence_signature
    original=engine._json_safe(meta['motion_observations']);frames=engine._json_safe(meta['neighbor_frames'])
    emit({'event':'detail','diagnostics':result,'geometry':geometry,
          'evidence_signature':evidence_signature(original,frames)})


if __name__=='__main__':
    try:
        request=json.load(sys.stdin)
        detail(request) if request.get('action')=='detail' else run(request)
    except Exception as exc:
        emit({'event':'error','message':str(exc),'error_type':type(exc).__name__});sys.exit(1)
