"""PG-DRGN CPU inference over dynamically constructed SQLite AIS windows."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json,math,os,time
import numpy as np
BASE=Path(__file__).resolve().parent
BUNDLE=BASE/"model"

TYPES = [
    {"id": "turn_back", "label": "早期折返", "tier": "Tier-I"},
    {"id": "deviation", "label": "航线偏离", "tier": "Tier-I"},
    {"id": "stall", "label": "异常停滞或徘徊", "tier": "Tier-I"},
    {"id": "sinuous", "label": "蛇行或不规则转向", "tier": "Tier-I"},
    {"id": "speed_drop", "label": "突然减速", "tier": "Tier-I"},
    {"id": "formation_departure", "label": "编队脱离", "tier": "Tier-II"},
    {"id": "group_inconsistency", "label": "群体运动不一致", "tier": "Tier-III"},
    {"id": "congestion", "label": "拥堵与航线转移迹象", "tier": "区域"},
    {"id": "observation_glitch", "label": "位置跳变与运动学不一致", "tier": "观测质量"},
]
TYPE_LOOKUP = {t["id"]: t for t in TYPES}


def _json_safe(value):
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return _json_safe(value.tolist())
    if isinstance(value, np.generic): return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


@lru_cache(maxsize=3)
def _model(name="ablation_4_full_cdg"):
    import torch
    if __package__:
        from .vendor import pgdrgn_net as sw
    else:
        from vendor import pgdrgn_net as sw
    torch.set_num_threads(2)
    path = BUNDLE / "vis_weights" / (name + ".pt")
    if not path.exists(): raise ValueError(f"缺少模型权重：{path.name}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = sw.Net(ck["kind"], ck["nf"], use_graph=ck.get("use_graph", False)).cpu()
    model.load_state_dict(ck["sd"]); model.eval()
    return model, ck, sw


def _prepare(arrays: dict) -> dict:
    raw = np.nan_to_num(np.asarray(arrays["raw"], np.float32))
    n, steps = raw.shape[:2]
    w = np.nan_to_num(np.asarray(arrays.get("w_self", arrays.get("w")), np.float32))
    if w.shape != (n, steps): raise ValueError("w_self 必须与窗口和时间步一致")
    w = np.clip(w, 0, 1)
    gated = raw.copy()
    gated[:, :, :2] = np.cumsum(np.diff(raw[:, :, :2], axis=1, prepend=0.0) * w[..., None], axis=1)
    for ch in range(2, 7):
        for t in range(1, steps):
            low = w[:, t] < 0.2
            gated[low, t, ch] = gated[low, t - 1, ch]
    result = dict(arrays)
    result.update(raw=raw, gated=gated, w=w,
                  rf=np.nan_to_num(np.asarray(arrays.get("r_field", np.zeros((n, steps))), np.float32)),
                  mask=np.asarray(arrays.get("mask", np.ones((n, steps))), bool),
                  nb_seq=np.nan_to_num(np.asarray(arrays.get("nb_seq", np.zeros((n, 32, 5, 4))), np.float32)),
                  nb_xy=np.nan_to_num(np.asarray(arrays.get("nb_xy", np.zeros((n, 32, 2))), np.float32)),
                  nb_v=np.nan_to_num(np.asarray(arrays.get("nb_v", np.zeros((n, 32, 2))), np.float32)),
                  nb_w=np.nan_to_num(np.asarray(arrays.get("nb_w", np.ones((n, 32))), np.float32)),
                  nb_mask=np.asarray(arrays.get("nb_mask", np.zeros((n, 32))), bool))
    return result


def _predict(data: dict, name="ablation_4_full_cdg"):
    import torch
    model, ck, sw = _model(name)
    x = sw.build(data, ck["cfg"]["use_kg"], ck["cfg"]["use_lf"])
    mu, sd = ck["stats"]
    with torch.inference_mode():
        x = torch.as_tensor(np.asarray((x - mu) / sd, np.float32))
        args = ()
        if ck.get("use_graph"):
            args = (torch.as_tensor(data["nb_seq"]), torch.as_tensor(data["nb_xy"]),
                    torch.as_tensor(data["nb_mask"]), torch.as_tensor(data["nb_w"]),
                    torch.as_tensor(data["rf"][np.arange(len(data['raw'])),np.maximum(0,data['mask'].sum(1)-1)]))
        logits, cls, features = model(x, *args)
        return torch.sigmoid(logits).numpy(), torch.softmax(cls, -1).numpy(), features.numpy()


def _field(data: dict, i: int):
    valid_indices = np.flatnonzero(data["mask"][i])
    if "field_valid" in data and (not len(valid_indices) or not data["field_valid"][i, valid_indices[-1]]):
        return None
    m = data["nb_mask"][i] & (data["nb_w"][i] > 0.3)
    xy, velocity = data["nb_xy"][i][m], data["nb_v"][i][m]
    if len(xy) < 4: return None
    radius = 8 * 1852.
    x = np.column_stack([np.ones(len(xy)), xy / radius])
    theta, _, rank, _ = np.linalg.lstsq(x, velocity, rcond=None)
    if rank < 3: return None
    return {"vector": theta[0], "div": float((theta[1, 0] + theta[2, 1]) / radius),
            "curl": float((theta[1, 1] - theta[2, 0]) / radius)}


def _physical_support(data: dict, i: int):
    good = data["mask"][i] & (data["w"][i] >= 0.3)
    indices = np.flatnonzero(good)
    if len(indices) < 4: return False, {"reliable_steps": len(indices)}
    speeds = data["raw"][i, indices, 4]
    # Training clean-pool observations were selected at SOG >= 2 kn. A vessel
    # that never reaches that speed on reliable observations is out of coverage;
    # stationary COG noise must not become an operational behaviour alert.
    if not np.any(speeds >= 2.0):
        return False, {"reliable_steps": len(indices), "max_reliable_speed_kn": float(speeds.max()),
                       "in_training_motion_domain": False, "reason": "低速/停泊超出训练底图覆盖"}
    angles = np.unwrap(np.arctan2(data["raw"][i, indices, 5], data["raw"][i, indices, 6]))
    residual = data["rf"][i, good & _field_step_mask(data, i)]
    mean_residual = float(np.mean(residual)) if len(residual) else 0.
    support = mean_residual >= 0.5 or np.ptp(speeds) >= 2 or np.ptp(angles) >= 0.45
    return bool(support), {"reliable_steps": len(indices), "speed_range_kn": float(np.ptp(speeds)),
                           "heading_range_deg": float(np.degrees(np.ptp(angles))),
                           "reliable_mean_residual_mps": mean_residual,
                           "field_reliable_steps": len(residual), "in_training_motion_domain": True}


def _record(data: dict, i: int, metadata: dict, scores, cls, dataset: dict) -> dict:
    good = data["mask"][i]
    valid_indices = np.flatnonzero(good)
    if not len(valid_indices): valid_indices = np.array([0])
    quality = good & (data["w"][i] < 0.3)
    reliable = good & (data["w"][i] >= 0.3)
    speeds = data["raw"][i, valid_indices, 4]
    field_reliable = reliable & _field_step_mask(data, i)
    residual = float(np.mean(data["rf"][i, field_reliable])) if field_reliable.any() else None
    cls_index = int(np.argmax(cls[i]))
    type_id = TYPES[cls_index]["id"]
    nbmask = data["nb_mask"][i]
    nbv = data["nb_v"][i][nbmask]
    nbspeed = float(np.mean(np.linalg.norm(nbv, axis=-1))) / 0.514444 if len(nbv) else None
    own_mean = float(np.mean(speeds))
    # The five-class head cannot identify relationship types. Such indications
    # are expressly labelled physical heuristics, not a trained nine-class head.
    relation_note = None
    if nbspeed is not None and len(nbv) >= 4:
        if residual is not None and abs(own_mean - nbspeed) > max(2., 0.45 * nbspeed) and residual > 0.6:
            type_id = "group_inconsistency"
            relation_note = "邻船速度差与可靠段流场残差启发式；不是模型训练的独立关系类别"
        elif residual is not None and residual > 1.5:
            type_id = "formation_departure"
            relation_note = "高可靠段流场残差启发式；未提供编队成员名单，不构成编队脱离确认"
    label = TYPE_LOOKUP[type_id]
    t = np.asarray(data.get("t", np.tile(np.arange(20), (len(data["raw"]), 1))))[i]
    time_values = [float(x) for x in t[valid_indices]]
    field = _field(data, i)
    last = int(valid_indices[-1])
    residual_vector = data.get("r_vec")
    rv = residual_vector[i, last] if residual_vector is not None and _field_step_mask(data, i)[last] else None
    fv = data["raw"][i, last, 2:4] - rv if rv is not None else (field["vector"] if field else None)
    widow_id = str(metadata.get("window_id", f"{dataset['id']}-WIN-{i:05d}"))
    residual_text = f"可靠段流场残差 {residual:.2f} m/s" if residual is not None else "可靠段流场残差不可用"
    why = f"窗口有效段平均航速 {own_mean:.2f} kn，{residual_text}，模型排序分数 {float(scores[i]):.3f}。"
    if nbspeed is not None: why += f"末时刻 {len(nbv)} 个邻船观测的平均速度为 {nbspeed:.2f} kn。"
    else: why += "该窗口缺少可用邻船观测，不能给出邻船对照结论。"
    if quality.any(): why += f"另有 {int(quality.sum())} 步可信度低于 0.3，单列为观测质量疑点。"
    support, physical = _physical_support(data, i)
    motion_domain = bool(reliable.any() and np.any(data["raw"][i, reliable, 4] >= 2.))
    return _json_safe({"id": widow_id, "vessel_id": str(metadata.get("mmsi") or widow_id),
            "window_id": widow_id, "source_index": metadata.get("source_index", i),
            "start_utc": metadata.get("start_utc"), "end_utc": metadata.get("end_utc"),
            "type": type_id, "type_label": ("疑似" + label["label"]) if relation_note else label["label"], "tier": label["tier"],
            "score": float(scores[i]), "field_residual_mps": residual,
            "mean_gate_weight": float(data["w"][i, good].mean()) if good.any() else 0,
            "why_flagged": why, "quality_only": False, "has_quality_issue": bool(quality.any()),
            "quality_steps": np.flatnonzero(quality), "raw_xy": data["raw"][i, valid_indices, :2],
            "repaired_xy": data["gated"][i, valid_indices, :2], "time_seconds": time_values,
            "gate_weights": data["w"][i, valid_indices],
            "neighbors": [{"xy": xy, "velocity": v} for xy, v in zip(data["nb_xy"][i][nbmask], nbv)],
            "field_vector": fv, "residual_vector": rv, "latlon": metadata.get("latlon"),
            "physical_support": support, "physical_evidence": physical,
            "in_training_motion_domain": motion_domain,
            "domain_status": "包含可靠在航步骤，已通过基础适用域筛选" if motion_domain else "低速/停泊或无可靠在航步骤，超出训练底图覆盖",
            "classification_method": relation_note or "五类单船行为分类头；行为类别为候选解释，不是业务定性",
            "diagnostics": {"pg_probability": [], "raw_probability": [],
                            "gate_weights": data["w"][i, valid_indices], "time_seconds": time_values,
                            "method": "选择记录后按真实权重计算前缀重放", "notices": []}})


def _field_step_mask(data,i):
    return data['mask'][i]&np.asarray(data.get('field_valid',np.ones_like(data['mask']))[i],bool)


@lru_cache(maxsize=1)
def _normal_scores():
    return np.sort(np.asarray(json.loads((BUNDLE/'normal_calibration.json').read_text(encoding='utf-8'))['scores']))


def calibration(far):
    scores=_normal_scores();n=len(scores);allowed=int(n*far);threshold=float(scores[max(0,n-allowed-1)])
    return {'threshold':threshold,'far_budget':far,'normal_samples':n,
            'empirical_validation_far':float(np.mean(scores>threshold)),
            'source':'独立正常验证集的已固化模型校准参数',
            'guarantee':'源域验证工作点，不承诺当前入库数据实际误报率'}


def _evaluate_legacy_window(window,far=.01):
    """All inputs here originate from a fresh SQLite query, never a demo file."""
    from .sqlite_stream import unproject
    data=_prepare(window['arrays']);scores,classes,_=_predict(data);cal=calibration(far)
    records=[];counts={t['id']:0 for t in TYPES}
    for i,meta in enumerate(window['metadata']):
        r=_record(data,i,meta,scores,classes,window['dataset'])
        # Five-class trained head; relationship labels need concrete comparative
        # motion evidence and are never inferred from residual magnitude alone.
        k=int(np.argmax(classes[i]));r['type']=TYPES[k]['id'];r['type_label']='疑似'+TYPES[k]['label'];r['tier']='Tier-I'
        own=np.array(meta['sog_kn']);nb=np.array([np.nan if v is None else v for v in meta['neighbor_mean_sog']])
        sufficient=np.isfinite(nb)
        if sufficient.sum()>=4:
            nb_change=float(np.nanmean(nb[-3:])-np.nanmean(nb[:3])) if np.isfinite(nb[:3]).any() and np.isfinite(nb[-3:]).any() else 0
            own_change=float(own[-3:].mean()-own[:3].mean())
            if abs(nb_change)>=2 and abs(own_change-nb_change)>=2 and r['field_residual_mps'] is not None and r['field_residual_mps']>.6:
                r.update(type='group_inconsistency',type_label='疑似群体运动不一致',tier='Tier-III',
                         classification_method='同向邻船均速变化与目标速度变化差异的物理判据')
            elif len(meta['neighbor_tracks'])>=4 and r['field_residual_mps'] is not None and r['field_residual_mps']>1.5:
                # A change in separation relative to the same observed neighbours
                # is required; no supplied formation membership means suspicion.
                distances=[]
                from .sqlite_stream import EARTH
                for nn in meta['neighbor_tracks']:
                    if len(nn['latlon'])<2:continue
                    a,b=np.array(nn['latlon'][0]),np.array(nn['latlon'][-1]);ownll=np.array(meta['latlon'])
                    factor=np.array([1.,math.cos(math.radians(ownll[0,0]))])*math.pi*EARTH/180
                    distances.append(float(np.linalg.norm((b-ownll[-1])*factor)-np.linalg.norm((a-ownll[0])*factor)))
                if len(distances)>=4 and np.median(distances)>300:
                    r.update(type='formation_departure',type_label='疑似编队脱离',tier='Tier-II',
                             classification_method='与相同同向邻船的间距扩大超过300米，结合可靠段流场残差；编队成员身份仍需人工确认')
        valid=data['mask'][i];ekin=data['e_kin'][i,valid];finite_ekin=ekin[np.isfinite(ekin)]
        quality=np.isfinite(ekin)&(data['w'][i,valid]<.3)
        r['quality_steps']=np.flatnonzero(quality).tolist();r['has_quality_issue']=bool(quality.any())
        peak=float(np.max(finite_ekin)) if len(finite_ekin) else None
        r.update(e_kin_mps=peak,e_kin_series=ekin.tolist(),sog_kn=meta['sog_kn'],cog_deg=meta['cog_deg'],
                 heading_deg=meta['heading_deg'],neighbor_mean_sog=meta['neighbor_mean_sog'],
                 neighbor_sog_low=meta['neighbor_sog_low'],neighbor_sog_high=meta['neighbor_sog_high'],
                 neighbor_tracks=meta['neighbor_tracks'],field_vectors=meta['field_vectors'],
                 neighbor_frames=meta['neighbor_frames'],neighbor_track_catalog=meta['neighbor_track_catalog'],
                 neighbor_context_version=2,neighbor_revision=meta['neighbor_revision'],
                 residual_vectors=[v.tolist() if ok else None for v,ok in zip(data['r_vec'][i,valid],data['field_valid'][i,valid])],
                 window_start_utc=meta['window_start_utc'],window_end_utc=meta['window_end_utc'],
                 repaired_latlon=unproject(data['gated'][i,valid,:2]+np.array(meta['origin_position']),meta['projection_origin']).tolist())
        eligible=(meta['valid_report_steps']>=8 and r['physical_support'] and r['in_training_motion_domain'])
        r['is_candidate']=bool(scores[i]>cal['threshold'] and eligible)
        r['quality_only']=r['has_quality_issue'] and not r['is_candidate']
        r['selection_role']='candidate' if r['is_candidate'] else 'quality' if r['quality_only'] else 'context'
        if r['is_candidate']:counts[r['type']]+=1
        if r['has_quality_issue']:counts['observation_glitch']+=1
        ownmin=float(np.min(own));comparable=float(np.nanmean(nb)) if sufficient.any() else None
        r['why_flagged']=f"MMSI {r['vessel_id']} 在 {r['start_utc']}—{r['end_utc']} 的最低航速为 {ownmin:.2f} kn"
        if comparable is not None:r['why_flagged']+=f"，可用同向邻船均速 {comparable:.2f} kn，末步可比邻船 {len(meta['neighbor_tracks'])} 艘"
        else:r['why_flagged']+='；可信同向邻船不足，不能用流场证据确认个体偏离'
        r['why_flagged']+=(f"。运动学残差峰值 {peak:.2f} m/s。" if peak is not None else '。报告速度/航向不足，运动学残差不可核验，不当作定位跳变。')
        if not meta['report_fields_complete']:r['why_flagged']+=f"其中 {len(own)-meta['valid_report_steps']} 步缺少有效报告速度或航向，已单独降低可信度，不计入定位跳变数量。"
        r['classification_method']+='；单船类别来自五类模型头，关系类别采用透明物理条件，均需人工确认'
        if len(own)<8:r['domain_status']='有效报文不足8条，仅作轨迹和质量核查'
        records.append(r)
    good=data['mask']&data['field_valid'];div=float(data['e_phys'][:,:,2][good].mean()) if good.any() else None
    curl=float(data['e_phys'][:,:,3][good].mean()) if good.any() else None
    coverage=float(good.sum()/max(1,data['mask'].sum()))
    congestion=coverage>=.5 and div is not None and div < -2e-5 and window['mean_sog_kn']<7
    if congestion:counts['congestion']=1
    summary={'vessels_in_window':window['vessels'],'moving_vessels':window['moving_vessels'],
        'mean_sog_kn':window['mean_sog_kn'],'divergence_mean':div,'curl_mean':curl,
        'field_valid_ratio':float(good.sum()/max(1,data['mask'].sum())),
        'fallback_rate':float(1-good.sum()/max(1,data['mask'].sum())),
        'point_count':window['point_count'],'candidate_count':sum(r['is_candidate'] for r in records),
        'quality_count':sum(r['has_quality_issue'] for r in records),'macro_label':'拥堵积压迹象' if congestion else '可信邻船覆盖不足' if coverage<.5 else '流场可核查'}
    return _json_safe({'records':records,'summary':summary,'type_counts':counts,'calibration':cal})


def evaluate_window(window,far=.01,exclude_stationary=True):
    from .business_engine import evaluate_business_window
    return evaluate_business_window(window,far,exclude_stationary)


def diagnose(window,record_id):
    data=_prepare(window['arrays']);idx=next((i for i,m in enumerate(window['metadata']) if m['window_id']==record_id),None)
    if idx is None:raise ValueError('当前数据库窗口中找不到该船舶记录')
    one={k:v[idx:idx+1].copy() if isinstance(v,np.ndarray) and v.ndim and len(v)==len(data['raw']) else v for k,v in data.items()}
    pg=[];baseline=[];size=int(one['mask'].sum())
    # Causal target prefixes, fixed final neighbour snapshot: label as a
    # retrospective diagnostic rather than a strict real-time guarantee.
    for step in range(size):
        p={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in one.items()}
        for key in ['raw','gated','w','rf']:
            if p[key].ndim>=2:p[key][:,step+1:]=p[key][:,step:step+1]
        pg.append(float(_predict(p)[0][0]));baseline.append(float(_predict(p,'ablation_0_raw_tcn')[0][0]))
    return {'pg_probability':pg,'raw_probability':baseline,'gate_weights':one['w'][0,:size].tolist(),
            'e_kin_series':one['e_kin'][0,:size].tolist(),'time_seconds':one['t'][0,:size].tolist(),
            'method':'当前 SQLite 船舶窗口的逐前缀模型诊断；邻船采用窗口末快照，非严格在线提前预警概率'}


def analyze_maritime_situation(ais_window_data,region_name='SQLite AIS',far_budget=.01):
    data=_prepare(ais_window_data);scores,cls,_=_predict(data)
    return _json_safe({'region_name':region_name,'scores':scores,'classes':cls,'calibration':calibration(far_budget)})

