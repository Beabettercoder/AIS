"""Create held-out source quartets under locked protocol; no score selection."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,math
import numpy as np
import pandas as pd

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');OLD=ROOT/'original_method_and_relation_v1_20261026/relation';OUT=ROOT/'relation_test_v1_20261027'
sys.path.insert(0,str(OLD/'vendor'));sys.path.insert(0,str(OLD))
from audit_neighborhood import Source,REGIONS,wrap,sha,NM
from build_relation_smoke import build_one
from expand_relation_data import validate_ranges
import build_speed_stress as stress

def high_select(wid,members):
    return sorted(range(len(members)),key=lambda j:hashlib.sha256(('speed-stress-member:'+wid+':'+str(members[j])).encode()).digest())[:math.ceil(.5*len(members))]

lock=json.loads((OUT/'freeze_lock.json').read_text())
for p,h in lock['files'].items():assert sha(p)==h,(p,sha(p),h)
registry=json.loads((ROOT/'release_20261026/source_registry.json').read_text())['regions']
groups=pd.read_parquet(ROOT/'release_20261026/global_all_mmsi_split.parquet',columns=['mmsi']);targets=pd.read_parquet(ROOT/'release_20261026/selected_windows.parquet',columns=['mmsi','split'])
allids=set(groups.mmsi.astype(str));train=set(targets.loc[targets.split=='train','mmsi'].astype(str));val=set(targets.loc[targets.split=='val','mmsi'].astype(str));test=set(targets.loc[targets.split=='test','mmsi'].astype(str));allowed=allids-train-val
assert not (test & (train|val))
previous=json.loads((OLD/'all_sources_v1/relation_manifest.json').read_text());prior_targets={str(x['mmsi']) for x in previous['samples']};prior_background=set();prior_report=set()
for item in previous['samples']:
    with np.load(OLD/'all_sources_v1'/item['file'],allow_pickle=False) as z:
        ids=z['neighbor_mmsi'];t=z['source_neighbor_t'];prior_background.update(map(str,z['cohort_mmsi']))
        prior_report.update((str(a),float(b)) for a,b in zip(ids,t))
assert not(test & prior_targets)
cases=OUT/'cases';cases.mkdir(exist_ok=False);records=[];rejects=[];regions={};test_bg=set();test_reports=set();source_ranges={}
for region in REGIONS:
    src=Source(registry[region]);p=ROOT/'labeling/labeled_v1/features_scene'/region/'test.npz'
    with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in ('window_id','base_window_id','mmsi','mask','time_s')}
    order=sorted(range(len(a['mmsi'])),key=lambda i:hashlib.sha256(str(a['window_id'][i]).encode()).hexdigest())
    stats=Counter();source_ranges[region]={'source_file':str(p),'source_sha256':sha(p),'windows':len(order)}
    for j in order:
        target=str(a['mmsi'][j]);wid=str(a['window_id'][j]);n=int(a['mask'][j].sum());t=a['time_s'][j,:n]
        assert target in test and target not in (train|val)
        tp,ts,tc=src.source_target(target,t);anchor=max(2,n//2-1);at=float(t[anchor]);ids=src.latest(at,target,allowed);dist=np.linalg.norm(src.p[ids]-tp[anchor],axis=1);angle=np.abs(wrap(src.c[ids]-tc[anchor]))
        ok=src.good[ids]&src.valid[ids]&(src.w[ids]>=.3)&(src.s[ids]>=3)&(dist<=4*NM)&(angle<=30)&(np.abs(src.s[ids]-ts[anchor])<=3)
        cohort=ids[ok];cohort=cohort[np.argsort(src.ids[cohort],kind='stable')]
        prefix_ok=bool(ts[anchor]>=3 and np.median(ts[:anchor+1])>=3 and anchor>=2 and t[-1]-at>=180)
        row=dict(window_id=wid,base_window_id=str(a['base_window_id'][j]),mmsi=target,region=region,split='test',anchor_step=anchor,anchor_time_s=at,source_target_speed_kn=float(ts[anchor]),source_target_course_deg=float(tc[anchor]),source_target_position_m=tp[anchor].tolist(),source_timestamps=t.tolist(),cohort_mmsi=src.ids[cohort].tolist(),cohort_report_time_s=src.t[cohort].tolist(),cohort_count=len(cohort),prefix_ok=prefix_ok,accepted=bool(prefix_ok and len(cohort)>=4))
        stats['candidates']+=1
        if not prefix_ok or len(cohort)<4:
            reason='prefix_ineligible_and_cohort_lt4' if not prefix_ok and len(cohort)<4 else ('prefix_ineligible' if not prefix_ok else 'cohort_lt4')
            rejects.append({'region':region,'window_id':wid,'mmsi':target,'reason':reason,'cohort_count':len(cohort)});stats[reason]+=1;continue
        stats['anchor_eligible']+=1;got,reason=build_one(src,row,allowed)
        if got is None:
            rejects.append({'region':region,'window_id':wid,'mmsi':target,'reason':reason,'cohort_count':len(cohort)});stats[reason]+=1;continue
        arrays,proof=got
        validate_ranges(arrays)
        base=cases/f'{region}_test_{wid}.npz';np.savez_compressed(base,**arrays)
        # Exact prior operators, shared across all quartet siblings.
        original_select=stress.select_members
        old_view,old_proof=stress.stress_one(arrays,wid)
        stress.select_members=high_select
        high_view,high_proof=stress.stress_one(arrays,wid)
        stress.select_members=original_select
        oldfile=cases/f'{region}_test_{wid}_existing.npz';highfile=cases/f'{region}_test_{wid}_high.npz'
        np.savez_compressed(oldfile,**old_view);np.savez_compressed(highfile,**high_view)
        mids=list(map(str,arrays['cohort_mmsi']));test_bg.update(mids);test_reports.update((str(x),float(t)) for x,t in zip(arrays['neighbor_mmsi'],arrays['source_neighbor_t']))
        records.append({'file':base.name,'sha256':sha(base),'existing_file':oldfile.name,'existing_sha256':sha(oldfile),'high_file':highfile.name,'high_sha256':sha(highfile),'window_id':wid,'base_window_id':row['base_window_id'],'region':region,'mmsi':target,'start_s':float(t[0]),'end_s':float(t[-1]),'cohort_mmsi':mids,'old_stress':old_proof,'high_stress':high_proof,'source_proof':proof})
        stats['accepted']+=1
    regions[region]=dict(stats);print('REGION',region,regions[region],flush=True)
assert len(records)+len(rejects)==sum(x['windows'] for x in source_ranges.values())
assert len({x['window_id'] for x in records})==len(records)
test_targets={r['mmsi'] for r in records}; assert not test_targets&prior_targets
result={'status':'completed','protocol_sha256':sha(OUT/'protocol.json'),'lock_sha256':sha(OUT/'freeze_lock.json'),'code_sha256':sha(Path(__file__)),'source_ranges':source_ranges,'regions':regions,'candidates':len(records)+len(rejects),'accepted_sources':len(records),'rejected_sources':len(rejects),'quartet_count':4*len(records),'test_target_mmsi':len(test_targets),'test_target_prior_quartet_target_overlap':len(test_targets&prior_targets),'background_mmsi_test':len(test_bg),'background_mmsi_trainval':len(prior_background),'shared_background_mmsi':len(test_bg&prior_background),'shared_background_source_reports':len(test_reports&prior_report),'test_background_member_reports':len(test_reports),'samples':records,'rejects':rejects}
(OUT/'test_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');print('COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['samples','rejects','source_ranges']}),flush=True)
