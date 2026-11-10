"""Re-cut raw AIS for a development-only 12->8 query-time forecasting pilot."""
from __future__ import annotations
import argparse, collections, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from features import (project, vectors, compatibility, make_view, Background,
                      VIEWS, SELF_NAMES, FIELD_NAMES, seeded)

REGIONS = ['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai']
SCHEMA = 'forecast_pilot_12x8_v1'

def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def save_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)

def save_npz(path, arrays):
    path=Path(path);tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**arrays);os.replace(tmp,path)

def locate_source(root, source):
    candidate = Path(source)
    if candidate.exists():return candidate
    return root / str(source).replace('\\','/').split('/AIS/',1)[-1]

def natural_maneuver(sog,cog):
    flags=[]
    for sl in (slice(0,12),slice(12,20)):
        speed=sog[sl];course=cog[sl];moving=speed>=2
        turn=bool(moving.sum()>=2 and np.ptp(np.unwrap(np.deg2rad(course[moving])))>=np.deg2rad(30))
        flags.append(turn or float(np.ptp(speed))>=2)
    return any(flags)

def candidates(table, target_ids, bbox, cfg):
    """Physical row numbers and real reports; no interpolation, no injected labels."""
    pools={};counts=collections.Counter();n_hist=12;n_total=20
    target_table=table[table.mmsi.isin(target_ids)]
    for mid, frame in target_table.groupby('mmsi',sort=True):
        rows=frame.sort_values('timestamp',kind='stable').drop_duplicates('timestamp').index.to_numpy()
        rec=table.loc[rows];t=rec.timestamp.to_numpy(float)
        xy=project(rec.lon.to_numpy(),rec.lat.to_numpy(),bbox)
        sog=rec.sog.to_numpy(float);cog=rec.cog.to_numpy(float)
        ok=np.isfinite(xy).all(1)&np.isfinite(t)&np.isfinite(sog)&np.isfinite(cog)
        ok&=(sog>=0)&(sog<102.2)&(cog>=0)&(cog<360)
        ok&=(np.abs(rec.lat.to_numpy())<=90)&(np.abs(rec.lon.to_numpy())<=180)
        # An invalid raw record ends a continuous segment instead of being silently bridged.
        starts=np.r_[True,(np.diff(t)>cfg['max_gap_s'])|(np.diff(t)<=0)|(~ok[:-1])|(~ok[1:])]
        segs=np.cumsum(starts);out=[]
        for seg in np.unique(segs):
            ii=np.flatnonzero((segs==seg)&ok)
            sampled=[];last=-np.inf
            for j in ii:
                if t[j]-last>=cfg['sample_min_s']:
                    sampled.append(j);last=t[j]
            for start in range(0,len(sampled)-n_total+1,n_total):
                jj=np.array(sampled[start:start+n_total]);counts['candidate_windows']+=1
                tt=t[jj];pp=xy[jj];ss=sog[jj]
                if np.diff(tt).max()>cfg['max_gap_s']:
                    counts['sampled_gap_excluded']+=1;continue
                if tt[11]-tt[0]>cfg['max_history_s'] or tt[-1]-tt[11]>cfg['max_future_s']:
                    counts['duration_excluded']+=1;continue
                if np.median(ss[:12])<1 and np.linalg.norm(pp[11]-pp[0])<300:
                    counts['stationary_history_excluded']+=1;continue
                # QC runs on original observations BEFORE any stress injection, including cutoff->future.
                future_speed=np.linalg.norm(np.diff(pp[11:],axis=0),axis=1)/np.diff(tt[11:])
                if np.any(future_speed>50):
                    counts['extreme_future_position_excluded']+=1;continue
                out.append(rows[jj].tolist())
        if out:pools[int(mid)]=out
    return pools,dict(counts)

def balanced_take(pools,cap,key):
    rng=seeded(key);ids=np.array(sorted(pools));rng.shuffle(ids)
    queues={int(m):[pools[int(m)][j] for j in rng.permutation(len(pools[int(m)]))] for m in ids}
    selected=[];depth=0
    while len(selected)<cap:
        added=False
        for mid in ids:
            q=queues[int(mid)]
            if depth<len(q):
                selected.append({'mmsi':int(mid),'rows':q[depth]});added=True
                if len(selected)>=cap:break
        if not added:break
        depth+=1
    return selected

def stack_records(records):
    return {k:np.stack([r[k] for r in records]) for k in records[0]} if records else {}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path('/mnt/sda/WangZijia/AIS'))
    parser.add_argument('--out',type=Path,default=Path('data'))
    parser.add_argument('--regions',nargs='+',choices=REGIONS,default=REGIONS)
    parser.add_argument('--train-per-region',type=int,default=700)
    parser.add_argument('--val-per-region',type=int,default=150)
    parser.add_argument('--max-history-s',type=float,default=1800)
    parser.add_argument('--max-future-s',type=float,default=1200)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.project_root;out=args.out
    out.mkdir(parents=True,exist_ok=True);shards=out/'shards';shards.mkdir(exist_ok=True)
    cfg=dict(schema=SCHEMA,regions=args.regions,history_steps=12,future_steps=8,
             sample_min_s=60,max_gap_s=300,max_history_s=args.max_history_s,
             max_future_s=args.max_future_s,train_per_region=args.train_per_region,
             val_per_region=args.val_per_region,radius_nm=8,sigma_nm=4,heading_deg=45,
             top_k=32,min_neighbor_kn=2,background_age_s=300,seed=20261105,
             stress_target_jump_m=1000,stress_target_drift_m=600,stress_speed_factor=4,
             neighbor_fraction=.5,neighbor_speed_factor=[3,5],neighbor_shift_m=600)
    registry_path=root/'outputs/balanced_global_v2/source_registry.json'
    labels_path=root/'outputs/balanced_global_v2/labeling/labeled_v1/label_manifest.parquet'
    split_path=root/'outputs/balanced_global_v2/release_20261026/global_all_mmsi_split.parquet'
    for p in (registry_path,labels_path,split_path):
        if not p.exists():raise FileNotFoundError(f'Required original server asset: {p}')
    registry=json.loads(registry_path.read_text(encoding='utf-8'))['regions']
    labels=pd.read_parquet(labels_path,columns=['region','split','mmsi']).drop_duplicates()
    ids={s:set(map(int,labels.loc[labels.split==s,'mmsi'])) for s in ['train','val','test']}
    assert not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
    global_split=pd.read_parquet(split_path)
    expected=global_split.set_index('mmsi')['split'].to_dict()
    for s, members in ids.items():
        assert all(expected.get(m)==s for m in members),f'Locked identity mismatch: {s}'
    cfg['source_registry_sha256']=sha(registry_path);cfg['identity_manifest_sha256']=sha(labels_path)
    cfg['global_split_sha256']=sha(split_path)
    cfg['builder_sha256']=sha(__file__);cfg['features_sha256']=sha(Path(__file__).with_name('features.py'))
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    marker=out/'build_config.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['fingerprint']!=fingerprint:raise RuntimeError('Configuration/code changed: use a NEW --out directory.')
        if not args.resume:raise RuntimeError('Output already exists. Use --resume or a new --out.')
    save_json(marker,dict(fingerprint=fingerprint,config=cfg));start_clock=time.time()
    status=lambda **kw:save_json(out/'progress.json',dict(elapsed_s=time.time()-start_clock,**kw))
    print('Stage 1/3: verify source files; select raw observed windows (test identities only, no test windows).',flush=True)
    selected={};audit={};cap_values=[]
    for region in tqdm(args.regions,desc='Source audit / regions',unit='region'):
        source=locate_source(root,registry[region]['canonical_file'])
        digest=sha(source)
        if digest!=registry[region]['sha256']:raise RuntimeError(f'Source hash changed: {source}')
        status(stage='source_selection',region=region)
        table=pd.read_parquet(source,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
        selected[region]={};audit[region]={'source_file':str(source),'source_sha256':digest,'raw_rows':len(table)}
        for split,cap in [('train',args.train_per_region),('val',args.val_per_region)]:
            target_ids=set(map(int,labels.loc[(labels.region==region)&(labels.split==split),'mmsi']))
            pools,counts=candidates(table,target_ids,registry[region]['core_bbox'],cfg)
            items=balanced_take(pools,cap,f'{region}/{split}/{cfg["seed"]}')
            if not items:raise RuntimeError(f'No qualifying {region}/{split} windows; inspect selection criteria.')
            selected[region][split]=items
            audit[region][split]=dict(**counts,eligible_windows=sum(map(len,pools.values())),
                selected_windows=len(items),selected_target_mmsi=len({x['mmsi'] for x in items}),
                requested_cap=cap,eligible_target_mmsi=len(pools))
            if split=='train':
                for item in items:
                    h=table.loc[item['rows'][:12]];t=h.timestamp.to_numpy(float)
                    pp=project(h.lon.to_numpy(),h.lat.to_numpy(),registry[region]['core_bbox'])
                    vv=vectors(h.sog.to_numpy(),h.cog.to_numpy());w,valid,_=compatibility(t,pp,vv)
                    cap_values.extend(np.linalg.norm(vv[valid&(w>=.7)],axis=1).tolist())
        tqdm.write(f'{region}: train {len(selected[region]["train"])} / validation {len(selected[region]["val"])} windows')
        del table
    if not cap_values:raise RuntimeError('No trusted training-history speed values for causal cap.')
    speed_cap=float(np.quantile(cap_values,.995))
    save_json(out/'selection.json',selected);save_json(out/'selection_audit.json',audit)
    print(f'Stage 2/3: causal history features; speed cap from TRAIN histories = {speed_cap:.3f} m/s.',flush=True)
    provenance=[];background_sets={'train':set(),'val':set()};all_background={'train':set(),'val':set()}
    total=sum(len(x['train'])+len(x['val'])*len(VIEWS) for x in selected.values())
    with tqdm(total=total,desc='History features / views',unit='window-view') as bar:
        for region in args.regions:
            source=locate_source(root,registry[region]['canonical_file']);bbox=registry[region]['core_bbox']
            table=pd.read_parquet(source,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
            bg=Background(table,bbox)
            for split in ['train','val']:
                shard=shards/f'{region}_{split}.npz';detail=shards/f'{region}_{split}_provenance.json'
                views=['base'] if split=='train' else VIEWS
                if args.resume and shard.exists() and detail.exists():
                    old=json.loads(detail.read_text());provenance.extend(old)
                    for row in old:
                        background_sets[split].update(row['views']['base']['background_mmsi'])
                        for va in row['views'].values():all_background[split].update(va['background_mmsi'])
                    bar.update(len(selected[region][split])*len(views));continue
                excluded=(ids['val']|ids['test']) if split=='train' else ids['test']
                records=[];details=[]
                for item in selected[region][split]:
                    rr=table.loc[item['rows']];tt=rr.timestamp.to_numpy(float)
                    pp=project(rr.lon.to_numpy(),rr.lat.to_numpy(),bbox)
                    ss=rr.sog.to_numpy(float);cc=rr.cog.to_numpy(float)
                    wid=hashlib.sha256(f'{region}/{item["mmsi"]}/{item["rows"]}/{SCHEMA}'.encode()).hexdigest()[:24]
                    record=dict(future_xy=pp[12:].astype('f8'),query_s=(tt[12:]-tt[11]).astype('f4'),
                        time_s=(tt[:12]-tt[0]).astype('f4'),mmsi=np.int64(item['mmsi']),
                        region=np.array(region),window_id=np.array(wid),maneuver=np.bool_(natural_maneuver(ss,cc)))
                    doc=dict(window_id=wid,region=region,split=split,mmsi=item['mmsi'],
                             row_indices=item['rows'],timestamps=tt.tolist(),source_file=str(source),
                             source_sha256=registry[region]['sha256'],views={})
                    for view in views:
                        feature,va=make_view(tt[:12],pp[:12],ss[:12],cc[:12],item['mmsi'],bg,excluded,wid,view,speed_cap,cfg)
                        record.update({f'{view}__{k}':v for k,v in feature.items()});doc['views'][view]=va
                        all_background[split].update(va['background_mmsi'])
                        if view=='base':background_sets[split].update(va['background_mmsi'])
                        bar.set_postfix(region=region,split=split,view=view);bar.update(1)
                    assert np.array_equal(record['base__raw'],record.get('neighbor_stress__raw',record['base__raw']))
                    records.append(record);details.append(doc)
                    if len(records)%10==0:status(stage='causal_features',region=region,split=split,windows_done=len(records),windows_total=len(selected[region][split]),completed_views=bar.n,total_views=total)
                save_npz(shard,stack_records(records));save_json(detail,details);provenance.extend(details)
            del bg,table
    print('Stage 3/3: merge caches and write provenance, horizon, isolation and context audit.',flush=True)
    dataset_files={};span_summary={};coverage={}
    for split in ['train','val']:
        arrays=[]
        for region in args.regions:
            with np.load(shards/f'{region}_{split}.npz',allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
        merged={k:np.concatenate([x[k] for x in arrays],axis=0) for k in arrays[0]}
        save_npz(out/f'{split}.npz',merged);dataset_files[split]=sha(out/f'{split}.npz')
        span_summary[split]={}
        for region in args.regions:
            use=merged['region']==region
            span_summary[split][region]=dict(windows=int(use.sum()),
                history_seconds_quantiles=np.quantile(merged['time_s'][use,-1],[0,.5,1]).tolist(),
                future_seconds_quantiles=np.quantile(merged['query_s'][use,-1],[0,.5,1]).tolist(),
                natural_maneuver_windows=int(merged['maneuver'][use].sum()))
        coverage[split]={view:{arm:dict(valid_step_fraction=float(merged[f'{view}__{arm}_valid'].mean()),
            affine_step_fraction=float((merged[f'{view}__{arm}'][:,:,-1]==2).mean()))
            for arm in ['f0','f1']} for view in (['base'] if split=='train' else VIEWS)}
    save_json(out/'window_provenance.json',provenance)
    final=dict(schema=SCHEMA,config=cfg,fingerprint=fingerprint,speed_cap_mps=speed_cap,
        speed_cap_sample_count=len(cap_values),npz_sha256=dataset_files,self_features=SELF_NAMES,field_features=FIELD_NAMES,
        official_target_counts={k:len(v) for k,v in ids.items()},target_split_overlap=0,
        base_background_shared_mmsi=len(background_sets['train']&background_sets['val']),
        all_view_background_shared_mmsi=len(all_background['train']&all_background['val']),
        train_background_with_heldout_targets=len(all_background['train']&(ids['val']|ids['test'])),
        val_background_with_test_targets=len(all_background['val']&ids['test']),
        horizon_summary=span_summary,field_coverage=coverage,test_data_constructed=False,
        future_reference='Unmodified raw AIS observations; not surveyed physical ground truth.',
        query_protocol='Actual future report timestamps supplied as query times, no future values in features.',
        background_policy='Train excludes val/test target identities. Val excludes test targets. Other background identities may recur.',
        stress_protocol='Train base only. Validation target jump/drift/speed and neighbor second-half speed+position corruption; unchanged future coordinates.')
    assert final['train_background_with_heldout_targets']==0 and final['val_background_with_test_targets']==0
    save_json(out/'data_manifest.json',final);status(stage='complete',fingerprint=fingerprint)
    print(f'COMPLETE: {out.resolve()} | train.npz, val.npz, data_manifest.json, window_provenance.json',flush=True)

if __name__=='__main__':main()
