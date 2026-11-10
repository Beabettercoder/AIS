"""Raw AIS forecasting data: configurable history, nested quotas, fixed MMSI split.

No model is trained here. Test feature construction requires a checkpoint-freeze
record supplied by the orchestrator. Inventory inspects availability, never scores.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frozen_pilot'))
import features as F
sys.path.insert(0,str(ROOT))
from reanchor import observed_history,reanchor_positions

REGIONS=['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai']
SCHEMA='forecast_configurable_history_v2_20261106'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
def save_npz(path,arrays):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,path)

def configuration(pilot,hist,future):
    if int(hist)<4 or int(future)<2:raise ValueError('history must be >=4, future >=2')
    p=read(Path(pilot)/'data/data_manifest.json')
    cfg=dict(p['config']);cfg.update(schema=SCHEMA,history_steps=int(hist),future_steps=int(future),
        max_history_s=150*int(hist),max_future_s=1200.,seed=20261105,
        threshold=.7,cap_rule='Frozen original 700-per-region training-history speed quantile',
        speed_cap_mps=p['speed_cap_mps'])
    # Quotas do not influence pool qualification or random ordering.
    cfg.pop('train_per_region',None);cfg.pop('val_per_region',None)
    return cfg,p

def metadata(project,pilot,hist,future):
    project=Path(project);pilot=Path(pilot);cfg,parent=configuration(pilot,hist,future)
    paths={
        'source_registry_sha256':project/'outputs/balanced_global_v2/source_registry.json',
        'identity_manifest_sha256':project/'outputs/balanced_global_v2/labeling/labeled_v1/label_manifest.parquet',
        'global_split_sha256':project/'outputs/balanced_global_v2/release_20261026/global_all_mmsi_split.parquet'}
    for key,path in paths.items():
        if sha(path)!=parent['config'][key]:raise RuntimeError(f'Frozen source/split registry changed: {path}')
    labels=pd.read_parquet(paths['identity_manifest_sha256'],columns=['region','split','mmsi']).drop_duplicates()
    ids={s:set(map(int,labels.loc[labels.split==s,'mmsi'])) for s in ['train','val','test']}
    assert not (ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
    global_split=pd.read_parquet(paths['global_split_sha256']).set_index('mmsi')['split'].to_dict()
    assert all(global_split.get(mid)==s for s,group in ids.items() for mid in group)
    return cfg,parent,read(paths['source_registry_sha256'])['regions'],labels,ids

def locate(project,source):
    p=Path(source)
    return p if p.exists() else Path(project)/str(source).replace('\\','/').split('/AIS/',1)[-1]

def candidates(table,target_ids,bbox,cfg):
    hist=cfg['history_steps'];future=cfg['future_steps'];total=hist+future
    pools={};counts=collections.Counter()
    subset=table[table.mmsi.isin(target_ids)]
    for mid,frame in subset.groupby('mmsi',sort=True):
        rows=frame.sort_values('timestamp',kind='stable').drop_duplicates('timestamp').index.to_numpy()
        r=table.loc[rows];t=r.timestamp.to_numpy(float);xy=F.project(r.lon.to_numpy(),r.lat.to_numpy(),bbox)
        sog=r.sog.to_numpy(float);cog=r.cog.to_numpy(float)
        ok=np.isfinite(xy).all(1)&np.isfinite(t)&np.isfinite(sog)&np.isfinite(cog)
        ok&=(sog>=0)&(sog<102.2)&(cog>=0)&(cog<360)
        ok&=(np.abs(r.lon.to_numpy())<=180)&(np.abs(r.lat.to_numpy())<=90)
        starts=np.r_[True,(np.diff(t)>cfg['max_gap_s'])|(np.diff(t)<=0)|(~ok[:-1])|(~ok[1:])]
        segs=np.cumsum(starts);accepted=[]
        for seg in np.unique(segs):
            ii=np.flatnonzero((segs==seg)&ok);sampled=[];last=-np.inf
            for j in ii:
                if t[j]-last>=cfg['sample_min_s']:sampled.append(j);last=t[j]
            for start in range(0,len(sampled)-total+1,total):
                jj=np.array(sampled[start:start+total]);counts['candidate_windows']+=1
                tt=t[jj];pp=xy[jj];ss=sog[jj]
                if np.diff(tt).max()>cfg['max_gap_s']:counts['sampled_gap_excluded']+=1;continue
                if tt[hist-1]-tt[0]>cfg['max_history_s'] or tt[-1]-tt[hist-1]>cfg['max_future_s']:
                    counts['duration_excluded']+=1;continue
                if np.median(ss[:hist])<1 and np.linalg.norm(pp[hist-1]-pp[0])<300:
                    counts['stationary_history_excluded']+=1;continue
                vpos=np.linalg.norm(np.diff(pp[hist-1:],axis=0),axis=1)/np.diff(tt[hist-1:])
                if np.any(vpos>50):counts['extreme_future_position_excluded']+=1;continue
                accepted.append(rows[jj].tolist())
        if accepted:pools[int(mid)]=accepted
    return pools,dict(counts)

def balanced_take(pools,quota,key):
    if int(quota)<1:raise ValueError('Quota must be a positive integer')
    rng=F.seeded(key);ids=np.array(sorted(pools));rng.shuffle(ids)
    queues={int(mid):[pools[int(mid)][j] for j in rng.permutation(len(pools[int(mid)]))] for mid in ids}
    out=[];depth=0
    while len(out)<quota:
        added=False
        for mid in ids:
            queue=queues[int(mid)]
            if depth<len(queue):
                out.append(dict(mmsi=int(mid),rows=queue[depth]));added=True
                if len(out)>=quota:break
        if not added:break
        depth+=1
    return out

def natural_maneuver(sog,cog,hist):
    for sl in [slice(0,hist),slice(hist,None)]:
        s=sog[sl];c=cog[sl];moving=s>=2
        if len(s) and np.ptp(s)>=2:return True
        if moving.sum()>=2 and np.ptp(np.unwrap(np.deg2rad(c[moving])))>=np.deg2rad(30):return True
    return False

def inventory(project,pilot,hist=12,future=8):
    project=Path(project);pilot=Path(pilot)
    cfg,parent,registry,labels,ids=metadata(project,pilot,hist,future)
    result=dict(config=cfg,regions={},official_targets={s:len(x) for s,x in ids.items()},
                target_mmsi_overlap=0,model_scoring_performed=False,training_performed=False)
    eligible_global={s:set() for s in ids}
    for region in tqdm(REGIONS,desc=f'Inventory H={hist} F={future}',unit='region'):
        source=locate(project,registry[region]['canonical_file']);assert sha(source)==registry[region]['sha256']
        table=pd.read_parquet(source,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
        report=dict(source_file=str(source),source_sha256=registry[region]['sha256'],raw_rows=len(table))
        for split in ['train','val','test']:
            targets=set(map(int,labels.loc[(labels.region==region)&(labels.split==split),'mmsi']))
            pools,counts=candidates(table,targets,registry[region]['core_bbox'],cfg)
            n=sum(map(len,pools.values()));eligible_global[split].update(pools)
            report[split]=dict(**counts,eligible_windows=n,eligible_target_mmsi=len(pools),
                selected_at_700=min(700,n),selected_at_2000=min(2000,n),selected_at_150=min(150,n))
        result['regions'][region]=report
    result['totals']={s:dict(eligible_windows=sum(r[s]['eligible_windows'] for r in result['regions'].values()),
        unique_eligible_target_mmsi=len(eligible_global[s]),
        quota_700_total=sum(r[s]['selected_at_700'] for r in result['regions'].values()),
        quota_2000_total=sum(r[s]['selected_at_2000'] for r in result['regions'].values())) for s in ids}
    return result

def build(project,pilot,out,split,quota,hist=12,future=8,freeze_path=None):
    project=Path(project);pilot=Path(pilot);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if split not in ['train','val','test']:raise ValueError('Unknown split')
    freeze_hash=None
    if split=='test':
        if freeze_path is None or not Path(freeze_path).is_file():raise RuntimeError('Test build requires the orchestrator checkpoint-freeze record')
        freeze_hash=sha(freeze_path)
    cfg,parent,registry,labels,ids=metadata(project,pilot,hist,future)
    identity=dict(schema=SCHEMA,config=cfg,split=split,quota=int(quota),
        parent_manifest_sha256=sha(pilot/'data/data_manifest.json'),data_code_sha256=sha(__file__),
        features_sha256=sha(ROOT/'frozen_pilot/features.py'),reanchor_sha256=sha(ROOT/'reanchor.py'),
        freeze_sha256=freeze_hash)
    manifest_path=out/f'{split}_manifest.json';path=out/f'{split}.npz'
    if manifest_path.exists():
        old=read(manifest_path)
        if old['identity']!=identity:raise RuntimeError(f'Data configuration changed: use new out directory ({out})')
        assert sha(path)==old['npz_sha256'];return old
    records=[];provenance=[];regional={};background_ids=set();views=['base'] if split=='train' else F.VIEWS
    for region in tqdm(REGIONS,desc=f'Build {split} H{hist} quota{quota}',unit='region'):
        source=locate(project,registry[region]['canonical_file']);assert sha(source)==registry[region]['sha256']
        table=pd.read_parquet(source,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
        targets=set(map(int,labels.loc[(labels.region==region)&(labels.split==split),'mmsi']))
        pools,counts=candidates(table,targets,registry[region]['core_bbox'],cfg)
        selected=balanced_take(pools,int(quota),f'{region}/{split}/{cfg["seed"]}')
        if not selected:raise RuntimeError(f'No qualified {region}/{split} windows at H={hist}. Do not silently duplicate rows.')
        regional[region]=dict(**counts,eligible_windows=sum(map(len,pools.values())),eligible_target_mmsi=len(pools),
            selected_windows=len(selected),selected_target_mmsi=len({r['mmsi'] for r in selected}),requested_quota=int(quota))
        bg=F.Background(table,registry[region]['core_bbox']);excluded=ids['val']|ids['test'] if split=='train' else ids['test'] if split=='val' else set()
        for item in tqdm(selected,desc=f'{region}/{split}',unit='window',leave=False):
            row=table.iloc[item['rows']];t=row.timestamp.to_numpy(float);p=F.project(row.lon.to_numpy(),row.lat.to_numpy(),registry[region]['core_bbox'])
            sog=row.sog.to_numpy(float);cog=row.cog.to_numpy(float)
            # Retain original IDs at the original H/F setting, enabling byte-exact 700-window checks.
            id_schema='forecast_pilot_12x8_v1' if (hist,future)==(12,8) else f'{SCHEMA}_h{hist}_f{future}'
            wid=hashlib.sha256(f'{region}/{item["mmsi"]}/{item["rows"]}/{id_schema}'.encode()).hexdigest()[:24]
            record=dict(future_xy=p[hist:].astype('f8'),query_s=(t[hist:]-t[hist-1]).astype('f4'),time_s=(t[:hist]-t[0]).astype('f4'),
                mmsi=np.int64(item['mmsi']),region=np.array(region),window_id=np.array(wid),maneuver=np.bool_(natural_maneuver(sog,cog,hist)))
            doc=dict(window_id=wid,region=region,split=split,mmsi=item['mmsi'],row_indices=item['rows'],timestamps=t.tolist(),
                source_file=str(source),source_sha256=registry[region]['sha256'],views={})
            for view in views:
                x,audit=F.make_view(t[:hist],p[:hist],sog[:hist],cog[:hist],item['mmsi'],bg,excluded,wid,view,cfg['speed_cap_mps'],cfg)
                visible,v=observed_history(F,t[:hist],p[:hist],sog[:hist],cog[:hist],wid,view)
                w,valid,_=F.compatibility(t[:hist],visible,v);gp,gv,_=F.causal_gate(t[:hist],visible,v,w,valid,cfg['speed_cap_mps'])
                cp,reset=reanchor_positions(t[:hist],visible,gp,w,valid,.7)
                x['gated'][:,:2]=cp.astype('f4');x['anchor_gate']=visible[0]+cp[-1];x['reanchor_reset']=reset
                record.update({view+'__'+k:v for k,v in x.items()});doc['views'][view]=audit
                background_ids.update(audit['background_mmsi'])
            if split!='train':
                np.testing.assert_array_equal(record['base__gated'],record['neighbor_stress__gated'])
                np.testing.assert_array_equal(record['base__anchor_gate'],record['neighbor_stress__anchor_gate'])
            records.append(record);provenance.append(doc)
        save(out/'build_progress.json',dict(split=split,region=region,completed_windows=len(records),history_steps=hist))
    arrays={k:np.stack([r[k] for r in records]) for k in records[0]}
    assert len(np.unique(arrays['window_id']))==len(records)
    assert set(map(int,arrays['mmsi'])).issubset(ids[split])
    save_npz(path,arrays);save(out/f'{split}_provenance.json',provenance)
    report=dict(identity=identity,npz_sha256=sha(path),windows=len(records),unique_target_mmsi=len(np.unique(arrays['mmsi'])),
        regions=regional,history_steps=hist,future_steps=future,background_id_count=len(background_ids),
        background_mmsi=sorted(background_ids),target_identity_policy='globally disjoint train/val/test target MMSIs; background reuse is permitted as declared',
        history_span_seconds=np.quantile(arrays['time_s'][:,-1],[0,.5,.9,1]).tolist(),
        future_span_seconds=np.quantile(arrays['query_s'][:,-1],[0,.5,.9,1]).tolist(),
        cap_fitted_on='original 700-per-region TRAIN histories only',future_reference='unmodified subsequent observed AIS positions')
    save(manifest_path,report);return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('/mnt/sda/WangZijia/AIS'))
    p.add_argument('--pilot',type=Path,default=Path('/mnt/sda/WangZijia/AIS/forecast_pilot_20261105'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--inventory',action='store_true')
    p.add_argument('--history-steps',type=int,default=12);p.add_argument('--future-steps',type=int,default=8)
    p.add_argument('--split',choices=['train','val','test'],default='train');p.add_argument('--quota',type=int)
    p.add_argument('--train-per-region',type=int,default=2000);p.add_argument('--val-per-region',type=int,default=150);p.add_argument('--test-per-region',type=int,default=150)
    p.add_argument('--freeze-path',type=Path);a=p.parse_args()
    if a.inventory:
        r=inventory(a.project,a.pilot,a.history_steps,a.future_steps);a.out.mkdir(parents=True,exist_ok=True)
        save(a.out/f'inventory_h{a.history_steps}_f{a.future_steps}.json',r)
        rows=[dict(history_steps=a.history_steps,region=region,split=split,**info[split]) for region,info in r['regions'].items() for split in ['train','val','test']]
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (a.out/f'inventory_h{a.history_steps}_f{a.future_steps}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
        print(json.dumps(r['totals'],ensure_ascii=False,indent=2))
    else:
        quota=a.quota if a.quota is not None else {'train':a.train_per_region,'val':a.val_per_region,'test':a.test_per_region}[a.split]
        r=build(a.project,a.pilot,a.out,a.split,quota,a.history_steps,a.future_steps,a.freeze_path)
        print(json.dumps({k:r[k] for k in ['windows','unique_target_mmsi','history_steps','future_steps','npz_sha256']},indent=2))

if __name__=='__main__':main()
