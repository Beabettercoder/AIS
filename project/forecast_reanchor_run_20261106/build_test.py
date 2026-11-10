"""Build real-observation forecasting test windows only after checkpoint freeze."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from common import *
from reanchor import observed_history,reanchor_positions

def build_test(pilot,work):
    work=Path(work);freeze=assert_frozen(work);parent=validate_parent(pilot)
    done=work/'data/test_manifest.json';dest=work/'data/test.npz'
    if done.exists():
        d=read(done);assert sha(dest)==d['npz_sha256'] and d['freeze_sha256']==sha(work/'checkpoint_freeze.json');return d
    cfg=parent['config'];project=pilot.parent
    registry=read(project/'outputs/balanced_global_v2/source_registry.json')['regions']
    labels=pd.read_parquet(project/'outputs/balanced_global_v2/labeling/labeled_v1/label_manifest.parquet',columns=['region','split','mmsi']).drop_duplicates()
    ids={s:set(map(int,labels.loc[labels.split==s,'mmsi'])) for s in ['train','val','test']}
    assert not (ids['train']&ids['test'] or ids['val']&ids['test'])
    all_records=[];provenance=[];regional={};test_background=set()
    for region in tqdm(cfg['regions'],desc='Build TEST after freeze',unit='region'):
        source=B.locate_source(project,registry[region]['canonical_file']);assert sha(source)==registry[region]['sha256']
        table=pd.read_parquet(source,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
        pool,counters=B.candidates(table,set(map(int,labels.loc[(labels.region==region)&(labels.split=='test'),'mmsi'])),registry[region]['core_bbox'],cfg)
        selection=B.balanced_take(pool,CONFIG['test_windows_per_region'],f'{region}/test/{CONFIG["test_sampling_seed"]}')
        if not selection:raise RuntimeError(f'No qualified test window: {region}; do not change rules based on scores.')
        bg=F.Background(table,registry[region]['core_bbox']);regional[region]=dict(**counters,selected=len(selection),unique_targets=len({s['mmsi'] for s in selection}))
        for item in tqdm(selection,desc=region,leave=False,unit='window'):
            row=table.iloc[item['rows']];tt=row.timestamp.to_numpy(float);pp=F.project(row.lon.to_numpy(),row.lat.to_numpy(),registry[region]['core_bbox'])
            ss=row.sog.to_numpy(float);cc=row.cog.to_numpy(float)
            wid=__import__('hashlib').sha256(f'{region}/{item["mmsi"]}/{item["rows"]}/{B.SCHEMA}'.encode()).hexdigest()[:24]
            record=dict(future_xy=pp[12:].astype('f8'),query_s=(tt[12:]-tt[11]).astype('f4'),time_s=(tt[:12]-tt[0]).astype('f4'),
                mmsi=np.int64(item['mmsi']),region=np.array(region),window_id=np.array(wid),maneuver=np.bool_(B.natural_maneuver(ss,cc)))
            doc=dict(window_id=wid,mmsi=item['mmsi'],region=region,split='test',row_indices=item['rows'],timestamps=tt.tolist(),
                     source_file=str(source),source_sha256=registry[region]['sha256'],views={})
            for view in CONFIG['views']:
                x,audit=F.make_view(tt[:12],pp[:12],ss[:12],cc[:12],item['mmsi'],bg,set(),wid,view,parent['speed_cap_mps'],cfg)
                visible,v=observed_history(F,tt[:12],pp[:12],ss[:12],cc[:12],wid,view)
                w,valid,_=F.compatibility(tt[:12],visible,v);gp,gv,_=F.causal_gate(tt[:12],visible,v,w,valid,parent['speed_cap_mps'])
                cp,reset=reanchor_positions(tt[:12],visible,gp,w,valid,.7)
                x['original_anchor_gate']=x['anchor_gate'].copy()
                x['gated'][:,:2]=cp.astype('f4');x['anchor_gate']=visible[0]+cp[-1];x['reanchor_reset']=reset
                record.update({view+'__'+k:v for k,v in x.items()});doc['views'][view]=audit
                test_background.update(audit['background_mmsi'])
            np.testing.assert_array_equal(record['base__raw'],record['neighbor_stress__raw'])
            np.testing.assert_array_equal(record['base__gated'],record['neighbor_stress__gated'])
            all_records.append(record);provenance.append(doc)
        save(work/'test_build_progress.json',dict(region=region,completed_windows=len(all_records),stage='building'))
    arrays=B.stack_records(all_records)
    for split in ['train','val']:
        d=load_npz(work/f'data/{split}.npz')
        assert not set(map(int,d['mmsi']))&set(map(int,arrays['mmsi']))
        assert not set(d['window_id'])&set(arrays['window_id'])
    B.save_npz(dest,arrays);save(work/'data/test_provenance.json',provenance)
    old_prov=read(pilot/'data/window_provenance.json')
    bgsets={s:set().union(*(set(v['views']['base']['background_mmsi']) for v in old_prov if v['split']==s)) for s in ['train','val']}
    result=dict(created_after_freeze=True,freeze_sha256=sha(work/'checkpoint_freeze.json'),npz_sha256=sha(dest),
        windows=len(all_records),target_mmsi=len(set(map(int,arrays['mmsi']))),regions=regional,target_train_val_test_overlap=0,
        shared_background_ids_with_train=len(test_background&bgsets['train']),shared_background_ids_with_val=len(test_background&bgsets['val']),
        horizon_seconds_quantiles=np.quantile(arrays['query_s'][:,-1],[0,.5,.9,1]).tolist(),
        reference='Unmodified subsequent AIS observations',source_history_note=CONFIG['test_provenance_note'])
    save(done,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,default=DEFAULT_PILOT);p.add_argument('--work',type=Path,default=Path('work'))
    a=p.parse_args();build_test(a.pilot,a.work)
