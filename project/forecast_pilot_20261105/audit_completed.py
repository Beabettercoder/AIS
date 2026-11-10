"""Independent raw-report provenance and completed-run alignment audit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));args=p.parse_args();root=args.root
    dm=json.loads((root/'data/data_manifest.json').read_text())
    rows=json.loads((root/'data/window_provenance.json').read_text())
    registry_path=Path(rows[0]['source_file']).parents[1]/'source_registry.json'
    # Source registry resides at balanced_global_v2 for both east and west sources.
    registry=json.loads(registry_path.read_text())['regions']
    data={};index={}
    for split in ['train','val']:
        path=root/f'data/{split}.npz';assert digest(path)==dm['npz_sha256'][split]
        with np.load(path,allow_pickle=False) as z:data[split]={k:z[k] for k in ['future_xy','query_s','time_s','window_id','mmsi','region']}
        index[split]={str(w):i for i,w in enumerate(data[split]['window_id'])}
    assert not(set(map(int,data['train']['mmsi']))&set(map(int,data['val']['mmsi'])))
    checked=0;maxerr=0.;byregion={}
    for region in tqdm(dm['config']['regions'],desc='Raw source reconstruction',unit='region'):
        selected=[r for r in rows if r['region']==region];path=Path(selected[0]['source_file'])
        assert digest(path)==registry[region]['sha256']==selected[0]['source_sha256']
        table=pd.read_parquet(path,columns=['mmsi','timestamp','lon','lat']).reset_index(drop=True)
        b=registry[region]['core_bbox'];lon0=(b[0]+b[2])/2;lat0=(b[1]+b[3])/2
        for row in selected:
            rec=table.iloc[row['row_indices']];ts=rec.timestamp.to_numpy(float)
            assert np.all(rec.mmsi.to_numpy()==row['mmsi']);assert np.array_equal(ts,np.array(row['timestamps']))
            true=np.column_stack([(rec.lon.to_numpy()-lon0)*111195*np.cos(np.deg2rad(lat0)),(rec.lat.to_numpy()-lat0)*111195])[12:]
            z=data[row['split']];i=index[row['split']][row['window_id']]
            maxerr=max(maxerr,float(np.max(np.abs(true-z['future_xy'][i]))))
            np.testing.assert_array_equal(true,z['future_xy'][i]);np.testing.assert_array_equal(ts[12:]-ts[11],z['query_s'][i])
            np.testing.assert_array_equal(ts[:12]-ts[0],z['time_s'][i])
            for va in row['views'].values():
                assert va['max_neighbor_time'] is None or va['max_neighbor_time']<=ts[11]
                assert row['mmsi'] not in va['background_mmsi']
            checked+=1
        byregion[region]=len(selected)
    protocols=[];groups={}
    for path in sorted((root/'parallel_runs').rglob('protocol.json')):
        x=json.loads(path.read_text());m=x['model'];seed=x['seed'];arm=x['arm']
        assert x['epochs']==(75 if m=='itransformer' else 25)
        res=json.loads(path.with_name('result.json').read_text());assert res['status']=='complete'
        groups.setdefault((m,seed),[]).append(x)
        ck=path.with_name('checkpoint.pt');assert ck.exists()
        protocols.append(dict(model=m,seed=seed,arm=arm,epochs=x['epochs'],parameters=x['parameter_count'],
                              initial_hash=x['initial_model_hash'],checkpoint_sha256=digest(ck)))
    assert len(protocols)==48 and len(groups)==12
    for group in groups.values():
        assert {x['arm'] for x in group}=={'P0','P1','P2','P3'}
        assert len({x['initial_model_hash'] for x in group})==1
        assert len({x['parameter_count'] for x in group})==1
        assert len({x['normalizer_hash'] for x in group})==1
    s=json.loads((root/'reports/summary.json').read_text());assert s['status']=='complete'
    post=json.loads((root/'reports/independent_postflight.json').read_text());assert post['status']=='pass'
    report=dict(status='pass',windows_reconstructed_from_raw=checked,future_positions_checked=checked*8,
        max_coordinate_reconstruction_error_m=maxerr,windows_by_region=byregion,
        observed_train_target_mmsi=len(np.unique(data['train']['mmsi'])),
        observed_val_target_mmsi=len(np.unique(data['val']['mmsi'])),target_train_val_overlap=0,
        paired_initializations='all 12 model-seed groups have four identical initial model hashes and widths',
        completed_runs=48,validation_prediction_arrays=s['unique_prediction_files'],
        test_opened=False,protocols=protocols,independent_checks=post['checks'])
    dest=root/'reports/原始报文与完整训练核验.json';dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['protocols','independent_checks']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
