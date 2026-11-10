"""Extract all observed trajectory points for the frozen balanced map release.

The output stores ragged tracks using offsets and never interpolates or samples
additional observations. source_rows means zero-based canonical Parquet row
position, independent of any persisted DataFrame index label.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,default=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/release_20261026'))
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'track_assets')
    args=parser.parse_args()
    manifest_path=args.release/'selected_windows.parquet'
    registry_path=args.release/'source_registry.json'
    report=json.loads((args.release/'selection_report.json').read_text())
    registry=json.loads(registry_path.read_text())['regions']
    assert report['status']=='verified_build'
    assert sha(manifest_path)==report['output_hashes']['selected_windows.parquet']
    selected=pd.read_parquet(manifest_path).reset_index(drop=True)
    assert len(selected)==21600 and not selected.window_id.duplicated().any()
    args.out.mkdir(parents=True,exist_ok=True)
    evidence={'schema':'frozen_selected_complete_tracks_v1','selection_sha256':sha(manifest_path),'source_registry_sha256':sha(registry_path),'script_sha256':sha(Path(__file__)),'source_row_semantics':'Zero-based physical row in the canonical Parquet, never DataFrame index labels','interpolation':False,'regions':{}}
    total=0
    for region in report['regions']:
        m=registry[region];path=Path(m['canonical_file']);source_hash=sha(path)
        assert source_hash==m['sha256']==report['source_files'][region]['canonical_sha256']
        source=pd.read_parquet(path,columns=['mmsi','timestamp','lon','lat']).reset_index(drop=True)
        # Materialize numeric columns exactly once before selecting tiny windows.
        values={key:source[key].to_numpy() for key in ['mmsi','timestamp','lon','lat']}
        windows=selected[selected.region==region]
        assert len(windows)==2700 and windows.mmsi.nunique()==300
        tracks=[];offsets=[0];window_ids=[];target_ids=[];splits=[];manifest_rows=[]
        for row in windows.itertuples():
            ix=np.array(json.loads(row.row_indices),dtype=np.int64)
            times=np.array(json.loads(row.timestamps),dtype=np.int64)
            assert 8<=len(ix)<=20 and len(ix)==row.n_obs==len(times)
            assert np.all(ix>=0) and np.all(ix<len(source))
            assert np.all(values['mmsi'][ix]==row.mmsi)
            assert np.array_equal(values['timestamp'][ix],times)
            assert np.all(np.diff(times)>=60)
            tracks.append(ix);offsets.append(offsets[-1]+len(ix))
            window_ids.append(row.window_id);target_ids.append(row.mmsi);splits.append(row.split);manifest_rows.append(row.Index)
        indices=np.concatenate(tracks)
        coords=np.column_stack((values['lon'][indices],values['lat'][indices])).astype(np.float64)
        assert np.isfinite(coords).all()
        assert np.all(np.abs(coords[:,0])<=180) and np.all(np.abs(coords[:,1])<=90)
        assert len(np.unique(indices))==len(indices),'The frozen release repeated a canonical target observation within a region'
        target=args.out/f'{region}.npz'
        np.savez_compressed(target,coordinates=coords,offsets=np.array(offsets,dtype=np.int64),
                            window_id=np.array(window_ids,dtype='U24'),mmsi=np.array(target_ids,dtype=np.int64),
                            split=np.array(splits,dtype='U5'),timestamps=values['timestamp'][indices].astype(np.int64),
                            source_rows=indices,selected_manifest_rows=np.array(manifest_rows,dtype=np.int64))
        # Re-read every output and verify ragged boundaries and precise values.
        with np.load(target,allow_pickle=False) as loaded:
            assert len(loaded['offsets'])==2701 and loaded['offsets'][-1]==len(coords)
            assert np.array_equal(loaded['coordinates'],coords)
            assert np.array_equal(loaded['source_rows'],indices)
            assert np.array_equal(loaded['timestamps'],values['timestamp'][indices])
            assert len(set(loaded['mmsi']))==300 and len(set(loaded['window_id']))==2700
        info={'canonical_file':str(path),'canonical_sha256':source_hash,'display_name':m['display_name'],'continent':m['continent'],'core_bbox':m['core_bbox'],'windows':len(windows),'vessels':int(windows.mmsi.nunique()),'points':len(coords),'observed_track_bbox':[float(coords[:,0].min()),float(coords[:,1].min()),float(coords[:,0].max()),float(coords[:,1].max())],'track_file':str(target),'track_sha256':sha(target),'split_windows':windows.groupby('split').size().to_dict()}
        evidence['regions'][region]=info;total+=len(coords)
        print(region,info['windows'],info['vessels'],info['points'],flush=True)
    assert total==372893
    evidence.update(windows=len(selected),points=total,regions_count=len(evidence['regions']),all_checks_passed=True)
    (args.out/'track_assets_manifest.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False))
    print('VERIFIED',len(selected),'tracks,',total,'observed points;',args.out,flush=True)


if __name__=='__main__':main()
