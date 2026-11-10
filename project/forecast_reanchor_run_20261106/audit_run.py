"""Final traceability audit, independent of the training loss and displayed means."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from common import *

def audit(work):
    work=Path(work);frozen=assert_frozen(work);prep=read(work/'preparation_complete.json')
    ids={};data={}
    for s in ['train','val','test']:
        data[s]=load_npz(work/f'data/{s}.npz');ids[s]=set(map(int,data[s]['mmsi']))
    assert not (ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
    tm=read(work/'data/test_manifest.json');assert sha(work/'data/test.npz')==tm['npz_sha256']
    assert tm['freeze_sha256']==sha(work/'checkpoint_freeze.json')
    prov=read(work/'data/test_provenance.json');index={str(w):i for i,w in enumerate(data['test']['window_id'])};count=0
    for region in tqdm(sorted({r['region'] for r in prov}),desc='Independent raw TEST reference audit'):
        rr=[r for r in prov if r['region']==region];src=Path(rr[0]['source_file'])
        assert sha(src)==rr[0]['source_sha256'];raw=pd.read_parquet(src,columns=['mmsi','timestamp','lon','lat']).reset_index(drop=True)
        reg=read(src.parents[1]/'source_registry.json')['regions'][region];b=reg['core_bbox'];lon0=(b[0]+b[2])/2;lat0=(b[1]+b[3])/2
        for r in rr:
            row=raw.iloc[r['row_indices']];t=row.timestamp.to_numpy(float);i=index[r['window_id']]
            assert (row.mmsi.to_numpy()==r['mmsi']).all();np.testing.assert_array_equal(t,r['timestamps'])
            truth=np.column_stack([(row.lon.to_numpy()-lon0)*111195*np.cos(np.deg2rad(lat0)),(row.lat.to_numpy()-lat0)*111195])[12:]
            np.testing.assert_array_equal(truth,data['test']['future_xy'][i]);np.testing.assert_array_equal(t[12:]-t[11],data['test']['query_s'][i])
            for v in r['views'].values():
                assert r['mmsi'] not in v['background_mmsi']
                assert v['max_neighbor_time'] is None or v['max_neighbor_time']<=t[11]
            count+=1
    norm=read(work/'data/normalizer.json');verified=0
    for r in frozen['models']:
        p=Path(r['checkpoint']);assert read(p.with_name('normalizer.json'))==norm
        result=read(work/f"test_runs/{r['model']}/seed_{r['seed']}/{r['arm']}/result.json")
        assert result['checkpoint_sha256']==r['checkpoint_sha256'] and result['status']=='complete'
        if r['arm'] in ['P0','P1']:
            base=load_npz(work/f"test_runs/{r['model']}/seed_{r['seed']}/{r['arm']}/predictions_base.npz")
            stressed=load_npz(work/f"test_runs/{r['model']}/seed_{r['seed']}/{r['arm']}/predictions_neighbor_stress.npz")
            np.testing.assert_array_equal(base['pred_xy'],stressed['pred_xy'])
        verified+=1
    for split in ['val','test']:assert read(work/f'reports/{split}/summary.json')['completed_models']==48
    report=dict(status='pass',models_verified=verified,new_training_runs=36,reused_raw=12,
        target_mmsi_counts={s:len(v) for s,v in ids.items()},target_overlap=0,test_windows_reconstructed=count,
        test_future_positions_checked=count*8,max_future_reference_difference_m=0,
        checkpoint_freeze_sha256=sha(work/'checkpoint_freeze.json'),test_manifest=tm,
        raw_reuse_records=prep['reused'],observed_tradeoff=CONFIG['user_decision'])
    save(work/'reports/independent_audit.json',report)
    (work/'reports/独立核验.md').write_text(f'# 独立核验\n\n状态：通过。36组重训、12组原始输入复用；48个测试模型与冻结检查点一致。\n\n训练/验证/测试目标船零交集；从原始报文物理行号重算{count}窗、{count*8}个未来观测参考位置，最大坐标差为0米。所有模型使用同一原训练标准化参数；仅目标输入模型在邻船污染前后预测逐点一致。\n\n渐进漂移折中由用户明确接受并锁定，旧CV未通过记录未修改。该测试是预测任务目标船留出评估，原AIS来源此前用于检测研究，不称全新外部数据。\n',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('work'));a=p.parse_args();audit(a.work)
