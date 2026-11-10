"""Verify both prespecified bridging experiments before recent-model runs.

This is a completeness/integrity gate, not a requirement for positive gains
and not a model selection procedure. Test labels are read only after the
bridge experiments have finished their independently frozen evaluations.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from recent_late_provenance import sha,verify_test_rho

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
sys.path.insert(0,str(ROOT/'formal_20261101/single'))
import train_single

def check():
    tcn=Path('/home/ps/AIS_bridge_20261107/tcn'); tft=Path('/home/ps/AIS_bridge_20261107/tft_eval128')
    log=Path('/home/ps/AIS_bridge_20261107/logs/tft128_finalize.log')
    required=[tcn/'frozen_validation.lock.json',tcn/'paired_test_summary.json',tft/'validation_frozen.json',tft/'val_summary.json',tft/'test_summary.json',log]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise RuntimeError('Bridges incomplete: '+', '.join(missing))
    if 'TFT BRIDGE COMPLETE' not in log.read_text(encoding='utf8'):raise RuntimeError('TFT eval128 finalization not complete')
    tcn_lock=json.loads((tcn/'frozen_validation.lock.json').read_text());tcns=json.loads((tcn/'paired_test_summary.json').read_text())
    if len(tcn_lock['runs'])!=21 or not {'best_val','last'}<=set(tcns):raise ValueError('TCN expected frozen 21 runs, best and last summaries')
    for k,v in {**tcn_lock['dependency_sha256'],**tcn_lock['training_validation_data_sha256']}.items():
        if sha(k)!=v:raise ValueError('TCN frozen dependency/data mismatch: '+k)
    if sha(HERE/'bridge_tcn_20261107.py')!=tcn_lock['code_sha256']:raise ValueError('TCN source mismatch')
    for key,record in tcn_lock['runs'].items():
        path=tcn/'validation'/key
        for name in ('best_val','last'):
            if sha(path/(name+'.pt'))!=record[name]:raise ValueError('TCN checkpoint mismatch: '+key+'/'+name)
        if sha(path/'val_scores.npz')!=record['validation_scores']:raise ValueError('TCN validation scores mismatch '+key)
    tft_lock=json.loads((tft/'validation_frozen.json').read_text());tfv=json.loads((tft/'val_summary.json').read_text());tfs=json.loads((tft/'test_summary.json').read_text())
    if tft_lock['count']!=24 or len(tft_lock['records'])!=24:raise ValueError('TFT expected frozen 24 runs')
    if len(tfv['rows'])!=24 or len(tfs['rows'])!=24 or len(tfv['paired'])!=7 or len(tfs['paired'])!=7:raise ValueError('TFT summary count mismatch')
    if sha(HERE/'bridge_tft.py')!=tft_lock['source_sha256']:raise ValueError('TFT source mismatch')
    if sha(tft/'training_sample_identity.npz')!=tft_lock['training_sample_identity_sha256']:raise ValueError('TFT training identity checksum mismatch')
    with np.load(ROOT/'framework_validation_v1_20261028/data/individual/train.npz',allow_pickle=False) as z:
        with np.load(tft/'training_sample_identity.npz',allow_pickle=False) as q:
            for k in ('window_id','mmsi','region','y'):
                if not np.array_equal(z[k],q[k]):raise ValueError('TFT training identity mismatch: '+k)
    val=train_single.load('val')
    # Only test identity metadata are needed here; test labels and scores are
    # not inspected to authorize the later 48 runs.
    with np.load(ROOT/'framework_validation_v1_20261028/data/individual/test.npz',allow_pickle=False) as z:
        test={'ids':z['window_id'].astype(str),'mmsi':z['mmsi'].astype(str),'region':z['region'].astype(str)}
    mapping={'window_id':'ids','mmsi':'mmsi','region':'region','y':'y'}
    for key in tcn_lock['runs']:
        path=tcn/'validation'/key
        for split,ref in (('val',val),('test',test)):
            with np.load(path/(split+'_scores.npz'),allow_pickle=False) as z:
                for col,oldkey in mapping.items():
                    if split=='test' and col=='y':continue
                    if not np.array_equal(z[col],ref[oldkey]):raise ValueError(f'TCN {split} identity mismatch {key}/{col}')
                if split=='val' and not np.isfinite(z['score']).all():raise ValueError('TCN nonfinite validation scores')
    for r in tft_lock['records']:
        d=tft/r['condition']/r['arm']/f"seed{r['seed']}"
        spec=json.loads((d/'protocol.json').read_text())
        if spec['batch_size']!=128 or spec['evaluation_batch']!='fixed 128 in ordered windows; TFT selects periodic frequencies across each batch, so score may depend on batch composition':raise ValueError('TFT fixed batch check failed')
        if sha(d/'best_val.pt')!=r['checkpoint_sha256'] or sha(d/'protocol.json')!=r['protocol_sha256']:raise ValueError('TFT checkpoint/protocol checksum mismatch')
        for split,ref in (('val',val),('test',test)):
            with np.load(d/(split+'_scores.npz'),allow_pickle=False) as z:
                for key,oldkey in mapping.items():
                    if split=='test' and key=='y':continue
                    if not np.array_equal(z[key],ref[oldkey]):raise ValueError(f'TFT {split} identity mismatch {key}')
                if split=='val' and not np.isfinite(z['score']).all():raise ValueError('TFT nonfinite validation scores')
    return {'tcn_frozen_runs':21,'tft_frozen_runs':24,'tft_evaluation_batch':128,'TCN_current_source_and_training_data_hashes':'passed',
            'TFT_source_checkpoint_protocol_training_val_test_ID_hashes':'passed','test_rho_provenance':verify_test_rho(),
            'evidence_sha256':{str(p):sha(p) for p in required},
            'gate':'completeness only; no score-based model selection; existing test previously exposed'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107/recent_original_late');a=p.parse_args()
    report=check();print('BRIDGE_PREFLIGHT_OK',json.dumps(report,ensure_ascii=False),flush=True)
    if a.write:
        dest=Path(a.out)/'bridge_gate_lock.json';dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        print('BRIDGE_GATE_WRITTEN',dest)
if __name__=='__main__':main()
