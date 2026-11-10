"""Independent reduced regression for the recorded Great Belt replay failure."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--neighborhood',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();folder=args.neighborhood
    failure=json.loads((folder/'replay_failure_diagnosis.json').read_text())
    spec=importlib.util.spec_from_file_location('scanner_regression_subject',folder/'audit_neighborhood.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=failure['raw_rows'];src=object.__new__(module.Source)
    src.t=np.asarray([r['t'] for r in rows],float);src.p=np.asarray([r['p'] for r in rows],float)
    src.s=np.asarray([r['s'] for r in rows],float);src.c=np.asarray([r['c'] for r in rows],float)
    src.good=np.isfinite(src.p).all(1)&np.isfinite(src.s)&np.isfinite(src.c)&(src.s>=0)&(src.s<102.2)&(src.c>=0)&(src.c<360)
    vessel=failure['neighbor']['mmsi'];src.ship={vessel:np.arange(len(rows))}
    change=dict(failure['changes'][0]);change['parameters']=dict(change['parameters'])
    finite_t=src.t[src.good]
    change['parameters']['report_index']=int(np.flatnonzero(finite_t==change['parameters']['time_s'])[0])
    md={'decision_time_s':failure['decision'],'neighbor_observation_noise':{'changes':[change]}}
    replay=src.overlay(md)
    # Expected validity depends on adjacency in the complete source sequence.
    dt=np.diff(src.t)
    expected=np.r_[False,src.good[:-1]&src.good[1:]&(dt>0)&(dt<=300)]
    target_index=int(np.flatnonzero(src.t==failure['neighbor']['time_s'])[0])
    assert not expected[target_index]
    assert set(replay)==set(range(len(rows)))
    assert [bool(replay[i][3]) for i in range(len(rows))]==expected.tolist()
    assert replay[target_index][4]==0 and replay[target_index][5]==0
    # This reproduces why deleting the invalid report bridges the gap.
    kept=np.flatnonzero(src.good);kept_index=int(np.flatnonzero(kept==target_index)[0])
    bridged_valid=0<src.t[kept[kept_index]]-src.t[kept[kept_index-1]]<=300
    assert bridged_valid
    report=dict(status='passed',window_id=failure['window_id'],neighbor_mmsi=vessel,
                report_time_s=float(src.t[target_index]),complete_sequence_validity=expected.tolist(),
                target_validity_replayed=False,target_error_mps=0.,target_reliability=0.,
                deleting_invalid_report_would_bridge=True,
                scope='Independent five-row regression from the recorded failure. The recorded finite-history report index is remapped to this reduced fixture; no source data or feature files are modified.',
                scanner_sha256=hashlib.sha256((folder/'audit_neighborhood.py').read_bytes()).hexdigest(),
                failure_report_sha256=hashlib.sha256((folder/'replay_failure_diagnosis.json').read_bytes()).hexdigest())
    with args.out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report))


if __name__=='__main__':main()
