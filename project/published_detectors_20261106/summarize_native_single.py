import argparse,json,csv
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from summarize_supervised import compare

def load(root,method,arm,seed):
    path=(root/'carla_all_windows'/arm if method=='carla' else root/'single_all_windows'/method/arm)/f'seed{seed}'/'test_scores.npz'
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/home/ps/AIS_published_detectors_20261107');p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args();root=Path(a.root);rows=[]
    for method in ('tft_gcn','simad','dmrad','carla'):
        data={arm:[load(root,method,arm,seed) for seed in range(3)] for arm in ('raw','ordinary','reliable')}
        for arm,ds in data.items():
            aps=[average_precision_score(d['y'],d['score']) for d in ds];aucs=[roc_auc_score(d['y'],d['score']) for d in ds]
            rows.append({'method':method,'arm':arm,'ap_mean':float(np.mean(aps)),'ap_seed_sd':float(np.std(aps,ddof=1)),'auroc_mean':float(np.mean(aucs))})
        for arm in ('raw','ordinary'):
            pairs=list(zip(data[arm],data['reliable']));delta,lo,hi,n,valid=compare(pairs,'score',a.bootstrap,42)
            rows.append({'method':method,'arm':'reliable minus '+arm,'ap_mean':delta,'ci_low':lo,'ci_high':hi,'mmsi_clusters':n,'valid_bootstrap':valid})
    out=root/'native_single_summary';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps({'results':rows,'note':'MMSI bootstrap conditional on three seeds; test previously exposed'},indent=2),encoding='utf8')
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    print('native single summary:',len(rows),'rows')
if __name__=='__main__':main()
