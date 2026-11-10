import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

def bootstrap(y,a,b,ids,draws,seed):
    ids=np.asarray(ids); ships,inv=np.unique(ids,return_inverse=True)
    groups=[np.flatnonzero(inv==i) for i in range(len(ships))]
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(draws):
        ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]); yy=y[ix]
        if len(np.unique(yy))<2: continue
        diffs.append(average_precision_score(yy,a[ix])-average_precision_score(yy,b[ix]))
    return [float(v) for v in np.quantile(diffs,[.025,.975])]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107');p.add_argument('--draws',type=int,default=2000);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    root=Path(a.out)/'relation_pilot';data={}
    for model in ('tft_gcn','simad','dmrad'):
        arr={}
        for arm in ('C0','F0','F1'):
            with np.load(root/model/arm/f'seed{a.seed}'/'test_scores.npz',allow_pickle=False) as z: arr[arm]={k:z[k] for k in z.files}
        for arm in ('F0','F1'):
            for k in ('source','case','y','mmsi','region'):
                if not np.array_equal(arr['C0'][k],arr[arm][k]): raise ValueError('unpaired '+model+' '+arm+' '+k)
        data[model]={}
        y=arr['C0']['y'];ids=arr['C0']['mmsi']
        for view in ('base','natural','suffix','combined_l100_m600'):
            ap={k:float(average_precision_score(y,arr[k][view])) for k in ('C0','F0','F1')}
            ci=bootstrap(y,arr['F1'][view],arr['F0'][view],ids,a.draws,7111)
            data[model][view]={'AP':ap,'F1_minus_F0':ap['F1']-ap['F0'],'CI95':ci}
    out=Path(a.out)/f'relation_summary_seed{a.seed}.json';out.write_text(json.dumps(data,indent=2),encoding='utf8');print(json.dumps(data,indent=2))
if __name__=='__main__':main()
