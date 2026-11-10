import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107');p.add_argument('--draws',type=int,default=2000);a=p.parse_args()
    root=Path(a.out)/'relation_pilot';result={}
    for model in ('tft_gcn','simad','dmrad','carla'):
        arr={}
        for seed in range(3):
            arr[seed]={}
            for arm in ('C0','F0','F1'):
                folder=(Path(a.out)/'carla_relation_pilot'/arm if model=='carla' else root/model/arm)
                with np.load(folder/f'seed{seed}/test_scores.npz',allow_pickle=False) as z:arr[seed][arm]={k:z[k] for k in z.files}
        base=arr[0]['C0']
        for seed in range(3):
            for arm in ('C0','F0','F1'):
                for key in ('source','case','y','mmsi','region'):
                    if not np.array_equal(base[key],arr[seed][arm][key]):raise ValueError('different test sample '+model+' '+key)
        y=base['y'];ships,inv=np.unique(base['mmsi'],return_inverse=True);groups=[np.flatnonzero(inv==i) for i in range(len(ships))]
        result[model]={}
        for view in ('base','natural','suffix','combined_l100_m600'):
            aps={arm:[float(average_precision_score(y,arr[s][arm][view])) for s in range(3)] for arm in ('C0','F0','F1')}
            deltas=np.asarray(aps['F1'])-aps['F0'];rng=np.random.default_rng(7111);boot=[]
            for _ in range(a.draws):
                ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]);yy=y[ix]
                if len(np.unique(yy))<2:continue
                boot.append(float(np.mean([average_precision_score(yy,arr[s]['F1'][view][ix])-average_precision_score(yy,arr[s]['F0'][view][ix]) for s in range(3)])))
            result[model][view]={'AP_by_seed':aps,'AP_mean':{k:float(np.mean(v)) for k,v in aps.items()},'F1_minus_F0_by_seed':deltas.tolist(),'F1_minus_F0_mean':float(deltas.mean()),'CI95_MMSI_conditional_on_3_seeds':[float(v) for v in np.quantile(boot,[.025,.975])]}
    (Path(a.out)/'relation_summary_three_seeds.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
