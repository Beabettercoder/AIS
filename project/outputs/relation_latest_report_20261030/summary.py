from pathlib import Path
import json, hashlib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

base=Path('/mnt/sda/WangZijia/AIS/outputs')
out=base/'relation_latest_report_20261030'
views=('base','natural','suffix','combined_l100_m600')
arms=('C0','F0','F1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
models={}
for name in ('tcn','gru','transformer','itransformer'):
    seeds=(0,1,2) if name in ('tcn','itransformer') else (0,)
    pred={}; reports={}; paths={}
    for seed in seeds:
        for method in ('cumulative','latest'):
            for arm in arms:
                if method=='latest':
                    p=out/'runs/aligned_embed'/('seed'+str(seed))/name/arm
                elif name in ('tcn','gru','transformer'):
                    p=(base/'heading_alignment_20261030/runs/aligned_embed'/name/arm if seed==0 else base/'tcn_suffix_stability_20261030/runs/aligned_embed'/('seed'+str(seed))/'tcn'/arm)
                else:
                    # No 75-epoch C0 was evaluated in previous pilot. New C0 is the matching target-only control.
                    if arm=='C0':
                        p=out/'runs/aligned_embed'/('seed'+str(seed))/name/arm
                    else:
                        p=base/'itrans_stability_20261030/extended75/runs'/('seed'+str(seed))/'aligned_embed/itransformer'/arm
                paths[(seed,method,arm)]=p
                with np.load(p/'val_scores.npz',allow_pickle=False) as z: pred[(seed,method,arm)]={k:z[k] for k in z.files}
                reports[(seed,method,arm)]=json.loads((p/'result.json').read_text())
    ref=pred[(0,'latest','C0')]
    for key,obj in pred.items():
        for k in ('y','case','source','mmsi','region'):np.testing.assert_array_equal(obj[k],ref[k])
        rr=reports[key]; assert rr['seed']==key[0] and rr['last_epoch']==(75 if name=='itransformer' else 25)
    for seed in seeds:
        for method in ('latest','cumulative'):
            assert len({reports[(seed,method,arm)]['init_sha256'] for arm in arms})==1
        assert reports[(seed,'latest','F1')]['init_sha256']==reports[(seed,'cumulative','F1')]['init_sha256']
    ids,inv=np.unique(ref['mmsi'].astype(str),return_inverse=True)
    rng=np.random.default_rng(20261029);draw=rng.integers(0,len(ids),size=(2000,len(ids)))
    result={'scope':'Previously exposed development validation; paired target-MMSI bootstrap samples fixed model(s), no correction for multiple comparisons or seed population.','seeds':list(seeds),'num_mmsi':len(ids),'views':{},'hashes':{}}
    for key,p in paths.items(): result['hashes']['%s_%s_%s'%key]={'prediction':sha(p/'val_scores.npz'),'checkpoint':sha(p/'final.pt'),'init':reports[key]['init_sha256']}
    for view in views:
        scores={key:obj[view+'_score'] for key,obj in pred.items()}
        vals={key:float(average_precision_score(ref['y'],z)) for key,z in scores.items()}
        # Cache descending sort and ties once; recompute weighted cumulative sums
        # for each target-MMSI draw. Match sklearn's average_precision_score.
        cache={}
        for key,z in scores.items():
            order=np.argsort(-z,kind='mergesort')
            ordered=z[order]
            ends=np.r_[np.flatnonzero(np.diff(ordered)!=0),len(z)-1]
            cache[key]=(order,ends,ref['y'][order].astype(float))
        def weighted_ap(key,weight):
            order,ends,y=cache[key]
            w=weight[order].astype(float)
            tp=np.cumsum(w*y)[ends]; total=np.cumsum(w)[ends]
            precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
            return float(np.sum(np.diff(np.r_[0.,tp])*precision)/tp[-1])
        for key in scores:
            assert abs(weighted_ap(key,np.ones(len(ref['y'])))-vals[key])<1e-12
        pairs=[('latest_f1_f0','latest','F1','latest','F0'),('old_f1_f0','cumulative','F1','cumulative','F0'),('latest_f1_c0','latest','F1','latest','C0'),('f0_latest_minus_old','latest','F0','cumulative','F0'),('f1_latest_minus_old','latest','F1','cumulative','F1')]
        diffs={label:[vals[(s,a,b)]-vals[(s,c,d)] for s in seeds] for label,a,b,c,d in pairs}
        ci={k:[] for k in diffs}
        for sample in draw:
            w=np.bincount(sample,minlength=len(ids))[inv]
            ap={key:weighted_ap(key,w) for key in scores}
            for label,a,b,c,d in pairs:
                ci[label].append(float(np.mean([ap[(s,a,b)]-ap[(s,c,d)] for s in seeds])))
        result['views'][view]={'values':{f'{s}_{m}_{a}':v for (s,m,a),v in vals.items()},'per_seed_diff':diffs,'mean_diff':{k:float(np.mean(v)) for k,v in diffs.items()},'bootstrap95':{k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in ci.items()}}
        print(name,view,result['views'][view]['mean_diff'],result['views'][view]['bootstrap95'],flush=True)
    models[name]=result
(out/'paired_summary.json').write_text(json.dumps(models,indent=2),encoding='utf-8')
