#!/usr/bin/env python3
"""Audit 144 matched runs and compute target-MMSI clustered paired AP CIs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

MODELS=('tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph')
TASKS={'single':('raw','ordinary','reliable'),'relation':('C0','F0','F1')}
VIEWS={'single':('base',),'relation':('base','natural','suffix','combined_l100_m600')}

def paired_bootstrap(ys,aa,bb,mmsi,nboot=2000,seed=1729):
    """Three-seed mean AP difference, paired resampling on target MMSI only."""
    units,inv=np.unique(mmsi,return_inverse=True)
    if len(units)<2:raise ValueError('Need at least 2 target MMSI clusters')
    # Pre-sort once per seed; sample multiplicities implement paired cluster
    # resampling exactly, including repeated ships and equal-score ties.
    def ordered(score):
        order=np.argsort(-score,kind='stable');s=score[order]
        ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
        return ys[order].astype(np.float64),inv[order],ends
    prepared=[(ordered(a),ordered(b)) for a,b in zip(aa,bb)]
    def weighted_ap(item,counts):
        label,cluster,ends=item;w=counts[cluster];tp=np.cumsum(w*label)[ends];total=np.cumsum(w)[ends]
        if tp[-1]==0:return float('nan')
        precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
        return float(np.sum(np.diff(np.r_[0.,tp])*precision)/tp[-1])
    rng=np.random.default_rng(seed);d=[]
    for _ in range(nboot):
        counts=np.bincount(rng.integers(0,len(units),len(units)),minlength=len(units))
        weights=counts[inv]
        if not np.any(weights*(ys==1)) or not np.any(weights*(ys==0)):continue
        d.append(np.mean([weighted_ap(a,counts)-weighted_ap(b,counts) for a,b in prepared]))
    if len(d)<.95*nboot:raise ValueError('More than 5% invalid bootstrap replicates')
    point=np.mean([average_precision_score(ys,a)-average_precision_score(ys,b) for a,b in zip(aa,bb)])
    lo,hi=np.quantile(d,[.025,.975]);return {'delta_AP':float(point),'lower95':float(lo),'upper95':float(hi),'mmsi_clusters':len(units),'boot_valid':len(d)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/home/ps/AIS_published_detectors_20261107');p.add_argument('--nboot',type=int,default=2000);p.add_argument('--seed',type=int,default=1729);a=p.parse_args()
    root=Path(a.root);work=root/'eight_matched';out=root/'eight_matched_summary';out.mkdir(parents=True,exist_ok=True)
    errors=[];records={};rows=[]
    for task,arms in TASKS.items():
        for model in MODELS:
            for arm in arms:
                for seed in range(3):
                    d=work/task/model/arm/f'seed{seed}';required=[d/'last.pt',d/'best_val.pt',d/'protocol.json',d/'history.json',d/'test_scores.npz',d/'test_metrics.json']
                    missing=[str(q) for q in required if not q.is_file()]
                    if missing:errors.append({'task':task,'model':model,'arm':arm,'seed':seed,'missing':missing});continue
                    try:
                        metric=json.loads((d/'test_metrics.json').read_text());z=np.load(d/'test_scores.npz',allow_pickle=False)
                        x={k:z[k] for k in z.files};y=x['y'];ids=('window_id',) if task=='single' else ('source','case')
                        for key in ('y','mmsi','region',*ids,*VIEWS[task]):
                            if key not in x or len(x[key])!=len(y):raise ValueError('missing/mismatched '+key)
                        for view in VIEWS[task]:
                            val=metric['views'][view];threshold=metric['val_thresholds']
                            row={'task':task,'model':model,'arm':arm,'seed':seed,'view':view,'n_windows':len(y),'n_mmsi':len(np.unique(x['mmsi'])),
                                 'AP':val['AP'],'AUROC':val['AUROC'],'threshold_1pct':threshold['1%'],'actual_FAR_1pct':val['FAR@1%'],'recall_1pct':val['Recall@1%'],
                                 'threshold_5pct':threshold['5%'],'actual_FAR_5pct':val['FAR@5%'],'recall_5pct':val['Recall@5%']}
                            if abs(average_precision_score(y,x[view])-row['AP'])>1e-8:raise ValueError('saved AP does not match raw predictions')
                            rows.append(row)
                        records[(task,model,arm,seed)]=x
                    except Exception as e:errors.append({'task':task,'model':model,'arm':arm,'seed':seed,'error':str(e)})
    status=root/'logs/eight_matched_status.tsv'
    if not status.exists():errors.append({'error':'missing queue status file'})
    else:
        with status.open(newline='',encoding='utf8') as f:s=list(csv.DictReader(f,delimiter='\t'))
        keys=[(r['task'],r['model'],r['arm'],int(r['seed'])) for r in s]
        if len(s)!=144 or len(set(keys))!=144:errors.append({'error':'expected 144 unique queue statuses','rows':len(s),'unique':len(set(keys))})
        errors.extend({'status':r} for r in s if r['status']!='OK')
    if errors or len(records)!=144:
        (out/'汇总失败.json').write_text(json.dumps({'expected':144,'complete':len(records),'errors':errors},ensure_ascii=False,indent=2),encoding='utf8')
        raise SystemExit(f'Incomplete or invalid results: {len(records)}/144; see 汇总失败.json')
    comparisons=[]
    for task,arms in TASKS.items():
        baseline,ordinary,reliable=arms
        ids=('window_id',) if task=='single' else ('source','case')
        for model in MODELS:
            ref=records[(task,model,baseline,0)]
            for arm in arms:
                for seed in range(3):
                    cur=records[(task,model,arm,seed)]
                    for k in ('y','mmsi','region',*ids):
                        if not np.array_equal(ref[k],cur[k]):raise ValueError(f'Unpaired metadata: {task}/{model}/{arm}/{seed}/{k}')
            for view in VIEWS[task]:
                for comparator in (ordinary,baseline):
                    aa=[records[(task,model,reliable,s)][view] for s in range(3)];bb=[records[(task,model,comparator,s)][view] for s in range(3)]
                    ci=paired_bootstrap(ref['y'],aa,bb,ref['mmsi'],a.nboot,a.seed)
                    comparisons.append({'task':task,'model':model,'view':view,'comparison':f'{reliable}-{comparator}',**ci})
    def write_csv(path,data):
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write_csv(out/'每种子指标.csv',rows);write_csv(out/'MMSI聚类配对区间.csv',comparisons)
    report={'completed_runs':len(records),'per_seed_rows':len(rows),'paired_comparisons':len(comparisons),'bootstrap_replicates':a.nboot,
        'bootstrap_interpretation':'MMSI-cluster paired resampling; delta is mean of 3 per-seed AP differences; does not include between-seed population uncertainty',
        'single_reliable_vs_raw':'changes target gate and field together; not attributable to reliable weighting alone',
        'data':'previously exposed test; resampled full offline windows','test_fars':'actual values using each model validation-normal thresholds'}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
