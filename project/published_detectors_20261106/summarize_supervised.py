"""Paired MMSI-cluster bootstrap for the supervised recent-method experiments."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

def root_for(root,method):return root/('supervised_carla_twostage' if method=='carla' else 'supervised')
def load(root,method,task,arm,seed):
    path=root_for(root,method)/task
    if method!='carla':path=path/method
    path=path/arm/f'seed{seed}'
    with np.load(path/'test_scores.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    return d,json.loads((path/'test_metrics.json').read_text(encoding='utf8'))
def compare(groups,view,reps,seed):
    first=groups[0][0];y=first['y'];m=first['mmsi'].astype(str)
    for a,b in groups:
        for d in (a,b):
            for key in ('y','mmsi','window_id','source','case'):
                if key in first and (key not in d or not np.array_equal(first[key],d[key])):raise ValueError('Unpaired '+key)
    unique,inv=np.unique(m,return_inverse=True);indices=[np.flatnonzero(inv==i) for i in range(len(unique))]
    def delta(ix):return float(np.mean([average_precision_score(y[ix],b[view][ix])-average_precision_score(y[ix],a[view][ix]) for a,b in groups]))
    observed=delta(np.arange(len(y)));rng=np.random.default_rng(seed);vals=[]
    for _ in range(reps):
        ix=np.concatenate([indices[i] for i in rng.integers(0,len(unique),len(unique))])
        if len(np.unique(y[ix]))==2: vals.append(delta(ix))
    lo,hi=np.quantile(vals,[.025,.975]);return observed,float(lo),float(hi),len(unique),len(vals)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/home/ps/AIS_published_detectors_20261107');p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args();root=Path(a.root);rows=[];missing=[]
    for method in ('tft_gcn','simad','dmrad','carla'):
      for task in ('single','relation'):
        ref='raw' if task=='single' else 'C0';ordinary='ordinary' if task=='single' else 'F0';trusted='reliable' if task=='single' else 'F1'
        for view in (('base',) if task=='single' else ('base','natural','suffix','combined_l100_m600')):
          try:
            entries={arm:[load(root,method,task,arm,s) for s in range(3)] for arm in (ref,ordinary,trusted)}
          except FileNotFoundError as e:missing.append(str(e));continue
          for arm in (ref,ordinary,trusted):
            ms=[item[1]['views'][view] for item in entries[arm]]
            row={'method':method,'task':task,'view':view,'arm':arm,'metric':'AP/AUROC','ap_mean':float(np.mean([x['AP'] for x in ms])),'ap_seed_sd':float(np.std([x['AP'] for x in ms],ddof=1)),'auroc_mean':float(np.mean([x['AUROC'] for x in ms]))}
            for k in ('FAR@1%','Recall@1%','FAR@5%','Recall@5%'):row[k]=float(np.mean([x[k] for x in ms]))
            rows.append(row)
          for arm in (ref,ordinary):
            paired=[(entries[arm][i][0],entries[trusted][i][0]) for i in range(3)]
            delta,lo,hi,n,valid=compare(paired,view,a.bootstrap,42)
            rows.append({'method':method,'task':task,'view':view,'arm':trusted+' minus '+arm,'metric':'paired AP','ap_mean':delta,'ci_low':lo,'ci_high':hi,'mmsi_clusters':n,'valid_bootstrap':valid})
    dest=root/'supervised_summary';dest.mkdir(parents=True,exist_ok=True)
    (dest/'summary.json').write_text(json.dumps({'results':rows,'missing':missing,'note':'MMSI cluster bootstrap on mean of three seeds; no seed-population uncertainty; previously exposed test'},indent=2),encoding='utf8')
    cols=list(dict.fromkeys(k for row in rows for k in row))
    with (dest/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    print('完成汇总:',len(rows),'行; 缺失:',len(missing))
if __name__=='__main__':main()
