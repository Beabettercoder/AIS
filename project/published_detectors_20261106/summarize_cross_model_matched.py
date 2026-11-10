#!/usr/bin/env python3
"""Audit and aggregate all paired 8-backbone / 4-recent supervised runs."""
import argparse,csv,json,time,traceback
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from summarize_eight_matched import paired_bootstrap

BACKBONES=('tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph')
RECENT=('tft_gcn','simad','dmrad','carla')
ARMS={'single':('raw','ordinary','reliable'),'relation':('C0','F0','F1')}
VIEWS={'single':('base',),'relation':('base','natural','suffix','combined_l100_m600')}
SIZES={'single':1680,'relation':880}

def base_path(root,task,model,arm,seed):
 if model in BACKBONES:return root/'eight_matched'/task/model/arm/f'seed{seed}'
 if model=='carla':return root/'supervised_carla_twostage'/task/arm/f'seed{seed}'
 return root/'supervised'/task/model/arm/f'seed{seed}'

def auc_ci(y,aa,bb,mmsi,nboot,seed):
 units,inv=np.unique(mmsi,return_inverse=True)
 def ordered(s):
  i=np.argsort(s,kind='stable');v=s[i];ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1]
  return y[i].astype(float),inv[i],ends
 prepared=[(ordered(a),ordered(b)) for a,b in zip(aa,bb)]
 def auc(item,counts):
  target,groups,ends=item;w=counts[groups];p=np.cumsum(w*target)[ends];n=np.cumsum(w*(1-target))[ends]
  gp=np.diff(np.r_[0.,p]);gn=np.diff(np.r_[0.,n]);return float(np.sum(gp*(n-.5*gn))/(p[-1]*n[-1]))
 rng=np.random.default_rng(seed);diff=[]
 for _ in range(nboot):
  c=np.bincount(rng.integers(0,len(units),len(units)),minlength=len(units));w=c[inv]
  if not np.any(w*(y==0)) or not np.any(w*(y==1)):continue
  diff.append(np.mean([auc(a,c)-auc(b,c) for a,b in prepared]))
 if len(diff)<.95*nboot:raise ValueError('More than 5% invalid bootstrap replicates')
 return tuple(float(v) for v in np.quantile(diff,[.025,.975])),len(diff)

def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(root,nboot,bootstrap_seed):
 out=root/'cross_model_matched_summary';out.mkdir(parents=True,exist_ok=True)
 records={};missing=[];errors=[];rows=[];refs={}
 for task,arms in ARMS.items():
  for model in BACKBONES+RECENT:
   for arm in arms:
    for seed in range(3):
     d=base_path(root,task,model,arm,seed);required=('best_val.pt','last.pt','protocol.json','history.json','test_scores.npz','test_metrics.json')
     absent=[x for x in required if not (d/x).is_file()]
     if absent:missing.append({'task':task,'model':model,'arm':arm,'seed':seed,'files':absent});continue
     try:
      m=json.loads((d/'test_metrics.json').read_text());
      with np.load(d/'test_scores.npz',allow_pickle=False) as z:x={k:z[k] for k in z.files}
      y=x['y'];identity=('window_id',) if task=='single' else ('source','case')
      for key in ('y','mmsi','region',*identity,*VIEWS[task]):
       if key not in x or len(x[key])!=len(y):raise ValueError('missing/mismatched '+key)
      if len(y)!=SIZES[task]:raise ValueError(f'Expected {SIZES[task]} rows; got {len(y)}')
      keys=[tuple(str(x[k][i]) for k in identity) for i in range(len(y))]
      if len(set(keys))!=len(keys):raise ValueError('duplicate window identity')
      if task not in refs:refs[task]=(keys,{k:x[k] for k in ('y','mmsi','region')})
      refkeys,refmeta=refs[task];idx={key:i for i,key in enumerate(keys)}
      if set(refkeys)!=set(keys):raise ValueError('window identity set differs')
      order=np.asarray([idx[key] for key in refkeys]);x={k:v[order] for k,v in x.items()}
      for key in ('y','mmsi','region'):
       if not np.array_equal(refmeta[key],x[key]):raise ValueError('metadata differs: '+key)
      for view in VIEWS[task]:
       s=x[view];v=m['views'][view]
       if not np.isfinite(s).all():raise ValueError('nonfinite scores')
       ap,auc=average_precision_score(y[order],s),roc_auc_score(y[order],s)
       if abs(ap-v['AP'])>1e-8 or abs(auc-v['AUROC'])>1e-8:raise ValueError('saved metric differs from predictions')
       thresholds=m['val_thresholds']
       rows.append({'task':task,'model':model,'arm':arm,'seed':seed,'view':view,'samples':len(y),'ships':len(np.unique(x['mmsi'])),'AP':ap,'AUROC':auc,
        'FAR_1pct':v['FAR@1%'],'Recall_1pct':v['Recall@1%'],'FAR_5pct':v['FAR@5%'],'Recall_5pct':v['Recall@5%'],
        'threshold_1pct':thresholds['1%'],'threshold_5pct':thresholds['5%']})
      records[(task,model,arm,seed)]=x
     except Exception as e:errors.append({'task':task,'model':model,'arm':arm,'seed':seed,'error':str(e)})
 expected=2*12*3*3
 state={'expected':expected,'completed_valid':len(records),'missing':missing,'errors':errors}
 (out/'progress.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf8')
 if missing or errors or len(records)!=expected:
  print(f'Awaiting complete validated result set: {len(records)}/{expected}, missing {len(missing)}, errors {len(errors)}',flush=True)
  return False
 pairs=[];summaries=[]
 for task,arms in ARMS.items():
  baseline,ordinary,reliable=arms; y=refs[task][1]['y'];mmsi=refs[task][1]['mmsi']
  for model in BACKBONES+RECENT:
   for view in VIEWS[task]:
    for arm in arms:
     ap=np.array([average_precision_score(y,records[(task,model,arm,s)][view]) for s in range(3)])
     auc=np.array([roc_auc_score(y,records[(task,model,arm,s)][view]) for s in range(3)])
     summaries.append({'task':task,'model':model,'arm':arm,'view':view,'AP_mean':ap.mean(),'AP_seed_sd':ap.std(ddof=1),'AUROC_mean':auc.mean(),'AUROC_seed_sd':auc.std(ddof=1)})
    for comparator in (baseline,ordinary):
     aa=[records[(task,model,reliable,s)][view] for s in range(3)]
     bb=[records[(task,model,comparator,s)][view] for s in range(3)]
     ci=paired_bootstrap(y,aa,bb,mmsi,nboot,bootstrap_seed);auc_bounds,valid=auc_ci(y,aa,bb,mmsi,nboot,bootstrap_seed)
     delta_auc=np.mean([roc_auc_score(y,a)-roc_auc_score(y,b) for a,b in zip(aa,bb)])
     pairs.append({'task':task,'model':model,'view':view,'comparison':reliable+'-'+comparator,'delta_AP':ci['delta_AP'],'AP_lower95':ci['lower95'],'AP_upper95':ci['upper95'],
      'delta_AUROC':delta_auc,'AUROC_lower95':auc_bounds[0],'AUROC_upper95':auc_bounds[1],'mmsi_clusters':ci['mmsi_clusters'],'bootstrap_valid':valid})
 write_csv(out/'每种子指标.csv',rows);write_csv(out/'三种子平均.csv',summaries);write_csv(out/'MMSI聚类配对区间.csv',pairs)
 report={'completed_runs':len(records),'metrics_rows':len(rows),'pair_rows':len(pairs),'bootstraps':nboot,
  'identity_check':'all model and arm sample IDs, label, target MMSI, region matched after reordering',
  'ci':'paired target-MMSI cluster bootstrap on mean of 3 seed differences; does not model seed population uncertainty',
  'training_scope':'common AIS inputs/labels and BCE training, but recent methods retain their native auxiliary objectives; CARLA has two phases',
  'evaluation_scope':'previously exposed test; window-level offline resampled inputs; target MMSI isolation',
  'causal_attribution':'reliable minus raw changes gate and field jointly; reliable minus ordinary compares field reliability under shared gate'}
 (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8');print('COMPLETED',json.dumps(report,ensure_ascii=False),flush=True)
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/home/ps/AIS_published_detectors_20261107');p.add_argument('--nboot',type=int,default=2000);p.add_argument('--seed',type=int,default=1729);p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=60);a=p.parse_args()
 while True:
  if run(Path(a.root),a.nboot,a.seed) or not a.watch:return
  time.sleep(a.interval)
if __name__=='__main__':main()
