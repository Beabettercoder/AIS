"""One locked evaluation of accepted source quartets, and paired MMSI bootstrap."""
from pathlib import Path
import json,sys,hashlib,csv
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');PRIOR=ROOT/'relation_robust_v1_20261027';OUT=ROOT/'relation_test_v1_20261027'
sys.path.insert(0,str(PRIOR));import run_relation as rr
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=json.loads((OUT/'freeze_lock.json').read_text())
for p,h in lock['files'].items():assert sha(p)==h,p
manifest=json.loads((OUT/'test_manifest.json').read_text());samples=manifest['samples'];assert len(samples)>0
orig=rr.weights
def absweights(*args):
 valid,wabs,wmix,e,rel=orig(*args);return valid,wabs,wabs,e,rel

class Model(nn.Module):
 def __init__(self,arch):
  super().__init__();self.arch=arch;self.body=nn.GRU(7,16,batch_first=True) if arch=='gru' else nn.Sequential(nn.Conv1d(7,16,3,padding=2),nn.ReLU(),nn.Conv1d(16,16,3,padding=2),nn.ReLU());self.out=nn.Linear(16,1)
 def forward(self,x,l):
  if self.arch=='gru':
   h,_=self.body(x);return self.out(h[torch.arange(len(l),device=l.device),l-1]).squeeze(-1)
  h=x.transpose(1,2);h=self.body[1](self.body[0](h)[:,:,:x.shape[1]]);h=self.body[3](self.body[2](h)[:,:,:x.shape[1]]);mask=(torch.arange(x.shape[1],device=x.device)[None,:]<l[:,None]).float();return self.out((h*mask[:,None,:]).sum(2)/l[:,None]).squeeze(-1)

views=['base','existing_speed','high_speed'];modes=['target','mean','median','mixed_median','absolute_median'];rows={v:[] for v in views};features={(v,m):[] for v in views for m in modes};maxlen=0
for item in samples:
 for view,key,hashkey in [('base','file','sha256'),('existing_speed','existing_file','existing_sha256'),('high_speed','high_file','high_sha256')]:
  p=OUT/'cases'/item[key];assert sha(p)==item[hashkey]
  with np.load(p,allow_pickle=False) as z:obs=rr.observed(z)
  for case in range(4):
   rr.weights=orig;st,f=rr.extract(obs,case)
   rr.weights=absweights;st_abs,f_abs=rr.extract(obs,case)
   st['absolute_weighted_median']=st_abs['mixed_weighted_median'];st['absolute_report_weighted_median']=st_abs['mixed_report_weighted_median']
   row={'source':item['window_id'],'mmsi':item['mmsi'],'region':item['region'],'case':case,'y':int(case in (1,2)),**st};rows[view].append(row)
   for mode in modes:
    x=f_abs['mixed_median'] if mode=='absolute_median' else f[mode]
    features[(view,mode)].append(x);maxlen=max(maxlen,len(x))
rr.weights=orig

def metrics(y,score,region):
 y=np.asarray(y);score=np.asarray(score);region=np.asarray(region)
 d={g:float(average_precision_score(y[region==g],score[region==g])) for g in sorted(set(region))}
 return {'pooled_AP':float(average_precision_score(y,score)),'pooled_AUROC':float(roc_auc_score(y,score)),'equal_region_AP':float(np.mean(list(d.values()))),'per_region_AP':d}

stats_names=['target_only','neighbor_only','mean','median','absolute_weighted_mean','absolute_weighted_median','mixed_weighted_median','mixed_report_weighted_median','absolute_report_weighted_median']
summary={'scope':'new controlled test quartets from previously pilot-exposed source test; no natural ground truth; complete window offline','test_manifest_sha256':sha(OUT/'test_manifest.json'),'lock_sha256':sha(OUT/'freeze_lock.json'),'sources':len(samples),'test_mmsi':manifest['test_target_mmsi'],'statistics':{},'neural':{},'bootstrap':{}}
for view in views:
 y=np.array([r['y'] for r in rows[view]]);region=np.array([r['region'] for r in rows[view]])
 summary['statistics'][view]={m:metrics(y,[r[m] for r in rows[view]],region) for m in stats_names}
with (OUT/'test_statistics.csv').open('x',newline='',encoding='utf8') as fh:
 writer=csv.DictWriter(fh,fieldnames=['view']+list(rows['base'][0]));writer.writeheader()
 for view in views:
  for row in rows[view]:writer.writerow({'view':view,**row})
torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');scores={};neuralrows=[]
for arch in ('tcn','gru'):
 for mode in modes:
  for seed in (0,1,2):
   checkpoint=(OUT if mode=='absolute_median' else PRIOR)/f'model_{arch}_{mode}_seed{seed}.pt'
   assert str(checkpoint) in lock['files'] and sha(checkpoint)==lock['files'][str(checkpoint)]
   c=torch.load(checkpoint,map_location='cpu',weights_only=False);model=Model(arch).to(device);model.load_state_dict(c['state_dict']);model.eval();mu=c['mean'];sd=c['std']
   for view in views:
    lst=features[(view,mode)];X=np.zeros((len(lst),maxlen,7),np.float32);L=np.array([len(x) for x in lst]);
    for i,x in enumerate(lst):X[i,:len(x)]=x
    preds=[]
    with torch.no_grad():
     for batch in np.array_split(np.arange(len(lst)),int(np.ceil(len(lst)/64))):
      yhat=model(torch.tensor((X[batch]-mu)/sd,device=device),torch.tensor(L[batch],device=device));preds.extend(torch.sigmoid(yhat).cpu().numpy().tolist())
    scores[(arch,mode,seed,view)]=np.array(preds);ys=[x['y'] for x in rows[view]];reg=[x['region'] for x in rows[view]]
    summary['neural'].setdefault(arch,{}).setdefault(mode,{}).setdefault(view,[]).append({'seed':seed,**metrics(ys,preds,reg)})
    for row,s in zip(rows[view],preds):neuralrows.append({'view':view,'architecture':arch,'mode':mode,'seed':seed,'source':row['source'],'mmsi':row['mmsi'],'region':row['region'],'case':row['case'],'y':row['y'],'score':s})
with (OUT/'test_neural_predictions.csv').open('x',newline='',encoding='utf8') as fh:
 writer=csv.DictWriter(fh,fieldnames=list(neuralrows[0]));writer.writeheader();writer.writerows(neuralrows)

subjects=np.array([r['mmsi'] for r in rows['base']]);unique=np.unique(subjects);groups={m:np.flatnonzero(subjects==m) for m in unique};rng=np.random.default_rng(20261027)
comparisons=[]
for view in views:
 base=rows[view];y=np.array([r['y'] for r in base]);med=np.array([r['median'] for r in base])
 for name in ['absolute_weighted_median','mixed_report_weighted_median','absolute_report_weighted_median']:
  comparisons.append((f'stat:{view}:{name}-median',y,np.array([r[name] for r in base]),med))
 for arch in ('tcn','gru'):
  for mode in ('absolute_median','mixed_median'):
   for seed in (0,1,2):comparisons.append((f'neural:{view}:{arch}:{mode}-median:seed{seed}',y,scores[(arch,mode,seed,view)],scores[(arch,'median',seed,view)]))
boots={name:[] for name,*_ in comparisons}
for b in range(2000):
 chosen=rng.choice(unique,size=len(unique),replace=True);ix=np.concatenate([groups[m] for m in chosen])
 for name,y,a,c in comparisons:boots[name].append(float(average_precision_score(y[ix],a[ix])-average_precision_score(y[ix],c[ix])))
for name,y,a,c in comparisons:
 z=np.array(boots[name]);summary['bootstrap'][name]={'difference':float(average_precision_score(y,a)-average_precision_score(y,c)),'ci95_percentile':np.quantile(z,[.025,.975]).tolist(),'draws':len(z),'resampling_unit':'target MMSI'}
(OUT/'test_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
(OUT/'test_bootstrap_draws.json').write_text(json.dumps(boots,ensure_ascii=False),encoding='utf8')
print('EVALUATION_COMPLETE',len(samples),len(unique),flush=True)
for view in views:
 print(view,{k:round(summary['statistics'][view][k]['pooled_AP'],4) for k in stats_names},flush=True)
 for arch in ('tcn','gru'):
  print(arch,{m:round(np.mean([r['pooled_AP'] for r in summary['neural'][arch][m][view]]),4) for m in modes},flush=True)
