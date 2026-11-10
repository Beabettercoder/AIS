"""Evaluate supplementary frozen report-weighted neural context, all sources."""
from pathlib import Path
import json,csv,hashlib,sys
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');OUT=ROOT/'relation_test_expanded_v1_20261027';sys.path.insert(0,str(OUT))
import report_context_common as c
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=json.loads((OUT/'report_context_lock.json').read_text())
for p,h in lock['files'].items():assert sha(p)==h,p
manifest=json.loads((OUT/'test_manifest.json').read_text());items=manifest['samples'];views=['base','existing_speed','high_speed'];features={v:[] for v in views};rows={v:[] for v in views};maxlen=0
for item in items:
 for view,key,hkey in [('base','file','sha256'),('existing_speed','existing_file','existing_sha256'),('high_speed','high_file','high_sha256')]:
  p=OUT/'cases'/item[key];assert sha(p)==item[hkey]
  with np.load(p,allow_pickle=False) as z:o=c.rr.observed(z)
  for k in range(4):
   x=c.extract(o,k);features[view].append(x);maxlen=max(maxlen,len(x));rows[view].append({'source':item['window_id'],'mmsi':item['mmsi'],'region':item['region'],'case':k,'y':int(k in (1,2))})
baseline={}
with (OUT/'test_neural_predictions.csv').open(encoding='utf8',newline='') as f:
 for r in csv.DictReader(f):
  if r['mode']=='median':baseline[(r['view'],r['architecture'],int(r['seed']),r['source'],int(r['case']))]=float(r['score'])
scores={};pred=[];result={'scope':'supplementary report-level weighted neural models, test previously examined during parallel work; exploratory','manifest_sha256':sha(OUT/'test_manifest.json'),'lock_sha256':sha(OUT/'report_context_lock.json'),'models':{},'bootstrap':{}}
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
for arch in ('tcn','gru'):
 for seed in (0,1,2):
  p=OUT/f'model_{arch}_report_weighted_median_seed{seed}.pt';ck=torch.load(p,map_location='cpu',weights_only=False);model=c.Model(arch).to(device);model.load_state_dict(ck['state_dict']);model.eval();mu=ck['mean'];sd=ck['std']
  for view in views:
   lst=features[view];X=np.zeros((len(lst),maxlen,7),np.float32);length=np.array([len(x) for x in lst]);
   for i,x in enumerate(lst):X[i,:len(x)]=x
   out=[]
   with torch.no_grad():
    for batch in np.array_split(np.arange(len(lst)),int(np.ceil(len(lst)/64))):out.extend(torch.sigmoid(model(torch.tensor((X[batch]-mu)/sd,device=device),torch.tensor(length[batch],device=device))).cpu().numpy().tolist())
   out=np.array(out);ys=np.array([r['y'] for r in rows[view]]);reg=np.array([r['region'] for r in rows[view]]);base=np.array([baseline[(view,arch,seed,r['source'],r['case'])] for r in rows[view]])
   s={'seed':seed,'pooled_AP':float(average_precision_score(ys,out)),'pooled_AUROC':float(roc_auc_score(ys,out)),'per_region_AP':{g:float(average_precision_score(ys[reg==g],out[reg==g])) for g in sorted(set(reg))},'median_baseline_pooled_AP':float(average_precision_score(ys,base))}
   s['equal_observed_region_AP']=float(np.mean(list(s['per_region_AP'].values())));result['models'].setdefault(arch,{}).setdefault(view,[]).append(s);scores[(arch,seed,view)]=(ys,out,base)
   for r,v in zip(rows[view],out):pred.append({'architecture':arch,'seed':seed,'view':view,**r,'score':float(v)})
with (OUT/'test_report_context_predictions.csv').open('x',newline='',encoding='utf8') as fh:
 w=csv.DictWriter(fh,fieldnames=list(pred[0]));w.writeheader();w.writerows(pred)
subjects=np.array([r['mmsi'] for r in rows['base']]);unique=np.unique(subjects);groups={u:np.flatnonzero(subjects==u) for u in unique};rng=np.random.default_rng(20261027);boots={k:[] for k in scores}
for _ in range(2000):
 ix=np.concatenate([groups[m] for m in rng.choice(unique,size=len(unique),replace=True)])
 for key,(y,a,b) in scores.items():boots[key].append(float(average_precision_score(y[ix],a[ix])-average_precision_score(y[ix],b[ix])))
for (arch,seed,view),(y,a,b) in scores.items():
 z=np.array(boots[(arch,seed,view)]);result['bootstrap'][f'{view}:{arch}:seed{seed}']={'difference':float(average_precision_score(y,a)-average_precision_score(y,b)),'ci95':np.quantile(z,[.025,.975]).tolist(),'draws':len(z),'cluster':'target MMSI'}
(OUT/'test_report_context_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print('REPORT_CONTEXT_COMPLETE')
for arch in ('tcn','gru'):
 for view in views:print(arch,view,'AP',np.mean([r['pooled_AP'] for r in result['models'][arch][view]]))
