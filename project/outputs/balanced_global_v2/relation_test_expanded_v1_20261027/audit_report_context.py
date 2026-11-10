"""Replay supplementary report-weighted checkpoints and independently recompute AP."""
from pathlib import Path
import sys,csv,json,hashlib,numpy as np,torch
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');OUT=ROOT/'relation_test_expanded_v1_20261027';sys.path.insert(0,str(OUT));import report_context_common as c
def ap(y,s):
 y=np.asarray(y,int);s=np.asarray(s,float);ix=np.argsort(-s,kind='stable');yy=y[ix];ss=s[ix];end=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1];tp=np.cumsum(yy)[end];return float(np.sum(np.diff(np.r_[0,tp/tp[-1]])*tp/(end+1)))
def auc(y,s):
 y=np.asarray(y,int);s=np.asarray(s,float);p=s[y==1];n=s[y==0];return float(np.mean((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])))
d=json.loads((OUT/'test_manifest.json').read_text());res=json.loads((OUT/'test_report_context_results.json').read_text());lock=json.loads((OUT/'report_context_lock.json').read_text())
for p,h in lock['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
with (OUT/'test_report_context_predictions.csv').open(encoding='utf8',newline='') as f:pred=list(csv.DictReader(f))
features={};counts=0
for view,key in [('base','file'),('existing_speed','existing_file'),('high_speed','high_file')]:
 for item in d['samples']:
  with np.load(OUT/'cases'/item[key],allow_pickle=False) as z:o=c.rr.observed(z)
  for k in range(4):features[(view,item['window_id'],k)]=c.extract(o,k);counts+=1
max_error=0;max_score=0;dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(1)
for arch in ('tcn','gru'):
 for seed in (0,1,2):
  ck=torch.load(OUT/f'model_{arch}_report_weighted_median_seed{seed}.pt',map_location='cpu',weights_only=False);model=c.Model(arch).to(dev);model.load_state_dict(ck['state_dict']);model.eval()
  for view in ('base','existing_speed','high_speed'):
   rows=[r for r in pred if r['architecture']==arch and int(r['seed'])==seed and r['view']==view];y=[int(r['y']) for r in rows];score=np.array([float(r['score']) for r in rows]);expected=res['models'][arch][view][seed];max_error=max(max_error,abs(ap(y,score)-expected['pooled_AP']),abs(auc(y,score)-expected['pooled_AUROC']))
   xx=[features[(view,r['source'],int(r['case']))] for r in rows];lens=np.array([len(x) for x in xx]);X=np.zeros((len(xx),max(lens),7),np.float32)
   for i,x in enumerate(xx):X[i,:len(x)]=x
   out=[]
   with torch.no_grad():
    for ids in np.array_split(np.arange(len(X)),int(np.ceil(len(X)/64))):out.extend(torch.sigmoid(model(torch.tensor((X[ids]-ck['mean'])/ck['std'],device=dev),torch.tensor(lens[ids],device=dev))).cpu().numpy().tolist())
   max_score=max(max_score,float(np.max(np.abs(np.array(out)-score))))
assert max_error<1e-12 and max_score<1e-5
o={'status':'passed','recomputed_observed_feature_cases':counts,'prediction_rows':len(pred),'max_manual_metric_abs_error':max_error,'max_checkpoint_replay_abs_error':max_score,'replay_device':str(dev),'scope':'supplementary model and post hoc sample expansion; not a pristine test'}
(OUT/'audit_report_context.json').write_text(json.dumps(o,indent=2),encoding='utf8');print(json.dumps(o,indent=2))
