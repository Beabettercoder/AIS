"""Independently recompute test AP, replay locked checkpoints and check siblings."""
from pathlib import Path
import csv,json,hashlib,sys
import numpy as np,torch
from torch import nn
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');OUT=ROOT/'relation_test_expanded_v1_20261027';ORIG=ROOT/'relation_test_v1_20261027';PRIOR=ROOT/'relation_robust_v1_20261027';sys.path.insert(0,str(PRIOR));import run_relation as rr
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ap(y,s):
 y=np.asarray(y,int);s=np.asarray(s,float);ix=np.argsort(-s,kind='stable');yy=y[ix];ss=s[ix];end=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1];tp=np.cumsum(yy)[end];precision=tp/(end+1);recall=tp/tp[-1];return float(np.sum(np.diff(np.r_[0,recall])*precision))
def auc(y,s):
 y=np.asarray(y,int);s=np.asarray(s,float);pos=s[y==1];neg=s[y==0];return float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])))
class Model(nn.Module):
 def __init__(self,arch):
  super().__init__();self.arch=arch;self.body=nn.GRU(7,16,batch_first=True) if arch=='gru' else nn.Sequential(nn.Conv1d(7,16,3,padding=2),nn.ReLU(),nn.Conv1d(16,16,3,padding=2),nn.ReLU());self.out=nn.Linear(16,1)
 def forward(self,x,l):
  if self.arch=='gru':h,_=self.body(x);return self.out(h[torch.arange(len(l),device=l.device),l-1]).squeeze(-1)
  h=x.transpose(1,2);h=self.body[1](self.body[0](h)[:,:,:x.shape[1]]);h=self.body[3](self.body[2](h)[:,:,:x.shape[1]]);mask=(torch.arange(x.shape[1],device=x.device)[None,:]<l[:,None]).float();return self.out((h*mask[:,None,:]).sum(2)/l[:,None]).squeeze(-1)

z=json.loads((OUT/'test_manifest.json').read_text());prev=json.loads((ORIG/'test_manifest.json').read_text());results=json.loads((OUT/'test_results.json').read_text());samples=z['samples'];old={x['window_id']:x for x in prev['samples']};new={x['window_id']:x for x in samples};assert set(old)<=set(new)
identical=0
for wid,a in old.items():
 b=new[wid]
 for k in ('file','existing_file','high_file'):
  with np.load(ORIG/'cases'/a[k],allow_pickle=False) as one,np.load(OUT/'cases'/b[k],allow_pickle=False) as two:
   assert set(one.files)==set(two.files)
   for key in one.files:assert np.array_equal(one[key],two[key]),(wid,k,key)
 identical+=1

with (OUT/'test_statistics.csv').open(encoding='utf8',newline='') as f:stat=list(csv.DictReader(f))
with (OUT/'test_neural_predictions.csv').open(encoding='utf8',newline='') as f:pred=list(csv.DictReader(f))
views=['base','existing_speed','high_speed'];methods=['target_only','neighbor_only','mean','median','absolute_weighted_mean','absolute_weighted_median','mixed_weighted_median','mixed_report_weighted_median','absolute_report_weighted_median'];max_metric=0
for view in views:
 rows=[r for r in stat if r['view']==view];y=[int(r['y']) for r in rows]
 for m in methods:
  s=[float(r[m]) for r in rows];expected=results['statistics'][view][m];max_metric=max(max_metric,abs(ap(y,s)-expected['pooled_AP']),abs(auc(y,s)-expected['pooled_AUROC']))
for arch in ('tcn','gru'):
 for mode in ('target','mean','median','mixed_median','absolute_median'):
  for seed in (0,1,2):
   for view in views:
    rows=[r for r in pred if r['architecture']==arch and r['mode']==mode and int(r['seed'])==seed and r['view']==view];y=[int(r['y']) for r in rows];s=[float(r['score']) for r in rows];expected=results['neural'][arch][mode][view][seed];max_metric=max(max_metric,abs(ap(y,s)-expected['pooled_AP']),abs(auc(y,s)-expected['pooled_AUROC']))
assert max_metric<1e-12

original_weights=rr.weights
def absweights(*args):
 valid,wabs,wmix,e,rel=original_weights(*args);return valid,wabs,wabs,e,rel
features={};target_equal=0;cohort_equal=0;max_stat=0
for view,filekey,hkey in [('base','file','sha256'),('existing_speed','existing_file','existing_sha256'),('high_speed','high_file','high_sha256')]:
 for x in samples:
  p=OUT/'cases'/x[filekey];assert sha(p)==x[hkey]
  with np.load(p,allow_pickle=False) as raw:o=rr.observed(raw)
  for key in ['target_t','target_pos','target_sog_kn','target_cog_deg']:
   assert np.array_equal(o[key][0],o[key][1]) and np.array_equal(o[key][2],o[key][3]);target_equal+=1
  for key in ['neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg']:
   assert np.array_equal(o[key][0],o[key][2]) and np.array_equal(o[key][1],o[key][3]);cohort_equal+=1
  for case in range(4):
   rr.weights=original_weights;st,f=rr.extract(o,case);rr.weights=absweights;sa,fa=rr.extract(o,case);rr.weights=original_weights
   st['absolute_weighted_median']=sa['mixed_weighted_median'];st['absolute_report_weighted_median']=sa['mixed_report_weighted_median']
   stored=next(r for r in stat if r['view']==view and r['source']==x['window_id'] and int(r['case'])==case)
   for m in methods:max_stat=max(max_stat,abs(float(stored[m])-st[m]))
   for mode in ('target','mean','median','mixed_median','absolute_median'):features[(view,mode,x['window_id'],case)]=fa['mixed_median'] if mode=='absolute_median' else f[mode]
assert max_stat<1e-12

torch.set_num_threads(1);max_pred=0;device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for arch in ('tcn','gru'):
 for mode in ('target','mean','median','mixed_median','absolute_median'):
  for seed in (0,1,2):
   p=(OUT if mode=='absolute_median' else PRIOR)/f'model_{arch}_{mode}_seed{seed}.pt';ck=torch.load(p,map_location='cpu',weights_only=False);model=Model(arch).to(device);model.load_state_dict(ck['state_dict']);model.eval()
   for view in views:
    rows=[r for r in pred if r['architecture']==arch and r['mode']==mode and int(r['seed'])==seed and r['view']==view];xx=[features[(view,mode,r['source'],int(r['case']))] for r in rows];length=np.array([len(a) for a in xx]);T=max(length);X=np.zeros((len(xx),T,7),np.float32)
    for i,x in enumerate(xx):X[i,:len(x)]=x
    out=[]
    with torch.no_grad():
     for batch in np.array_split(np.arange(len(X)),int(np.ceil(len(X)/64))):out.extend(torch.sigmoid(model(torch.tensor((X[batch]-ck['mean'])/ck['std'],device=device),torch.tensor(length[batch],device=device))).cpu().numpy().tolist())
    out=np.array(out)
    err=float(np.max(np.abs(out-np.array([float(r['score']) for r in rows]))))
    if err>1e-5:print('REPLAY_MISMATCH',arch,mode,seed,view,err,flush=True)
    max_pred=max(max_pred,err)
assert max_pred<1e-5
out={'status':'passed','sources':len(samples),'target_mmsi':z['test_target_mmsi'],'original_sources_in_expanded':identical,'sibling_target_array_checks':target_equal,'sibling_neighbor_array_checks':cohort_equal,'scored_rows':len(stat),'neural_rows':len(pred),'max_manual_metric_abs_error':max_metric,'max_replayed_stat_error':max_stat,'max_checkpoint_replay_probability_error':max_pred,'replay_device':str(device),'limitations':'metric recompute and model replay do not prove scene independence; report shared background'}
(OUT/'independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
