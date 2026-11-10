"""Recheck saved relation-model metrics, paired bootstrap and sampled GPU inference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');RUN=ROOT/'relation_dual_pressure_v2_20261027';STUDY=RUN/'model_study';OUT=ROOT/'relation_independent_audit_20261027'
VIEWS=('base','natural','suffix');MODES=('target','mean','median','absolute_median','absolute_report_median','mixed_report_median');ARCH=('tcn','gru');SEEDS=(0,1,2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def aps(scores,y,w):
 order=np.argsort(-scores,kind='stable');scores=scores[order];y=y[order];w=w[:,order]
 starts=np.r_[0,np.flatnonzero(scores[1:]!=scores[:-1])+1]
 mass=np.add.reduceat(w,starts,axis=1);pos=np.add.reduceat(w*y[None,:],starts,axis=1)
 cm=np.cumsum(mass,axis=1);cp=np.cumsum(pos,axis=1)
 return (np.divide(cp,cm,out=np.zeros_like(cp),where=cm>0)*pos).sum(axis=1)/np.maximum(cp[:,-1],1)
class IndependentModel(nn.Module):
 def __init__(self,kind):
  super().__init__();self.arch=kind
  self.body=nn.GRU(7,16,batch_first=True) if kind=='gru' else nn.Sequential(nn.Conv1d(7,16,3,padding=2),nn.ReLU(),nn.Conv1d(16,16,3,padding=2),nn.ReLU())
  self.out=nn.Linear(16,1)
 def forward(self,x,l):
  if self.arch=='gru':
   h,_=self.body(x);return self.out(h[torch.arange(len(l),device=l.device),l-1]).squeeze(-1)
  h=x.transpose(1,2)
  h=self.body[1](self.body[0](h)[:,:,:x.shape[1]])
  h=self.body[3](self.body[2](h)[:,:,:x.shape[1]])
  mask=(torch.arange(x.shape[1],device=x.device)[None,:]<l[:,None]).float()
  return self.out((h*mask[:,None,:]).sum(2)/l[:,None]).squeeze(-1)

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
code=json.loads((STUDY/'code_lock.json').read_text(encoding='utf8'));lock=json.loads((STUDY/'checkpoint_lock.json').read_text(encoding='utf8'))
assert code['dataset_manifest_sha256']==sha(RUN/'dataset_manifest_combined.json') and code['script_sha256']==sha(STUDY/'model_study.py')
for p,h in code['dependency_hashes'].items():assert sha(p)==h
assert lock['count']==108 and lock['code_lock_sha256']==sha(STUDY/'code_lock.json') and lock['source_manifest_sha256']==sha(RUN/'dataset_manifest_combined.json')
freezetime=datetime.fromisoformat(lock['frozen_utc']);assert freezetime<datetime.fromtimestamp((STUDY/'features_test.npz').stat().st_mtime,timezone.utc)
features={}
for split in ('train','val','test'):
 p=STUDY/f'features_{split}.npz';m=json.loads((STUDY/f'features_{split}_manifest.json').read_text(encoding='utf8'))
 assert m['feature_file_sha256']==sha(p) and m['dataset_manifest_sha256']==sha(RUN/'dataset_manifest_combined.json') and m['code_lock_sha256']==sha(STUDY/'code_lock.json')
 with np.load(p,allow_pickle=False) as z:features[split]={k:z[k] for k in z.files}
for a,b in [('train','val'),('train','test'),('val','test')]:
 assert not set(features[a]['mmsi'])&set(features[b]['mmsi'])
 assert not set(features[a]['source'])&set(features[b]['source'])
for split,z in features.items():
 assert len(z['y'])%4==0 and np.array_equal(z['y'].reshape(-1,4),np.tile([0,1,1,0],(len(z['y'])//4,1)))
 for v in VIEWS:
  a=z[v+'__target'];assert np.array_equal(a[0::4],a[1::4]) and np.array_equal(a[2::4],a[3::4])
  for mode in MODES[1:]:
   a=z[v+'__'+mode][...,4:];assert np.array_equal(a[0::4],a[2::4]) and np.array_equal(a[1::4],a[3::4])
train=features['train'];valid=np.arange(20)[None,:]<train['length'][:,None];expected_scaler={}
for exposure in VIEWS:
 for mode in MODES:
  x=train[exposure+'__'+mode][valid];mu=x.mean(0);std=x.std(0);std[std<1e-6]=1;expected_scaler[(exposure,mode)]=(mu,std)
expected_orders={}
for seed in SEEDS:
 rng=np.random.default_rng(seed);expected_orders[seed]=[hashlib.sha256(rng.permutation(len(train['y'])).astype('<i8').tobytes()).hexdigest() for _ in range(20)]
replay_set={('base','target',0),('base','median',1),('natural','absolute_median',2),('suffix','absolute_report_median',0),('suffix','mixed_report_median',1),('natural','mean',2)}
maxmetric=0.;maxscaler=0.;maxreplay=0.;runs=0;score_arrays={'val':{},'test':{}};replays=[];point={}
for arch in ARCH:
 dev=torch.device('cuda:0' if arch=='tcn' else 'cuda:1')
 for exposure in VIEWS:
  for mode in MODES:
   for seed in SEEDS:
    p=STUDY/'runs'/arch/exposure/mode/f'seed{seed}';rec=json.loads((p/'run.json').read_text(encoding='utf8'));te=json.loads((p/'test_metrics.json').read_text(encoding='utf8'))
    checkpoint=p/'checkpoint.pt';assert sha(checkpoint)==lock['checkpoints'][str(checkpoint)]==rec['checkpoint_sha256']==te['checkpoint_sha256']
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['epochs']==20 and len(rec['loss_per_epoch'])==20
    assert rec['batch_order_sha256_per_epoch']==expected_orders[seed]
    mu,sd=expected_scaler[(exposure,mode)];maxscaler=max(maxscaler,float(np.max(np.abs(mu-ck['mean']))),float(np.max(np.abs(sd-ck['std']))));assert np.array_equal(mu,ck['mean']) and np.array_equal(sd,ck['std'])
    assert rec['parameters']==(1153 if arch=='tcn' else 1217)
    for split,rep in [('val',rec['val_metrics']),('test',te['metrics'])]:
     filename=p/f'{split}_predictions.npz';assert sha(filename)==(rec['val_predictions_sha256'] if split=='val' else te['test_predictions_sha256'])
     with np.load(filename,allow_pickle=False) as z:
      for k in ('source','mmsi','region','y','case'):assert np.array_equal(z[k],features[split][k])
      y=z['y'];region=z['region']
      for view in VIEWS:
       pred=z[view].astype(float);key=(arch,exposure,mode,seed,view);score_arrays[split][key]=pred
       met=rep[view];ap=float(average_precision_score(y,pred));au=float(roc_auc_score(y,pred));ma=float(np.mean([average_precision_score(y[region==r],pred[region==r]) for r in np.unique(region)]))
       for a,b in [(ap,met['pooled_AP']),(au,met['pooled_AUROC']),(ma,met['macro_region_AP'])]:maxmetric=max(maxmetric,abs(a-b));assert abs(a-b)<1e-12
       for r,v in met['regions'].items():assert abs(float(average_precision_score(y[region==r],pred[region==r]))-v)<1e-12
       if mode=='target':assert np.array_equal(pred[0::4],pred[1::4]) and np.array_equal(pred[2::4],pred[3::4])
       if (exposure,mode,seed) in replay_set:
        # Training explicitly disables cuDNN TF32 and enables deterministic
        # algorithms; the separately launched test process uses PyTorch defaults.
        training_backend=(split=='val')
        torch.backends.cudnn.allow_tf32=not training_backend
        torch.backends.cudnn.deterministic=training_backend
        torch.use_deterministic_algorithms(training_backend)
        model=IndependentModel(arch).to(dev);model.load_state_dict(ck['state_dict']);model.eval()
        X=((features[split][view+'__'+mode]-mu)/sd).astype(np.float32);L=features[split]['length'];replayed=[]
        with torch.no_grad():
         for lo in range(0,len(X),256):
          xx=torch.tensor(X[lo:lo+256],device=dev);ll=torch.tensor(L[lo:lo+256],dtype=torch.long,device=dev)
          replayed.extend(torch.sigmoid(model(xx,ll)).cpu().numpy().tolist())
        err=float(np.max(np.abs(pred-np.array(replayed))));maxreplay=max(maxreplay,err);assert err<5e-5,(arch,exposure,mode,seed,split,view,err)
        replays.append({'architecture':arch,'exposure':exposure,'mode':mode,'seed':seed,'split':split,'view':view,'max_probability_error':err})
    runs+=1
boot_err=0.;ci_err=0.;comparison_rows=0
for split in ('val','test'):
 z=features[split];y=z['y'].astype(int);ids,inv=np.unique(z['mmsi'],return_inverse=True);rng=np.random.default_rng(20261027)
 counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000).astype(np.float32);weight=counts[:,inv]
 report=json.loads((STUDY/f'bootstrap_{split}.json').read_text(encoding='utf8'));assert report['checkpoint_lock_sha256']==sha(STUDY/'checkpoint_lock.json') and report['feature_sha256']==sha(STUDY/f'features_{split}.npz')
 with np.load(STUDY/f'bootstrap_draws_{split}.npz',allow_pickle=False) as saved:
  assert np.array_equal(saved['mmsi_ids'],ids) and np.array_equal(saved['mmsi_draw_counts'],counts)
  allboot={};allpoint={}
  for key,p in score_arrays[split].items():
   name='__'.join(map(str,key));draw=aps(p,y,weight);err=float(np.max(np.abs(draw-saved[name])));boot_err=max(boot_err,err);assert err<1e-10,(split,key,err)
   allboot[key]=draw;allpoint[key]=float(average_precision_score(y,p))
  for row in report['comparisons']:
   arch,exp,view,mode=row['arch'],row['exposure'],row['view'],row['mode'];ref='median';seed=row['seed']
   if seed=='mean_fixed_3':
    delta=np.mean([allboot[(arch,exp,mode,s,view)]-allboot[(arch,exp,ref,s,view)] for s in SEEDS],axis=0)
    value=np.mean([allpoint[(arch,exp,mode,s,view)]-allpoint[(arch,exp,ref,s,view)] for s in SEEDS])
   else:
    seed=int(seed);delta=allboot[(arch,exp,mode,seed,view)]-allboot[(arch,exp,ref,seed,view)];value=allpoint[(arch,exp,mode,seed,view)]-allpoint[(arch,exp,ref,seed,view)]
   err=float(np.max(np.abs(np.quantile(delta,[.025,.975])-row['ci95'])));ci_err=max(ci_err,err);assert err<1e-10 and abs(value-row['pooled_AP_delta'])<1e-12
   comparison_rows+=1
out={'status':'passed','runs':runs,'epochs_each':20,'train_sources':len(set(features['train']['source'])),'val_sources':len(set(features['val']['source'])),
 'test_sources':len(set(features['test']['source'])),'val_test_predictions_checked':sum(len(v) for split in score_arrays.values() for v in split.values()),
 'max_metric_error':maxmetric,'max_train_scaler_error':maxscaler,'gpu_checkpoint_replays':len(replays),'max_gpu_probability_error':maxreplay,
 'bootstrap_comparisons_checked':comparison_rows,'bootstrap_draws_each':2000,'max_individual_bootstrap_error':boot_err,'max_bootstrap_CI_error':ci_err,
 'freeze_before_test_feature_utc':lock['frozen_utc'],'data_hash':sha(RUN/'dataset_manifest_combined.json'),'checkpoint_lock_sha256':sha(STUDY/'checkpoint_lock.json'),
 'backend':{'pytorch_version':torch.__version__,'cudnn_version':torch.backends.cudnn.version(),
            'validation':'cudnn.allow_tf32=False, cudnn.deterministic=True, deterministic_algorithms=True',
            'test':'cudnn.allow_tf32=True, cudnn.deterministic=False, deterministic_algorithms=False; torch defaults in separate evaluation process',
            'cuda_matmul_allow_tf32':False,'visible_gpus':'physical 6,7'},
 'scope':'independent model class for sampled GPU replay; all saved validation/test metrics and all saved cluster draws recalculated',
 'limits':'fixed historical source, three trained seeds, synthetic quartet labels, shared background and spatial/temporal contexts; initial-state hashes were not saved',
 'replays':replays,'script_sha256':sha(Path(__file__))}
with (OUT/'model_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k!='replays'},ensure_ascii=False,indent=2),flush=True)
