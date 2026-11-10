"""Evaluate one frozen GRU checkpoint under two documented cuDNN settings."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');STUDY=ROOT/'relation_dual_pressure_v2_20261027/model_study';OUT=ROOT/'relation_independent_audit_20261027'
RUN=STUDY/'runs/gru/base/median/seed1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
class Model(nn.Module):
 def __init__(self):super().__init__();self.body=nn.GRU(7,16,batch_first=True);self.out=nn.Linear(16,1)
 def forward(self,x,l):
  h,_=self.body(x);return self.out(h[torch.arange(len(l),device=l.device),l-1]).squeeze(-1)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
ck=torch.load(RUN/'checkpoint.pt',map_location='cpu',weights_only=False)
with np.load(STUDY/'features_test.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
with np.load(RUN/'test_predictions.npz',allow_pickle=False) as z:saved={k:z[k] for k in z.files}
y=data['y'];L=data['length'];assert np.array_equal(y,saved['y'])
views={}
for view in ('base','natural','suffix'):
 X=((data[view+'__median']-ck['mean'])/ck['std']).astype(np.float32);runs={}
 for label,tf32,det in [('test_default',True,False),('validation_backend',False,True)]:
  torch.backends.cudnn.allow_tf32=tf32;torch.backends.cudnn.deterministic=det;torch.use_deterministic_algorithms(det)
  model=Model().cuda();model.load_state_dict(ck['state_dict']);model.eval();pred=[]
  with torch.no_grad():
   for i in range(0,len(X),256):
    p=torch.sigmoid(model(torch.tensor(X[i:i+256],device='cuda'),torch.tensor(L[i:i+256],dtype=torch.long,device='cuda')))
    pred.extend(p.cpu().numpy().tolist())
  runs[label]=np.asarray(pred)
 a=runs['test_default'];b=runs['validation_backend'];d=np.abs(a-b)
 assert np.array_equal(a,saved[view])
 ap0=float(average_precision_score(y,a));ap1=float(average_precision_score(y,b))
 views[view]={'pooled_AP_original_test_default':ap0,'pooled_AP_validation_backend':ap1,'pooled_AP_difference_validation_minus_original':ap1-ap0,
   'pooled_AUROC_original_test_default':float(roc_auc_score(y,a)), 'pooled_AUROC_validation_backend':float(roc_auc_score(y,b)),
   'max_probability_absolute_difference':float(d.max()),'mean_probability_absolute_difference':float(d.mean()),
   'original_saved_max_probability_difference':float(np.max(np.abs(a-saved[view])))}
out={'status':'complete','scope':'posthoc backend numerical sensitivity on previously exposed test, one pre-existing GRU base/median seed1 checkpoint; not a new model or selection',
 'test_default':'cuDNN TF32=True, cudnn.deterministic=False, deterministic_algorithms=False',
 'validation_backend':'cuDNN TF32=False, cudnn.deterministic=True, deterministic_algorithms=True',
 'pytorch_version':torch.__version__,'cudnn_version':torch.backends.cudnn.version(),'device':torch.cuda.get_device_name(0),
 'checkpoint_sha256':sha(RUN/'checkpoint.pt'),'original_prediction_sha256':sha(RUN/'test_predictions.npz'),
 'features_sha256':sha(STUDY/'features_test.npz'),'views':views,'script_sha256':sha(Path(__file__))}
with (OUT/'backend_precision_probe.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps(out,ensure_ascii=False,indent=2))
