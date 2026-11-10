"""Reproduce previous neural train protocol with w_abs weighted-median context.

Only pre-existing train/validation quartets are opened. Test sources are not read.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score

ROOT = Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OLD = ROOT/'original_method_and_relation_v1_20261026/relation'
PRIOR = ROOT/'relation_robust_v1_20261027'
OUT = ROOT/'relation_test_expanded_v1_20261027'
sys.path.insert(0,str(PRIOR)); import run_relation as rr
import report_context_common as common
Model=common.Model

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
manifest=json.loads((OLD/'all_sources_v1/relation_manifest.json').read_text())
features=[]; labels=[];splits=[];metadata=[]
for it in manifest['samples']:
    path=OLD/'all_sources_v1'/it['file']; assert sha(path)==it['sha256']
    with np.load(path,allow_pickle=False) as z:
        obs=rr.observed(z);y=z['y'].astype(int); split=str(z['split']);source=str(z['source_window_id']);mmsi=str(z['target_mmsi'])
    assert split in ('train','val')
    for k in range(4):
        f=common.extract(obs,k);features.append(f);labels.append(y[k]);splits.append(split);metadata.append([source,mmsi,k])
maxlen=max(map(len,features));x=np.zeros((len(features),maxlen,7),np.float32);length=np.array([len(a) for a in features]);
for i,a in enumerate(features):x[i,:len(a)]=a
splits=np.array(splits);labels=np.array(labels);tr=np.flatnonzero(splits=='train');val=np.flatnonzero(splits=='val');mask=np.arange(maxlen)[None,:]<length[:,None];valid=x[tr][mask[tr]];mu=valid.mean(0);sd=valid.std(0);sd[sd<1e-6]=1
assert len(tr)==1092 and len(val)==448
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');runs=[];pred=[]
for arch in ('tcn','gru'):
 for seed in (0,1,2):
  torch.manual_seed(seed);np.random.seed(seed);model=Model(arch).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(seed);losslog=[]
  for epoch in range(20):
   model.train(); losses=[]
   for ids in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/64))):
    X=torch.tensor((x[ids]-mu)/sd,device=device);L=torch.tensor(length[ids],device=device);Y=torch.tensor(labels[ids],dtype=torch.float32,device=device)
    opt.zero_grad();out=model(X,L);loss=nn.functional.binary_cross_entropy_with_logits(out,Y);loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
   losslog.append(float(np.mean(losses)))
  model.eval();path=OUT/f'model_{arch}_report_weighted_median_seed{seed}.pt'
  torch.save({'state_dict':model.state_dict(),'mean':mu,'std':sd,'architecture':arch,'mode':'report_weighted_median','seed':seed,'loss':losslog},path)
  with torch.no_grad():
   ids=val;score=torch.sigmoid(model(torch.tensor((x[ids]-mu)/sd,device=device),torch.tensor(length[ids],device=device))).cpu().numpy()
  for i,s in zip(ids,score):pred.append({'architecture':arch,'seed':seed,'source':metadata[i][0],'mmsi':metadata[i][1],'case':metadata[i][2],'y':int(labels[i]),'score':float(s)})
  runs.append({'arch':arch,'seed':seed,'checkpoint':path.name,'sha256':sha(path),'parameters':sum(q.numel() for q in model.parameters()),'val_pooled_ap':float(average_precision_score(labels[val],score)),'loss':losslog})
  print('TRAINED',arch,seed,runs[-1]['val_pooled_ap'],flush=True)
(OUT/'report_context_train_results.json').write_text(json.dumps({'scope':'train/val only, no test', 'runs':runs,'val_predictions':pred,'train_sources':len(tr)//4,'val_sources':len(val)//4,'source_manifest_sha256':sha(OLD/'all_sources_v1/relation_manifest.json')},indent=2),encoding='utf8')
