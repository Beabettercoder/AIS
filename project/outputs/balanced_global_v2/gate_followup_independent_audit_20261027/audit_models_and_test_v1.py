"""Reconstruct candidate inputs and model inference; recompute test clustered contrasts."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score,roc_auc_score
from independent_gate import independent
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');SRC=ROOT/'gating_training_v2_20261027/data';BASE=ROOT/'causal_gate_main_v1_20261027';RUN=ROOT/'causal_gate_followup_v2_20261027';SCENE=ROOT/'labeling/labeled_v1/features_scene';OUT=ROOT/'gate_followup_independent_audit_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai');KNOT=1852/3600
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
class Enc(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  if kind=='gru':self.rnn=nn.GRU(12,64,batch_first=True,bidirectional=True);self.dim=256
  else:self.net=nn.Sequential(nn.Conv1d(12,64,5,padding=2),nn.GELU(),nn.Conv1d(64,64,5,padding=4,dilation=2),nn.GELU(),nn.Conv1d(64,64,5,padding=8,dilation=4),nn.GELU());self.dim=128
 def forward(self,x):
  h=self.rnn(x)[0].transpose(1,2) if self.kind=='gru' else self.net(x.transpose(1,2))
  return torch.cat([h.mean(-1),h.max(-1).values],-1)
class Core(nn.Module):
 def __init__(self,kind):super().__init__();self.enc=Enc(kind);self.bin=nn.Linear(self.enc.dim,1);self.cls=nn.Linear(self.enc.dim,5)
class Model(nn.Module):
 def __init__(self,kind):super().__init__();self.core=Core(kind)
 def forward(self,x):return self.core.bin(self.core.enc(x)).squeeze(-1)
def pack(g,w,v):
 n,T=w.shape
 return np.concatenate([g,np.zeros((n,T,2),np.float32),w[...,None],v[...,None],np.zeros((n,T,1),np.float32)],-1).astype(np.float32)

plan=json.loads((RUN/'frozen_test_plan.json').read_text(encoding='utf8'));selection=json.loads((RUN/'development_selection.json').read_text(encoding='utf8'));pars=json.loads((RUN/'fitted_parameters.json').read_text(encoding='utf8'))
assert plan['status']=='frozen_before_test' and plan['candidate']==selection['candidate'] and plan['selection_sha256']==sha(RUN/'development_selection.json')
for p,h in plan['locks'].items():assert sha(p)==h,p
assert plan['builder_sha256']==sha(RUN/'build_candidates.py') and plan['evaluator_sha256']==sha(RUN/'evaluate_locked_test.py')
assert datetime.fromisoformat(plan['frozen_utc'])<datetime.fromtimestamp((RUN/'test_results/predictions.npz').stat().st_mtime,timezone.utc)
parts={s:[] for s in ('train','val')}
for r in REGIONS:
 for split in parts:
  with np.load(SRC/r/f'{split}.npz',allow_pickle=False) as z:d={k:z[k] for k in ('window_id','mmsi','region','y','mask','w_self','kinematic_valid')}
  with np.load(RUN/'inputs'/r/f'{split}.npz',allow_pickle=False) as z:g=z[selection['candidate']].copy();wid=z['window_id'].copy()
  assert np.array_equal(wid,d['window_id']);d['x']=pack(g,d['w_self'],d['kinematic_valid']);parts[split].append(d)
data={s:{k:np.concatenate([p[k] for p in parts[s]]) for k in parts[s][0]} for s in parts}
assert not set(data['train']['mmsi'].astype(str))&set(data['val']['mmsi'].astype(str))
x=data['train']['x'][data['train']['mask']];mu=x.mean(0,keepdims=True);sd=x.std(0,keepdims=True)+np.float32(1e-6)
assert len(x)==136071 and len(data['train']['y'])==7840 and len(data['val']['y'])==1680
expected_orders={}
for seed in (0,1,2):
 rng=np.random.RandomState(seed);cur=np.arange(7840,dtype=np.int64);arr=[]
 for epoch in range(1,26):
  rng.shuffle(cur);b=np.asarray([epoch],dtype='<i8').tobytes()+cur.astype('<i8').tobytes();arr.append(hashlib.sha256(b).hexdigest())
 expected_orders[seed]=arr
max_valmetric=0.;val_scores={}
for arch in ('tcn','gru'):
 for seed in (0,1,2):
  p=RUN/'runs'/arch/f'seed_{seed}'/'candidate';result=json.loads((p/'result.json').read_text(encoding='utf8'));prior=json.loads((BASE/'runs'/arch/f'seed_{seed}'/'causal'/'result.json').read_text(encoding='utf8'))
  assert result['status']=='completed_validation_only' and result['epochs']==25 and result['primary_epoch']==25 and result['gradient_training_windows']==7840
  assert result['epoch_batch_order_sha256']==expected_orders[seed]==prior['epoch_batch_order_sha256']
  for k in ('encoder','binary_head','type_head'):assert result['initial_state_hashes'][k]==prior['initial_state_hashes'][k]
  with np.load(p/'normalizer.npz',allow_pickle=False) as z:assert np.array_equal(z['mean'],mu) and np.array_equal(z['std'],sd)
  for filename,h in result['output_hashes'].items():assert sha(p/filename)==h
  for row in result['history']:
   assert row['train_total']==row['train_binary'] and all(row['train_'+k]==0 for k in ('type_ce','js','aug_binary','hidden_smooth'))
  with (p/'val_predictions_final.csv').open(encoding='utf8',newline='') as f:rr=list(csv.DictReader(f))
  assert np.array_equal([r['window_id'] for r in rr],data['val']['window_id'].astype(str)) and np.array_equal([int(r['y']) for r in rr],data['val']['y'])
  scores=np.array([float(r['score']) for r in rr]);val_scores[(arch,seed)]=scores;y=data['val']['y'];region=data['val']['region'].astype(str)
  with np.load(p/'epoch_validation_predictions.npz',allow_pickle=False) as z:assert np.array_equal(scores,z['scores'][-1])
  for a,b in [(average_precision_score(y,scores),result['final_validation']['pooled']['AP']),(roc_auc_score(y,scores),result['final_validation']['pooled']['AUROC']),(np.mean([average_precision_score(y[region==r],scores[region==r]) for r in REGIONS]),result['final_validation']['macro']['AP'])]:
   err=abs(float(a)-b);max_valmetric=max(max_valmetric,err);assert err<1e-12

test=[]
for r in REGIONS:
 with np.load(SCENE/r/'test.npz',allow_pickle=False) as z:d={k:z[k] for k in ('window_id','mmsi','region','y','mask','self_gate_weight','kinematic_valid','time_s','target_pos','target_sog_kn','target_cog_deg')}
 n,T=d['mask'].shape;raw=np.zeros((n,T,7),np.float32);raw[...,:2]=(d['target_pos']-d['target_pos'][:,0:1]).astype(np.float32)
 angle=np.deg2rad(d['target_cog_deg']);speed=d['target_sog_kn'];raw[...,2]=speed*KNOT*np.sin(angle);raw[...,3]=speed*KNOT*np.cos(angle);raw[...,4]=speed;raw[...,5]=np.sin(angle);raw[...,6]=np.cos(angle);raw[~d['mask']]=0
 w=d['self_gate_weight'].astype(np.float32).copy();w[~d['mask']]=0;g=raw.copy()
 for i in range(n):
  for j in range(1,T):
   if w[i,j]<.2:g[i,j,2:7]=g[i,j-1,2:7]
  pos,_=independent(raw[i],w[i],d['mask'][i],d['kinematic_valid'][i],d['time_s'][i],pars['speed_cap_mps'],pars['acceleration_q99_mps2'],selection['candidate']);g[i,:,:2]=pos
 d['x']=pack(g,w,d['kinematic_valid']);test.append(d)
te={k:np.concatenate([p[k] for p in test]) for k in test[0]};y=te['y'].astype(int);regions=te['region'].astype(str);mmsi=te['mmsi'].astype(str);ids=te['window_id'].astype(str)
with np.load(RUN/'test_results/predictions.npz',allow_pickle=False) as z:preds={k:z[k] for k in z.files}
with np.load(BASE/'formal_test_results/all_test_predictions.npz',allow_pickle=False) as z:baseline={k:z[k] for k in z.files}
assert np.array_equal(ids,preds['window_id']) and np.array_equal(y,preds['y']) and np.array_equal(mmsi,preds['mmsi'])
assert np.array_equal(ids,baseline['window_id']) and np.array_equal(y,baseline['y']) and np.array_equal(mmsi,baseline['mmsi'].astype(str))
result=json.loads((RUN/'test_results/test_results.json').read_text(encoding='utf8'));assert result['plan_sha256']==sha(RUN/'frozen_test_plan.json')
unique,inv=np.unique(mmsi,return_inverse=True);rng=np.random.default_rng(20261027);draw=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000).astype(float);weights=draw[:,inv]
def boot(scores):
 pooled=aps(scores,y,weights);regional=[]
 for r in REGIONS:
  m=regions==r;yr=y[m];w=weights[:,m];assert ((w*(yr==0)).sum(1)>0).all() and ((w*(yr==1)).sum(1)>0).all();regional.append(aps(scores[m],yr,w))
 return pooled,np.mean(regional,axis=0)
torch.set_num_threads(4);boots={};metrics={};max_forward=0.;max_testmetric=0.;max_ci=0.;max_threshold=0.
for i,run in enumerate(plan['runs']):
 arch,seed=run['architecture'],int(run['seed']);assert (str(preds['architecture'][i]),int(preds['seed'][i]))==(arch,seed)
 p=RUN/'runs'/arch/f'seed_{seed}'/'candidate';ck=torch.load(p/'checkpoint_final.pt',map_location='cpu',weights_only=False);model=Model(arch);model.load_state_dict(ck['state_dict']);model.eval()
 assert np.array_equal(ck['normalizer_mean'].numpy(),mu) and np.array_equal(ck['normalizer_std'].numpy(),sd)
 X=(te['x']-mu)/sd;outputs=[]
 with torch.no_grad():
  for j in range(0,len(X),128):outputs.extend(torch.sigmoid(model(torch.from_numpy(X[j:j+128]))).numpy().tolist())
 expected=preds['scores'][i];diff=float(np.max(np.abs(expected-np.array(outputs))));max_forward=max(max_forward,diff);assert diff<1e-6,(arch,seed,diff)
 obs=result['runs'][i];assert (obs['architecture'],obs['seed'])==(arch,seed)
 for A,B in [(average_precision_score(y,expected),obs['pooled_AP']),(roc_auc_score(y,expected),obs['pooled_AUROC']),(np.mean([average_precision_score(y[regions==r],expected[regions==r]) for r in REGIONS]),obs['macro_AP'])]:
  err=abs(float(A)-B);max_testmetric=max(max_testmetric,err);assert err<1e-12
 val=val_scores[(arch,seed)];vy=data['val']['y'];neg=np.sort(val[vy==0])[::-1]
 for alpha in ('0.01','0.05'):
  k=int(np.floor(float(alpha)*len(neg)));th=float(np.nextafter(neg[k],np.inf));assert run['thresholds'][alpha]['threshold']==th
  err=max(abs(float(np.mean(expected[y==0]>=th))-obs[alpha]['actual_FAR']),abs(float(np.mean(expected[y==1]>=th))-obs[alpha]['recall']));max_threshold=max(max_threshold,err);assert err==0
 cand=boot(expected);boots[(arch,seed,'candidate')]=cand;metrics[(arch,seed,'candidate')]=(obs['pooled_AP'],obs['macro_AP'])
 for arm in ('raw','old','comp','causal'):
  idx=np.flatnonzero((baseline['architecture']==arch)&(baseline['seed']==seed)&(baseline['arm']==arm));assert len(idx)==1;ref=baseline['scores'][idx[0]];rboot=boot(ref);boots[(arch,seed,arm)]=rboot
  pair=obs['paired'][arm];pdelta=cand[0]-rboot[0];mdelta=cand[1]-rboot[1]
  for vals,key in [(pdelta,'pooled_AP_difference_CI95'),(mdelta,'macro_AP_difference_CI95')]:
   err=float(np.max(np.abs(np.quantile(vals,[.025,.975])-pair[key])));max_ci=max(max_ci,err);assert err<1e-10
for row in result['seed_mean_paired']:
 arch=row['architecture'];arm=row['reference']
 for index,key in [(0,'pooled_AP_difference_CI95'),(1,'macro_AP_difference_CI95')]:
  delta=np.mean([boots[(arch,s,'candidate')][index]-boots[(arch,s,arm)][index] for s in (0,1,2)],axis=0)
  err=float(np.max(np.abs(np.quantile(delta,[.025,.975])-row[key])));max_ci=max(max_ci,err);assert err<1e-10
out={'status':'passed','validation_runs':6,'validation_predictions':6*1680,'test_runs':6,'test_predictions':6*1680,
 'max_validation_metric_error':max_valmetric,'max_test_metric_error':max_testmetric,'max_test_probability_replay_error':max_forward,
 'max_test_far_recall_error':max_threshold,'max_paired_bootstrap_ci_error':max_ci,'cluster_draws':2000,'test_mmsi':len(unique),
 'freeze_utc':plan['frozen_utc'],'candidate':selection['candidate'],'scope':'independent target observation reconstruction and independent model class CPU forward; seed-specific pooled/region AP, AUROC, fixed thresholds and paired cluster CIs',
 'limitations':'test previously exposed; source/mask/eligibility flags from existing dataset; synthetic anomaly labels and shared backgrounds; no region or seed population uncertainty',
 'helper_sha256':sha(OUT/'independent_gate.py'),'script_sha256':sha(Path(__file__))}
with (OUT/'audit_models_and_test.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
