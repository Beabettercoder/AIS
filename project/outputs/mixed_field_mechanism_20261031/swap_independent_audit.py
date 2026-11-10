"""Independent source/data/model/metric check; no train and no test."""
import os,sys,json,hashlib,importlib.util
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from sklearn.metrics import average_precision_score
O=Path('/mnt/sda/WangZijia/AIS/outputs'); IN=O/'mixed_context_swap_20261031'; OUT=O/'mixed_field_mechanism_20261031'; M=O/'mixed_stress_probe_20261031'; H=O/'heading_evidence_pilot_20261030'
spec=importlib.util.spec_from_file_location('heading_ref_audit',H/'train_heading_evidence.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
lock=json.loads((IN/'frozen_protocol.json').read_text())
bad=[k for k,v in lock['sha256'].items() if not Path(k).exists() or sha(Path(k))!=v]
assert not bad,bad
torch.set_num_threads(2)
device=torch.device('cuda:5')
tr,va=ref.read('train'),ref.read('val'); mu,sd=ref.fit(tr)
views=('base','natural','suffix','combined_l100_m600')
arms=('F0_heading','F1_heading')
prepared={}
for view in views:
 for arm in arms:prepared[(view,arm)]=ref.prep(va,view,arm,mu,sd,device)
 x0,m0,t0=prepared[(view,arms[0])];x1,m1,t1=prepared[(view,arms[1])]
 assert x0.shape==(1088,20,40)
 assert torch.equal(x0[:,:,:8],x1[:,:,:8])
 assert torch.equal(m0,m1) and torch.equal(t0,t1)
stored_summary=json.loads((IN/'summary.json').read_text())
pairs={'F0_model_switch_field':('F0_on_F1','F0_on_F0'),'F1_model_switch_field':('F1_on_F1','F1_on_F0'),'F0_field_switch_model':('F1_on_F0','F0_on_F0'),'F1_field_switch_model':('F1_on_F1','F0_on_F1'),'native_F1_minus_F0':('F1_on_F1','F0_on_F0')}
combos={'F0_on_F0':('F0_heading','F0_heading'),'F0_on_F1':('F0_heading','F1_heading'),'F1_on_F0':('F1_heading','F0_heading'),'F1_on_F1':('F1_heading','F1_heading')}
result={'scope':'Independent reconstruction of train-Base normalizer, full validation inputs, every mixed-trained checkpoint and four context-swaps. No test data. Bootstrap repeated via sklearn sample_weight, not supplied weighted-AP function. Off-diagonal feature/model combinations are potentially out of distribution.','frozen_files_checked':len(lock['sha256']),'frozen_files_mismatched':bad,'models':{},'bootstrap':{}}
for name in ('tcn','gru','transformer','itransformer'):
 seeds=(0,1,2) if name in ('tcn','itransformer') else (0,)
 z=np.load(IN/(name+'_scores.npz'),allow_pickle=False)
 for k in ('y','mmsi','region','source','case'):
  assert np.array_equal(z[k].astype(str),va[k].astype(str)),(name,k)
 max_replay=0.;max_diag=0.;max_ap=0.;cnt=0
 for seed in seeds:
  for net in arms:
   p=M/'runs'/name/f'seed{seed}'/net
   with np.load(p/'val_scores.npz') as n: native={k:n[k] for k in n.files}
   model=ref.base.get_model(name,40,name=='itransformer').to(device)
   model.load_state_dict(torch.load(p/'final.pt',map_location=device,weights_only=True));model.eval()
   for v in views:
    for feature in arms:
     x,m,t=prepared[(v,feature)];buf=[]
     with torch.no_grad():
      for j in range(0,len(x),256):buf.extend(torch.sigmoid(model(x[j:j+256],m[j:j+256],t[j:j+256])['logits']).cpu().numpy())
     xpred=np.asarray(buf);key=f's{seed}__{net}__{feature}__{v}';saved=z[key]
     max_replay=max(max_replay,float(np.max(np.abs(xpred-saved))))
     if feature==net:max_diag=max(max_diag,float(np.max(np.abs(xpred-native[v+'_score']))))
     label=('F0' if net=='F0_heading' else 'F1')+'_on_'+('F0' if feature=='F0_heading' else 'F1')
     ap=average_precision_score(va['y'],saved)
     max_ap=max(max_ap,abs(float(ap)-stored_summary[name]['views'][v]['per_seed_ap'][f'{seed}:{label}']))
     cnt+=1
 assert max_replay<1e-5 and max_diag<1e-5 and max_ap<1e-12
 result['models'][name]={'predictions_checked':cnt,'max_replay_abs':max_replay,'native_max_abs':max_diag,'max_sklearn_AP_delta':max_ap,'saved_scores_sha256':sha(IN/(name+'_scores.npz'))}
 print('model',name,cnt,max_replay,max_diag,flush=True)
 if name not in ('tcn','itransformer'):continue
 uniq,inv=np.unique(va['mmsi'].astype(str),return_inverse=True)
 draws=np.random.default_rng(20261031).integers(0,len(uniq),size=(2000,len(uniq)))
 view='suffix'; score={}
 for seed in seeds:
  for label,(net,feature) in combos.items():score[(seed,label)]=z[f's{seed}__{net}__{feature}__{view}']
 diffs={k:[] for k in pairs};y=va['y'].astype(int)
 for draw in draws:
  w=np.bincount(draw,minlength=len(uniq))[inv]
  aps={k:average_precision_score(y,p,sample_weight=w) for k,p in score.items()}
  for label,(a,b) in pairs.items():diffs[label].append(float(np.mean([aps[(s,a)]-aps[(s,b)] for s in seeds])))
 obs={}
 for label,vals in diffs.items():
  ci=np.quantile(vals,[.025,.975]);record=stored_summary[name]['views'][view]['contrasts'][label]['ci95']
  delta=float(np.max(np.abs(ci-record)));assert delta<1e-11,(name,label,ci,record)
  obs[label]={'CI':ci.tolist(),'max_abs_delta_vs_summary':delta}
 result['bootstrap'][name]={'view':view,'draws':len(draws),'contrasts':obs}
 print('bootstrap',name,'maxdelta',max(v['max_abs_delta_vs_summary'] for v in obs.values()),flush=True)
(OUT/'swap_independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('AUDIT PASS')

