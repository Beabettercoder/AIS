from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score

root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');work=root/'causal_gate_main_v1_20261027';out=root/'independent_audit_causal_v1_20261027';dest=work/'formal_test_results';plan=json.loads((work/'frozen_test_plan.json').read_text());results=json.loads((dest/'test_results.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert results['frozen_plan_sha256']==sha(work/'frozen_test_plan.json')
with np.load(dest/'all_test_predictions.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
y=data['y'];group=data['region'].astype(str);mmsi=data['mmsi'].astype(str);maxerr=0.;thresholderr=0.;runmap={}
for i,r in enumerate(results['results']):
 spec=plan['runs'][i];key=(r['architecture'],r['seed'],r['arm']);assert key==(spec['architecture'],spec['seed'],spec['arm'])==(str(data['architecture'][i]),int(data['seed'][i]),str(data['arm'][i]));assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
 s=data['scores'][i];ap=average_precision_score(y,s);auc=roc_auc_score(y,s);reg={k:average_precision_score(y[group==k],s[group==k]) for k in np.unique(group)}
 maxerr=max(maxerr,abs(ap-r['pooled_AP']),abs(auc-r['pooled_AUROC']),abs(np.mean(list(reg.values()))-r['macro_AP']),*(abs(v-r['region_AP'][k]) for k,v in reg.items()))
 for alpha in ('0.01','0.05'):
  t=spec['thresholds'][alpha]['threshold'];neg=y==0;pos=y==1;far=np.mean(s[neg]>=t);rec=np.mean(s[pos]>=t);thresholderr=max(thresholderr,abs(far-r[alpha]['actual_FAR']),abs(rec-r[alpha]['recall']));assert r[alpha]['threshold']==t
 runmap[key]=i
assert len(results['results'])==72 and data['scores'].shape==(72,1680) and maxerr<1e-10 and thresholderr<1e-12

# Forward replay on four representative checkpoints. Input preparation and
# architecture necessarily use the frozen experiment's own code: this checks
# saved checkpoint/normalizer/prediction consistency, not independent features.
sys.path.insert(0,str(work));import evaluate_test as evaluator
torch.set_num_threads(4);prepared,proof=evaluator.prep();assert np.array_equal(prepared['window_id'],data['window_id'])
maxforward=0.;checked=[]
for arch,arm in [('tcn','raw'),('gru','causal'),('tcn','weighted_median'),('gru','causal_common_bundle')]:
 key=(arch,0,arm);ix=runmap[key];spec=plan['runs'][ix];saved=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False);model=evaluator.train.hm.HistoricalModel(arch,arm);model.load_state_dict(saved['state_dict']);model.eval()
 x=(prepared['features'][arm]-saved['normalizer_mean'].numpy())/saved['normalizer_std'].numpy();score=[]
 with torch.no_grad():
  for lo in range(0,len(y),128):
   s,*_=model(torch.from_numpy(x[lo:lo+128]),torch.from_numpy(prepared['mask'][lo:lo+128]));score.extend(torch.sigmoid(s).numpy().tolist())
 err=float(np.max(np.abs(np.asarray(score)-data['scores'][ix])));maxforward=max(maxforward,err);checked.append({'architecture':arch,'seed':0,'arm':arm,'max_error':err})
assert maxforward<1e-6

# Recalculate 2000 cluster bootstrap intervals for pooled AP and both fixed
# threshold metrics for two prespecified representative runs.
ids,inv=np.unique(mmsi,return_inverse=True);rng=np.random.default_rng(20261027);counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000);weights=counts[:,inv];bs=[]
for key in [('tcn',0,'raw'),('gru',0,'causal_common_bundle')]:
 ix=runmap[key];s=data['scores'][ix];r=results['results'][ix];p=plan['runs'][ix];draw={'AP':[],'recall_0.01':[],'FAR_0.01':[],'recall_0.05':[],'FAR_0.05':[]}
 for w in weights:
  draw['AP'].append(float(average_precision_score(y,s,sample_weight=w)))
  for alpha in ('0.01','0.05'):
   pred=s>=p['thresholds'][alpha]['threshold'];neg=y==0;pos=y==1
   draw['FAR_'+alpha].append(float((w*neg*pred).sum()/max(1,(w*neg).sum())))
   draw['recall_'+alpha].append(float((w*pos*pred).sum()/max(1,(w*pos).sum())))
 for name,arr in draw.items():
  interval=np.quantile(arr,[.025,.975]);err=float(np.max(np.abs(interval-np.array(r['bootstrap_95'][name]))));assert err<1e-9,(key,name,err);bs.append({'key':key,'metric':name,'interval_max_error':err})
result={'status':'passed','runs':72,'test_windows_per_run':len(y),'test_predictions':72*len(y),'max_AP_AUROC_region_error':maxerr,'max_fixed_threshold_far_recall_error':thresholderr,
 'checkpoint_hashes_match_plan':True,'forward_replay':checked,'max_forward_error':maxforward,'forward_replay_scope':'4 full checkpoints, experiment preparation and architecture shared',
 'bootstrap_audited_runs':2,'bootstrap_draws_each':2000,'bootstrap':bs}
(out/'audit_main_test.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='bootstrap'}))
