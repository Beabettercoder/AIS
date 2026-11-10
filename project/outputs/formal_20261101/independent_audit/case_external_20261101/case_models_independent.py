from pathlib import Path
import json,hashlib,sys
import numpy as np
import torch

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs'); D=ROOT/'formal_20261101/cases/v6'; S=ROOT/'formal_20261101/single'; O=ROOT/'formal_20261101/independent_audit/case_external_20261101'
sys.path.insert(0,str(S));from eight_backbone_adapter import Model8
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 # Reconstruct train-only normalizer without importing any training/score routine.
 data=np.load(ROOT/'framework_validation_v1_20261028/data/individual/train.npz',allow_pickle=False)
 field=np.load(ROOT/'framework_validation_v1_20261028/extra_data/field_train.npz',allow_pickle=False)
 for k in ('window_id','mmsi','mask','time_s'):assert np.array_equal(data[k],field[k]),k
 m=data['mask'].astype(bool);v=field['field_common_valid'].astype(bool)&m;self7=field['gated7_legacy'].astype(np.float64)
 target=np.concatenate((data['base__C0'][...,:7][m],self7[m]),axis=0).astype(np.float64)
 vals=[]
 for arm in ('F0','F1'):
  bg=field['base__'+arm][...,7:9].astype(np.float64);div=field[arm+'__divergence_per_s'].astype(np.float64);curl=field[arm+'__curl_per_s'].astype(np.float64)
  feat=np.concatenate((self7[...,2:4]-bg,div[...,None],curl[...,None]),axis=-1).astype(np.float32);vals.append(feat[v])
 c=np.concatenate(vals,axis=0).astype(np.float64)
 def stats(x):
  mean=x.mean(0);sd=x.std(0);sd[sd<1e-7]=1.;return mean,sd
 sm,ss=stats(target);fm,fs=stats(c)
 protocol=json.loads((S/'frozen_train_protocol.json').read_text());norm=protocol['normalizer_arrays']
 diffs={k:float(np.max(np.abs(a-np.asarray(norm[k])))) for k,a in (('sm',sm),('ss',ss),('fm',fm),('fs',fs))}
 assert max(diffs.values())<1e-10,diffs
 fr=json.loads((S/'frozen_checkpoint_thresholds.json').read_text());names=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];cases=['MATSON','SARA'];device=torch.device('cuda:2');torch.set_num_threads(2)
 runs={}; overall=0.;flips={'1pct':0,'5pct':0}; nvalues=0
 inputs={};expected={}
 for subset in ('nonoverlap','rolling'):
  path=D/('realcase_inputs.npz' if subset=='nonoverlap' else 'rolling_inputs.npz'); scorepath=D/('case_scores.npz' if subset=='nonoverlap' else 'rolling_scores.npz')
  z=np.load(path,allow_pickle=False); pred=np.load(scorepath,allow_pickle=False);assert list(pred['models'])==names and np.array_equal(pred['seeds'],[0,1,2]);expected[subset]={k:pred[k] for k in cases}
  for case in cases:
   mask=z[case+'__mask'];x=(z[case+'__self'].astype(np.float64)-sm)/ss;x[~mask]=0
   field=(z[case+'__field'].astype(np.float64)-fm)/fs;field[~z[case+'__field_valid']]=0
   args=(x.astype(np.float32),mask,z[case+'__time_s'].astype(np.float32),field.astype(np.float32),z[case+'__rho'].astype(np.float32))
   assert all(np.isfinite(a).all() for a in args);inputs[(subset,case)]=tuple(torch.as_tensor(a,device=device) for a in args)
  # Full asset locks were independently checked by the source audit.
 for i,n in enumerate(names):
  for seed in (0,1,2):
   item=fr['arms'][f'{n}/{seed}/f1_bundle']; path=Path(item['checkpoint']);assert sha(path)==item['checkpoint_sha256']
   model=Model8(n).to(device);model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.eval()
   for subset in ('nonoverlap','rolling'):
    for case in cases:
     with torch.inference_mode():score=torch.sigmoid(model(*inputs[(subset,case)],True)[0]).cpu().numpy()
     prior=expected[subset][case][i,seed];assert len(score)==len(prior);maxdiff=float(np.max(np.abs(score-prior)));overall=max(overall,maxdiff);nvalues+=len(score)
     for label,k in (('1pct','0.01'),('5pct','0.05')):
      th=item['thresholds'][k]['threshold'];flips[label]+=int(np.count_nonzero((score>th)!=(prior>th)))
     runs[f'{subset}/{case}/{n}/{seed}']={'n':len(score),'max_abs_cpu_gpu':maxdiff}
   del model
  print('REPLAYED',n,flush=True)
 assert nvalues==2640 and overall<.002 and flips=={'1pct':0,'5pct':0},(nvalues,overall,flips)
 out={'scope':'Independent reconstructed train-only normalizer and GPU replay of all 24 F1 models on every fixed case window; comparison to prior CPU scores. Threshold changes counted, not a real-world FAR guarantee.', 'normalizer_max_abs_diff':diffs,'model_score_count':nvalues,'max_abs_score_difference':overall,'threshold_crossing_differences':flips,'records':runs}
 dest=O/'case_model_independent.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2),encoding='utf8');print('MODEL_AUDIT',json.dumps({k:v for k,v in out.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
