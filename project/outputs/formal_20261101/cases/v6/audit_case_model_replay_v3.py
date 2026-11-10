"""Independent same-host GPU replay of all single-vessel event scores; no labels."""
from pathlib import Path
import json,hashlib,sys,torch,numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs');S=R/'formal_20261101/single';D=R/'formal_20261101/cases/v6';sys.path[:0]=[str(S),str(R/'original_mainline_pilot_20261029')]
from eight_backbone_adapter import Model8
from pilot_train import load,fit
fr=json.loads((S/'frozen_checkpoint_thresholds.json').read_text());p=json.loads((S/'frozen_train_protocol.json').read_text());n=fit(load('train'))
for k in ('sm','ss','fm','fs'):np.testing.assert_array_equal(np.asarray(p['normalizer_arrays'][k]),np.asarray(n[k]))
models=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];device=torch.device('cuda:3');torch.set_num_threads(2);results={}; maxdiff=0.;cross1=0;cross5=0;N=0
for source,score in [('nonoverlap', 'case_scores.npz'),('rolling','rolling_scores.npz')]:
 dfile=D/('realcase_inputs.npz' if source=='nonoverlap' else 'rolling_inputs.npz');afile=D/score;assert afile.exists()
 with np.load(dfile,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 with np.load(afile,allow_pickle=False) as z:orig={k:z[k].copy() for k in ('MATSON','SARA')}
 for j,name in enumerate(models):
  for seed in range(3):
   info=fr['arms'][f'{name}/{seed}/f1_bundle'];w=Path(info['checkpoint']);assert hashlib.sha256(w.read_bytes()).hexdigest()==info['checkpoint_sha256'];m=Model8(name).to(device);m.load_state_dict(torch.load(w,map_location=device,weights_only=True));m.eval()
   for case in ('MATSON','SARA'):
    mask=d[case+'__mask'];x=(d[case+'__self']-n['sm'])/n['ss'];x[~mask]=0;field=(d[case+'__field']-n['fm'])/n['fs'];field[~d[case+'__field_valid']]=0
    args=[torch.as_tensor(v,device=device) for v in (x.astype(np.float32),mask,d[case+'__time_s'].astype(np.float32),field.astype(np.float32),d[case+'__rho'].astype(np.float32))]
    with torch.no_grad():q=torch.sigmoid(m(*args,True)[0]).cpu().numpy()
    p=orig[case][j,seed];assert len(p)==len(q);diff=np.abs(p-q);maxdiff=max(maxdiff,float(diff.max()))
    for b in ('0.01','0.05'):
     th=info['thresholds'][b]['threshold'];changes=int(((p>th)!=(q>th)).sum());cross1+=changes if b=='0.01' else 0;cross5+=changes if b=='0.05' else 0
    N+=len(p);results[f'{source}/{case}/{name}/{seed}']={'n':len(p),'max_abs_cpu_gpu':float(diff.max())}
   del m
print(json.dumps({'n':N,'max_abs_diff':maxdiff,'crossing_1pct_differences':cross1,'crossing_5pct_differences':cross5}),flush=True)
r={'scope':'all24 frozen single F1 models, 2 cases, non-overlap+rolling; CPU-vs-GPU3 score and threshold-replay verification','n_scores':N,'max_abs_cpu_gpu':maxdiff,'crossing_1pct_differences':cross1,'crossing_5pct_differences':cross5,'normalizer_reconstructed_exactly_from_train':True,'records':results,'warnings':['event labels are weak; thresholds not real FAR','floating-point device differences do not establish clinical or operational calibration']};p=D/'case_model_gpu_audit.json';assert not p.exists();p.write_text(json.dumps(r,indent=2))
