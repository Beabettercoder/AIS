"""Post-hoc inference-only sensitivity: toggle cuDNN TF32 on 108 fixed models."""
from pathlib import Path
from datetime import datetime,timezone
import os,sys,json,csv,hashlib
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');study=root/'relation_dual_pressure_v2_20261027/model_study';out=root/'relation_independent_audit_20261027/tf32_sensitivity_all';sys.path.insert(0,str(root/'relation_test_expanded_v1_20261027'));from report_context_common import Model
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
lock=json.loads((study/'checkpoint_lock.json').read_text());assert lock['count']==108
feat=study/'features_test.npz';fm=json.loads((study/'features_test_manifest.json').read_text());assert sha(feat)==fm['feature_file_sha256']
with np.load(feat,allow_pickle=False) as z:data={k:z[k] for k in z.files}
y=data['y'];L=data['length'];device=torch.device('cuda:0');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.use_deterministic_algorithms(False)
rows=[];all_on=[];all_off=[];keys=[]
for arch in ('tcn','gru'):
 for exposure in ('base','natural','suffix'):
  for mode in ('target','mean','median','absolute_median','absolute_report_median','mixed_report_median'):
   for seed in (0,1,2):
    run=study/'runs'/arch/exposure/mode/f'seed{seed}';cp=run/'checkpoint.pt';assert sha(cp)==lock['checkpoints'][str(cp)]
    state=torch.load(cp,map_location='cpu',weights_only=False);model=Model(arch).to(device);model.load_state_dict(state['state_dict']);model.eval()
    with np.load(run/'test_predictions.npz',allow_pickle=False) as saved:saved_pred={v:saved[v].copy() for v in ('base','natural','suffix')};assert np.array_equal(saved['y'],y)
    on_views=[];off_views=[]
    for view in ('base','natural','suffix'):
     x=((data[view+'__'+mode]-state['mean'])/state['std']).astype(np.float32);preds=[]
     for setting in (True,False):
      torch.backends.cudnn.allow_tf32=setting;pp=[]
      with torch.no_grad():
       for lo in range(0,len(x),256):
        logits=model(torch.tensor(x[lo:lo+256],device=device),torch.tensor(L[lo:lo+256],dtype=torch.long,device=device));pp.extend(torch.sigmoid(logits).cpu().numpy().tolist())
      preds.append(np.asarray(pp,dtype=np.float32))
     on,off=preds;on_views.append(on);off_views.append(off)
     ap_on=float(average_precision_score(y,on));ap_off=float(average_precision_score(y,off))
     rows.append({'architecture':arch,'training_exposure':exposure,'context':mode,'seed':seed,'view':view,
       'pooled_AP_original':float(average_precision_score(y,saved_pred[view])),'pooled_AP_on':ap_on,'pooled_AP_off':ap_off,'delta_AP_off_minus_on':ap_off-ap_on,
       'AUROC_on':float(roc_auc_score(y,on)),'AUROC_off':float(roc_auc_score(y,off)),
       'max_score_abs_off_vs_on':float(np.max(np.abs(off-on))),'max_score_abs_on_vs_original':float(np.max(np.abs(on-saved_pred[view])))})
    keys.append((arch,exposure,mode,seed));all_on.append(on_views);all_off.append(off_views)
 print('ARCH_COMPLETE',arch,len(rows),flush=True)
with (out/'per_model_view.csv').open('x',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(out/'replayed_predictions.npz',tf32_on=np.asarray(all_on,dtype=np.float32),tf32_off=np.asarray(all_off,dtype=np.float32),architecture=np.asarray([k[0] for k in keys]),exposure=np.asarray([k[1] for k in keys]),context=np.asarray([k[2] for k in keys]),seed=np.asarray([k[3] for k in keys]),views=np.asarray(['base','natural','suffix']),mmsi=data['mmsi'],source=data['source'],case=data['case'],y=y)
summary={'status':'complete','scope':'posthoc backend sensitivity on all previously frozen 108 checkpoints; no retraining, no selection and original predictions retained','time_utc':datetime.now(timezone.utc).isoformat(),
 'hardware':torch.cuda.get_device_name(0),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
 'pytorch':torch.__version__,'cudnn':torch.backends.cudnn.version(),'test_models':len(keys),'test_model_views':len(rows),'test_examples':len(y),
 'common_settings':'matmul TF32 off, cudnn benchmark off, cudnn deterministic false, deterministic algorithms false; cuDNN TF32 on/off is the only toggle',
 'max_abs_AP_change':max(abs(r['delta_AP_off_minus_on']) for r in rows),'max_abs_probability_change':max(r['max_score_abs_off_vs_on'] for r in rows),
 'max_original_saved_vs_tf32_on':max(r['max_score_abs_on_vs_original'] for r in rows),'checkpoint_lock_sha256':sha(study/'checkpoint_lock.json'),'test_features_sha256':sha(feat),'script_sha256':sha(Path(__file__)),
 'per_model_view_sha256':sha(out/'per_model_view.csv'),'replayed_predictions_sha256':sha(out/'replayed_predictions.npz')}
with (out/'summary.json').open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
