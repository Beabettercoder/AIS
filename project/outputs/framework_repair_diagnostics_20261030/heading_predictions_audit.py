import sys,importlib.util,hashlib,json
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,roc_auc_score
root=Path('/mnt/sda/WangZijia/AIS/outputs'); path=root/'heading_alignment_20261030/train_heading.py';sp=importlib.util.spec_from_file_location('heading',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m); m.VARIANT='aligned_embed'; dev=torch.device('cuda:0');torch.set_num_threads(2);tr=m.read('train');va=m.read('val');mu,sd=m.fit(tr);result={}
for model_name in ('tcn','gru','transformer','itransformer'):
 for arm in m.ARMS:
  p=root/'heading_alignment_20261030/runs/aligned_embed'/model_name/arm
  meta=json.loads((p/'result.json').read_text()); pred=np.load(p/'val_scores.npz');model=m.get_model(model_name,m.NF,model_name=='itransformer').to(dev);model.load_state_dict(torch.load(p/'final.pt',map_location=dev,weights_only=True));model.eval();
  assert np.array_equal(pred['y'],va['y']) and np.array_equal(pred['mmsi'],va['mmsi']) and np.array_equal(pred['source'],va['source']) and np.array_equal(pred['case'],va['case'])
  key=f'{model_name}/{arm}';result[key]={}
  for view in m.VIEWS:
   scores=pred[view+'_score'];expected=meta['views'][view]; ap=average_precision_score(va['y'],scores);auc=roc_auc_score(va['y'],scores); assert abs(ap-expected['pooled_AP'])<1e-12 and abs(auc-expected['AUROC'])<1e-12
   x,mask,time=m.prep(va,view,arm,mu,sd,dev);arr=[]
   with torch.no_grad():
    for start in range(0,len(x),256):arr.extend(torch.sigmoid(model(x[start:start+256],mask[start:start+256],time[start:start+256])['logits']).cpu().numpy())
   err=float(np.max(np.abs(np.array(arr)-scores)));result[key][view]={'pooled_ap':float(ap),'checkpoint_replay_max_abs_err':err}; assert err<2e-5,(key,view,err)
p=root/'framework_repair_diagnostics_20261030/heading_predictions_audit.json';p.write_text(json.dumps(result,indent=2));print('PASS 12 models 4 views sklearn + same gpu forward replay',max(vv['checkpoint_replay_max_abs_err'] for v in result.values() for vv in v.values()))
