"""Frozen OOD-only run; all predetermined models and windows, no threshold tuning or labels."""
from pathlib import Path
import json,hashlib,sys
from collections import defaultdict
import numpy as np,torch
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'hawaii_event_feasibility_20261031';sys.path.insert(0,str(R/'heading_alignment_20261030'));import train_heading as old

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
lock=json.loads((O/'frozen_pre_score_lock.json').read_text());assert sha(Path(__file__))==lock['score_code_sha256'];assert sha(O/'frozen_ood_models_thresholds_v2.json')==lock['model_lock_sha256'];assert sha(O/'ood_inputs_v2.npz')==lock['inputs_sha256'];assert sha(O/'ood_windows_v2.json')==lock['windows_sha256'];assert sha(O/'feature_parity_audit.json')==lock['feature_audit_sha256']
frozen=json.loads((O/'frozen_ood_models_thresholds_v2.json').read_text());records=json.loads((O/'ood_windows_v2.json').read_text())['all_windows'];
with np.load(O/'ood_inputs_v2.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
assert len(records)==len(a['mask'])==358 and all(x['out_of_training_domain'] for x in records);torch.set_num_threads(2);dev=torch.device('cuda:5');scores=[];metas=[]
for i,m in enumerate(frozen['models']):
 assert sha(m['model'])==m['model_sha256'] and sha(m['val_predictions'])==m['val_prediction_sha256'] and sha(m['model_protocol'])==m['model_protocol_sha256']
 name=m['name'];arm={'C0':'C0','F0_heading':'F0','F1_heading':'F1'}[m['arm']];model=old.get_model(name,40,name=='itransformer').to(dev);model.load_state_dict(torch.load(m['model'],map_location=dev,weights_only=True));model.eval();p=[]
 with torch.inference_mode():
  for start in range(0,len(records),128):
   z=slice(start,start+128);x=torch.tensor(a[arm][z],device=dev);mask=torch.tensor(a['mask'][z],device=dev);t=a['time_s'][z].copy();t-=t[:,:1];t[~a['mask'][z]]=0;times=torch.tensor(t,dtype=torch.float32,device=dev);log=model(x,mask,times)['logits'];p.extend(torch.sigmoid(log).cpu().numpy().tolist())
 scores.append(np.asarray(p,np.float64));metas.append({**m,'output_index':i});print('PREDICTED',i,name,m['seed'],m['training'],m['arm'],flush=True)
score=np.stack(scores);assert score.shape==(len(frozen['models']),len(records)) and np.isfinite(score).all();np.savez_compressed(O/'ood_scores.npz',score=score)
report={'protocol':'all fixed event-linked windows, all 40 fixed models; event labels weak and window outcome unknown; no recall/FAR estimate and no threshold fit on event data','score_file_sha256':sha(O/'ood_scores.npz'),'input_sha256':sha(O/'ood_inputs_v2.npz'),'by_event_model':[]}
for i,m in enumerate(metas):
 for name in sorted({x['name'] for x in records}):
  ids=np.array([j for j,x in enumerate(records) if x['name']==name]);p=score[i,ids];counts={k:int((p>v['threshold']).sum()) for k,v in m['thresholds'].items()};report['by_event_model'].append({'event_name':name,'model':m['name'],'seed':m['seed'],'training':m['training'],'arm':m['arm'],'window_count':len(ids),'score_min':float(p.min()),'score_median':float(np.median(p)),'score_max':float(p.max()),'validation_threshold_exceedance_count':counts})
(O/'ood_descriptive_scores.json').write_text(json.dumps(report,indent=2),encoding='utf8');print('SCORES_DONE',sha(O/'ood_scores.npz'),flush=True)
