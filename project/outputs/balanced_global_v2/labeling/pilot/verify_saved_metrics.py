"""Independently recompute all 14 pilot metric tables from prediction CSVs."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,precision_score,recall_score
BASE=Path(__file__).resolve().parent.parent/'labeled_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def metrics(data,threshold):
 per={}
 for r in sorted(set(x['region'] for x in data)):
  rr=[x for x in data if x['region']==r];y=np.array([int(x['y']) for x in rr]);p=np.array([float(x['score']) for x in rr]);pred=p>=threshold
  assert len(rr)==80 and y.sum()==32 and np.isfinite(p).all()
  per[r]={'AP':average_precision_score(y,p),'AUROC':roc_auc_score(y,p),'F1':f1_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred)}
 return {k:float(np.mean([v[k] for v in per.values()])) for k in per[next(iter(per))]}
result={'status':'passed','scope':'Actual 14-run validation/test CSVs, independent sklearn recomputation, validation-only threshold replay','runs':[]}
for dirname,expected in [('pilot_scene_seed0',12),('pilot_hard_seed0',2)]:
 folder=BASE/dirname;summary=json.loads((folder/'pilot_summary.json').read_text());records=summary['results'];assert len(records)==expected
 signatures=[]
 for m in records:
  d=folder/(m['backbone']+'__'+m['feature_group']);detail=json.loads((d/'metrics.json').read_text());va=rows(d/'validation_predictions.csv');te=rows(d/'test_predictions.csv')
  assert len(va)==len(te)==640
  for split,rr in [('validation',va),('test',te)]:
   got=metrics(rr,m['threshold'])
   for key,value in got.items():
    assert abs(value-detail[split]['macro'][key])<1e-12,(dirname,d.name,split,key)
    if split=='test':assert abs(value-m['test']['macro'][key])<1e-12
  y=np.array([int(r['y']) for r in va]);p=np.array([float(r['score']) for r in va]);thresholds=np.unique(p);best=-1.;chosen=None
  for th in thresholds:
   pred=p>=th;tp=np.sum(pred&(y==1));fp=np.sum(pred&(y==0));fn=np.sum((~pred)&(y==1));f=2*tp/(2*tp+fp+fn)
   if f>best+1e-12 or (abs(f-best)<=1e-12 and (chosen is None or th>chosen)):best=float(f);chosen=float(th)
  assert abs(chosen-m['threshold'])<1e-12 and abs(best-detail['validation_selected_F1'])<1e-12
  h=detail['history'];selected=max(h,key=lambda x:(x['validation_macro_AP'],-x['epoch']))['epoch'];assert selected==m['selected_epoch']
  signature=hashlib.sha256(json.dumps(sorted((r['base_window_id'],r['region'],r['mmsi'],r['y']) for r in te)).encode()).hexdigest();signatures.append(signature)
  for filename,value in detail['output_hashes'].items():assert sha(d/filename)==value
  result['runs'].append({'run':dirname+'/'+d.name,'test_prediction_rows':len(te),'validation_prediction_rows':len(va),'recomputed_test_macro':metrics(te,m['threshold']),'validation_threshold_replayed':chosen,'selected_epoch_verified':selected,'test_prediction_sha256':sha(d/'test_predictions.csv'),'validation_prediction_sha256':sha(d/'validation_predictions.csv'),'checkpoint_sha256':sha(d/'checkpoint.pt'),'paired_test_signature':signature})
 assert len(set(signatures))==1
assert len(result['runs'])==14 and len(set(r['paired_test_signature'] for r in result['runs']))==1
(BASE/'independent_pilot_metric_audit.json').write_text(json.dumps(result,indent=2));print('PASSED 14 real runs, 17920 saved prediction rows, thresholds and checkpoint rules',flush=True)
