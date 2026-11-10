"""Fixed logistic nuisance probes; train-only scaler and no model selection."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'1')
import argparse,hashlib,json,platform,time,warnings
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.exceptions import ConvergenceWarning

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/labeling/labeled_v1/shortcut_audit')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def metric(y,p):
 if len(set(y))!=2:raise ValueError('Need both labels to assess nuisance ROC/AP')
 return {'n':len(y),'positives':int(y.sum()),'prevalence_ap_baseline':float(y.mean()),'auroc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p))}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,default=ROOT);args=ap.parse_args();started=time.time()
 f=args.folder/'pre_noise_nuisance_features.npz';audit=json.loads((args.folder/'pre_noise_invariant_audit.json').read_text())
 if audit['status']!='passed' or audit['output_npz_sha256']!=sha(f):raise ValueError('Unverified or changed feature extraction')
 with np.load(f,allow_pickle=False) as z:data={k:z[k] for k in z.files}
 y=data['y'].astype(int);splits=data['split'];regions=data['region'];train=splits=='train'
 if set(splits)!={'train','val','test'}:raise ValueError('Unexpected split')
 report={'status':'completed','scope':'pre-noise nuisance/shortcut diagnostic; not anomaly detection performance','classifier':'StandardScaler fit only on TRAIN -> LogisticRegression(C=1,solver=lbfgs,max_iter=2000,class_weight=None,random_state=20261026)','hyperparameter_selection':'None; no validation/test-based tuning, feature selection or outcome targeting','evaluation':'Per-region ROC-AUC/AP; equal-weight macro average; prevalence is AP chance-reference','review_rule':'Flag any test domain or macro AUROC >0.65; preserve the actual score','models':{},'input_hashes':{'features':sha(f),'invariant_audit':sha(args.folder/'pre_noise_invariant_audit.json'),'classifier_script':sha(Path(__file__))},'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'flags':[]}
 predictions={k:data[k] for k in ['sample_id','window_id','region','split','mmsi','y']}
 for name,key,names_key in [('prefix','prefix_X','prefix_feature_names'),('original_innovation','innovation_X','innovation_feature_names')]:
  X=data[key];scaler=StandardScaler();Xs=scaler.fit_transform(X[train]);transformed=scaler.transform(X)
  model=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,class_weight=None,random_state=20261026)
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',ConvergenceWarning);model.fit(Xs,y[train])
  convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
  p=model.predict_proba(transformed)[:,1];predictions[name+'_score']=p
  entry={'feature_names':data[names_key].tolist(),'train_rows_for_scaler':int(train.sum()),'constant_train_features':[str(n) for n,v in zip(data[names_key],scaler.var_) if v==0],'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'coef_standardized':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),'n_iter':model.n_iter_.tolist(),'convergence_warnings':convergence,'splits':{}}
  for split in ['train','val','test']:
   use=splits==split;by={}
   for region in sorted(set(regions)):
    ix=use&(regions==region);by[str(region)]=metric(y[ix],p[ix])
    if split=='test' and by[str(region)]['auroc']>.65:report['flags'].append({'model':name,'scope':'test/'+str(region),'auroc':by[str(region)]['auroc'],'reason':'AUROC exceeds review threshold .65'})
   macro={k:float(np.mean([v[k] for v in by.values()])) for k in ['auroc','ap','prevalence_ap_baseline']}
   entry['splits'][split]={'per_region':by,'macro':macro,'pooled':metric(y[use],p[use])}
   if split=='test' and macro['auroc']>.65:report['flags'].append({'model':name,'scope':'test/macro','auroc':macro['auroc'],'reason':'Macro AUROC exceeds review threshold .65'})
  report['models'][name]=entry
  np.savez_compressed(args.folder/(name+'_logistic_model.npz'),coef=model.coef_,intercept=model.intercept_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,feature_names=data[names_key])
  print(name,json.dumps(entry['splits']['test']['macro']),flush=True)
 np.savez_compressed(args.folder/'per_sample_nuisance_predictions.npz',**predictions)
 report['prediction_file_sha256']=sha(args.folder/'per_sample_nuisance_predictions.npz');report['elapsed_seconds']=time.time()-started
 report['requires_review']=bool(report['flags']);report['invariant_checks_passed']=True
 (args.folder/'nuisance_logistic_results.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'requires_review':report['requires_review'],'flags':report['flags'],'results':str(args.folder/'nuisance_logistic_results.json')},indent=2),flush=True)


if __name__=='__main__':main()
