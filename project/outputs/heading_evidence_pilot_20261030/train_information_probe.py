"""Exploratory group-held-out probe on training sources, not a test benchmark."""
import sys,json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
sys.path.insert(0,'/mnt/sda/WangZijia/AIS/outputs/heading_evidence_pilot_20261030')
import train_heading_evidence as p
d=p.read('train');a,valid=p.base.features(d,'base','F1');h,ha=p.heading_extra(d,'base','F1_heading');mask=d['mask'];n=mask.sum(1);idx=np.arange(len(n));half=np.floor(n/2).astype(int)
def pool(x):
 pre=(np.arange(x.shape[1])[None,:]<half[:,None])&mask;post=(np.arange(x.shape[1])[None,:]>=half[:,None])&mask
 q=lambda m:(x*m[...,None]).sum(1)/np.maximum(m.sum(1),1)[:,None]
 return np.concatenate([q(pre),q(post),q(post)-q(pre),np.abs(q(post)-q(pre))],1)
X=pool(a);H=pool(h); y=d['y'];groups=d['mmsi']; tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=20261030).split(X,y,groups))
res={}
for label,vec in [('original',X),('original_plus_heading',np.concatenate([X,H],1))]:
 model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
 model.fit(vec[tr],y[tr]);pred=model.predict_proba(vec[te])[:,1];res[label]={'AP':float(average_precision_score(y[te],pred)),'AUROC':float(roc_auc_score(y[te],pred))}
 low=d['raw7'][te,0,4]<2
 if low.sum() and 0<y[te][low].sum()<low.sum():res[label]['initial_target_under_2kn']={'n':int(low.sum()),'AP':float(average_precision_score(y[te][low],pred[low])),'AUROC':float(roc_auc_score(y[te][low],pred[low]))}
for split,ii in [('inner_train',tr),('inner_validation',te)]:
 print(split,'windows',len(ii),'mmsi',len(np.unique(groups[ii])))
assert not set(groups[tr])&set(groups[te])
out={'scope':'training split only, exploratory MMSI-held-out logistic probe; synthetic relationship labels','features':'time-half summaries and absolute differences, original model field vs same plus Heading five channels','results':res,'seed':20261030,'train_windows':len(tr),'heldout_windows':len(te),'train_mmsi':len(np.unique(groups[tr])),'heldout_mmsi':len(np.unique(groups[te]))}
open('/mnt/sda/WangZijia/AIS/outputs/heading_evidence_pilot_20261030/train_information_probe.json','w').write(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
