"""Cluster paired uncertainty for frozen neural predictions, without retraining."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from sklearn.metrics import average_precision_score

OUT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027/model_study')
VIEWS=('base','natural','suffix'); MODES=('target','mean','median','absolute_median','absolute_report_median','mixed_report_median')
ARCH=('tcn','gru'); SEEDS=(0,1,2)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def boot_ap(y,score,weights):
    order=np.argsort(-score,kind='stable');s=score[order];Y=y[order];w=weights[:,order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    cum=np.cumsum(w,axis=1)[:,ends];tp=np.cumsum(w*Y[None,:],axis=1)[:,ends]
    prev=np.concatenate((np.zeros((len(weights),1)),tp[:,:-1]),axis=1)
    return np.sum((tp-prev)*np.divide(tp,cum,out=np.zeros_like(tp),where=cum>0),axis=1)/np.maximum(tp[:,-1],1)
def main(split):
    cl=OUT/'checkpoint_lock.json';assert cl.exists();lock=json.loads(cl.read_text())
    assert lock['count']==108
    if split=='test':assert (OUT/'features_test_manifest.json').exists()
    with np.load(OUT/f'features_{split}.npz',allow_pickle=False) as z:
        y=z['y'].astype(int);region=z['region'];mmsi=z['mmsi'];source=z['source'];case=z['case']
    ids,inv=np.unique(mmsi,return_inverse=True);rng=np.random.default_rng(20261027)
    counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000).astype(np.float32)
    weights=counts[:,inv];apdraw={};point={};summaries=[];runs=0
    for arch in ARCH:
      for exposure in VIEWS:
        for mode in MODES:
          for seed in SEEDS:
            d=OUT/'runs'/arch/exposure/mode/f'seed{seed}'
            with np.load(d/f'{split}_predictions.npz',allow_pickle=False) as z:
                assert np.array_equal(z['y'],y) and np.array_equal(z['source'],source) and np.array_equal(z['case'],case)
                assert np.array_equal(z['mmsi'],mmsi) and np.array_equal(z['region'],region)
                for view in VIEWS:
                    score=z[view].astype(float);key=f'{arch}__{exposure}__{mode}__{seed}__{view}'
                    draws=boot_ap(y,score,weights);apdraw[key]=draws;point[key]=float(average_precision_score(y,score))
                    for j in (0,1,2): assert abs(draws[j]-average_precision_score(y,score,sample_weight=weights[j]))<1e-10
            runs+=1
        for view in VIEWS:
          for mode in MODES:
            if mode=='median':continue
            diffs=[];pd=[]
            for seed in SEEDS:
                x=f'{arch}__{exposure}__{mode}__{seed}__{view}';b=f'{arch}__{exposure}__median__{seed}__{view}'
                delta=apdraw[x]-apdraw[b];diffs.append(delta);pd.append(point[x]-point[b])
                summaries.append(dict(arch=arch,exposure=exposure,view=view,mode=mode,seed=seed,
                    comparison='mode-minus-median',pooled_AP_delta=point[x]-point[b],ci95=np.quantile(delta,[.025,.975]).tolist()))
            delta=np.mean(diffs,axis=0);summaries.append(dict(arch=arch,exposure=exposure,view=view,mode=mode,seed='mean_fixed_3',
                comparison='mode-minus-median',pooled_AP_delta=float(np.mean(pd)),ci95=np.quantile(delta,[.025,.975]).tolist()))
    assert runs==108
    np.savez_compressed(OUT/f'bootstrap_draws_{split}.npz',**apdraw,mmsi_ids=ids,mmsi_draw_counts=counts)
    report={'split':split,'unit':'target MMSI, shared draws, 2000 replicates', 'target_mmsi':len(ids),
      'seed_scope':'mean_fixed_3 conditions on fixed trained models; does not account for model training seed population or shared backgrounds',
      'note':'offline synthetic quartet labels; three siblings and repeated windows are not independent',
      'checkpoint_lock_sha256':sha(cl),'feature_sha256':sha(OUT/f'features_{split}.npz'),
      'draws_sha256':sha(OUT/f'bootstrap_draws_{split}.npz'),'comparisons':summaries}
    with (OUT/f'bootstrap_{split}.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('BOOT_DONE',split,runs,len(ids),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['val','test']);a=p.parse_args();main(a.split)
