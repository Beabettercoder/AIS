"""Independently recompute factorial validation predictions, inputs and pairing."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OLD=ROOT/'gating_training_v2_20261027/data'
PRIOR=ROOT/'causal_gate_main_v1_20261027'
FIELD=ROOT/'robust_background_v1_20261027'
RUN=ROOT/'factorial_ablation_v1_20261027'
OUT=ROOT/'factorial_independent_audit_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
ARMS=('raw_bundle','raw_common','raw_common_bundle','causal_bundle','old_bundle','comp_bundle','raw_bundle_self','causal_bundle_self','causal_common_bundle_self','raw_bundle_bgz','causal_bundle_bgz','causal_common_bundle_bgz')
BASE={'raw_bundle':'raw','raw_common':'raw_common','raw_common_bundle':'raw_common','causal_bundle':'causal','old_bundle':'old','comp_bundle':'comp','raw_bundle_self':'raw','causal_bundle_self':'causal','causal_common_bundle_self':'causal_common','raw_bundle_bgz':'raw','causal_bundle_bgz':'causal','causal_common_bundle_bgz':'causal_common'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
assert tuple(json.loads((RUN/'protocol.json').read_text(encoding='utf8'))['arms'])==ARMS
samples={}; train_features={}; masks=[]
for region in REGIONS:
    with np.load(OLD/region/'train.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
    with np.load(PRIOR/'gate_data'/region/'train.npz',allow_pickle=False) as z:g={k:z[k] for k in z.files}
    with np.load(FIELD/region/'train.npz',allow_pickle=False) as z:f={k:z[k] for k in z.files}
    assert np.array_equal(d['window_id'],g['window_id']) and np.array_equal(d['window_id'],f['window_id'])
    mask=d['mask'];masks.append(mask);n,T=mask.shape
    blank=np.zeros((n,T,5),np.float32)
    raw=np.concatenate((d['raw_legacy7'],blank),axis=2)
    common=raw.copy();common[...,11]=f['common_valid'].astype(np.float32)
    causal=np.concatenate((g['causal7'],np.zeros((n,T,2),np.float32),d['w_self'][...,None],d['kinematic_valid'][...,None],np.zeros((n,T,1),np.float32)),axis=2).astype(np.float32)
    causal_common=causal.copy();causal_common[...,11]=f['common_valid'].astype(np.float32)
    old=np.concatenate((d['old_gate9'][...,:7],np.zeros((n,T,2),np.float32),d['w_self'][...,None],d['kinematic_valid'][...,None],np.zeros((n,T,1),np.float32)),axis=2).astype(np.float32)
    comp=np.concatenate((d['comp_gate9'][...,:7],np.zeros((n,T,2),np.float32),d['w_self'][...,None],d['kinematic_valid'][...,None],np.zeros((n,T,1),np.float32)),axis=2).astype(np.float32)
    for name,x in [('raw',raw),('raw_common',common),('causal',causal),('causal_common',causal_common),('old',old),('comp',comp)]:train_features.setdefault(name,[]).append(x)
    with np.load(OLD/region/'val.npz',allow_pickle=False) as z:
        samples[region]={k:z[k].copy() for k in ('window_id','base_window_id','mmsi','y','region')}
mask=np.concatenate(masks);expected_norm={}
for base,parts in train_features.items():
    v=np.concatenate(parts)[mask];expected_norm[base]={'mean':v.mean(0,keepdims=True),'std':v.std(0,keepdims=True)+np.float32(1e-6)}

total=0;max_metric=0.;max_norm=0.;max_saved_score=0.;result=[]
for arch in ('tcn','gru'):
 for seed in (0,1,2):
    prior=PRIOR/'runs'/arch/f'seed_{seed}'/'raw'
    prior_result=json.loads((prior/'result.json').read_text(encoding='utf8'))
    initial=prior_result['initial_state_hashes'];batch=prior_result['epoch_batch_order_sha256']
    for arm in ARMS:
        p=RUN/'runs'/arch/f'seed_{seed}'/arm
        d=json.loads((p/'result.json').read_text(encoding='utf8'))
        assert d['status']=='completed_validation_only' and d['epochs']==25 and d['primary_epoch']==25
        assert d['gradient_training_windows']==7840 and d['optimizer_steps_per_epoch']==62
        assert d['protocol_sha256']==sha(RUN/'protocol.json')
        assert d['epoch_batch_order_sha256']==batch
        for key in ('encoder','binary_head','type_head'):assert d['initial_state_hashes'][key]==initial[key],(arch,seed,arm,key)
        with np.load(p/'normalizer.npz',allow_pickle=False) as z:
            for key in ('mean','std'):
                err=float(np.max(np.abs(z[key]-expected_norm[BASE[arm]][key])));max_norm=max(max_norm,err);assert err==0
        for filename,h in d['output_hashes'].items():assert sha(p/filename)==h,(p,filename)
        with (p/'val_predictions_final.csv').open(encoding='utf8',newline='') as f:rr=list(csv.DictReader(f))
        order=np.concatenate([samples[r]['window_id'] for r in REGIONS]);y=np.concatenate([samples[r]['y'] for r in REGIONS]);region=np.concatenate([samples[r]['region'] for r in REGIONS]);mmsi=np.concatenate([samples[r]['mmsi'].astype(str) for r in REGIONS])
        assert len(rr)==1680 and np.array_equal(order,[r['window_id'] for r in rr]) and np.array_equal(y,[int(r['y']) for r in rr]) and np.array_equal(mmsi,[r['mmsi'] for r in rr])
        score=np.array([float(r['score']) for r in rr]);
        with np.load(p/'epoch_validation_predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(order,z['window_id']);diff=float(np.max(np.abs(score-z['scores'][-1])));max_saved_score=max(max_saved_score,diff);assert diff==0
        pooled=float(average_precision_score(y,score));auc=float(roc_auc_score(y,score));macro=float(np.mean([average_precision_score(y[region==r],score[region==r]) for r in REGIONS]))
        for err in (abs(pooled-d['final_validation']['pooled']['AP']),abs(auc-d['final_validation']['pooled']['AUROC']),abs(macro-d['final_validation']['macro']['AP'])):max_metric=max(max_metric,err);assert err<1e-12
        for ep in d['history']:
            if arm=='raw_common':assert ep['train_total']==ep['train_binary'] and all(ep['train_'+k]==0 for k in ('type_ce','js','aug_binary','hidden_smooth'))
            else:
                val=ep['train_binary']+.5*ep['train_type_ce']+.2*ep['train_js']+.3*ep['train_aug_binary']+.01*ep['train_hidden_smooth']
                assert abs(val-ep['train_total'])<2e-6
        result.append({'backbone':arch,'seed':seed,'arm':arm,'pooled_AP':pooled,'pooled_AUROC':auc,'macro_AP':macro});total+=len(rr)
out={'status':'passed','runs':len(result),'validation_predictions':total,'max_metric_error':max_metric,
     'max_normalizer_error':max_norm,'max_csv_vs_final_epoch_prediction_error':max_saved_score,
     'reused_prior_check':'all 72 runs shared prior raw seed-specific encoder/head initialization and batch-order hashes; prior prediction metrics not repeated here',
     'optimizer_steps':62,'epochs':25,'training_windows':7840,'result':result,'script_sha256':sha(Path(__file__))}
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'audit_validation.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k!='result'},indent=2))
