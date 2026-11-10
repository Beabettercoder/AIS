"""Independently check all test metrics, thresholds and factorial bootstrap interactions."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
RUN=ROOT/'factorial_ablation_v1_20261027';PRIOR=ROOT/'causal_gate_main_v1_20261027';OUT=ROOT/'factorial_independent_audit_20261027'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

plan=json.loads((RUN/'frozen_test_plan.json').read_text(encoding='utf8'))
oldplan=json.loads((PRIOR/'frozen_test_plan.json').read_text(encoding='utf8'))
summary=json.loads((RUN/'historical_test_results/test_results.json').read_text(encoding='utf8'))
assert summary['frozen_plan_sha256']==sha(RUN/'frozen_test_plan.json') and plan['status']=='frozen_before_test'
prior=['raw','old','comp','causal','causal_common','causal_common_bundle']
runlist=[r for r in oldplan['runs'] if r['arm'] in prior]+plan['runs']
assert len(plan['runs'])==72 and len(runlist)==108
with np.load(RUN/'historical_test_results/all_test_predictions.npz',allow_pickle=False) as z: saved={k:z[k] for k in z.files}
y=saved['y'].astype(int);rgn=saved['region'].astype(str);mmsi=saved['mmsi'].astype(str);P=saved['scores']
assert len(y)==1680 and P.shape==(108,1680) and len(set(mmsi))==311
byresult={(r['architecture'],int(r['seed']),r['arm']):r for r in summary['results']}
allscore={}; max_metric=0.; max_threshold=0.; max_prior=0.
with np.load(PRIOR/'formal_test_results/all_test_predictions.npz',allow_pickle=False) as z:
    assert np.array_equal(saved['window_id'],z['window_id']) and np.array_equal(y,z['y'])
    oldscores={(str(a),int(s),str(k)):p.copy() for a,s,k,p in zip(z['architecture'],z['seed'],z['arm'],z['scores']) if str(k) in prior}
for i,run in enumerate(runlist):
    key=(run['architecture'],int(run['seed']),run['arm']);assert (str(saved['architecture'][i]),int(saved['seed'][i]),str(saved['arm'][i]))==key
    a=P[i];allscore[key]=a;row=byresult[key]
    vals=[(average_precision_score(y,a),row['pooled_AP']),(roc_auc_score(y,a),row['pooled_AUROC']),
          (np.mean([average_precision_score(y[rgn==r],a[rgn==r]) for r in np.unique(rgn)]),row['macro_AP'])]
    for obs,expected in vals:max_metric=max(max_metric,abs(float(obs)-expected));assert abs(float(obs)-expected)<1e-12
    base=PRIOR if key[2] in prior else RUN
    p=base/'runs'/key[0]/f'seed_{key[1]}'/key[2]
    assert sha(p/'checkpoint_final.pt')==run['checkpoint_sha256'] and sha(p/'val_predictions_final.csv')==run['validation_predictions_sha256']
    with (p/'val_predictions_final.csv').open(encoding='utf8',newline='') as f:vr=list(csv.DictReader(f))
    neg=np.sort(np.array([float(x['score']) for x in vr if int(x['y'])==0]))[::-1]
    for keyalpha in ('0.01','0.05'):
        j=int(np.floor(float(keyalpha)*len(neg)));th=float(np.nextafter(neg[min(j,len(neg)-1)],np.inf))
        assert run['thresholds'][keyalpha]['threshold']==th
        far=float(np.mean(a[y==0]>=th));recall=float(np.mean(a[y==1]>=th))
        max_threshold=max(max_threshold,abs(far-row[keyalpha]['actual_FAR']),abs(recall-row[keyalpha]['recall']))
        assert far==row[keyalpha]['actual_FAR'] and recall==row[keyalpha]['recall']
    if key in oldscores:
        max_prior=max(max_prior,float(np.max(np.abs(a-oldscores[key]))));assert np.array_equal(a,oldscores[key])

# Reconstruct clustering from the documented seed, using tie-group masses rather than
# the experiment's cumulative-endpoint formula. Bootstrap arrays are conditional on
# target MMSI and three fixed model seeds; shared background is not resampled.
ids,inv=np.unique(mmsi,return_inverse=True);rng=np.random.default_rng(20261027)
weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000).astype(float)[:,inv]
def aps(scores,y,w):
    idx=np.argsort(-scores,kind='stable');scores=scores[idx];yy=y[idx];w=w[:,idx]
    starts=np.r_[0,np.flatnonzero(scores[1:]!=scores[:-1])+1]
    mass=np.add.reduceat(w,starts,axis=1);pos=np.add.reduceat(w*yy[None,:],starts,axis=1)
    cm=np.cumsum(mass,axis=1);cp=np.cumsum(pos,axis=1)
    return (np.divide(cp,cm,out=np.zeros_like(cp),where=cm>0)*pos).sum(axis=1)/np.maximum(cp[:,-1],1)
boot={}
for key,a in allscore.items():
    pooled=aps(a,y,weights);regional=[];valid=np.ones(len(weights),bool)
    for region in np.unique(rgn):
        mask=rgn==region;yr=y[mask];w=weights[:,mask]
        valid &= ((w*(yr==1)).sum(1)>0)&((w*(yr==0)).sum(1)>0)
        regional.append(aps(a[mask],yr,w))
    macro=np.mean(regional,axis=0);macro[~valid]=np.nan;boot[key]={'AP':pooled,'macro_AP':macro}
comparison_error=0.;interaction_error=0.
for row in summary['paired_AP']:
    arch=row['architecture'];metric=row['metric'];A=row['a'];B=row['b']
    bs=np.mean(np.stack([boot[(arch,s,A)][metric]-boot[(arch,s,B)][metric] for s in (0,1,2)]),axis=0)
    ci=np.nanquantile(bs,[.025,.975]);err=float(np.max(np.abs(ci-np.array(row['CI95']))));comparison_error=max(comparison_error,err);assert err<1e-10
defs={'gate_by_bundle_support_off':('causal_bundle','causal','raw_bundle','raw'),
      'gate_by_bundle_support_on':('causal_common_bundle','causal_common','raw_common_bundle','raw_common'),
      'support_by_bundle_raw':('raw_common_bundle','raw_common','raw_bundle','raw'),
      'support_by_bundle_causal':('causal_common_bundle','causal_common','causal_bundle','causal')}
for row in summary['interactions']:
    arch=row['architecture'];metric=row['metric'];A,B,C,D=defs[row['name']]
    bs=np.mean(np.stack([boot[(arch,s,A)][metric]-boot[(arch,s,B)][metric]-boot[(arch,s,C)][metric]+boot[(arch,s,D)][metric] for s in (0,1,2)]),axis=0)
    ci=np.nanquantile(bs,[.025,.975]);err=float(np.max(np.abs(ci-np.array(row['CI95']))));interaction_error=max(interaction_error,err);assert err<1e-10
out={'status':'passed','runs':len(runlist),'new_checkpoints':len(plan['runs']),'reused_checkpoints':len(oldplan['runs']) if False else 36,
     'test_predictions':len(runlist)*len(y),'test_mmsi':len(ids),'bootstrap_draws':len(weights),
     'max_metric_error':max_metric,'max_threshold_metric_error':max_threshold,'max_prior_prediction_error':max_prior,
     'paired_interval_count':len(summary['paired_AP']),'max_paired_interval_error':comparison_error,
     'interaction_interval_count':len(summary['interactions']),'max_interaction_interval_error':interaction_error,
     'validation_and_checkpoint_hashes':'all passed','scope':'test sources historically exposed; MMSI-cluster CI fixes seeds and omits shared-background dependence',
     'limitations':'this audit checks stored predictions, thresholds and intervals, not an independently rebuilt model forward',
     'script_sha256':sha(Path(__file__))}
with (OUT/'audit_test.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps(out,ensure_ascii=False,indent=2))
