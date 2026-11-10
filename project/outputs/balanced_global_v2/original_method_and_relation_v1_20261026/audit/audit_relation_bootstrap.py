"""Replay paired target-MMSI bootstrap by explicit row resampling."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

SCORERS=('target_only','neighbor_only','relative_mean','relative_median','relative_reliable')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(ok,msg):
    if not ok:raise AssertionError(msg)
def load(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return [x for x in csv.DictReader(f) if x['split']=='val']
def interval(v):return dict(lower_2_5=float(np.percentile(v,2.5)),upper_97_5=float(np.percentile(v,97.5)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--relation',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();r=a.relation
    check(not a.out.exists(),'Refusing overwrite')
    protocol=json.loads((r/'bootstrap_protocol_v1.json').read_text());report=json.loads((r/'bootstrap_v1/paired_mmsi_bootstrap.json').read_text())
    check(report['protocol_sha256']==sha(r/'bootstrap_protocol_v1.json'),'protocol hash')
    base=load(r/'base_statistics_v2/scores.csv');stress=load(r/'speed_stress_statistics_v2/scores.csv')
    fields=('base_window_id','mmsi','region','case_index','y')
    check([[v[k] for k in fields] for v in base]==[[v[k] for k in fields] for v in stress],'paired observations')
    check(sha(r/'base_statistics_v2/scores.csv')==report['base_scores_sha256'] and sha(r/'speed_stress_statistics_v2/scores.csv')==report['stress_scores_sha256'],'score hashes')
    y=np.array([int(v['y']) for v in base]);ids=np.array([v['mmsi'] for v in base]);unique=np.unique(ids)
    groups=[np.flatnonzero(ids==u) for u in unique]
    check(len(unique)==report['validation_target_mmsi']==58,'cluster unit')
    for group in groups:check(y[group].sum()*2==len(group),'cluster sibling balance')
    scores=np.array([[[float(v[k]) for v in rows] for k in SCORERS] for rows in [base,stress]])
    with np.load(r/'bootstrap_v1/bootstrap_draw_metrics.npz',allow_pickle=False) as z:
        stored=z['metrics'].copy();check(np.array_equal(z['mmsi'],unique),'saved cluster identities')
    expected=np.empty_like(stored);rng=np.random.default_rng(protocol['seed']);digest=hashlib.sha256()
    for b in range(protocol['replicates']):
        draw=rng.integers(len(unique),size=len(unique));digest.update(draw.astype('<i8').tobytes())
        rows=np.concatenate([groups[k] for k in draw])
        for view in range(2):
            for k in range(len(SCORERS)):
                expected[b,view,k]=[average_precision_score(y[rows],scores[view,k,rows]),roc_auc_score(y[rows],scores[view,k,rows])]
        if (b+1)%500==0:print('BOOTSTRAP_AUDITED',b+1,flush=True)
    maxerr=float(np.max(abs(expected-stored)));check(maxerr<1e-12,'replicate metrics differ')
    check(digest.hexdigest()==report['draws_sha256'],'cluster draws changed')
    points=np.empty((2,5,2))
    for view,name in enumerate(['base','speed_stress']):
        for k,scorer in enumerate(SCORERS):
            points[view,k]=[average_precision_score(y,scores[view,k]),roc_auc_score(y,scores[view,k])]
            for m,metric in enumerate(['AP','AUROC']):
                obj=report['metric_intervals'][name][scorer][metric];ci=interval(expected[:,view,k,m])
                check(abs(obj['point']-points[view,k,m])<1e-12,'point metric')
                for key,value in ci.items():check(abs(value-obj[key])<1e-12,'metric interval')
    for obj in report['contrasts']:
        if obj['view']=='speed_stress-minus-base':
            k=SCORERS.index(obj['scorer']);d=expected[:,1,k]-expected[:,0,k];pt=points[1,k]-points[0,k]
        else:
            view=['base','speed_stress'].index(obj['view']);fr=SCORERS.index(obj['from']);to=SCORERS.index(obj['to']);d=expected[:,view,to]-expected[:,view,fr];pt=points[view,to]-points[view,fr]
        for m,metric in enumerate(['AP','AUROC']):
            check(abs(obj[metric]['point']-pt[m])<1e-12,'contrast point')
            for key,value in interval(d[:,m]).items():check(abs(value-obj[metric][key])<1e-12,'contrast interval')
    output=dict(status='passed',scope='All2000 paired target-MMSI cluster replicates independently reconstructed by explicit row duplication; all pooled AP/AUROC confidence intervals and paired contrasts checked. No macro-region interval inferred.',clusters=len(unique),replicates=protocol['replicates'],maximum_replicate_metric_error=maxerr,draws_sha256=digest.hexdigest(),source_report_sha256=sha(r/'bootstrap_v1/paired_mmsi_bootstrap.json'),auditor_sha256=sha(Path(__file__)))
    with a.out.open('x',encoding='utf-8') as f:json.dump(output,f,indent=2)
    print(json.dumps(output))
if __name__=='__main__':main()
