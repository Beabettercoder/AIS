"""Reuse the original target-MMSI bootstrap draws for exploratory followup."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from observed_statistics_v2 import sha

def rows(p):
    with p.open(newline='',encoding='utf-8') as f:return [x for x in csv.DictReader(f) if x['split']=='val']
def ci(v):return {'lower_2_5':float(np.percentile(v,2.5)),'upper_97_5':float(np.percentile(v,97.5))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--followup',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    reference=read_json=a.root/'bootstrap_v1/paired_mmsi_bootstrap.json';old=json.loads(reference.read_text());B=old['replicates'];rng=np.random.default_rng(old['seed'])
    origrows=rows(a.root/'base_statistics_v2/scores.csv');groupsid=np.array([x['mmsi'] for x in origrows]);unique=np.unique(groupsid);group={m:i for i,m in enumerate(unique)};g=np.array([group[x] for x in groupsid]);y=np.array([int(x['y']) for x in origrows])
    keys=lambda rs:[tuple(r[k] for k in ['base_window_id','mmsi','region','case_index','y']) for r in rs]
    rr=[rows(a.followup/v/'scores.csv') for v in ['base','speed_stress']];assert all(keys(rs)==keys(origrows) for rs in rr)
    scores=np.array([[float(x['relative_report_weighted']) for x in rs] for rs in rr]);values=np.empty((B,2,2));digest=hashlib.sha256()
    with np.load(a.root/'bootstrap_v1/bootstrap_draw_metrics.npz',allow_pickle=False) as z:
        assert np.array_equal(z['mmsi'],unique);original_values=z['metrics'].copy();scorers=z['scorers'].astype(str).tolist()
    for j in range(B):
        draw=rng.integers(len(unique),size=len(unique));digest.update(draw.astype('<i8').tobytes());w=np.bincount(draw,minlength=len(unique))[g]
        for v in range(2):values[j,v]=[average_precision_score(y,scores[v],sample_weight=w),roc_auc_score(y,scores[v],sample_weight=w)]
    assert digest.hexdigest()==old['draws_sha256']
    report={'status':'passed','scope':'Post-diagnostic exploratory followup; paired target-MMSI draws exactly reused from original bootstrap. No new model/data/threshold selection.','original_bootstrap_report_sha256':sha(reference),'original_bootstrap_arrays_sha256':sha(a.root/'bootstrap_v1/bootstrap_draw_metrics.npz'),'draws_sha256':digest.hexdigest(),'replicates':B,'target_mmsi':len(unique),'validation_sources':len(set(x['base_window_id'] for x in origrows)),'code_sha256':sha(Path(__file__)),'points_and_intervals':{},'paired_contrasts':[]}
    for v,view in enumerate(['base','speed_stress']):
        point=[float(average_precision_score(y,scores[v])),float(roc_auc_score(y,scores[v]))]
        report['points_and_intervals'][view]={metric:{'point':point[m],**ci(values[:,v,m])} for m,metric in enumerate(['AP','AUROC'])}
        for comparator in ['relative_mean','relative_reliable','relative_median']:
            j=scorers.index(comparator);report['paired_contrasts'].append({'view':view,'new_scorer':'relative_report_weighted','comparator':comparator,**{metric:{'point':point[m]-old['metric_intervals'][view][comparator][metric]['point'],**ci(values[:,v,m]-original_values[:,v,j,m])} for m,metric in enumerate(['AP','AUROC'])}})
    report['stress_minus_base']={metric:{'point':report['points_and_intervals']['speed_stress'][metric]['point']-report['points_and_intervals']['base'][metric]['point'],**ci(values[:,1,m]-values[:,0,m])} for m,metric in enumerate(['AP','AUROC'])}
    np.savez_compressed(a.out/'draw_metrics.npz',metrics=values,mmsi=unique,views=np.array(['base','speed_stress']),metric_names=np.array(['AP','AUROC']))
    (a.out/'paired_bootstrap.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps({'status':'passed','contrasts':report['paired_contrasts']}),flush=True)
if __name__=='__main__':main()
