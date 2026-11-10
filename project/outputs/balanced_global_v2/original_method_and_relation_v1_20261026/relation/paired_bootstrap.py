"""Target-MMSI cluster bootstrap over paired fixed validation predictions."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from observed_statistics_v2 import NAMES

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return [r for r in csv.DictReader(f) if r['split']=='val']
def interval(values):
    return {'lower_2_5':float(np.percentile(values,2.5)),'upper_97_5':float(np.percentile(values,97.5))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--stress',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text());base=load(a.base/'scores.csv');stress=load(a.stress/'scores.csv')
    keys=lambda rows:[tuple(r[k] for k in ['base_window_id','mmsi','region','case_index','y']) for r in rows]
    assert keys(base)==keys(stress)
    y=np.array([int(r['y']) for r in base]);ids=np.array([r['mmsi'] for r in base]);unique=np.unique(ids);lookup={x:i for i,x in enumerate(unique)};groups=np.array([lookup[x] for x in ids])
    for uid in unique:
        labels=y[ids==uid];assert labels.sum()*2==len(labels)
    score=np.array([[[float(r[n]) for r in rows] for n in NAMES] for rows in [base,stress]])
    B=protocol['replicates'];rng=np.random.default_rng(protocol['seed']);values=np.empty((B,2,len(NAMES),2));draw_hash=hashlib.sha256()
    for j in range(B):
        draw=rng.integers(len(unique),size=len(unique));draw_hash.update(draw.astype('<i8').tobytes());counts=np.bincount(draw,minlength=len(unique));w=counts[groups]
        for view in range(2):
            for k,n in enumerate(NAMES):
                values[j,view,k]=[average_precision_score(y,score[view,k],sample_weight=w),roc_auc_score(y,score[view,k],sample_weight=w)]
        if (j+1)%500==0:print('BOOTSTRAP',j+1,flush=True)
    for k in [0,1]:assert np.max(np.abs(values[:,:,k,:]-.5))<1e-12
    out={'status':'passed','protocol_sha256':sha(a.protocol),'code_sha256':sha(Path(__file__)),'base_scores_sha256':sha(a.base/'scores.csv'),'stress_scores_sha256':sha(a.stress/'scores.csv'),'replicates':B,'seed':protocol['seed'],'draws_sha256':draw_hash.hexdigest(),'validation_target_mmsi':len(unique),'validation_source_windows':len(set(r['base_window_id'] for r in base)),'validation_scenarios':len(base),'metric_intervals':{},'contrasts':[],'scope':protocol['scope'],'limitation':protocol['limitation']}
    for v,view in enumerate(['base','speed_stress']):
        out['metric_intervals'][view]={}
        for k,n in enumerate(NAMES):
            point=[float(average_precision_score(y,score[v,k])),float(roc_auc_score(y,score[v,k]))]
            out['metric_intervals'][view][n]={metric:{'point':point[m],**interval(values[:,v,k,m])} for m,metric in enumerate(['AP','AUROC'])}
        for fr,to in [('relative_mean','relative_reliable'),('relative_mean','relative_median'),('relative_median','relative_reliable')]:
            i,jj=NAMES.index(fr),NAMES.index(to)
            out['contrasts'].append({'view':view,'from':fr,'to':to,**{metric:{'point':out['metric_intervals'][view][to][metric]['point']-out['metric_intervals'][view][fr][metric]['point'],**interval(values[:,v,jj,m]-values[:,v,i,m])} for m,metric in enumerate(['AP','AUROC'])}})
    for k,n in enumerate(NAMES):out['contrasts'].append({'view':'speed_stress-minus-base','scorer':n,**{metric:{'point':out['metric_intervals']['speed_stress'][n][metric]['point']-out['metric_intervals']['base'][n][metric]['point'],**interval(values[:,1,k,m]-values[:,0,k,m])} for m,metric in enumerate(['AP','AUROC'])}})
    np.savez_compressed(a.out/'bootstrap_draw_metrics.npz',metrics=values,views=np.array(['base','speed_stress']),scorers=np.array(NAMES),metric_names=np.array(['AP','AUROC']),mmsi=unique)
    (a.out/'paired_mmsi_bootstrap.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({'status':'passed','clusters':len(unique),'replicates':B}),flush=True)
if __name__=='__main__':main()
