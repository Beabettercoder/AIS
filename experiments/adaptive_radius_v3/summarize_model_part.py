"""Paired target-MMSI bootstrap for one task/backbone, after frozen test scoring."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from radius_core import ARMS
import train_single_v2 as single
import train_relation_v2 as relation
sys.path.insert(0,'/mnt/sda/WangZijia/AIS/published_detectors_20261106')
from summarize_eight_matched import paired_bootstrap
VIEWS=('base','natural','suffix','combined_l100_m600');PORTS=('houston','ny','la','shanghai')
PAIRS=(('target_only','tight_plain'),('tight_plain','wide_plain'),('wide_plain','adaptive_plain_nofallback'),('adaptive_plain_nofallback','adaptive_plain_fallback'),('wide_plain','adaptive_plain_fallback'),('adaptive_plain_fallback','adaptive_qualified_fallback'),('adaptive_qualified_fallback','adaptive_credible_fallback'),('adaptive_plain_fallback','adaptive_credible_fallback'),('wide_plain','adaptive_credible_fallback'),('tight_plain','adaptive_credible_fallback'),('adaptive_credible_fallback','adaptive_credible_deg6'),('adaptive_plain_fallback','adaptive_plain_deg6'))
def write(path,rows):
 keys=list(dict.fromkeys(k for x in rows for k in x))
 with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=('single','relation'),required=True);p.add_argument('--name',choices=('tcn','gru','transformer','graph'),required=True);p.add_argument('--root',type=Path,default=Path('/home/ps/AIS_adaptive_radius_relation_v3_20261109'));p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args();root=a.root
 manifest=json.loads((root/'frozen_manifest.json').read_text());base,_=(single.data(root,'test') if a.task=='single' else relation.data(root,'test'))
 y=base['y'];ships=base['mmsi'].astype(str);ports=np.isin(base['region'],PORTS)
 sparse=np.array([np.median(v[m])<4 for v,m in zip(base['eligible_count_8nm'],base['mask'])]) if a.task=='single' else base['sparse_source']
 groups={'all':np.ones(len(y),bool),'ports':ports,'restricted_channels':~ports,'sparse_overlapping':sparse}
 refkeys=('y','mmsi','region','window_id') if a.task=='single' else ('y','mmsi','region','source','case')
 records={};metrics_data={};rows=[];pairs=[]
 for ck in ('last','best_val'):
  for arm in ARMS:
   for seed in (0,1,2):
    key=f'{a.task}/{a.name}/{arm}/seed{seed}';folder=root/'runs'/key
    if key not in manifest['runs']:raise ValueError('missing frozen key '+key)
    if single.sha(folder/f'{ck}.pt')!=manifest['runs'][key][ck]:raise ValueError('checkpoint mismatch '+key)
    with np.load(folder/f'test_{ck}_scores.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    m=json.loads((folder/f'test_{ck}_metrics.json').read_text())
    for k in refkeys:
     if not np.array_equal(base[k],d[k]):raise ValueError('sample identity mismatch '+key+k)
    for view in VIEWS:
     if len(d[view])!=len(y) or not np.isfinite(d[view]).all():raise ValueError('invalid score '+key+view)
     if abs(float(average_precision_score(y,d[view]))-m['views'][view]['AP' if a.task=='relation' else 'pooled_AP'])>1e-10:raise ValueError('AP mismatch '+key+view)
    records[(ck,arm,seed)]=d;metrics_data[(ck,arm,seed)]=m
 for ck in ('last','best_val'):
  for view in VIEWS:
   gg=dict(groups)
   if view!='base':
    hit=records[(ck,'target_only',0)][view+'_hit']
    for arm in ARMS:
     for seed in (0,1,2):
      if not np.array_equal(hit,records[(ck,arm,seed)][view+'_hit']):raise ValueError('stress hit differs across arms')
    gg['direct_hit_all']=hit.any(1)
   for group,sel in gg.items():
    yy=y[sel];ss=ships[sel]
    if len(np.unique(yy))<2 or len(np.unique(ss))<2:continue
    for arm in ARMS:
     for seed in (0,1,2):
      pp=records[(ck,arm,seed)][view][sel];threshold=metrics_data[(ck,arm,seed)]['thresholds'];row={'task':a.task,'model':a.name,'checkpoint':ck,'view':view,'group':group,'arm':arm,'seed':seed,'windows':int(sel.sum()),'positive':int(yy.sum()),'ships':len(np.unique(ss)),
        'AP':float(average_precision_score(yy,pp)),'AUROC':float(roc_auc_score(yy,pp))}
      for k,t in threshold.items():row['actual_FAR_'+k]=float(np.mean(pp[yy==0]>=t));row['recall_'+k]=float(np.mean(pp[yy==1]>=t))
      rows.append(row)
    if ck!='last':continue
    for before,after in PAIRS:
     aa=[records[(ck,after,s)][view][sel] for s in (0,1,2)];bb=[records[(ck,before,s)][view][sel] for s in (0,1,2)]
     try:result=paired_bootstrap(yy,aa,bb,ss,a.bootstrap,1729)
     except ValueError as e:result={'reason':str(e)}
     pairs.append({'task':a.task,'model':a.name,'view':view,'group':group,'before':before,'after':after,**result})
  print('SUMMARY_CHECKPOINT',a.task,a.name,ck,flush=True)
 out=root/'summary_parts';out.mkdir(exist_ok=True);write(out/f'{a.task}_{a.name}_metrics.csv',rows);write(out/f'{a.task}_{a.name}_pairs.csv',pairs)
 (out/f'{a.task}_{a.name}.json').write_text(json.dumps({'status':'complete','task':a.task,'model':a.name,'rows':len(rows),'pairs':len(pairs),'bootstrap':a.bootstrap,'note':'target MMSI clustered conditional on the three trained seeds; heldout test previously used; comparisons not multiplicity adjusted'},indent=2),encoding='utf8')
 print('SUMMARY_PART_COMPLETE',a.task,a.name,len(rows),len(pairs),flush=True)
if __name__=='__main__':main()
