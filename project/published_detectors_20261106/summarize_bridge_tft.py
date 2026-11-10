import argparse,json,csv
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from summarize_eight_matched import paired_bootstrap

p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
root=Path('/home/ps/AIS_bridge_20261107/tft_eval128');manifest=json.loads((root/'validation_frozen.json').read_text(encoding='utf8'))
groups=[('aux0_count',x) for x in ('raw','gate','direct','late','bundle','original')]+[('aux0_legacy','original'),('aux0.1_count','direct')]
ref=None;data={};rows=[]
for condition,arm in groups:
 for seed in range(3):
  file=root/condition/arm/f'seed{seed}'/(a.split+'_scores.npz')
  with np.load(file,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  if ref is None:ref=d
  elif not all(np.array_equal(ref[k],d[k]) for k in ('y','mmsi','region','window_id')):raise ValueError(f'Unpaired: {file}')
  data[(condition,arm,seed)]=d
  rows.append({'condition':condition,'arm':arm,'seed':seed,'AP':float(average_precision_score(d['y'],d['score'])),'AUROC':float(roc_auc_score(d['y'],d['score']))})
pairs=[(('aux0_count','raw'),('aux0_count','gate')),(('aux0_count','gate'),('aux0_count','direct')),(('aux0_count','direct'),('aux0_count','late')),(('aux0_count','late'),('aux0_count','bundle')),(('aux0_count','bundle'),('aux0_count','original')),(('aux0_count','original'),('aux0_legacy','original')),(('aux0_count','direct'),('aux0.1_count','direct'))]
pairrows=[]
for before,after in pairs:
 old=[data[(*before,s)]['score'] for s in range(3)];new=[data[(*after,s)]['score'] for s in range(3)]
 ci=paired_bootstrap(ref['y'],new,old,ref['mmsi'],a.bootstrap,1729)
 diffs=[float(average_precision_score(ref['y'],n)-average_precision_score(ref['y'],o)) for n,o in zip(new,old)]
 pairrows.append({'before':'/'.join(before),'after':'/'.join(after),'seed_deltas':diffs,**ci})
means=[]
for condition,arm in groups:
 group=[x for x in rows if x['condition']==condition and x['arm']==arm]
 means.append({'condition':condition,'arm':arm,'AP_mean':float(np.mean([x['AP'] for x in group])),'AP_sd':float(np.std([x['AP'] for x in group],ddof=1)),'AUROC_mean':float(np.mean([x['AUROC'] for x in group]))})
out=root/(a.split+'_summary.json');out.write_text(json.dumps({'rows':rows,'means':means,'paired':pairrows,'ci':'95% target MMSI cluster bootstrap, conditional on three seeds','note':'original-time TFT arm has right padding; TFT blocks do not consume masks or timestamps'},indent=2),encoding='utf8')
print(a.split,'summary saved',out,flush=True)
