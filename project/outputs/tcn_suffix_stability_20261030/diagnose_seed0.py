from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score

base=Path('/mnt/sda/WangZijia/AIS/outputs')
src=base/'heading_alignment_20261030/runs/aligned_embed/tcn'
dest=base/'tcn_suffix_stability_20261030'
manifest=json.loads((base/'balanced_global_v2/relation_dual_pressure_v2_20261027/dataset_manifest_combined.json').read_text())
rows={str(r['window_id']):r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='val'}
with np.load(base/'relation_repair_20261030/val_relation_field.npz') as z: d={k:z[k] for k in z.files}
pred={}
for arm in ('F0','F1'):
 with np.load(src/arm/'val_scores.npz') as z: pred[arm]={k:z[k] for k in z.files}
for key in ('y','case','source','mmsi','region'):
 np.testing.assert_array_equal(pred['F0'][key],pred['F1'][key]);np.testing.assert_array_equal(pred['F0'][key],d[key])
ids=d['source'].astype(str); y=d['y']; u=np.unique(ids); fields=d['suffix__field'];length=d['length']
first=np.array([np.flatnonzero(ids==i)[0] for i in u]);counts=np.array([len(rows[i]['cohort_mmsi']) for i in u]); hits=np.array([sum(m['indices']['suffix']['suffix_grid_hits'] for m in rows[i]['stress']['members']) for i in u]); selected=np.array([rows[i]['stress']['selected_count'] for i in u]); regions=d['region'][first]; nboats=len(np.unique(d['mmsi']))
groups={'all':np.ones(len(u),bool),'3_neighbors':counts==3,'4plus_neighbors':counts>=4,'suffix_hits_0':hits==0,'suffix_hits_1_2':(hits>0)&(hits<=2),'suffix_hits_3plus':hits>=3}
for region in sorted(set(regions)):groups['region_'+region]=regions==region
result={'n_sources':len(u),'n_target_mmsi':nboats,'case_count':len(y),'groups':{},'source_records':[]}
for group,mask in groups.items():
 chosen=u[mask]; ix=np.isin(ids,chosen); result['groups'][group]={'sources':int(mask.sum()),'target_mmsi':int(len(np.unique(d['mmsi'][ix]))),'cases':int(ix.sum())}
 if ix.sum() and len(np.unique(y[ix]))==2:
  for view in ('base','natural','suffix','combined_l100_m600'):
   ap={a:float(average_precision_score(y[ix],pred[a][view+'_score'][ix])) for a in ('F0','F1')};result['groups'][group][view]={'F0':ap['F0'],'F1':ap['F1'],'difference_pp':100*(ap['F1']-ap['F0'])}
for i,idx,n,h,s in zip(u,first,counts,hits,selected):
 ix=ids==i; modes=fields[idx,0,:int(length[idx]),7];scores={a:pred[a]['suffix_score'][ix].tolist() for a in ('F0','F1')}
 result['source_records'].append({'source':i,'mmsi':str(d['mmsi'][idx]),'region':str(d['region'][idx]),'candidate_members':int(n),'suffix_grid_hits':int(h),'selected_members':int(s),'valid_steps':int((modes>0).sum()),'zero_order_steps':int((modes==1).sum()),'affine_steps':int((modes==2).sum()),'label':y[ix].tolist(),'scores':scores})
(dest/'seed0_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='source_records'},indent=2))
