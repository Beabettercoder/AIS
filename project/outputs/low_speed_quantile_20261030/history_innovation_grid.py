"""One training-only grid, 15 fixed candidates, no validation/test use.
Weights and counters from history_innovation_records; cluster bootstrap for the first and last rule.
"""
import numpy as np,json
from pathlib import Path
O=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_quantile_20261030')
a=np.load(O/'history_innovation_records.npz',allow_pickle=False)['suffix']
lo=a['h']<2;sets={'altered_truebase_low':lo&a['changed']&(a['base_speed']<2),'unaltered_historylow_accel1kn':lo&~a['changed']&~a['prev']&~a['pos']&a['natural_accel'],'unaltered_current_low':lo&~a['changed']&~a['prev']&~a['pos']&(a['base_speed']<2)}
res={}; grids=[]
for e in [.5,.8,1.2,1.5,2.]:
 for k in [1.,2.,3.]:
  w=a['old']*np.where(lo&(a['ecur']>=e),np.exp(-.5*np.maximum(0,a['ratio']-k)**2),1.0)
  result={'e_cur_min_mps':e,'ratio_onset':k,'stats':{}}
  for group,mask in sets.items():
   d=(w<.3).astype(float)-(a['old']<.3).astype(float)
   result['stats'][group]={'n':int(mask.sum()),'old_rate':float((a['old'][mask]<.3).mean()),'new_rate':float((w[mask]<.3).mean()),'mean_weight_change':float((w[mask]-a['old'][mask]).mean())}
   if (e,k) in [(.5,1.),(2.,3.)]:
    q=a[mask]; y=d[mask]; ids,inv=np.unique(q['target'],return_inverse=True); num=np.bincount(inv,weights=y); den=np.bincount(inv); rng=np.random.default_rng(20261030); draws=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000); v=(draws@num)/np.maximum(draws@den,1);result['stats'][group]['paired_cluster_95']=np.quantile(v,[.025,.975]).tolist();result['stats'][group]['target_count']=len(ids)
  grids.append(result)
res={'scope':'training exploration only; 400 SHA-sorted train sources; changed flag is synthetic diagnostic label never input; baseline natural accuracy unknown; repeated member reports across contexts; no validation/test accessed','grid':grids}
(O/'history_innovation_grid.json').write_text(json.dumps(res,indent=2),encoding='utf8');print(json.dumps({'first':grids[0],'last':grids[-1]},indent=2))
