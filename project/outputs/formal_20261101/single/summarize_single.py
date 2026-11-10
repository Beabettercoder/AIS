"""Complete paired single-vessel summary; no test-based model selection."""
from pathlib import Path
import sys,json,csv,hashlib,numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');N=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];A=['raw','causal_bundle','f0_bundle','f1_bundle'];B=2000
R=json.loads((P/'test_launch_result.json').read_text());assert len(R)==24 and all(x['exit']==0 for x in R)
def scores(name,seed,arm,split):
 p=P/'runs'/name/f'seed{seed}'
 path=p/arm/'final_val.npz' if split=='validation' else p/'test'/(arm+'_scores.npz')
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def apweight(s,y,W):
 order=np.argsort(-s,kind='stable');s=s[order];y=y[order];w=W[:,order];edges=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1];pos=np.add.reduceat(w*y,edges,axis=1);allw=np.add.reduceat(w,edges,axis=1);cumpos=np.cumsum(pos,axis=1);cumall=np.cumsum(allw,axis=1);p=np.divide(cumpos,cumall,out=np.zeros_like(cumpos),where=cumall>0);tp=cumpos[:,-1];return np.divide((pos*p).sum(1),tp,out=np.full(len(W),np.nan),where=tp>0)
def std(x):return float(np.std(x,ddof=1))
def stats(x):return {'mean':float(np.mean(x)),'sd':std(x),'seeds':[float(y) for y in x]}
def ci(x):return [float(a) for a in np.nanquantile(x,[.025,.975])]
out={'scope':'8-region previously exposed development test; fixed epoch 25; target-MMSI bootstrap reflects fixed models/sample uncertainty, not new unseen test or seed uncertainty','bootstrap_draws':B,'models':{}}
rows=[]
for n in N:
 out['models'][n]={}
 for split in ('validation','test'):
  data={s:{a:scores(n,s,a,split) for a in A} for s in (0,1,2)};ref=data[0]['raw'];y=ref['y'];mmsi=ref['mmsi'].astype(str);regions=ref['region'].astype(str);ids=ref['window_id'].astype(str)
  assert len(y)==1680 and len(set(ids))==1680
  for s in data:
   for a in A:
    for k in ('y','mmsi','region','window_id'):assert np.array_equal(data[s][a][k],ref[k]),(n,split,s,a,k)
  unique,inv=np.unique(mmsi,return_inverse=True);rng=np.random.default_rng(20261101);draw=rng.integers(0,len(unique),size=(B,len(unique)));counts=np.stack([np.bincount(d,minlength=len(unique)) for d in draw]);W=counts[:,inv].astype(np.float64)
  performance={};sim={};sim_macro={}
  for a in A:
   pooled=[];macro=[];auroc=[];bs=[];bs_macro=[]
   for s in (0,1,2):
    z=data[s][a]['score'];pooled.append(average_precision_score(y,z));macro.append(np.mean([average_precision_score(y[regions==r],z[regions==r]) for r in sorted(set(regions))]));auroc.append(roc_auc_score(y,z));bs.append(apweight(z,y,W));bs_macro.append(np.nanmean(np.stack([apweight(z[regions==r],y[regions==r],W[:,regions==r]) for r in sorted(set(regions))]),axis=0))
   performance[a]={'pooled_AP':stats(pooled),'macro_AP':stats(macro),'AUROC':stats(auroc),'pooled_AP_mmsi_CI':ci(np.mean(bs,axis=0)),'macro_AP_mmsi_CI':ci(np.mean(bs_macro,axis=0))};sim[a]=np.array(bs);sim_macro[a]=np.array(bs_macro)
   if split=='test':
    ops={};
    for budget in ('0.001','0.01','0.05'):
     items=[json.loads((P/'runs'/n/f'seed{s}'/'test/results.json').read_text())['metrics'][a]['operating_points'][budget] for s in (0,1,2)]
     ops[budget]={'actual_FAR':stats([q['actual_test_FAR'] for q in items]),'recall':stats([q['test_recall'] for q in items])}
    performance[a]['operating_points']=ops
   row={'backbone':n,'split':split,'arm':a,'pooled_AP_mean':float(np.mean(pooled)),'pooled_AP_sd':std(pooled),'macro_AP_mean':float(np.mean(macro)),'macro_AP_sd':std(macro),'AUROC_mean':float(np.mean(auroc))}
   if split=='test':
    for budget in ('0.001','0.01','0.05'):
     row['recall_at_val_FAR_'+budget]=performance[a]['operating_points'][budget]['recall']['mean'];row['actual_FAR_'+budget]=performance[a]['operating_points'][budget]['actual_FAR']['mean']
   rows.append(row)
  paired={}
  for a in ('raw','causal_bundle','f0_bundle'):
   dmean=float(np.mean([performance['f1_bundle']['pooled_AP']['seeds'][s]-performance[a]['pooled_AP']['seeds'][s] for s in (0,1,2)]));paired['f1_minus_'+a]={'mean':dmean,'CI95':ci(np.mean(sim['f1_bundle']-sim[a],axis=0)),'seed_differences':[float(performance['f1_bundle']['pooled_AP']['seeds'][s]-performance[a]['pooled_AP']['seeds'][s]) for s in (0,1,2)]}
  paired_macro={}
  for a in ('raw','causal_bundle','f0_bundle'):
   ds=[float(performance['f1_bundle']['macro_AP']['seeds'][s]-performance[a]['macro_AP']['seeds'][s]) for s in (0,1,2)]
   paired_macro['f1_minus_'+a]={'mean':float(np.mean(ds)),'CI95':ci(np.nanmean(sim_macro['f1_bundle']-sim_macro[a],axis=0)),'seed_differences':ds}
  out['models'][n][split]={'performance':performance,'paired_pooled_AP':paired,'paired_macro_AP':paired_macro,'MMSI_count':len(unique),'windows':len(y)}
  print(n,split,'F1',performance['f1_bundle']['pooled_AP']['mean'],paired['f1_minus_raw'],flush=True)
# Aggregate exposed test perturbation strata without recasting event generation as real labels.
strata=[]
for n in N:
 for a in A:
  allr=[json.loads((P/'runs'/n/f'seed{seed}'/'test/results.json').read_text())['metrics'][a] for seed in (0,1,2)]
  for field in ('strata_noise_type','strata_noise_scope','strata_observation_noise_effective','strata_target_noise_effective','strata_neighbor_noise_effective'):
   for group in sorted(allr[0].get(field,{})):
    entries=[q[field][group] for q in allr];values=[q['AP'] for q in entries]
    strata.append({'backbone':n,'arm':a,'stratum_type':field.removeprefix('strata_'),'group':group,'n':entries[0]['n'],'positive':entries[0]['positive'],'negative':entries[0]['negative'],'AP_mean':float(np.mean(values)) if all(x is not None for x in values) else None,'AP_sd':std(values) if all(x is not None for x in values) else None})
out['test_strata']=strata
(P/'single_formal_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False));cols=sorted(set(k for r in rows for k in r))
with (P/'single_formal_table.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
with (P/'single_formal_strata.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=list(strata[0]));w.writeheader();w.writerows(strata)
lines=['# 单主体八骨干三种子：固定末轮结果','','同一8区域训练/验证/测试划分；每臂固定25轮，seed0/1/2。测试来源此前已曝光，不是独立盲测。阈值仅由验证负类确定；FAR是实际测试误报。区间是固定模型的目标MMSI分组bootstrap，不代表训练种子总体不确定性。原始输入+模型与F1同时改变多个组件，不能将差异单独归因于可信场。','','|骨干|Raw pooled AP|F1 pooled AP|F1−Raw pp [95% CI]|Raw宏平均 AP|F1宏平均 AP|F1 AUC|F1验证1%阈值下实际FAR/召回|','|---|---:|---:|---|---:|---:|---:|---|']
for n in N:
 d=out['models'][n]['test'];r=d['performance']['raw'];f=d['performance']['f1_bundle'];q=d['paired_pooled_AP']['f1_minus_raw'];op=f['operating_points']['0.01'];lines.append(f"|{n}|{r['pooled_AP']['mean']:.4f} ± {r['pooled_AP']['sd']:.4f}|{f['pooled_AP']['mean']:.4f} ± {f['pooled_AP']['sd']:.4f}|{q['mean']*100:+.2f} [{q['CI95'][0]*100:+.2f},{q['CI95'][1]*100:+.2f}]|{r['macro_AP']['mean']:.4f}|{f['macro_AP']['mean']:.4f}|{f['AUROC']['mean']:.4f}|{op['actual_FAR']['mean']:.3f} / {op['recall']['mean']:.3f}|")
lines+=['','所有四臂、验证/测试、0.1%/1%/5%阈值、逐地区和受控污染分层详见 single_formal_summary.json、single_formal_strata.csv 与各模型 test/results.json。测试正负样本分组由既有受控标签指定，非真实事故真值。','']
(P/'单主体八骨干三种子正式结果.md').write_text('\n'.join(lines),encoding='utf8')
