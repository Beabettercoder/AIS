"""Fixed-checkpoint 4-arm paired decomposition; exposed test, no selection."""
from pathlib import Path
import json,hashlib,csv,numpy as np
from sklearn.metrics import average_precision_score
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');N=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];A=['raw','causal_bundle','f0_bundle','f1_bundle'];V=2000
old=json.loads((P/'single_formal_summary.json').read_text());lock=json.loads((P/'frozen_checkpoint_thresholds.json').read_text());assert len(lock['arms'])==96

def sh(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def apw(s,y,W):
 o=np.argsort(-s,kind='stable');s=s[o];yy=y[o];w=W[:,o];e=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1];pos=np.add.reduceat(w*yy,e,axis=1);tot=np.add.reduceat(w,e,axis=1);cp=np.cumsum(pos,axis=1);ct=np.cumsum(tot,axis=1);precision=np.divide(cp,ct,out=np.zeros_like(cp),where=ct>0);return np.divide((pos*precision).sum(1),cp[:,-1],out=np.full(len(W),np.nan),where=cp[:,-1]>0)
def ci(x):return [float(v) for v in np.nanquantile(x,[.025,.975])]
allout={'scope':'Previously exposed test; fixed 3 models; 2,000 target-MMSI paired bootstrap. Comparisons may combine input and training changes.','analyst':'pair_decomposition.py','paired':{},'source_summary_sha256':sh(P/'single_formal_summary.json'),'checkpoint_lock_sha256':sh(P/'frozen_checkpoint_thresholds.json')};rows=[]
for n in N:
 data={s:{} for s in (0,1,2)}
 for seed in data:
  for a in A:
   with np.load(P/'runs'/n/f'seed{seed}'/'test'/(a+'_scores.npz'),allow_pickle=False) as z:data[seed][a]={k:z[k] for k in z.files}
 r=data[0]['raw'];y=r['y'];m=r['mmsi'].astype(str);regions=r['region'].astype(str)
 for seed in data:
  for a in A:
   for k in ('y','mmsi','region','window_id'):assert np.array_equal(data[seed][a][k],r[k])
 ids,inv=np.unique(m,return_inverse=True);rng=np.random.default_rng(20261101);draw=rng.integers(0,len(ids),(V,len(ids)));W=np.stack([np.bincount(row,minlength=len(ids))[inv] for row in draw]).astype(float)
 sim={}
 for a in A:
  pool=[];macro=[]
  for seed in (0,1,2):
   x=data[seed][a]['score'];pool.append(apw(x,y,W));macro.append(np.nanmean(np.stack([apw(x[regions==r],y[regions==r],W[:,regions==r]) for r in sorted(set(regions))]),axis=0))
  sim[a]={'pooled':np.array(pool),'macro':np.array(macro)}
 pairs=[('causal_minus_raw','causal_bundle','raw'),('f0_minus_causal','f0_bundle','causal_bundle'),('f1_minus_f0','f1_bundle','f0_bundle')];allout['paired'][n]={}
 for k,a,b in pairs:
  rec={}
  for metric in ('pooled','macro'):
   key='pooled_AP' if metric=='pooled' else 'macro_AP';pa=old['models'][n]['test']['performance'][a][key]['seeds'];pb=old['models'][n]['test']['performance'][b][key]['seeds'];d=np.array(pa)-np.array(pb);v=np.nanmean(sim[a][metric]-sim[b][metric],axis=0);rec[metric]={'mean':float(d.mean()),'per_seed':[float(x) for x in d],'CI95':ci(v)}
  allout['paired'][n][k]=rec;rows.append({'backbone':n,'comparison':k,'pooled_delta_pp':100*rec['pooled']['mean'],'pooled_lo_pp':100*rec['pooled']['CI95'][0],'pooled_hi_pp':100*rec['pooled']['CI95'][1],'macro_delta_pp':100*rec['macro']['mean'],'macro_lo_pp':100*rec['macro']['CI95'][0],'macro_hi_pp':100*rec['macro']['CI95'][1]})
 print('PAIRED',n,flush=True)
(P/'single_pair_decomposition.json').write_text(json.dumps(allout,indent=2,allow_nan=False))
with (P/'single_pair_decomposition.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fm(obj):return f"{obj['mean']:.4f} ± {obj['sd']:.4f}"
def delta(obj):return f"{obj['mean']*100:+.2f} [{obj['CI95'][0]*100:+.2f}, {obj['CI95'][1]*100:+.2f}]"
lines=['# 单主体：八骨干四组与逐项配对','','所有结果来自已曝光开发测试源；每种骨干三seed固定第25轮。AP为每seed平均±样本标准差。注册参数包含Raw时未调用的场适配器和辅助训练头，不能等同各臂的活跃参数。配对95%区间对固定三个模型重采样目标MMSI 2,000次，不覆盖种子总体、共享背景依赖及多重比较。','', '各组区别：Raw为自船BCE；Causal+bundle同时改变目标门控和训练包；F0在此基础上增加普通局部场且其四维控制器状态随场变化；F1将普通场换为可信场，相关控制器状态也随之变化。故各步是整体配置对照，不可把前两步分别称门控或场的单独纯因果效应。Graph单主体参考节点关闭；iTransformer/PatchTST的平滑使用时间代理。','', '## 四组 pooled AP 与八区等权 AP','', '|骨干|注册参数|Raw pooled / macro|Causal+bundle pooled / macro|F0 pooled / macro|F1 pooled / macro|','|---|---:|---|---|---|---|']
params=json.loads((P/'parameter_inventory.json').read_text())
for n in N:
 q=old['models'][n]['test']['performance'];vals=[fm(q[a]['pooled_AP'])+' / '+fm(q[a]['macro_AP']) for a in A];lines.append('|'+n+'|'+str(params[n]['total_registered'])+'|'+'|'.join(vals)+'|')
lines+=['','## pooled AP 逐项配对差（百分点及95%区间）','','|骨干|Causal+bundle − Raw|F0 − Causal+bundle|F1 − F0|','|---|---|---|---|']
for n in N:
 v=allout['paired'][n];lines.append('|'+n+'|'+'|'.join(delta(v[k]['pooled']) for k in ('causal_minus_raw','f0_minus_causal','f1_minus_f0'))+'|')
lines+=['','## 八区等权 AP 逐项配对差（百分点及95%区间）','','|骨干|Causal+bundle − Raw|F0 − Causal+bundle|F1 − F0|','|---|---|---|---|']
for n in N:
 v=allout['paired'][n];lines.append('|'+n+'|'+'|'.join(delta(v[k]['macro']) for k in ('causal_minus_raw','f0_minus_causal','f1_minus_f0'))+'|')
lines+=['','固定模型差异的逐seed和全部数字见 single_pair_decomposition.json/csv；0.1%/1%/5%验证阈值下实际测试FAR及召回、逐地区与污染分层见同目录 single_formal_summary.json、single_formal_strata.csv。','']
(P/'单主体四组完整对照与配对区间.md').write_text('\n'.join(lines),encoding='utf8')
