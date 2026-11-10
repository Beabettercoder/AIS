"""Render mean/SD and paired MMSI bootstrap for the complete model study."""
from pathlib import Path
import csv,json,statistics
import numpy as np
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027/model_study')
DATA=OUT.parent
VIEWS=('base','natural','suffix');MODES=('target','mean','median','absolute_median','absolute_report_median','mixed_report_median');ARCH=('tcn','gru')
rows=[]
for arch in ARCH:
 for exposure in VIEWS:
  for mode in MODES:
   results=[json.loads((OUT/'runs'/arch/exposure/mode/f'seed{s}'/'run.json').read_text()) for s in (0,1,2)]
   tests=[json.loads((OUT/'runs'/arch/exposure/mode/f'seed{s}'/'test_metrics.json').read_text()) for s in (0,1,2)]
   for split,z,key in [('val',results,'val_metrics'),('test',tests,'metrics')]:
    for view in VIEWS:
     for metric in ('pooled_AP','pooled_AUROC','macro_region_AP'):
      a=[x[key][view][metric] for x in z]
      rows.append({'split':split,'arch':arch,'train_exposure':exposure,'eval_view':view,'mode':mode,'metric':metric,'mean':statistics.mean(a),'sd':statistics.stdev(a),'seed0':a[0],'seed1':a[1],'seed2':a[2]})
with (OUT/'model_summary.csv').open('x',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
own=[]
for split in ('val','test'):
 with np.load(OUT/f'bootstrap_draws_{split}.npz',allow_pickle=False) as draws:
  for arch in ARCH:
   for exp in VIEWS:
    for mode in MODES:
     for view in VIEWS:
      series=[];pp=next(x for x in rows if x['split']==split and x['arch']==arch and x['train_exposure']==exp and x['eval_view']==view and x['mode']==mode and x['metric']=='pooled_AP')
      for seed in (0,1,2):
       a=draws[f'{arch}__{exp}__{mode}__{seed}__{view}'];series.append(a);ci=np.quantile(a,[.025,.975])
       own.append({'split':split,'arch':arch,'train_exposure':exp,'eval_view':view,'mode':mode,'seed':seed,'pooled_AP':pp[f'seed{seed}'],'CI95_low':ci[0],'CI95_high':ci[1]})
      ci=np.quantile(np.mean(series,axis=0),[.025,.975]);own.append({'split':split,'arch':arch,'train_exposure':exp,'eval_view':view,'mode':mode,'seed':'mean_fixed_3','pooled_AP':pp['mean'],'CI95_low':ci[0],'CI95_high':ci[1]})
with (OUT/'model_AP_own_cluster_CI.csv').open('x',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(own[0]));w.writeheader();w.writerows(own)
def get(split,arch,exp,view,mode):return next(x for x in rows if x['split']==split and x['arch']==arch and x['train_exposure']==exp and x['eval_view']==view and x['mode']==mode and x['metric']=='pooled_AP')
lines=['# 群体关系：三种训练暴露与双压力神经网络对照','',
 '本报告为 relaxed 样本版本的 108 次训练；原资格版本的静态统计另见关系数据主报告。该数据为受控四格、完整窗口离线评估，基础来源有既往曝光和背景共享，不是新自然异常金标。',
 '2 个骨干 × 6 种上下文 × 3 种训练暴露 × 3 个种子，每次固定 20 轮；三种评估视图来自同一来源/固定成员/标签，训练只读 train，归一化只用对应训练暴露。各组训练、验证完成后固定全部108个检查点及归一化，随后才提取本轮 test 神经特征并生成预测；底层测试观测此前已用于来源构造、审计和部分历史 pilot。',
 '基础为 base；自然压力在后锚点随机报告施加速度扰动，后段压力从目标后段能对齐的有效邻船报文中选取；压力是离线受控视图，实际扰动及命中率请同时参考数据主报告。',
 '普通均值、中位数、绝对可信加权中位数基于当前时刻的邻船速度；报文加权后中位数以当前及之前、同一前/后段的成员报告进行加权，再对成员取中位数。成员资格与锚点依赖整窗后续报文，故整体不是在线因果模型。',
 '', '## 测试 pooled AP：三种子均值 ± 样本标准差', '',
 '| 骨干 | 训练暴露 | 上下文 | 基础视图 | 自然压力 | 后段压力 |','|---|---|---|---:|---:|---:|']
for arch in ARCH:
 for exp in VIEWS:
  for mode in MODES:
   a=[get('test',arch,exp,v,mode) for v in VIEWS]
   lines.append(f'| {arch} | {exp} | {mode} | '+' | '.join(f"{x['mean']:.4f} ± {x['sd']:.4f}" for x in a)+' |')
lines += ['', '所有验证与测试的 pooled AP、AUROC、存在样本的区域等权 AP、每个种子值见 `model_summary.csv`；模型自身的目标 MMSI 聚类 AP 区间见 `model_AP_own_cluster_CI.csv`。各模型权重与逐样本得分在 `runs/`。区域等权只计算有样本区域。',
 '', '## 目标 MMSI 聚类配对差', '', '下表比较每个模型与同骨干、同训练暴露、同种子的普通中位数上下文，差值为三个固定种子平均；区间来自同一组 2000 次目标 MMSI 聚类抽样，单位为百分点。它不覆盖模型种子总体不确定性与共享背景/时空相关性。', '',
 '| 骨干 | 训练暴露 | 测试视图 | 上下文减普通中位数 | 平均差 pp [95% CI] |','|---|---|---|---|---:|']
b=json.loads((OUT/'bootstrap_test.json').read_text())
for x in b['comparisons']:
 if x['seed']!='mean_fixed_3' or x['mode'] not in ('absolute_median','absolute_report_median','mixed_report_median'):continue
 d=x['pooled_AP_delta']*100;lo,hi=np.array(x['ci95'])*100
 lines.append(f"| {x['arch']} | {x['exposure']} | {x['view']} | {x['mode']} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
lines += ['', '其中训练、验证和测试目标 MMSI 互不重叠，但背景船与报文可能共享；具体数量见数据主报告。低覆盖区域不能仅因有少量样本便代表该区域的泛化。',
 '', '## 文件','', '- `code_lock.json`：协议、模型代码、原始函数和样本清单哈希。', '- `checkpoint_lock.json`：首次提取测试特征之前固定的 108 个模型与训练归一化。', '- `features_{train,val,test}.npz` 与各 manifest：逐源、逐视图、逐四格网络输入及对齐元数据。', '- `runs/`：每轮训练损失和批序哈希、权重与逐样本验证/测试预测。', '- `bootstrap_{val,test}.json` 及对应 `bootstrap_draws_*.npz`：配对区间与MMSI抽样次数。', '- `model_summary.csv`：全部评估指标的每种子和均值。']
(OUT/'关系神经模型双压力报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('REPORT_READY',len(rows),flush=True)
