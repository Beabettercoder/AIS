"""Summarize frozen, observed-only relations and target-MMSI paired intervals."""
from pathlib import Path
import json
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
summaries={s:json.loads((OUT/f'statistics_summary_{s}.json').read_text()) for s in ['train','val','test']}
def fmt(x):return f'{x:.4f}'
lines=['# 群体关系两版双压力：简单统计对照','','日期：2026-10-27。全部方法同一来源、同一版本、同一成员和观察视图；四格标签由受控 XOR 构造，结果不是自然异常人工金标或实时部署指标。','', '原始规则与放宽规则的成员/样本集不同，不应把二者性能差值解释为单一阈值效应。测试船舶 MMSI 与训练/验证目标互斥，但底层源数据曾参与 pilot，且背景船和源报文仍跨划分共享。','', '## Pooled AP','','| 版本 | 划分 | 视图 | 来源 / 目标船 | 单目标 / 单邻船 | 均值 | 普通中位数 | 绝对可信成员中位数 | 绝对可信报文加权后中位数 | 混合可信报文加权后中位数 |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for v in ['original','relaxed']:
 for s in ['train','val','test']:
  for view in ['base','natural','suffix']:
   d=summaries[s]['per_version'][v][view];m=d['metrics'];lines.append(f"| {v} | {s} | {view} | {d['sources']} / {d['target_mmsi']} | {fmt(m['target_only']['pooled_AP'])} / {fmt(m['neighbor_only']['pooled_AP'])} | {fmt(m['mean']['pooled_AP'])} | {fmt(m['median']['pooled_AP'])} | {fmt(m['absolute_weighted_median']['pooled_AP'])} | {fmt(m['absolute_report_weighted_median']['pooled_AP'])} | {fmt(m['mixed_report_weighted_median']['pooled_AP'])} |")
lines += ['', '所有四格的单侧标签按构造平衡，AP=0.5 是负对照而非自然海事异常的真实基线。8 区域测试分布不均，放宽版 LA/Hau 各仅1来源，区域等权 AP 可在 JSON 查看，不代表充分跨区泛化。', '', '## 配对 MMSI bootstrap','','验证/测试每视图使用相同的 2,000 次目标 MMSI 聚类抽样，对各方法与该视图普通中位数的 pooled AP 差计算百分位 95% 区间。以下单位为百分点，保留原始三种比较；这些区间条件于本批目标船，不覆盖共享背景与场景相关性。','','| 版本 / 划分 | 视图 | 绝对可信成员中位数 − 中位数 | 绝对可信报文中位数 − 中位数 | 混合可信报文中位数 − 中位数 |','|---|---|---:|---:|---:|']
for v in ['original','relaxed']:
 for s in ['val','test']:
  for view in ['base','natural','suffix']:
   b=summaries[s]['per_version'][v][view]['paired_mmsi_bootstrap'];cells=[]
   for method in ['absolute_weighted_median','absolute_report_weighted_median','mixed_report_weighted_median']:
    x=b[method];cells.append(f"{100*x['diff_AP']:+.2f} [{100*x['CI95'][0]:+.2f}, {100*x['CI95'][1]:+.2f}]")
   lines.append('| '+v+' / '+s+' | '+view+' | '+' | '.join(cells)+' |')
lines += ['', '本轮放宽 test 定向压力约 59.64% 邻船成员被选，162/220 来源所选成员超过半数；该压力比恰好 50% 更强，简单中位数及软权重在多数成员受扰时不能保证稳健。后段定向满足索引命中不保证每名成员数值改变，实际数值检查见 `effective_pressure_audit.json`；所有来源都保留。','', '训练、验证、测试每个四格变体分数及完整 AUROC/分区域 AP 保存于 `statistics_{train,val,test}.csv`、`statistics_summary_{train,val,test}.json`；验证/测试每次抽样差值见 `statistics_bootstrap_{val,test}.json`。模型结果见 `model_study/关系神经模型双压力报告.md`；数据构造和拒绝原因见 `八区域关系扩充构造清点.md`。']
(OUT/'关系双压力简单统计报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('STATISTICS_REPORT_DONE',flush=True)
