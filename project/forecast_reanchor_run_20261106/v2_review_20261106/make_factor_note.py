"""Summarize existing v2 test outputs only; no training or checkpoint changes."""
import json,hashlib
from pathlib import Path
ROOT=Path('/mnt/sda/WangZijia/AIS/forecast_reanchor_run_20261106/work_v2')
OUT=ROOT.parent/'v2_review_20261106';OUT.mkdir(exist_ok=True)
EXPS=['training_constant','data_small','main','capacity_h128','capacity_h256']
MODELS=['tcn','gru','transformer','itransformer']
NAMES={'tcn':'TCN','gru':'GRU','transformer':'Transformer','itransformer':'iTransformer'}
E={e:json.loads((ROOT/e/'reports/test/summary.json').read_text()) for e in EXPS}
F=json.loads((ROOT/'comparisons/test/factor_comparisons.json').read_text())
def row(e,m,a,v='base'):return next(r for r in E[e]['summary'] if (r['model'],r['arm'],r['view'],r['subset'])==(m,a,v,'all'))
def fmt(r):return f"{r['ADE_m_mean']:.2f} / {r['FDE_m_mean']:.2f}"
def ci(r):return f"{r['delta']:+.2f} [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]"
lines=['# 240组预测实验：训练、数据与容量因素解读','',
 '本报告只读取已经生成的测试结果，不改动模型、不重新训练。全部测试数值为1,183窗、303艘目标船上的三种子均值。ADE/FDE单位为米，越低越好。', '',
 '## 核心判断','',
 '训练与选型升级改善了多数完整配置的基础观测误差；扩大训练配额进一步降低均值，但多数区间跨零。隐藏维度由64增至128/256没有带来稳定的基础观测收益。更大模型不是本轮提高精度的主要来源。', '',
 '主配置四种Raw模型的基础测试ADE均低于原始恒速外推；完整P3在GRU、Transformer、iTransformer上均值低于原始恒速，但四种骨干的P3均高于各自Raw。这里没有把“均值低于恒速”写成已经完成CV配对显著性检验。', '',
 '## 1. 基础观测：完整P3的ADE / FDE','',
 '|骨干|原训练策略，5600窗|改进训练，5600窗|改进训练，14897窗，64维|同数据128维|同数据256维|',
 '|---|---:|---:|---:|---:|---:|']
for m in MODELS:lines.append('|'+NAMES[m]+'|'+'|'.join(fmt(row(e,m,'P3')) for e in EXPS)+'|')
lines+=['','训练策略对照同时改变学习率调度和选型（常量学习率＋末轮 → 余弦调度＋验证最佳），只能解释组合收益。数据量对照保持轮数而非总更新步数，因此扩数据也增加了优化更新次数。','',
 '## 2. 主配置Raw与P3，及恒速参照','',
 '|骨干|Raw ADE/FDE|P3 ADE/FDE|','|---|---:|---:|']
for m in MODELS:lines.append('|'+NAMES[m]+'|'+fmt(row('main',m,'P0'))+'|'+fmt(row('main',m,'P3'))+'|')
lines+=['',f"原始恒速外推：{fmt(row('main','CV_raw','CV_raw'))}；门控重锚恒速外推：{fmt(row('main','CV_gate','CV_gate'))}。",'',
 '## 3. 基础观测的逐因素ADE差值与95%区间','',
 '差值＝候选－对照；负值代表改善。区间按目标MMSI聚类，以三个已训练模型为条件，未做多重比较校正。','',
 '|骨干／配置|训练升级|训练配额扩大|128－64|256－64|','|---|---:|---:|---:|---:|']
comparisons=['data_small - training_constant','main - data_small','capacity_h128 - main','capacity_h256 - main']
for m in MODELS:
 for arm in ['P0','P3']:
  values=[]
  for c in comparisons:
   x=next(x for x in F['rows'] if (x['comparison'],x['model'],x['arm'],x['view'],x['subset'],x['aggregation'],x['metric'])==(c,m,arm,'base','all','pooled','ADE_m'))
   values.append(ci(x))
  lines.append('|'+NAMES[m]+'/'+arm+'|'+'|'.join(values)+'|')
lines+=['','完整P3的训练升级：TCN、Transformer、iTransformer的ADE区间低于零，GRU区间上限为+0.02米，跨零。扩大数据后四种P3均值下降，仅TCN的区间低于零。128或256相对64的基础P0/P3差值区间全部跨零，不能据此称扩大容量提高精度。','',
 '## 4. 邻船污染下可信场的净作用：P3－P2 ADE','',
 '|骨干|原训练，5600窗|改进训练，5600窗|主配置64维|128维|256维|','|---|---:|---:|---:|---:|---:|']
for m in MODELS:
 vals=[]
 for e in EXPS:
  r=next(x for x in E[e]['paired'] if (x['model'],x['view'],x['subset'],x['contrast'],x['aggregation'],x['metric'])==(m,'neighbor_stress','all','P3-P2','pooled','ADE_m'))
  vals.append(ci(r))
 lines.append('|'+NAMES[m]+'|'+'|'.join(vals)+'|')
lines+=['','主配置64维的四个ADE区间全部跨零；128/256维下TCN与iTransformer的区间低于零。不能概括为可信场在所有骨干及容量上都显著受益。','',
 '## 5. 不同污染条件的整体表现','',
 '下表仅为四骨干P3的等权平均ADE，描述趋势，不另作总体显著性声明。','',
 '|配置|基础|末点跳变|后段漂移|速度尖峰|邻船污染|','|---|---:|---:|---:|---:|---:|']
for e in EXPS:
 vals=[sum(row(e,m,'P3',v)['ADE_m_mean'] for m in MODELS)/4 for v in ['base','target_jump','target_drift','target_speed','neighbor_stress']]
 lines.append('|'+e+'|'+'|'.join(f'{x:.2f}' for x in vals)+'|')
lines+=['','训练升级对基础观测有益，但速度尖峰下的P3平均ADE由1,838.85升至1,952.35米，不能称全部条件都改善。主配置门控P1在末点跳变、速度尖峰下仍大幅优于Raw，主要防护作用来自门控；基础观测和邻船污染下完整P3仍未超过同骨干Raw。','',
 '## 解释与适用范围','',
 '实测支持优先改进训练及数据利用，并未支持为提高显存占用而继续加宽模型。关于容量收益不足的原因，目前只能提出泛化、观测信息量或优化配置等候选解释，不能把结果直接等同于“船舶惯性决定一切”或“背景场无用”。测试来源已在研究中重复使用，报告保留这一事实。','']
(OUT/'因素解读.md').write_text('\n'.join(lines),encoding='utf-8')
print(OUT/'因素解读.md')
