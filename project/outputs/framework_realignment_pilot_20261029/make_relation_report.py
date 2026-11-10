"""Generate explicit, versioned development-only relationship pilot report."""
import json
from pathlib import Path
P=Path('/mnt/sda/WangZijia/AIS/outputs/framework_realignment_pilot_20261029')
names=['tcn','gru','transformer','itransformer'];views=[('base','Base'),('natural','自然速度压力'),('suffix','后段速度压力'),('combined_l100_m600','后段速度+位置压力')]
sections=[('原均值池化旁路','relation_development_summary.json'),('前后均值差旁路（探索）','temporal_development_summary.json'),('真实时间归一变化率旁路（探索）','rate_development_summary.json')]
lines=['# 群体关系任务：留一场四骨干单种子开发验证','', '实验以训练集 2034 个来源、验证集 272 个来源构建配对四格。只使用原始观测构造留一局部仿射场，保持固定同源邻船，速度和位置压力作用于观测后重新计算可信度与场。四个视图均为已曝光开发验证集，不属于独立盲测。','', '每个视图的 F0 和 F1 均使用同一因果门控目标流、同一有效场支持和同一连续 rho；F0 使用距离权重，F1 使用距离×运动学可信度权重，因此本试验 F1 是因果门控+可信场组合，未单列 F2。目标流 C0 使用原始自船输入。正常化只使用训练集；25 轮，seed 0。','', '关系标签只有二分类，训练包以二分类 JS 替代五类别 JS，关闭五类辅助 CE，保留控制器、统计扰动、增强 BCE 和平滑项；这是历史包的任务适配，不可称为原五类训练包完全复现。','', '场拟合至少需要四名对齐邻船，F0/F1 均有效时才使用；其余位置回退至目标流。验证集 base、自然、后段速度、后段复合压力有效场步占比分别为 56.04%、56.02%、56.00%、55.40%。时序旁路至少需要前后两半各有一个有场步，验证集 608/1088 样本满足；这些样本对应 152/272 来源。','', '后两种旁路在首批结果后探索，二者同时加入背景速度与前后汇聚。时间归一版以标准化场特征的前后均值差除以 rho 加权的前后时间重心之差（秒），故单位为标准化特征值/秒；该汇聚为全窗离线处理。','', '下表均为 pooled AP；方括号为 F1−F0 的 2000 次目标 MMSI 聚类配对 bootstrap 95% 区间，单位为百分点。它仅量化已训练单个种子的样本不确定性，未校正多重比较。','']
for title,file in sections:
 d=json.loads((P/file).read_text()); lines += ['## '+title,'','| 骨干 | 视图 | C0 | F0 | F1 | F1−F0 (pp) | 95% 区间 (pp) |','|---|---|---:|---:|---:|---:|---:|']
 for n in names:
  model=d['models'][n]
  for view,label in views:
   a=model['values'][view];ci=model['intervals'][view]['f1_vs_f0'];p0=a['raw']['pooled_AP'];p1=a['f0_bundle']['pooled_AP'];p2=a['f1_bundle']['pooled_AP']
   lines.append(f"| {n} | {label} | {p0:.4f} | {p1:.4f} | {p2:.4f} | {(p2-p1)*100:+.2f} | [{ci['lo']*100:+.3f}, {ci['hi']*100:+.3f}] |")
 lines.append('')
lines += ['## 核验与范围','', '输入核验：全部字段有限；F0/F1 的支持权重一致；无场值清零；目标输入在同一来源对应四格内保持配对。训练/验证目标 MMSI 及来源不重叠；背景船和报文仍有共享。','', 'C0 与 F1 同时改变了目标门控、训练包和场输入，不能把它们之间的差异单独归因于场。各 JSON 另保存 causal_bundle 对照；其训练期控制器仍读取背景物理状态。F0 与 F1 使用同样的连续可信支持，但场拟合权重及由场计算的控制器状态不同。','', '本任务为受控合成关系标签；单种子、已曝光开发集、有限来源覆盖和多次探索下的结果不能声称稳定跨骨干增益。Bootstrap 不是跨随机种子稳定性证明；后续应先在开发集锁定协议，再在新来源独立测试。','', '对应逐样本预测、权重、实验配置、各视图统计和哈希保存在 relation_runs、relation_temporal_runs、relation_rate_runs 及对应 JSON 文件。']
(P/'关系任务开发验证报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(P/'关系任务开发验证报告.md')
