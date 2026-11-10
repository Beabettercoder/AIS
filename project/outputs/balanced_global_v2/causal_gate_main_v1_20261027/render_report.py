from pathlib import Path
import json,numpy as np
O=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/causal_gate_main_v1_20261027')
read=lambda p:json.loads((O/p).read_text(encoding='utf8'))
plan=read('frozen_test_plan.json');train=read('training_protocol.json');traj=read('trajectory_summary.json');natural=read('observed_natural_strata.json');test=read('formal_test_results/test_results.json');boot=read('formal_test_results/posthoc_primary_bootstrap.json');cap=read('gate_cap.json')
data={}
for arch in ('tcn','gru'):
 for seed in (0,1,2):
  for arm in train['arms']:
   r=read(f'runs/{arch}/seed_{seed}/{arm}/result.json');data[(arch,seed,arm)]=r
def avgstd(x):return f'{np.mean(x):.4f} ± {np.std(x,ddof=1):.4f}'
L=['# 因果门控、稳健背景与主基准测试报告','','日期：2026-10-27。结果使用 8 区域新基准；验证集经过多轮开发，现有测试集此前用于 pilot/paired-view，不是未曝光盲测。','',
 '## 1. 候选及对照','','旧门控对增量加权并累加；旧补偿插入当前相邻报告速度均值。新候选仅使用 t−1 及以前、w≥0.7 且运动学有效的最多 5 个报告速度，取分量中位数作为参考，最少需 3 个；不足时对低权重步回退旧门控。',
 'w<0.3 时比较当前位置速度和当前相邻报告速度均值分别与历史参考的距离：若前者小于 2.5 m/s 且不到后者一半则保留位置增量；若后者满足同样条件则用限幅报告速度补偿；否则用限幅历史参考补偿。其他有效步沿用旧补偿。参考或报告速度模长上限为训练集可信报告速度模长的 99.5 分位数，%.4f m/s；因果候选只改变位置两通道，其余沿用旧门控。'%cap['cap_mps'],
 '位置与报告速度均可能异常、历史参考可滞后真实变化，分支结果是候选判别，不是故障真值。','',
 '## 2. 轨迹反事实诊断','','下面以目标船受控污染之前生成的轨迹为参考，不将其输入模型；这不是人工核实的真实传感器金标。单元格为终点误差均值 / 中位数 / P90，单位 m。','',
 '| 验证窗口分层 | 数量 | 原始位置 | 原门控 | 旧补偿 | 新因果候选 |','|---|---:|---:|---:|---:|---:|']
for group,label in [('target_noise','目标受控污染'),('speed_spike','航速尖峰'),('teleport','位置跳变'),('drift','位置漂移'),('target_not_noise','无目标受控污染')]:
 t=traj[group];parts=[]
 for method in ('raw','old','comp','causal'):
  z=t[method+'_endpoint'];parts.append('%.1f / %.1f / %.1f'%(z['mean'],z['median'],z['p90']))
 L.append('| '+label+' | '+str(t['windows'])+' | '+' | '.join(parts)+' |')
L+=['','新候选降低本构造中航速尖峰误差，但在位置跳变和漂移分层上的平均误差高于旧补偿，不能说保留了旧补偿的全部位置修复收益。',
 '另在 y=0 且目标未加受控污染的原始观测中，按相对首点观测航向变化≥45°选出 %d 窗、航速变化≥3kn选出 %d 窗；新候选终点误差均值分别 %.1f m、%.1f m，旧补偿 %.1f m、%.1f m。两组可重叠，只是观测变化分层，不能作为人工行为金标。'%(natural['summary']['turn_45_deg']['windows'],natural['summary']['speed_change_3_kn']['windows'],natural['summary']['turn_45_deg']['causal_endpoint']['mean'],natural['summary']['speed_change_3_kn']['causal_endpoint']['mean'],natural['summary']['turn_45_deg']['comp_endpoint']['mean'],natural['summary']['speed_change_3_kn']['comp_endpoint']['mean']),
 '','## 3. 开发验证：72次配对训练','','7,840训练窗、1,680验证窗，TCN/GRU 各 12 组×3种子，固定25轮。所有模型12输入槽位、同一train/all、批次顺序、优化预算和训练数据归一化；标准控制真正只用二元 BCE。历史训练包包括额外类型 CE、增广 BCE、类型 JS、隐藏平滑和控制器，不是只有一致性损失。下表为三种子均值 ± 样本标准差。','',
 '| 设置 | TCN 区域macro AP | GRU 区域macro AP | TCN pooled AP / AUROC | GRU pooled AP / AUROC |','|---|---:|---:|---:|---:|']
for arm in train['arms']:
 fields=[]
 for arch in ('tcn','gru'):
  z=[data[(arch,i,arm)]['final_validation']['macro']['AP'] for i in (0,1,2)];fields.append(avgstd(z))
 for arch in ('tcn','gru'):
  z=[data[(arch,i,arm)]['final_validation']['pooled'] for i in (0,1,2)];fields.append('%.4f / %.4f'%(np.mean([x['AP'] for x in z]),np.mean([x['AUROC'] for x in z])))
 L.append('| '+arm+' | '+' | '.join(fields)+' |')
L+=['','`causal_common` 残差为零但包含背景支持标志；各场方法使用同一支持标志和二维残差槽。它并非纯粹不含邻域信息的单船基线。原仿射场使用可信筛选邻船，普通均值、中位数及可信加权中位数使用同一组未按可信度预过滤的普通邻船；因此原场与新场还存在成员组成差别。',
 '以相同支持标志下的无场配置为参照，本轮所测的普通/稳健背景没有在两种骨干上稳定提高验证 AP。因果候选轨迹误差改善也没有转化为跨骨干一致的检测增益。','',
 '## 4. 冻结后历史测试集评估','','训练结束后仅依照 72 组固定第25轮的验证区域macro AP、两个骨干和三个种子等权均值选择一个共享主配置，结果为 `'+plan['primary_arm']+'`。将全部 12 个预定对照、两个骨干和三个种子一并测试；没有利用测试集选择配置。测试集曾用于早期开发，不能称未曝光盲测。测试共 %d 窗、%d 个目标 MMSI。'%(test['test_windows'],test['test_unique_target_mmsi']),
 '','| 骨干 / 方案 | 测试区域macro AP | pooled AP | AUROC | 验证1%阈值下召回 / 实际FAR | 验证5%阈值下召回 / 实际FAR |','|---|---:|---:|---:|---:|---:|']
for arch in ('tcn','gru'):
 for arm in train['arms']:
  z=[r for r in test['results'] if r['architecture']==arch and r['arm']==arm]
  if len(z)!=3:raise ValueError((arch,arm,len(z)))
  fmt=lambda k:avgstd([x[k] for x in z]);op=lambda a:'%.3f / %.3f'%(np.mean([x[a]['recall'] for x in z]),np.mean([x[a]['actual_FAR'] for x in z]))
  L.append('| '+arch+' / '+arm+' | '+fmt('macro_AP')+' | '+fmt('pooled_AP')+' | '+fmt('pooled_AUROC')+' | '+op('0.01')+' | '+op('0.05')+' |')
L+=['','阈值按每个模型的验证阴性预测确定，分数并列时采取保守处理；表内 FAR/召回是三种子均值，完整逐种子值与 95% 区间见 `formal_test_results/test_results.json`。因测试集分布和采样不同，测试实际 FAR 不保证等于验证预算。',
 '','对已经在测试前固定的主方案与 raw，另外事后计算同一组 2,000 次目标 MMSI 聚类抽样的配对区间。下列数字为三种子差值均值及其按同一组船舶抽样得到的 95% 区间：','',
 '| 骨干 | 区域macro AP差（百分点） | pooled AP差（百分点） |','|---|---:|---:|']
for v in boot['three_seed_mean']:
 fm=lambda x:'%+.3f [%+.3f, %+.3f]'%(100*x[0],100*x[1][0],100*x[1][1])
 L.append('| '+v['backbone']+' | '+fm((v['macro_AP_difference'],v['macro_AP_difference_CI95']))+' | '+fm((v['pooled_AP_difference'],v['pooled_AP_difference_CI95']))+' |')
L+=['','抽样条件于当前目标船，不覆盖共享背景船/场景相关性或训练种子总体不确定性；每个种子单独区间见 `posthoc_primary_bootstrap.json`。主方案还包含支持可用标记和历史训练包，其相对 raw 的差异不能归因于新稳健背景场。','',
 '## 5. 证据与复现','','- `protocol.json`、`gate_cap.json`：训练前固定的候选与训练集限幅。','- `training_protocol.json`：12维输入和72组纯BCE/训练包对照。','- `gate_data/`、`trajectory_validation_rows.json`、`observed_natural_strata.json`：逐窗输入和误差。','- `runs/`：每轮验证预测、参数权重、标准化、批次哈希。','- `frozen_test_plan.json`：测试前冻结主配置、72个检查点和验证阈值。','- `test_feature_preflight.json`：测试输入构建器在验证集12组逐元素一致。','- `formal_test_results/`：测试逐窗预测、各指标、2000次聚类结果与事后主方案配对区间。','- `test_background/`：测试背景及逐成员来源；构造在冻结后执行。','']
(O/'主基准因果门控与测试报告.md').write_text('\n'.join(L),encoding='utf8');print('REPORT_COMPLETE',flush=True)
