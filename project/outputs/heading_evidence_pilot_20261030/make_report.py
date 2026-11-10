"""Human-readable report; all numbers read from saved per-window predictions."""
import json
from pathlib import Path
R=Path('/mnt/sda/WangZijia/AIS/outputs');O=R/'heading_evidence_pilot_20261030';x=json.loads((O/'paired_summary.json').read_text());probe=json.loads((O/'train_report_probe.json').read_text());linear=json.loads((O/'train_information_probe.json').read_text())
views=('base','natural','suffix','combined_l100_m600');names=('tcn','gru','transformer','itransformer');viewzh={'base':'Base','natural':'自然速度','suffix':'后段速度','combined_l100_m600':'速度+位置复合'}
S=[]
S.append('# Heading 额外证据：覆盖、受控一致性及小规模配对\n')
S.append('仅使用既有训练来源和已曝光开发验证来源；无测试标签、无八骨干正式训练。Heading是单独AIS字段，未证明来自独立传感器；实验不读取原始干净速度、位置和受控污染标记作为模型输入。受控标签是合成关系标签，不是真实故障或人工确认的真实机动。\n')
S.append('## 1. 全量覆盖与四格一致性\n')
text=(R/'heading_coverage_20261030/Heading全量覆盖与生成一致性.md').read_text(encoding='utf8');start=text.index('| 地区 |');end=text.index('\n\n',start);S.append(text[start:end]+'\n')
S.append('每区 train/val 来源数、数据源、低速覆盖、重复报文和缺失值定义详见 [Heading全量覆盖与生成一致性](../heading_coverage_20261030/Heading全量覆盖与生成一致性.md)。LA/Haugesund/Shanghai 的验证来源分别仅1/0/2，表是请求级覆盖而非独立样本比例。所有2306来源、Base/自然/后段视图目标和邻船单侧Heading+mask签名均一正一负。四格生成器改变 SOG、位置但保持 COG；本实验同身份与UTC秒共用源Heading，这是合成假设而非对 Heading 与合成轨迹的物理真值证明。训练与验证目标MMSI隔离，背景船报文跨分区重用。\n')
S.append('## 2. 训练域报文与简单模型诊断\n')
z=probe['paired_invariance'];S.append(f"后段实际改速 {z['n_changed']} 条重复训练报文中，{z['n_heading_pair_valid']} 条有合法Heading；原Heading以同一源缓存跨压力视图复用，COG逐项比较最大差 {z['cog_maxdiff_deg']:.1f}°。因此单独相对角在该对速度倍率污染的前后保持不变，不能直接判别速度污染；与可信权重聚合后的上下文仍可能随其他输入改变。\n")
S.append('| 训练记录组 | 重复报文 | 目标船 | 邻船 | Heading/COG有效 | 角差中位/90分位 | 船首角速度绝对值中位/90分位（度/秒） |\n|---|---:|---:|---:|---:|---|---|')
for k,label in [('changed_current_baseline_under_2kn','受控前<2kn且后段改速'),('unchanged_current_baseline_under_2kn','受控前<2kn且未改速'),('unchanged_history_under_2kn_speed_change_ge1kn','历史<2kn、当前未改且前后SOG差≥1kn')]:
 g=probe['groups'][k];S.append(f"| {label} | {g['n']} | {g['target']} | {g['neighbor']} | {g['heading_valid']} | {g['angle_median']:.1f}° / {g['angle_p90']:.1f}° | {g['yaw_median']:.3f} / {g['yaw_p90']:.3f} |")
S.append('\n这些是不同条件下的重复报文描述，未改速不等于人工确认无噪机动；组间不同不能当独立真实故障分类实验。训练集按目标MMSI划分的线性探针，原场摘要 AP %.4f，加入Heading摘要 AP %.4f；初始目标SOG<2kn的128窗口子集 %.4f→%.4f。该探针属探索，不能预判深网。\n'%(linear['results']['original']['AP'],linear['results']['original_plus_heading']['AP'],linear['results']['original']['initial_target_under_2kn']['AP'],linear['results']['original_plus_heading']['initial_target_under_2kn']['AP']))
S.append('## 3. 配对模型与训练设置\n')
S.append('原35维运动和场特征保持不变，新增5维邻船Heading侧证：相对COG的圆周sin/cos、Heading有效权重比例、过去相邻有效Heading的角速度与角速度有效权重比例。每个时点只用已到达且年龄≤300秒的报文；角速度用合法相邻Heading及秒差计算。原门控、阶段累计、零阶回退不变；无效Heading不填成真实0。F0与F1对应距离及距离×可信度聚合。\n')
S.append('四种比较均使用40维网络、相同种子初值和预算：C0上下文置零；F0+H普通场及Heading；F1−H可信场而Heading五槽归零；F1+H可信场和Heading。原先35维模型作为参考，但40维零槽仍改变网络参数，尤其iTransformer额外token参与attention，不能称同一个模型。每种子原始35维数值见 paired_summary.json。\n')
S.append('TCN/iTransformer各3 seed，固定25/75轮；GRU/Transformer各1 seed、25轮；AdamW和BCE，训练Base，仅以训练集拟合标准化，固定末轮。各视图 pooled AP，区间为目标MMSI聚类配对Bootstrap 2000次；固定模型样本不确定性，不涵盖随机种子总体、共享背景依赖及多重比较。\n')
S.append('| 骨干 | 视图 | C0 | F0+H | F1−H | F1+H | Heading增量：F1+H−F1−H pp [95%CI] | 可信配置：F1+H−F0+H pp [95%CI] |\n|---|---|---:|---:|---:|---:|---|---|')
for name in names:
 for v in views:
  z=x[name]['groups']['all']['views'][v];m=z['mean'];c=z['bootstrap95'];diff=z['mean_diff'];f=lambda k:f"{diff[k]*100:+.2f} [{c[k][0]*100:+.2f},{c[k][1]*100:+.2f}]"
  S.append(f"| {name} | {viewzh[v]} | {m['C0']:.4f} | {m['F0_heading']:.4f} | {m['F1_noheading']:.4f} | {m['F1_heading']:.4f} | {f('heading_vs_noheading')} | {f('reliability_config')} |")
S.append('\n## 4. Heading可用与受控前低速子组（探索）\n')
S.append('来源至少一时刻有Heading：232来源/76目标船/928格；全窗没有Heading：40来源/22目标船/160格（同一目标船可能出现在两组）。受控前被选邻船至少一条后段报文 SOG<2kn：33来源/24目标船/132格；该分组使用构造前速度和受控计划，仅用于事后评测，模型不能预知。全部子组在同一已曝光验证集上分析、未经多重校正。\n')
S.append('| 骨干 | 后段子组 | 来源 | 目标船 | F1−H | F1+H | Heading增量 pp [95%CI] |\n|---|---|---:|---:|---:|---:|---|')
for name in names:
 for group,label in [('source_any_heading','来源有Heading'),('source_no_heading','来源无Heading'),('suffix_selected_baseline_low_neighbor','受控前被选邻船<2kn')]:
  d=x[name]['groups'][group];z=d['views']['suffix'];m=z['mean'];v=z['mean_diff']['heading_vs_noheading'];c=z['bootstrap95']['heading_vs_noheading'];ci=f"[{c[0]*100:+.2f},{c[1]*100:+.2f}]" if c else '不可估计'
  S.append(f"| {name} | {label} | {d['n_sources']} | {d['n_mmsi']} | {m['F1_noheading']:.4f} | {m['F1_heading']:.4f} | {v*100:+.2f} {ci} |")
S.append('\n## 5. iTransformer逐seed收敛\n')
S.append('| Seed | F0+H Base AP / BCE | F1−H Base AP / BCE | F1+H Base AP / BCE | F1+H−F1−H 后段 pp |\n|---|---|---|---|---:|')
for seed in (0,1,2):
 cells=[]
 for a in ('F0_heading','F1_noheading','F1_heading'):
  r=json.loads((O/'runs'/'itransformer'/f'seed{seed}'/a/'result.json').read_text());cells.append(f"{r['views']['base']['pooled_AP']:.4f} / {r['loss_history'][-1]:.4f}")
 diff=x['itransformer']['groups']['all']['views']['suffix']['per_seed_diff']['heading_vs_noheading'][seed]*100
 S.append(f"| {seed} | {' | '.join(cells)} | {diff:+.2f} |")
S.append('\nseed 1 的无Heading F1与seed 0 的F0+H未充分收敛，额外token使架构与35维旧版不同。iTransformer 大幅平均差异混入对照训练不稳定；不能将其直接解释为Heading信息或跨骨干收益。来源无Heading组中不同训练模型也可能因参数不同给出不同分数，不能当作缺失报文本身携带信息。\n')
S.append('TCN与Transformer后段 F1+H−F0+H 区间跨零，且iTransformer有对照不收敛；未满足此前四骨干、三种压力条件，不启动八骨干×三种子完整实验。候选未改变原门控，不宣称已解决低速观测污染。全部数据、源码、预先固定协议、模型/逐窗预测、组别及配对区间保留本目录；独立核验另行提供。\n')
(O/'Heading额外证据配对试验报告.md').write_text('\n'.join(S),encoding='utf8')
