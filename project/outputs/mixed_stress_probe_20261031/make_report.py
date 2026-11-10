from pathlib import Path
import json, hashlib
D=Path('/mnt/sda/WangZijia/AIS/outputs/mixed_stress_probe_20261031')
x=json.loads((D/'summary.json').read_text()); views=[('base','Base'),('natural','自然速度压力'),('suffix','后段速度压力'),('combined_l100_m600','后段速度＋位置压力')]
text=['# 阶段累计框架：训练期污染暴露受控配对小试','',
'## 研究问题与冻结设置','',
'原先仅Base训练时，TCN在自然压力的可信场强于普通场，而后段压力下普通场也较强。该试验研究给普通场和可信场相同的训练期污染暴露后，二者差异如何变化；不预设可信场必须优于普通场。', '',
'与对应Base-only基线使用相同40维输入、原阶段累计流场、原运动学可信门控、邻船Heading、同一数据分区、训练集Base归一化、同骨干相同初始权重、BCE、优化器、batch顺序和固定末轮。TCN/GRU/Transformer保持窗口打散；iTransformer采用既有同源四格成批。TCN和iTransformer各3 seeds（25/75轮），GRU与Transformer各1 seed（25轮）。', '',
'新增的唯一训练条件为：每个epoch按来源随机等概率选择Base、自然压力或后段速度压力中的一个视图，同源四格共享视图；F0/F1相同seed在每epoch使用完全相同的选择。压力发生于原始邻船观测并按已存协议重新计算物理特征，模型无视图或污染标签输入。种子控制、batch组织和数据混合协议在训练前冻结。', '',
'本协议在此前开发观察后提出，是探索性对照，不是独立预注册确认。输入含邻船Heading，40维适配及iTransformer通道身份嵌入不是DeepSeek原35脚本的逐行复现。F0/F1除可靠权重外还包含阶段内报文聚合与支持特征差异，只能解释为整体配置差异。','',
'## pooled AP 与 95% 目标MMSI配对区间','',
'所有报告的差值单位为百分点；均值为固定已训练seed模型的等权平均。区间采用同一目标MMSI配对重采样2,000次；不包含种子总体不确定性、跨来源背景船复用和多重比较。已有开发验证集此前多次曝光，训练与验证目标MMSI无交集，标签是受控四格，不属于真实传感器故障真值。','',
'| 骨干 | 视图 | Base-only F0/F1 | Mixed F0/F1 | Mixed F1−F0 pp [95% CI] | F0 mixed−Base pp [95% CI] | F1 mixed−Base pp [95% CI] |','|---|---|---|---|---|---|---|']
for name in ('tcn','gru','transformer','itransformer'):
 for key,label in views:
  v=x[name]['views'][key];a=v['mean_ap'];d=v['mean_diff'];c=v['bootstrap95']
  def fmt(key):return f"{d[key]*100:+.2f} [{c[key][0]*100:+.2f}, {c[key][1]*100:+.2f}]"
  text.append(f"| {name} | {label} | {a['base_F0_heading']:.4f} / {a['base_F1_heading']:.4f} | {a['mixed_F0_heading']:.4f} / {a['mixed_F1_heading']:.4f} | {fmt('mixed_f1_minus_f0')} | {fmt('f0_mixed_minus_base')} | {fmt('f1_mixed_minus_base')} |")
text+=['','## 后段速度：逐随机种子差异','', '| 骨干 | Mixed F1−F0，逐seed（pp） | Base-only F1−F0，逐seed（pp） |','|---|---|---|']
for name in ('tcn','gru','transformer','itransformer'):
 v=x[name]['views']['suffix'];d=v['per_seed_diff'];text.append('| '+name+' | '+' / '.join(f'{z*100:+.2f}' for z in d['mixed_f1_minus_f0'])+' | '+' / '.join(f'{z*100:+.2f}' for z in d['base_f1_minus_f0'])+' |')
text+=['','## 解释与门槛','',
'四骨干后段视图中，普通场获得混合暴露后的平均AP均提高；F1−F0没有形成四骨干一致的正区间。因此训练数据中是否包含同类观测退化是影响模型比较的重要条件。在当前受控协议下，不能将先前结果的全部差异归因于可信门控，也不能从本试验推断任何方法在未见污染或真实事故中必然有效。','',
'混合训练可能让模型接触与验证压力相同的扰动构造；这不等于真实世界泛化。新F1在Base或其他视图可能下降，必须同时报告绝对AP。GRU和Transformer只有一个seed，区间不是种子稳定性保证。四骨干三个压力视图全部正向的严格门槛未满足，因此不启动八骨干完整训练。','',
'## 资产与核验','',
'frozen_protocol.json、frozen_manifest.sha256、train_mixed.py、smoke.py、worker.py和launch.py为训练前冻结；analysis_manifest.sha256记录训练过程中、查看正式汇总之前编写的summary.py。runs保存16个混合配置的终轮权重、逐窗预测、初始化哈希、每epoch损失和每epoch视图源数；summary.json保存固定基线、每seed及四视图配对区间。基线模型保留在 heading_evidence_pilot_20261030 与 itrans_sourcebalanced_20261031，详见每份结果中的baseline哈希。GPU4服务未动；无测试访问。', '']
(D/'训练暴露机制配对小试报告.md').write_text('\n'.join(text),encoding='utf8')
