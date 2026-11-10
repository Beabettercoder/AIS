"""Render reproducible factorial and ep-condition summary from saved results."""
from pathlib import Path
import json,statistics
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');NEW=ROOT/'factorial_ablation_v1_20261027';OLD=ROOT/'causal_gate_main_v1_20261027'
dev=json.loads((NEW/'validation_summary_v2.json').read_text(encoding='utf8'));test=json.loads((NEW/'historical_test_results/test_results.json').read_text(encoding='utf8'))
results={(r['architecture'],r['seed'],r['arm']):r for r in test['results']}
arms=['raw','raw_common','raw_bundle','raw_common_bundle','causal','causal_common','causal_bundle','causal_common_bundle','old','old_bundle','comp','comp_bundle',
      'raw_bundle_self','raw_bundle_bgz','causal_bundle_self','causal_bundle_bgz','causal_common_bundle_self','causal_common_bundle_bgz']
def testmean(arch,arm,k):return statistics.mean(results[(arch,s,arm)][k] for s in (0,1,2))
lines=['# 主基准：门控、邻域支持与训练条件配对消融','',
'日期：2026-10-27。本报告补全 8 格矩阵，并区分历史背景条件与标准化的背景/单船状态条件。72 次新增训练；以前 36 组匹配对照按原样复用。旧测试来源曾用于 pilot 及后续研究，不能称未曝光盲测。', '',
'## 一、设计与单船状态','',
'全部采用同一 12 维网络输入、8 区域 7840 训练与1680验证窗、25轮、batch 128、相同种子与批次；TCN/GRU 各三种子。保留相同编码器初始参数。',
'基础组只用二分类 BCE；历史训练包含类型 CE、增强 BCE、JS 和隐藏平滑与4D控制器。控制器增加10训练参数，预测时没有直接读取控制器状态。历史 ep 为背景速度、变化率、散度和旋度，未经标准化；因此 raw_bundle 虽然推理输入只有目标船，训练过程仍使用邻域物理信息。',
'两组附加条件采用各自训练数据拟合的四维 z-score，裁至 [-5,5]；单船 ep 是目标观测速度模长、相邻速度模长变化率、运动学不一致和可信权重；背景 ep 是历史背景四维。未知的单船分量用零填充，合格掩码仅单独保存、未输入控制器，因此零填与标准化后的均值相同。单船状态不输入标签、注入类型或污染前真值。两组状态含义、适格掩码和缺失模式不同，不能将差异解释成某一物理分量的因果贡献；历史背景 ep 原始版本本身未经标准化，不与标准化版混同。',
'两个标准化版本使用相同损失、网络输入与基础架构；不提供改变单个辅助损失、单个背景物理量或纯孤立正则化效应的证明。', '',
'## 二、验证与历史测试结果','',
'验证集为开发集；每模型预先从验证阴性确定1%/5%阈值，评估测试所有预先指定对照，不按测试重新选参数。下表均为3种子平均，FAR/召回对应1%验证阈值的实际测试经验结果。', '',
'| 骨干 | 配置 | 验证区域macro AP | 测试区域macro AP | 测试pooled AP | 测试AUROC | 测试1%阈值召回 / 实测FAR |',
'|---|---|---:|---:|---:|---:|---:|']
for arch in ('tcn','gru'):
 for arm in arms:
  vals=[results[(arch,s,arm)] for s in (0,1,2)];v=dev['metrics'][arch][arm]['mean_macro_AP']
  row=f"| {arch.upper()} | {arm} | {v:.4f} | {testmean(arch,arm,'macro_AP'):.4f} | {testmean(arch,arm,'pooled_AP'):.4f} | {testmean(arch,arm,'pooled_AUROC'):.4f} | {statistics.mean(r['0.01']['recall'] for r in vals):.3f} / {statistics.mean(r['0.01']['actual_FAR'] for r in vals):.3f} |"
  lines.append(row)
lines += ['','以上为每模型按验证阴性冻结阈值，不是以测试真值重新选择恰为1% FAR。5%工作点、区域和逐种子结果在 historical_test_results/test_results.json 中。',
          '', '## 三、预设配对','',
          '下表均为相同骨干和种子配对差值，单位百分点；区间对固定三个种子的平均差，以同一组2000次目标MMSI聚类抽样得到，不覆盖训练种子总体、共享邻船和场景相关性；已曝光测试集的结果只能如实作为历史测试证据。', '',
          '| 骨干 | 对照（前者减后者） | 验证区域AP差 | 测试区域AP差 [95% CI] | 测试pooled AP差 [95% CI] |','|---|---|---:|---:|---:|']
pairs=[('causal','raw'),('raw_common','raw'),('causal_common','causal'),('raw_bundle','raw'),('causal_bundle','causal'),('raw_common_bundle','raw_common'),('causal_common_bundle','causal_common'),('raw_common_bundle','raw_bundle'),('causal_common_bundle','causal_bundle'),('old_bundle','old'),('comp_bundle','comp'),('raw_bundle_self','raw_bundle_bgz'),('causal_bundle_self','causal_bundle_bgz'),('causal_common_bundle_self','causal_common_bundle_bgz'),('raw_bundle_bgz','raw_bundle'),('causal_bundle_bgz','causal_bundle'),('causal_common_bundle_bgz','causal_common_bundle')]
paired={(r['architecture'],r['a'],r['b'],r['metric']):r for r in test['paired_AP']}
for arch in ('tcn','gru'):
 for a,b in pairs:
  v=dev['paired'][arch][a+' minus '+b]['mean_pp'];macro=paired[(arch,a,b,'macro_AP')];pool=paired[(arch,a,b,'AP')]
  fmt=lambda x:f"{100*x['mean_difference']:+.3f} [{100*x['CI95'][0]:+.3f}, {100*x['CI95'][1]:+.3f}]"
  lines.append(f'| {arch.upper()} | {a} − {b} | {v:+.3f} | {fmt(macro)} | {fmt(pool)} |')
lines += ['', '## 四、完整三因素矩阵的条件效应','',
          '三因素为输入（raw/causal 门控包）、支持标记（关/开）、历史训练包（关/开）。每项效应按其他两个因素的四格等权平均；交互按对应配对差的差计算。三种子的标准差是训练随机性，不应视为独立船舶或海域的采样区间。', '',
          '| 骨干 | 条件效应（pp） | 验证平均 | 测试平均 | 各测试种子 |','|---|---|---:|---:|---|']
def arms_for(g,s,b):return ('causal' if g else 'raw')+('_common' if s else '')+('_bundle' if b else '')
for arch in ('tcn','gru'):
 for factor in ('gate','support','bundle'):
  delta_test=[];delta_dev=[]
  for seed in (0,1,2):
   tt=[];vv=[]
   for a in (0,1):
    for b in (0,1):
     off={'gate':(0,a,b),'support':(a,0,b),'bundle':(a,b,0)}[factor];on={'gate':(1,a,b),'support':(a,1,b),'bundle':(a,b,1)}[factor]
     ai=arms_for(*on);bi=arms_for(*off);tt.append(results[(arch,seed,ai)]['macro_AP']-results[(arch,seed,bi)]['macro_AP']);vv.append(dev['metrics'][arch][ai]['macro_AP'][seed]-dev['metrics'][arch][bi]['macro_AP'][seed])
   delta_test.append(100*statistics.mean(tt));delta_dev.append(100*statistics.mean(vv))
  lines.append(f'| {arch.upper()} | {factor} | {statistics.mean(delta_dev):+.3f} | {statistics.mean(delta_test):+.3f} | '+', '.join(f'{v:+.3f}' for v in delta_test)+' |')
lines += ['', '旧 common 标记无包情况下过去均值下降，不意味着训练包开启时效果必然相同；上表和配对表提供条件效应。',
          '', '### 明确的差中差交互', '',
          '交互表示同一骨干和种子内的差中差；三个种子平均及其固定种子、目标MMSI聚类95%区间如下。不同差值/交互未经多重比较调整，应作为开发探索。', '',
          '| 骨干 | 交互 | 验证macro（pp） | 测试macro（pp）及95% CI | 测试pooled（pp）及95% CI | 测试macro三个种子（pp） |',
          '|---|---|---:|---:|---:|---|']
inter={(r['architecture'],r['name'],r['metric']):r for r in test['interactions']}
for arch in ('tcn','gru'):
 for name,entry in dev['interactions'][arch].items():
  m=inter[(arch,name,'macro_AP')];p=inter[(arch,name,'AP')]
  fmt=lambda x:f"{100*x['mean_interaction']:+.3f} [{100*x['CI95'][0]:+.3f}, {100*x['CI95'][1]:+.3f}]"
  lines.append(f"| {arch.upper()} | {entry['formula']} | {entry['mean_pp']:+.3f} | {fmt(m)} | {fmt(p)} | "+', '.join(f'{100*x:+.3f}' for x in m['seed_interactions'])+' |')
lines += [
          '', '## 五、文件与核验','',
          '- `protocol.json`、`prepare_conditions.py`、`condition_stats.json`：固定设计与训练期单船/背景状态及范围。',
          '- `train_conditions.npz`、`val_conditions.npz`：四维输入与资格掩码；仅为训练条件，不含标签或污染前真值。',
          '- `runs/`：72次新增训练的权重、预测、标准化、初始权重哈希与每轮批次哈希。',
          '- `validation_summary_v2.json`：全部开发结果及完整配对差。初版 `validation_summary.json` 保留以追溯，差值列表较少。',
          '- `frozen_test_plan.json`、`historical_test_results/`：测试前固定的权重和验证阈值、逐窗预测、测试结果与聚类配对区间。',
          '- 使用旧对照时依赖此前 `causal_gate_main_v1_20261027`、`gating_training_v2_20261027` 和关联数据，不应把本目录误当独立全数据包。','']
path=NEW/'主基准正交消融与训练条件报告.md';path.write_text('\n'.join(lines),encoding='utf8');print('REPORT_COMPLETE',flush=True)
