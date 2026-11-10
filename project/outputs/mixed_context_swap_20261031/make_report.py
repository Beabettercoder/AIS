from pathlib import Path
import json
OUT=Path(__file__).resolve().parent
x=json.loads((OUT/'summary.json').read_text())
names={'tcn':'TCN（3 seeds）','gru':'GRU（1 seed）','transformer':'Transformer（1 seed）','itransformer':'iTransformer（3 seeds）'}
views={'base':'Base','natural':'自然速度压力','suffix':'后段速度压力','combined_l100_m600':'后段速度＋位置压力'}
def ci(d):return f"{100*d['mean']:+.2f} [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}]"
lines=['# 同训练污染暴露：场输入与模型交叉诊断','',
 '## 研究问题与范围','',
 '本实验复用 mixed_stress_probe_20261031 的混合污染训练 F0/F1 终轮模型，在先冻结配置后进行 2×2 输入交换，不重新训练，不读取测试。训练集和已多次用于开发的验证集目标 MMSI 隔离，背景船可能跨分区复用；四格是受控关系标签而非真实事故或故障真值。',
 'F0 表示普通局部场，F1 表示可信局部场。将原方案对应的整套 40 维特征交换，其中场残差、邻船 Heading 与支持量均跟随所选输入变化；同一目标观测、窗口、标签、mask、时间和训练集拟合的共享归一化保持一致。非对角线属于模型训练时未必见过的输入分布，只用于探索模型与输入的依赖，不能作为正式训练指标，也不能将结果归因于单个权重。',
 'TCN 与 iTransformer 各使用 seed 0/1/2，GRU 和 Transformer 各使用 seed 0；与原模型一样使用固定第25轮或第75轮。四个视图全部保留。每个配置对角线预测与已存原生预测同 GPU 重放完全一致，最大误差为0。',
 '全部 AP 为 pooled AP。区间使用同一目标 MMSI 聚类配对 Bootstrap 2,000 次，对固定种子模型的AP差先求均值，不包含种子总体不确定性、跨分区背景船重用或多重比较。','',
 '## 自然速度与后段速度：完整 2×2','',
 '| 骨干 | 压力 | F0模型＋F0输入 | F0模型＋F1输入 | F1模型＋F0输入 | F1模型＋F1输入 | F0模型：换F1输入（pp, 95% CI） | F1模型：换F1输入（pp, 95% CI） |','|---|---|---:|---:|---:|---:|---|---|']
for key in ('tcn','gru','transformer','itransformer'):
 for view in ('natural','suffix'):
  z=x[key]['views'][view];a=z['mean_ap'];c=z['contrasts']
  lines.append(f"| {names[key]} | {views[view]} | {a['F0_on_F0']:.4f} | {a['F0_on_F1']:.4f} | {a['F1_on_F0']:.4f} | {a['F1_on_F1']:.4f} | {ci(c['F0_model_switch_field'])} | {ci(c['F1_model_switch_field'])} |")
lines+=['','## 四视图原生对角线对照','',
 '| 骨干 | 视图 | F0模型＋F0输入 | F1模型＋F1输入 | F1−F0（pp, 95% CI） |','|---|---|---:|---:|---|']
for key in ('tcn','gru','transformer','itransformer'):
 for view in views:
  z=x[key]['views'][view];a=z['mean_ap'];lines.append(f"| {names[key]} | {views[view]} | {a['F0_on_F0']:.4f} | {a['F1_on_F1']:.4f} | {ci(z['contrasts']['native_F1_minus_F0'])} |")
lines+=['','## 解释','',
 '相同混合训练暴露下，TCN 自然压力中，同一 F0 模型换用 F1 输入提高约4.40 pp、同一 F1 模型换用 F1 输入提高约9.69 pp；后段压力分别变化−1.03 pp和+0.40 pp，方向和幅度依赖于所用模型和视图。自然压力下可信上下文提供的信号可以被两个模型不同程度利用；后段中 F0 训练配置本身已较强，直接交换上下文不能建立一致的增益。',
 '该比较不证明训练中模型学到了某个确定物理机制。输入中同时交换多个场特征，非对角线可能偏离相应模型训练分布；每个网络只在各自配置下训练，所以模型差异也不能解释为独立因果效应。特别是 GRU/Transformer 仅一个seed；所有区间为条件化固定模型的样本区间。',
 '## 资产与核验','',
 '`frozen_protocol.json` 在实验评分前保存所用模型、输入和代码哈希；首版 `frozen_protocol_pre_syntax_fix.json` 对应的脚本存在语法错误，训练/数据均未加载，修复语法并重新冻结后运行。`run_swap.py`、四个 `*_scores.npz`、`summary.json` 记录逐窗配对和每个seed的结果。原模型检查点、验证预测及对应输入的哈希在冻结清单和汇总中。','']
(OUT/'同污染训练的场模型交叉诊断.md').write_text('\n'.join(lines),encoding='utf8')
