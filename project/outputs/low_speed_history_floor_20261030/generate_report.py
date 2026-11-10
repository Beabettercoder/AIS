from pathlib import Path
import json,hashlib,os
import numpy as np
B=Path('/mnt/sda/WangZijia/AIS/outputs');O=B/'low_speed_history_floor_20261030'; F=O/'fixed_normalizer'; D=B/'low_speed_quantile_20261030'
x=json.loads((F/'paired_summary.json').read_text()); q=json.loads((D/'calibration_full_cluster.json').read_text()); h=json.loads((D/'history_probe.json').read_text()); g=json.loads((D/'history_innovation_grid.json').read_text()); proto=json.loads((F/'frozen_protocol.json').read_text())
lines=['# 历史低速有界可信度：训练证据与小规模负向对照','', '## 核心判定','', '温和历史速度判据不是可以直接部署的已验证修复：训练集内低速改速与未改机动代理明显重叠，候选额外降权在未改大速度变化样本更强；完整模型在后段压力未满足四骨干约定条件。所有试验仅使用已有开发训练/验证来源，既有验证源已曝光；未读取测试标签，未启动八骨干完整实验。','', '## 已知方法与预训练风险','', '以原阶段累计邻船历史、原因果观测门控与相同目标输入为基准。e_mid=||Δp/Δt−(v_t+v_{t−1})/2||。当过去3条已到达SOG的中位数 h<2kn 且 e_cur=||Δp/Δt−v_t||≥0.5m/s 时，原 w_abs 乘以 m=0.7+0.3 exp[−0.5 max(0, |SOG_t−h|/(h+0.5kn)−1)^2]；m属于[0.7,1]。历史不足、输入无效则回退原门控；没有使用污染标签、原始真值或未来观测。过去三条的总跨度没有硬上限，必须作为限制。','', '该规则在看到训练集内低速相容残差分析和15种门槛网格后，按主任务要求在任何候选模型训练之前锁定为一种温和负向对照，不能称为独立预注册发现。其门槛范围、样本筛选和历史依据均记录于 frozen_protocol.json 与 low_speed_quantile_20261030 的脚本。','', '## 无标签训练分位数与受控报文分析','',f"2034个训练来源、584目标MMSI的 Base 四格观测（含原生未知噪声），以邻船MMSI和相邻两报文时间、位置、速度去重后，min(位置差分速度,当前报告速度)<2kn 的有效记录 {q['unique_records']} 条、邻船 {q['neighbor_mmsi_count']} 艘（去重前 {q['before_dedup']} 条）。中点残差 q95={q['q95']:.3f} m/s（邻船MMSI聚类区间 [{q['neighbor_mmsi_cluster_bootstrap_2000_q95_CI'][0]:.3f},{q['neighbor_mmsi_cluster_bootstrap_2000_q95_CI'][1]:.3f}]）、q99={q['q99']:.3f} [{q['neighbor_mmsi_cluster_bootstrap_2000_q99_CI'][0]:.3f},{q['neighbor_mmsi_cluster_bootstrap_2000_q99_CI'][1]:.3f}]。这不是标注正常数据，分位数不是误报率保证；低速报文且位置异常可进入该分组。", '',f"若按前3条已到达SOG中位数<2kn分组，无标签训练Base去重 {h['unique_base_calibration_records']} 条、{h['neighbor_mmsi']} 艘邻船，q95={h['q95']:.3f}、q99={h['q99']:.3f} m/s。受控评估中同时满足基础当前<2kn、受扰当前与该历史分组的1462重复报文，按95/99分位软门控的w<0.3仅从0升至0.274%；同历史组未改且前报文未改39590条从0.389%升至1.076%。分组统计与分位校准不是独立测试。", '', '训练400来源的温和候选诊断：候选触发条件为历史低速且位置/当前速度不一致≥0.5m/s，以下统计分母包括满足历史低速分组的全部有效报文：受控改速且基础当前<2kn 278条（44目标）平均额外降权5.6%；受控未改、前一报文未改且相邻SOG变化≥1kn的2590条（129目标）平均额外降权19.7%；受控未改且相邻航向差≥45°的2966条平均额外降权10.4%。四格/跨来源报文可重复，未改是未施加当前受控压力，不能等同真实无噪声或人工确认机动。','', '15项训练内阈值网格也保留于 history_innovation_grid.json：改速低速的强抑制与未改机动代理的强抑制均需同时报告，不依据开发验证集挑参数。经验上目标信息在低速被误改与真机动之间明显重叠，不能从这项有限结果声称数学不可能区分。','', '## 多骨干配对结果','', '8组候选F1训练；TCN/iTransformer 3 seeds 固定25/75轮，GRU/Transformer各1 seed25轮，BCE、原先模型和初值、训练集原方案固定归一化；F0/C0预测从原有匹配模型复用。候选/原物理F0、mask及固定归一化输入独立核验逐项一致。开发验证集 272来源、86目标MMSI、1088四格。每行记录 pooled AP；差为百分点，区间是固定模型按目标MMSI成对bootstrap 2000次，不包括训练种子总体不确定性、背景船依赖或多重比较校正。','', '| 骨干 | 视图 | 原 F0 / F1 | 候选 F1 | 候选 F1−原 F0 pp [95% CI] | 候选 F1−原 F1 pp [95% CI] |', '|---|---|---|---:|---|---|']
views=[('base','Base'),('natural','自然速度'),('suffix','后段速度'),('combined_l100_m600','后段速度+位置')]
for model in ['tcn','gru','transformer','itransformer']:
 z=x[model]; seeds=z['seeds']
 for code,nm in views:
  a=z['views'][code];val=a['values'];avg=lambda k:float(np.mean([val[f'{seed}_{k}'] for seed in seeds])); mean=a['mean_diff'];cis=a['bootstrap95']
  fmt=lambda k:f"{mean[k]*100:+.2f} [{cis[k][0]*100:+.2f},{cis[k][1]*100:+.2f}]"
  lines.append(f"| {model} ({len(seeds)} seed) | {nm} | {avg('cumulative_F0'):.4f} / {avg('cumulative_F1'):.4f} | {avg('safe_F1'):.4f} | {fmt('safe_f1_f0')} | {fmt('f1_safe_minus_old')} |")
lines+=['','## 逐种子与收敛', '', '| 骨干 | 视图 | 候选 F1−原F0逐种子 pp | 候选 F1−原F1逐种子 pp |', '|---|---|---|---|']
for model in ['tcn','itransformer']:
 for code,nm in views:
  a=x[model]['views'][code]['per_seed_diff']; fmt=lambda k:' / '.join(f'{v*100:+.2f}' for v in a[k]); lines.append(f"| {model} | {nm} | {fmt('safe_f1_f0')} | {fmt('f1_safe_minus_old')} |")
lines+=['','iTransformer seed0 候选 F1 在75轮后 Base AP约0.5116，末轮损失约0.693；其余seed需要看各 result.json，均值与区间受到这个未收敛种子影响，不能将模型差异简单归因于门控。TCN后段F1−F0区间跨零。该方案还有训练集内已识别的未改机动代理降权风险。故既定条件未达到，未启动八骨干训练或测试。','', '## 资产与核验','', '本目录包含候选 builder.py、train/val 场数据、fixed_normalizer/frozen_protocol.json、训练脚本、权重、逐窗口预测、日志与 paired_summary.json；代码、数据、预测和权重由 complete_manifest.json 记录 SHA-256。无标签分位与15候选训练内观察代码和结果位于 ../low_speed_quantile_20261030/。原匹配基线模型留在 heading_alignment_20261030、tcn_suffix_stability_20261030、itrans_stability_20261030、neighbor_gate_safe_20261030。原始可信场与本轮可解释性结论并非全新独立海域实证。']
(O/'低速历史门控探索与配对报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
files=[]
for root in [O,D]:
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.name=='complete_manifest.json' or '__pycache__' in p.parts:continue
  hh=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1<<20),b''):hh.update(block)
  files.append({'path':str(p.relative_to(B)),'size':p.stat().st_size,'sha256':hh.hexdigest()})
(O/'complete_manifest.json').write_text(json.dumps({'scope':'candidate and train-only diagnostic evidence','files':files},indent=2),encoding='utf8');print('report',O/'低速历史门控探索与配对报告.md','manifest',len(files))

