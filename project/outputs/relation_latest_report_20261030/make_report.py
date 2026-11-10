from pathlib import Path
import hashlib,json
out=Path('E:/AIS/outputs/relation_latest_report_20261030')
d=json.loads((out/'paired_summary.json').read_text(encoding='utf-8'))
labels={'base':'Base','natural':'自然速度','suffix':'后段速度','combined_l100_m600':'后段速度+位置'}
def ci(v):return f'[{100*v[0]:+.2f}, {100*v[1]:+.2f}]'
def vals(row,method,arm,seeds): return sum(row['values'][f'{s}_{method}_{arm}'] for s in seeds)/len(seeds)
lines=['# 最新报文与阶段累计流场：冻结配对试验', '',
'本试验在观察到原阶段累计报文均值可能稀释短时速度污染之后，预先冻结唯一候选：将每个邻船当前时刻有效且不晚于目标时刻、年龄不超过300秒的最新一条报文作为该邻船速度。F0与F1使用同一报文、同一距离权重和成员集合，F1额外采用该报文运动学可信度。两臂共同沿用一阶仿射有效条件、零阶加权均值回退和无场标记。该改动同时改变训练和评估的成员聚合，且成员权重使用最近值，与原累计方案的效果比较是整个配置比较。','',
'相同源观测、四格和 Base / 自然速度 / 后段速度 / 后段速度+位置四视图；训练仅用Base且训练集统计标准化。TCN、iTransformer各3种子，分别固定25轮和75轮；GRU、Transformer各seed0、固定25轮。均为BCE与相同架构、等宽35通道，相同骨干和seed初始化一致；iTransformer使用同样通道身份嵌入。窗口级样本/标签、目标MMSI、成员计数与原数据一致。不读取测试标签。验证集此前曝光，结果为开发集比较。','',
'区间为固定各模型的目标MMSI配对聚类2,000次bootstrap；不包括种子总体不确定性、跨背景船依赖和多重比较校正。GRU、Transformer仅单种子。','',
'## 原累计与最新报文的完整对照','',
'表中数值为不同seed模型pooled AP的平均，括号为 F1−F0 的目标MMSI区间，单位为百分点；并列报告原配置，不能只看新方案的相对收益。','','| 骨干 | 视图 | 累计 F0 / F1 | 最新 F0 / F1 | 最新 F1−F0 pp [95% CI] | 最新 F1 相对累计 F1 pp [95% CI] |','|---|---|---|---|---|---|']
for name in ('tcn','gru','transformer','itransformer'):
  seeds=d[name]['seeds']
  for view in labels:
    row=d[name]['views'][view];avg=row['mean_diff'];bounds=row['bootstrap95']
    old=f"{vals(row,'cumulative','F0',seeds):.4f} / {vals(row,'cumulative','F1',seeds):.4f}"
    new=f"{vals(row,'latest','F0',seeds):.4f} / {vals(row,'latest','F1',seeds):.4f}"
    lines.append(f"| {name} | {labels[view]} | {old} | {new} | {100*avg['latest_f1_f0']:+.2f} {ci(bounds['latest_f1_f0'])} | {100*avg['f1_latest_minus_old']:+.2f} {ci(bounds['f1_latest_minus_old'])} |")
lines += ['', '## 各种子 F1−F0 差值', '', '| 骨干 | 视图 | 阶段累计 pp | 最新报文 pp |', '|---|---|---|---|']
for name in ('tcn','gru','transformer','itransformer'):
  for view in labels:
    dif=d[name]['views'][view]['per_seed_diff']; fmt=lambda values:' / '.join(f'{100*x:+.2f}' for x in values)
    lines.append(f"| {name} | {labels[view]} | {fmt(dif['old_f1_f0'])} | {fmt(dif['latest_f1_f0'])} |")
lines += ['', '## 结论与边界', '',
'按此前四骨干门槛，本候选未通过：TCN 的后段 F1−F0 平均为 +1.35 pp（区间 [+0.01, +2.59]），但其 F1 相对原阶段累计绝对 AP 下降 2.18 pp；iTransformer 的后段 F1−F0 平均为 −2.74 pp（区间 [−5.20, −0.05]），Transformer 后段区间跨零。',
'最新报文使后段污染暴露更加集中：此前独立观测计算中 F0/F1 的平均背景速度扰动从累计版0.133/0.054 m/s变为0.512/0.281 m/s，且持续的时步比例更短。它能检验稀释机制，但也让模型更加依赖短时报告，不能预先假设更好。',
'iTransformer 75轮、最新报文 F1 的三个seed Base AP请看上表与逐seed JSON，其中seed1在75轮结束训练损失仍接近 ln2、Base AP约0.514；它不是稳定收敛。这会影响跨种子平均和F1−F0差异的解释。GRU、Transformer的候选结果仅一个seed。',
'原方案在同一开发集已被多次使用，聚类bootstrap区间不是独立盲测，也不是种子方差或严格显著性证明。F0/F1区别包括最新报文可靠度权重与支持特征，不是纯单参数控制。没有达到既定条件，不启动八骨干三种子正式训练。', '',
'## 文件与独立核验', '',
'`frozen_protocol.json` 和 `frozen_addendum.json` 在模型训练前落盘；`builder.py`、`train.py`、`summary.py`、训练/验证数据、各seed预测与权重、`paired_summary.json` 和运行日志保存在本目录。独立代理复算了候选24组模型四视图的AP/AUROC、预测和权重哈希，并复算F1−F0与F1−C0的2,000次聚类区间。`summary.py` 在任何候选模型结果产生前修正了新增 GRU/Transformer 的25轮断言，并增加同seed两方案初始权重一致性检查；不改变训练。','']
(out/'最新报文配对验证报告.md').write_text('\n'.join(lines),encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='complete_manifest.json']
manifest={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(files)}
(out/'complete_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('files',len(files),'report',sha(out/'最新报文配对验证报告.md'))
