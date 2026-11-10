from pathlib import Path
import json,hashlib
root=Path(__file__).parent
summary=json.loads((root/'development_summary.json').read_text(encoding='utf8'))
lines=['# 航向对齐与通道身份：四骨干单种子关系试验','',
'四骨干 C0/F0/F1 共12次训练，iTransformer 另有3个 F1 配对消融。训练种子0，固定25轮，BCE；训练只使用 base 视图，评估使用此前已曝光的开发验证集。没有读取测试集。', '',
'35维输入保留先前24维原始坐标通道并添加11个航向对齐、速度模长、相对变化和变化率通道；只有新增通道满足坐标旋转不变测试。所有构造完成后使用训练数据拟合标准化，C0 的上下文槽全部置零。', '',
'iTransformer 之前已使用按通道顺序 flatten 后接线性投影，并非平均池化。本次仅在该模型新增可学习 var_embed，其他骨干未修改。下表为 pooled AP，括号为 2000 次目标 MMSI 配对聚类抽样的 F1−F0 区间（百分点）。', '',
'| 骨干 | 视图 | C0 | F0 | F1 | F1−F0 pp [95% CI] | F1−C0 pp [95% CI] |','|---|---|---:|---:|---:|---|---|']
for name,m in summary['models'].items():
 for v,d in m['values'].items():
  f0=d['F0']['pooled_AP'];f1=d['F1']['pooled_AP'];c0=d['C0']['pooled_AP'];ci=m['intervals'][v]['f1_vs_f0']
  ci0=m['intervals'][v]['f1_vs_c0']
  lines.append(f"| {name} | {v} | {c0:.4f} | {f0:.4f} | {f1:.4f} | {(f1-f0)*100:+.2f} [{ci['lo']*100:+.2f}, {ci['hi']*100:+.2f}] | {(f1-c0)*100:+.2f} [{ci0['lo']*100:+.2f}, {ci0['hi']*100:+.2f}] |")
lines.extend(['','## iTransformer 的新增变量配对消融','', '| 额外航向通道 | var_embed | Base | 自然 | 后段速度 | 复合压力 |', '|---|---|---:|---:|---:|---:|'])
for variant in ('old_noembed','old_embed','aligned_noembed','aligned_embed'):
 p=root/'runs'/variant/'itransformer'/'F1'/'result.json';r=json.loads(p.read_text(encoding='utf8')); values=[r['views'][v]['pooled_AP'] for v in ('base','natural','suffix','combined_l100_m600')];lines.append(f"| {'开' if variant.startswith('aligned') else '关'} | {'开' if variant.endswith('_embed') else '关'} | "+' | '.join(f'{z:.4f}' for z in values)+' |')
ablation=json.loads((root/'ablation_summary.json').read_text(encoding='utf8'))
lines.extend(['', '以下区间为 iTransformer 各消融的配对目标 MMSI 聚类 Bootstrap，单位为百分点；由固定单个种子模型获得，未作多重比较校正。', '', '| 视图 | 配对比较 | AP 差 [95% CI] |','|---|---|---|'])
for view,data in ablation['views'].items():
 for pair,ci in data['paired_difference'].items():
  lines.append(f"| {view} | {pair} | {ci['difference']*100:+.2f} [{ci['lo']*100:+.2f}, {ci['hi']*100:+.2f}] |")
lines.extend(['', '所谓关闭额外通道是保留35维输入并把新增11槽设零，并不等于旧24维模型；单种子可见航向特征与嵌入组合有较强交互，尚不能归为单一数学根因。iTransformer 的 F0 在 Base 下为0.5352，且训练损失接近随机分类，F1−F0 的高差异需结合 F0 训练未充分收敛解读。', '',
'按预先沟通的严格门槛：四骨干 Base F1 AP 均≥0.75；但 TCN 后段速度压力 F1−F0 区间跨零，故全部压力条件一致优于 F0 的条件未达到。95%区间仅反映固定已训练模型、目标 MMSI 聚类后的样本不确定性，未包括种子差异或多重比较校正。', '',
'压力与背景源数据保持与上一轮一致；F1 相对 F0 同时改变报文级聚合、拟合可信权重和支持特征。四格为受控合成任务，train/val 目标船和来源不重叠，但部分背景船跨分区共享。整窗离线建模不能描述为在线预警。',''])
(root/'航向对齐四骨干单种子报告.md').write_text('\n'.join(lines),encoding='utf8')
paths=[p for p in root.rglob('*') if p.is_file() and p.name not in ('manifest.json',)]
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print('files',len(paths),'report',root/'航向对齐四骨干单种子报告.md')
