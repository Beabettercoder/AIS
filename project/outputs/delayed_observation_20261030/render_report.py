from pathlib import Path
import json,hashlib
root=Path(r'E:\AIS\outputs\delayed_observation_20261030')
summary=json.loads((root/'fixed_normalizer'/'paired_summary.json').read_text(encoding='utf8'))
lines=[
'# 一报文延迟确认：低速观测证据与原阶段累计模型配对',
'',
'## 试验定位',
'',
'这是一项基于训练来源探索的负向对照。候选使用同一 AIS 轨迹的后续位置报文，不是独立传感器；前述训练诊断发现它可能连带压低未受控改速的机动代理，不能称为已验证的低速污染修复。模型验证来源此前已经曝光，未读取测试标签。',
'',
'## 证据与时间约束',
'',
'对邻船历史报文 i，过去三条报告航速中位数低于 2 kn 且 i 与前、后报告时间间隔均在 (0,300] 秒，计算 e_cur=||Δp_i/Δt_i−v_i||、e_next=||Δp_(i+1)/Δt_(i+1)−v_i||；score=max(0,2 e_cur−e_next−0.5 m/s)，额外乘子 0.7+0.3 exp(−score²/2) 乘以原 w_abs。只有在邻船下一条报文 i+1 已不晚于当前目标时点时，才对历史报文 i 生效；最新报文和不足历史的报文均采用原门控，不使用注入标签或干净轨迹真值。F0和F1仍使用同一组成员、原阶段累计、原仿射/零阶回退。',
'',
'按固定目标窗口的阶段划分，删除目标时点之后的邻船报文所做 96 次前缀比较通过；独立代理还对16个验证来源作384次未来邻船值扰动检查，当前场逐位一致。原流程使用完整目标窗口的 n//2 确定阶段，邻船cohort也在离线资料中固定，因此这些检查不构成完整在线系统的严格前缀证明。',
'',
'训练内400来源的历史低速、受控前当前低速组，受控改速230条平均潜在额外降权8.35%；限定当前仍低速的未施压速度变化代理410条为1.40%。扩大到包含当前已加速的未施压历史低速代理2462条，平均潜在额外降权11.44%，其中76.12%至少一个同阶段目标时刻有已到达后继，按所有报文计的实际回看机会平均降权约8.69%。这些分母不同、报文与四格可重复、未施压并不代表真实无噪声或真实机动标注；候选阈值系基于训练内探索。',
'',
'## 模型配对',
'',
'训练使用 Base 和原累计train/Base F0/F1固定标准化，复用相同输入的原C0/F0及原F1预测；仅候选F1重新训练。TCN和iTransformer各3 seeds、25/75轮，GRU和Transformer各1 seed、25轮。区间按目标MMSI聚类配对Bootstrap 2000次，仅固定模型样本不确定性，不覆盖随机种子总体、共享背景船依赖和多重比较。',
'',
'| 骨干 | 视图 | 原 F0 / 原 F1 | 候选 F1 | 候选 F1−F0 pp [95% CI] | 候选 F1−原 F1 pp [95% CI] |',
'|---|---|---|---:|---|---|']
names={'base':'Base','natural':'自然速度压力','suffix':'后段速度压力','combined_l100_m600':'后段速度+位置压力'}
for name in ('tcn','gru','transformer','itransformer'):
 for view in names:
  a=summary[name]['views'][view];seeds=summary[name]['seeds'];vs=a['values'];av=lambda code:sum(vs[f'{s}_{code}'] for s in seeds)/len(seeds)
  def fmt(key):
   val=a['mean_diff'][key]*100;lo,hi=[x*100 for x in a['bootstrap95'][key]]
   return f'{val:+.2f} [{lo:+.2f},{hi:+.2f}]'
  lines.append(f'| {name} ({len(seeds)} seed) | {names[view]} | {av("cumulative_F0"):.4f} / {av("cumulative_F1"):.4f} | {av("safe_F1"):.4f} | {fmt("safe_f1_f0")} | {fmt("f1_safe_minus_old")} |')
lines+=['','## 每种子差异与训练状态','','| 骨干 | 视图 | 候选F1−原F0，每seed pp | 候选F1−原F1，每seed pp |','|---|---|---|---|']
for name in ('tcn','gru','transformer','itransformer'):
 for view in names:
  a=summary[name]['views'][view]['per_seed_diff'];fmt=lambda key:' / '.join(f'{100*x:+.2f}' for x in a[key])
  lines.append(f'| {name} | {names[view]} | {fmt("safe_f1_f0")} | {fmt("f1_safe_minus_old")} |')
lines+=['','iTransformer seed1 在75轮后训练 BCE≈0.693、Base AP≈0.502，仍未收敛；其余两seed Base AP≈0.847/0.859。故其候选F1与原F0均值差混入了优化不稳定，不能简单归因门控。TCN 后段 F1−F0 +0.33pp、95%区间 [−1.02,+1.45]pp，跨零。候选训练中的机动代理降权风险尚存；四骨干条件未满足，不触发八骨干训练。','','## 可用外部字段与资产','','已有不同海域 AIS 原始文件含 Heading 和部分航行状态，Heading 有效覆盖不均；ROT 在丹麦原始文件存在而在现有标准化数据被丢弃。不同字段不等同于独立传感器，且若接入受控合成场景，必须逐源对齐并核验与机动注入的物理一致性，不能把合成前真值直接提供给候选。该试验没有加入这些字段。','','模型、逐样本预测、固定训练归一化、训练/验证场缓存、冻结协议、构造脚本与日志在当前目录。候选特征使用目标时刻已到达的报文，不读取受控标记；C0/F0输入是否与原版逐点等价、源报文场重算、预测与区间由独立核验文档记录。']
report=root/'延迟报文证据小规模配对报告.md'
report.write_text('\n'.join(lines)+'\n',encoding='utf8')
entries=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.name!='complete_manifest.json' and '__pycache__' not in p.parts:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
  entries.append({'path':p.relative_to(root).as_posix(),'sha256':h.hexdigest(),'size':p.stat().st_size})
(root/'complete_manifest.json').write_text(json.dumps({'files':entries,'count':len(entries)},indent=2,ensure_ascii=False),encoding='utf8')
print(report,'files',len(entries))
