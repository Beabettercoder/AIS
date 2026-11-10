import json,hashlib
from pathlib import Path
r=Path('/mnt/sda/WangZijia/AIS/outputs/target_heading_20261031')
x=json.loads((r/'paired_summary.json').read_text())
def pct(v):return f'{v*100:+.2f}'
def cell(v,c):return f'{pct(v)} [{pct(c[0])}, {pct(c[1])}]'
names={'tcn':'TCN','gru':'GRU','transformer':'Transformer','itransformer':'iTransformer'}
views={'base':'Base','natural':'自然速度','suffix':'后段速度','combined_l100_m600':'速度+位置复合'}
s=['# 目标船 Heading：原阶段累计下的四骨干配对','',
'## 范围与设计','',
'模型使用之前的邻船 Heading 五通道与原阶段累计场；本次仅额外引入目标船 Heading 与受控目标 COG 的圆周 sin/cos、Heading 有效 mask、与前次合法 Heading 的角速度（单位度/秒）及角速度有效 mask。按窗口 ID、目标 MMSI 与 UTC 秒严格匹配，非法值掩码处理；不读取原始干净速度/位置和注入标记作为特征。','',
'四组均为45维：C0_target 只有目标流及目标 Heading；F0_target 是普通场与邻船、目标 Heading；F1_no_target 是可信场与邻船 Heading、目标 Heading 5槽归零；F1_target 是可信场与邻船和目标 Heading。即本试验是目标 Heading 在已经包含邻船 Heading 的上下文上的增量，不是单独只用目标 Heading 的消融。','',
'每骨干同 seed 四臂相同初始化、预算和训练集归一化；只在训练 Base 上训练、固定末轮。TCN 3种子25轮、GRU和Transformer各1种子25轮采用窗口随机批次；iTransformer 3种子75轮采用来源成批（同源四格同批），该探索性设置来自之前40维小试，不能称独立预注册证明。验证集 272来源/86目标，已有开发暴露，不读取测试。','',
'目标MMSI配对聚类bootstrap 2,000次；区间只反映固定已训练模型的样本不确定性，不包含训练种子总体或跨来源背景船重用和多重比较。目标单侧四格配对 C0 AP 恒约0.5，是构造预期，不是优化失败。','',
'## 全体验证结果：pooled AP','',
'| 骨干 | 视图 | C0目标H | F0双H | F1仅邻船H | F1双H | 新增目标H差pp [95% CI] | 可信配置差pp [95% CI] |','|---|---|---:|---:|---:|---:|---|---|']
for name in names:
 d=x[name]
 for v,label in views.items():
  a=d['groups']['all']['views'][v];m=a['mean_AP'];q=a['mean_diff'];ci=a['bootstrap95']
  s.append(f"| {names[name]} | {label} | {m['C0_target']:.4f} | {m['F0_target']:.4f} | {m['F1_no_target']:.4f} | {m['F1_target']:.4f} | {cell(q['target_heading_gain'],ci['target_heading_gain'])} | {cell(q['reliability_config'],ci['reliability_config'])} |")
s+=['','## 原观测低速目标窗口的关系任务子组','',
'子组按原始未注入目标源报文是否存在SOG<2kn分层，且该目标窗口至少某时刻Heading有效；两条件不必发生在同一时刻。此筛选只用于事后分析，不输入模型；样本为13来源/11目标船/52格，标签依然是受控群体关系标签，并非真实机动或传感器故障标签。真实AIS的规则筛选轨迹另见真实样本独立报告。','',
'| 骨干 | 原场F1+邻船H | 再加目标H | 目标H增量pp [95% CI] |','|---|---:|---:|---|']
for name in names:
 a=x[name]['groups']['raw_target_low_and_heading_available']['views']['suffix'];m=a['mean_AP'];q=a['mean_diff'];ci=a['bootstrap95']
 s.append(f"| {names[name]} | {m['F1_no_target']:.4f} | {m['F1_target']:.4f} | {cell(q['target_heading_gain'],ci['target_heading_gain'])} |")
s+=['','## 每seed 训练状态','',
'| 骨干 | seed | 目标自船C0 | 普通场+目标H | 可信场无目标H | 可信场+目标H |','|---|---:|---:|---:|---:|---:|']
for name in names:
 d=x[name]
 for seed in d['seeds']:
  q=d['run_loss_final']
  s.append(f"| {names[name]} | {seed} | {q[str(seed)+'_C0_target']:.4f} | {q[str(seed)+'_F0_target']:.4f} | {q[str(seed)+'_F1_no_target']:.4f} | {q[str(seed)+'_F1_target']:.4f} |")
s+=['','## 判断与可复核资产','',
'在后段速度视图，TCN 的可信场双Heading相对普通场双Heading配对区间跨零，预先约定的四骨干严格门槛不满足，不启动八骨干正式训练。目标Heading增量的统计范围是合成关系分类，而不是传感器故障真值。Heading与原始合成轨迹的物理一致性未有人工真值验证。','',
'paired_summary.json 含所有视图、每seed预测、配对区间及分组数量；runs保存全部32模型、预测与协议；input_audit.json、model_smoke.json记录源对齐及前后向检查，冻结方案及代码SHA见 frozen_protocol_window.json、frozen_protocol_itrans.json。','']
(r/'目标Heading增量配对报告.md').write_text('\n'.join(s),encoding='utf8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
items={str(p.relative_to(r)):{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(r.rglob('*')) if p.is_file() and p.name!='complete_manifest.json' and '__pycache__' not in p.parts}
(r/'complete_manifest.json').write_text(json.dumps({'file_count':len(items),'files':items},indent=2),encoding='utf8')
print('files',len(items))
