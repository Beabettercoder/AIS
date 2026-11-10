"""Render four-step and field stability tables without selecting results."""
from pathlib import Path
import csv,json
import numpy as np
root=Path('/home/ps/AIS_adaptive_radius_relation_v2p1_20261109');parts=root/'summary_parts';reports=root/'reports'
def read(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
metrics=[];pairs=[]
for task in ('single','relation'):
 for model in ('tcn','gru','transformer','graph'):
  status=json.loads((parts/f'{task}_{model}.json').read_text())
  if status['status']!='complete':raise ValueError('incomplete summary')
  metrics+=read(parts/f'{task}_{model}_metrics.csv');pairs+=read(parts/f'{task}_{model}_pairs.csv')
coverage=read(reports/'test_field_coverage.csv');drift=read(reports/'test_field_drift_pairs.csv')
audit=json.loads((reports/'independent_results_audit.json').read_text());normal=json.loads((reports/'normalization_independence.json').read_text());source=json.loads((reports/'pretrain_audit.json').read_text())
if audit['status']!='passed' or any(normal[x]['status']!='passed' for x in ('single','relation')):raise ValueError('audit incomplete')
names={'tcn':'TCN','gru':'GRU','transformer':'Transformer','graph':'GAT-TCN'}
areas={'ports':'港口','restricted_channels':'受限水道','sparse_overlapping':'低密度航段（重叠）'}
views={'base':'基础观测','natural':'全窗时间对齐速度污染','suffix':'后段速度污染','combined_l100_m600':'后段速度＋位置污染'}
steps=('tight_plain','wide_plain','adaptive_plain_fallback','adaptive_credible_fallback')
def lookup(rows,**terms):
 z=[x for x in rows if all(x[k]==v for k,v in terms.items())]
 if len(z)!=1:raise ValueError('lookup mismatch '+str(terms)+' n='+str(len(z)))
 return z[0]
def pct(x):return '—' if x in ('',None) else f'{float(x)*100:.2f}%'
def ci(row,scale=100.):
 if not row.get('delta_AP'):return '样本不足'
 return f"{float(row['delta_AP'])*scale:+.2f} [{float(row['lower95'])*scale:+.2f}, {float(row['upper95'])*scale:+.2f}]"
def ap(task,model,view,group,arm):
 z=[float(r['AP']) for r in metrics if r['task']==task and r['model']==model and r['checkpoint']=='last' and r['view']==view and r['group']==group and r['arm']==arm]
 if len(z)!=3:raise ValueError('missing seed AP '+str((task,model,view,group,arm)))
 return f'{np.mean(z)*100:.2f}%'
def pair(task,model,view,group,before,after):return ci(lookup(pairs,task=task,model=model,view=view,group=group,before=before,after=after))
def num(row,key):return '—' if not row.get(key) else f"{float(row[key]):.3f}"

L=['# 动态邻域范围：普通场与可信场分离后的专项实验','',
 '本报告包含单主体与群体关系两项任务：TCN、GRU、Transformer、GAT-TCN 各三个种子、九种配置，共 216 组训练。固定第25轮为主，验证最优权重另存；测试集此前已用于研究。','',
 '普通场仅使用时间、距离、航向和距离权重；邻船可信度不参与它的筛选、拟合、支持度或训练场特征标准化。目标船自身的门控在各组相同。可信场使用运动学有效性与 w≥0.3 筛选，并在拟合中加入可信权重。两者的半径由各自可用邻船独立决定。','',
 '港口与受限水道按区域划分；低密度航段按基础观测的8海里普通邻船数量定义，与前两类重叠，并非地理上已核实的远洋。关系任务的稀疏分层按每个来源的无干预情况确定，四格保持同组。','',
 '四种观测视图均是同一样本：基础观测、全窗时间对齐速度污染、后段速度污染、后段速度＋位置污染。后三种均为受控干预；基础观测可能已有此前构造的污染。单主体与关系任务的自然时间对齐干预分别构造，二者污染命中率不应默认相同。','',
 '## 一、四步配置及场覆盖','','步骤依次为：①固定3.5海里普通场，无回退；②固定8海里普通场，无回退；③动态范围＋普通场＋零阶回退；④动态范围＋可信筛选和加权＋零阶回退。覆盖率的分母为有效目标时间步；包含回退的覆盖率不等于仿射场覆盖率。']
for task,tn in (('relation','群体关系'),('single','单主体')):
 L+=['',f'### {tn}','', '|视图|区域|固定3.5|固定8|动态普通＋回退|动态可信＋回退|其中可信方案零阶回退|','|---|---|---:|---:|---:|---:|---:|']
 for view,vn in views.items():
  for group,gn in areas.items():
   q=[lookup(coverage,task=task,view=view,group=group,arm=arm) for arm in steps]
   L.append(f"|{vn}|{gn}|{pct(q[0]['coverage'])}|{pct(q[1]['coverage'])}|{pct(q[2]['coverage'])}|{pct(q[3]['coverage'])}|{pct(q[3]['fallback_fraction'])}|")
L+=['','## 二、流场污染前后的稳定性','',
 '比较同一窗口、同一目标时刻的估计背景速度：偏移＝‖受扰场−基础场‖，单位 m/s。基础场不是已知真值。下表只统计固定8普通、动态普通、动态可信三种方法在基础和受扰视图均有参考、且该时刻命中扰动的共同时间步；无场和未命中比例不计入偏移均值，应结合覆盖表阅读。','',
 '|任务|扰动|区域|共同步数/目标船|固定8普通偏移|动态普通偏移|动态可信偏移|可信−动态普通及95%区间(m/s)|','|---|---|---|---:|---:|---:|---:|---:|']
for task,tn in (('relation','关系'),('single','单主体')):
 for view in ('natural','suffix','combined_l100_m600'):
  for group,gn in areas.items():
   a=lookup(drift,task=task,view=view,group=group,before='wide_plain',after='adaptive_plain_fallback',subset='common_three_hit')
   b=lookup(drift,task=task,view=view,group=group,before='adaptive_plain_fallback',after='adaptive_credible_fallback',subset='common_three_hit')
   if not b.get('mean_difference_mps'):
    L.append(f"|{tn}|{views[view]}|{gn}|{b['paired_steps']}/—|—|—|—|样本不足|");continue
   if abs(float(a['mean_after_mps'])-float(b['mean_before_mps']))>1e-5:raise ValueError('common support drift disagreement')
   delta=f"{float(b['mean_difference_mps']):+.3f} [{float(b['lower95']):+.3f}, {float(b['upper95']):+.3f}]"
   L.append(f"|{tn}|{views[view]}|{gn}|{b['paired_steps']}/{b['ships']}|{num(a,'mean_before_mps')}|{num(b,'mean_before_mps')}|{num(b,'mean_after_mps')}|{delta}|")
L+=['','正值表示后者偏移更大，负值表示更小；偏移变小仅说明对该扰动的响应更稳定，不等于更接近真实流场。','',
 '## 三、四步检测 AP 与配对差异','',
 'AP 为三个种子的测试集平均，百分数；差值单位为百分点，括号为按目标船 MMSI 聚类的95%区间。步骤③−②同时改变范围与回退；步骤④−③同时改变邻船资格、加权、所选范围及单主体训练包中背景状态，因此是整组机制差异。']
for task,tn in (('relation','群体关系'),('single','单主体')):
 for view,vn in views.items():
  L+=['',f'### {tn}：{vn}','', '|区域|骨干|① 固定3.5|② 固定8|③ 动态普通＋回退|④ 动态可信＋回退|③−②及95%区间|④−③及95%区间|','|---|---|---:|---:|---:|---:|---:|---:|']
  for group,gn in areas.items():
   for model,mn in names.items():
    z=[ap(task,model,view,group,arm) for arm in steps];delta1=pair(task,model,view,group,'wide_plain','adaptive_plain_fallback');delta2=pair(task,model,view,group,'adaptive_plain_fallback','adaptive_credible_fallback')
    L.append('|'+gn+'|'+mn+'|'+'|'.join(z)+'|'+delta1+'|'+delta2+'|')
L+=['','## 四、4–5 艘邻船时是否限制斜率','',
 '下表比较相同可信动态选圈下：4–5 艘邻船保留仿射斜率，与将其降为零阶均值。数值为“限制斜率 − 原方案”的 AP 差及95%区间，百分点。','',
 '|任务|区域|骨干|基础观测|全窗速度污染|后段速度污染|复合污染|','|---|---|---|---:|---:|---:|---:|']
for task,tn in (('relation','关系'),('single','单主体')):
 for group,gn in areas.items():
  for model,mn in names.items():
   z=[pair(task,model,view,group,'adaptive_credible_fallback','adaptive_credible_deg6') for view in views]
   L.append('|'+tn+'|'+gn+'|'+mn+'|'+'|'.join(z)+'|')
L+=['','## 五、其他消融和使用边界','',
 '结果文件 `summary_parts/*_pairs.csv` 另列：只切换半径、只加入回退、只增加可信资格、资格固定后增加权重、4点与6点放开斜率的对照。各对照都使用相同窗口和对应种子；未对多项比较做多重校正。','',
 '- 关系任务使用此前固定的同向邻船队列，最多覆盖原候选池内8海里；它并非重新搜索海图上所有船舶。','- 低密度分层与港口/水道重叠，不能作为独立远洋样本；距离与航向相近也不能证明同一水道。','- 自然/后段的直接污染命中情况与无场比例分别保存在字段覆盖和漂移明细，不把未命中窗口视为直接抗污染证据。','- 置信区间按目标MMSI聚类，条件于已训练的三个种子，不包含训练种子总体不确定性。','- 关系任务的标签是四格受控构造；单主体使用受控机动标签，不直接等同于真实异常事件。','- GAT-TCN 为当前任务中的目标及参考节点适配，不等同于完整邻船节点图。','- 同步保留既有版本的所有数据和结果；本版只重新拟合普通场的共同训练标准化，不把之前间接使用可信场统计的实验冒充严格普通场。','',
 f"独立核验：{audit['runs']} 组、{audit['checkpoints']} 份测试评分、{audit['recomputed_metrics']} 个指标重算通过；目标船自身对照在关系任务中 AP/AUROC 与50%的最大误差为 {audit['relation_target_only_max_abs_difference_from_0p5']:.2e}。"]
(reports/'专项实验中文汇总.md').write_text('\n'.join(L)+'\n',encoding='utf8');print('CHINESE_REPORT_COMPLETE',flush=True)
