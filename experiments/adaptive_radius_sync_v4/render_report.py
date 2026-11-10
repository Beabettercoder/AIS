from pathlib import Path
import csv,json,statistics,hashlib
r=Path('/home/ps/AIS_adaptive_sync_20261110');out=[]
A=('tight_plain','wide_plain','adaptive_plain_fallback','adaptive_credible_fallback')
D={'target_only':'仅目标','tight_plain':'固定3.5海里普通场','wide_plain':'固定8海里普通场','adaptive_plain_fallback':'动态普通场＋回退','adaptive_credible_fallback':'动态可信场＋回退'}
V={'base':'基础观测','natural':'自然时间速度污染','suffix':'后段速度污染','combined_l100_m600':'后段速度与位置污染'}
G={'all':'全部','ports':'港口','restricted_channels':'受限水道','sparse_overlapping':'低密度（重叠分组）'}
N={'tcn':'TCN','gru':'GRU','transformer':'Transformer','graph':'GAT-TCN'}
def rows(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def line(x=''):out.append(x)
def pct(x):return f'{100*float(x):.2f}%'
def diff(q):return f"{100*float(q['delta_AP']):+.2f} [{100*float(q['lower95']):+.2f}, {100*float(q['upper95']):+.2f}]" if q and q.get('lower95') else '样本不足'
line('# 关系任务：扩圈邻船同步机动与六船仿射门槛')
line('')
line('## 结果概览')
line('1. 测试集基础观测下，固定小半径、固定大半径、动态普通场＋回退、动态可信场＋回退的有参考覆盖率依次为 64.73%、92.35%、97.22%、97.02%。受限水道依次为 22.16%、73.93%、88.34%、88.04%；覆盖率包含零阶回退。')
line('2. 自然时间速度污染下，动态可信场相对动态普通场的 AP：TCN +5.02 个百分点 [2.30, 7.63]，GRU +6.37 [3.24, 9.41]，Transformer +3.34 [−0.15, 6.77]，GAT-TCN +5.37 [2.61, 8.21]。其中 Transformer 区间跨零，其余三个区间大于零。')
line('3. 在后段速度及速度＋位置复合污染下，四骨干的可信场与动态普通场 AP 差异区间均跨零；基础观测下 Transformer 的可信场低 1.78 个百分点 [−3.51, −0.10]。')
line('4. 自然速度污染下，在三组都有参考且污染命中的共同时间步，背景速度平均变化由 0.673 降至 0.262 米/秒；后段速度污染为 0.381 降至 0.317 米/秒；复合污染为 0.307 与 0.325 米/秒，配对差值区间跨零。')
line('5. 与上一版扩圈实验相比，同一批基础测试样本的动态可信场 AP 提高 3.99～5.52 个百分点。同步群体机动与六船门槛同时改变，不能将这一差值单独归因于某一项。原四格干预是受控生成，该结果并不代表真实航海事件识别率。')
line('')
line('本实验保留原有 2,034 / 272 / 220 个训练、验证、测试来源及目标船身份划分。在每个四格样本中，让扩圈后可能入选的同向邻船与原指定船群同步执行平滑减速：300 秒内最多减速 30%，同时积分修正位置。位置和速度的原有相容残差由构造保持。')
line('')
line('污染对象按来源和船舶身份的确定性哈希选择，概率约 50%，与四格标签无关。自然时序和后段时序分别施加连续两条报文的 3–5 倍速度干扰；复合视图同时施加 600 米位置偏移。实际选中率和命中率见下表。')
line('')
line('四个骨干、三种子，每个场配置 25 轮；以第 25 轮为主要对照，验证集最优权重结果另存。所有一阶场至少需要 6 艘邻船且通过几何条件检查；允许回退的配置在船数不足时使用零阶加权均值。普通场只用距离加权；可信场还使用运动学筛选和加权。GAT-TCN 使用目标船与背景摘要两个节点的分类适配实现。')
line('')
line('原四格来源的入选仍依赖原有可信邻船条件，测试集此前使用过。同步减速是受控构造，不意味着真实船舶发生同样机动。低密度组与地理分组重叠，不等同开放海域；邻船背景可能跨集合复用。')
line('')
line('## 一、干预覆盖与污染命中')
line('|集合|来源数|原指定船次|新增背景船次|可受扰船次|选中受扰船次|选中比例|')
line('|---|---:|---:|---:|---:|---:|---:|')
for split in ('train','val','test'):
 a=[]
 for p in sorted((r/'relation').glob(f'{split}_*.proof.json')):
  meta=json.loads(p.with_name(p.name.replace('.proof.json','.json')).read_text())
  if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['proof_sha256']:raise ValueError('proof hash mismatch '+str(p))
  a+=json.loads(p.read_text())
 vals={k:sum(x[k] for x in a) for k in ('original_cohort','additional_ambient','eligible','selected')}
 line(f"|{split}|{len(a)}|{vals['original_cohort']}|{vals['additional_ambient']}|{vals['eligible']}|{vals['selected']}|{pct(vals['selected']/vals['eligible']) if vals['eligible'] else '—'}|")
line('')
line('此处的船次是各来源内的船舶数量之和，不能当作全局独立船舶数。污染报告比例为实现条件允许的船次，实际影响参考场的时间步数列在第三节。')
line('')
line('## 二、参考场覆盖率')
line('覆盖率为能提供仿射场或零阶参考的有效时间步占比；括号内为仿射场覆盖率。')
line('|视图|分组|窗口 / 目标船|固定3.5海里|固定8海里|动态普通场＋回退|动态可信场＋回退|')
line('|---|---|---:|---:|---:|---:|---:|')
cov=rows(r/'reports/test_field_coverage.csv')
for view in V:
 for group in G:
  xx=[next(z for z in cov if z['view']==view and z['group']==group and z['arm']==arm) for arm in A]
  vals=[f"{pct(z['coverage'])}（{pct(int(z['affine_steps'])/int(z['target_steps']))}）" if int(z['target_steps']) else '—' for z in xx]
  line('|'+ '|'.join([V[view],G[group],f"{xx[0]['windows']} / {xx[0]['ships']}",*vals])+'|')
line('')
line('## 三、污染前后背景速度变化')
line('同一时间步比较污染前后背景速度，仅使用三组方法均有参考、且污染影响了候选报文的共同时间步；数值越小表示对这类污染更稳定，单位为米/秒。这是配对变化，不是真实流场误差。')
line('|视图|分组|共同时间步 / 船|固定8海里普通场 → 动态普通场|动态普通场 → 动态可信场|可信−普通及95%区间|')
line('|---|---|---:|---|---|---|')
dr=rows(r/'reports/test_field_drift_pairs.csv')
for view in list(V)[1:]:
 for group in G:
  ps=[]
  for a,b in [('wide_plain','adaptive_plain_fallback'),('adaptive_plain_fallback','adaptive_credible_fallback')]:
   ps.append(next((x for x in dr if x['view']==view and x['group']==group and x['before']==a and x['after']==b and x['subset']=='common_three_hit'),None))
  x,y=ps
  if not y or not y.get('mean_difference_mps'):line(f'|{V[view]}|{G[group]}|—|—|—|样本不足|');continue
  xy=f"{float(x['mean_before_mps']):.3f} → {float(x['mean_after_mps']):.3f}" if x and x.get('mean_difference_mps') else '—'
  yz=f"{float(y['mean_before_mps']):.3f} → {float(y['mean_after_mps']):.3f}"
  ci=f"{float(y['mean_difference_mps']):+.3f} [{float(y['lower95']):+.3f}, {float(y['upper95']):+.3f}]"
  line('|'+ '|'.join([V[view],G[group],f"{y['paired_steps']} / {y['ships']}",xy,yz,ci])+'|')
line('')
line('## 四、关系检测 AP：四配置对照')
line('每项 AP 为三个种子的平均值；差值单位为百分点，括号为目标船聚类的配对 95% Bootstrap 区间。')
line('|视图|分组|骨干|固定小|固定大|动态普通|动态可信|固定大−固定小|动态普通−固定大|动态可信−动态普通|')
line('|---|---|---|---:|---:|---:|---:|---|---|---|')
for view in V:
 for group in G:
  for name in N:
   rr=rows(r/'summary_parts'/f'relation_{name}_metrics.csv');pp=rows(r/'summary_parts'/f'relation_{name}_pairs.csv');vals=[];ds=[]
   for arm in A:
    x=[float(z['AP']) for z in rr if z['checkpoint']=='last' and z['view']==view and z['group']==group and z['arm']==arm];vals.append(pct(statistics.mean(x)) if len(x)==3 else '—')
   for before,after in zip(A,A[1:]):ds.append(diff(next((z for z in pp if z['view']==view and z['group']==group and z['before']==before and z['after']==after),None)))
   line('|'+ '|'.join([V[view],G[group],N[name],*vals,*ds])+'|')
line('')
line('固定小、固定大是平行基线；动态普通与固定大还同时改变回退设置。实验额外保留“动态无回退”和“仅可信筛选”配置，逐种子 AP、AUROC、阈值误报和相应配对区间见数据目录。重复比较未作多重校正。')
line('')
for split in ('train','val','test'):
 import train_relation_v2 as rel
 d,_=rel.data(r,split);mask=d['mask']
 for view in list(V)[1:]:
  hit=d[view+'__hit']&mask;line(f"- {split} / {V[view]}：受扰船的报文进入参考候选的时间步 {int(hit.sum())} / {int(mask.sum())}（{pct(hit.sum()/mask.sum())}）。")
line('')
line('使用邻船数量均值、首尾数量变化与有场比例等简单统计的探索性对照，完整可信场四种视图的 AP 接近 50%（最大约 50.4%），未发现这些简单支持度统计单独区分四格标签；这不排除更复杂的捷径。')
line('结果经过样本身份、三种子配对、训练规范、模型权重和预测指标独立检查。四格受控任务中的正负标签由目标变化与背景变化的组合生成，不等同于经真实事件标注的异常。')
(r/'关系任务同步干预_中文报告.md').write_text('\n'.join(out)+'\n',encoding='utf8');print('REPORT_COMPLETE',len(out),flush=True)
