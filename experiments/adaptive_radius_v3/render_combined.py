from pathlib import Path
import csv,json,statistics
S=Path('/home/ps/AIS_adaptive_radius_relation_v2p1_20261109');R=Path('/home/ps/AIS_adaptive_radius_relation_v3_20261109')
A=('tight_plain','wide_plain','adaptive_plain_fallback','adaptive_credible_fallback')
D={'target_only':'仅目标','tight_plain':'固定3.5海里普通场','wide_plain':'固定8海里普通场','adaptive_plain_fallback':'动态普通场与回退','adaptive_credible_fallback':'动态可信场与回退','adaptive_plain_deg6':'动态普通场（6船才拟合斜率）','adaptive_credible_deg6':'动态可信场（6船才拟合斜率）'}
V={'base':'基础观测','natural':'自然时间分布速度污染','suffix':'后段速度污染','combined_l100_m600':'后段速度和位置复合污染'}
G={'all':'全部','ports':'港口','restricted_channels':'受限水道','sparse_overlapping':'低密度（与区域重叠）'}
N={'tcn':'TCN','gru':'GRU','transformer':'Transformer','graph':'GAT-TCN'}
def rows(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pct(x):return f'{100*float(x):.2f}%'
def delta(r):return f"{100*float(r['delta_AP']):+.2f} [{100*float(r['lower95']):+.2f}, {100*float(r['upper95']):+.2f}]" if r and r.get('lower95') else '样本不足'
def data(task,root,model):return rows(root/'summary_parts'/f'{task}_{model}_metrics.csv'),rows(root/'summary_parts'/f'{task}_{model}_pairs.csv')
out=[]
def line(s=''):out.append(s)
line('# 动态背景场半径：普通场与可信场配对实验')
line('')
line('本实验比较固定与动态邻域，以及运动学可信度和稀疏回退的作用。普通场在邻船筛选、聚合与训练统计量中均不使用运动学可信度；可信场筛除低可信报文并进行可信度加权。四组顺次比较包含不同因素，尤其固定小半径与固定大半径是两个平行基线，后文附额外消融用于分离各因素。')
line('')
line('## 主要结果')
line('1. 单主体受限水道的有参考覆盖率：固定 3.5 海里为 7.61%，固定 8 海里为 28.07%，动态普通场及回退为 70.14%，动态可信场及回退为 69.26%。这里的提升主要包含零阶回退，不能全部称为仿射场覆盖率。')
line('2. 关系任务全部样本的覆盖率依次为 75.60%、95.39%、97.22%、97.02%。低密度关系子集只有 8 个四格窗口、2 艘目标船，不能代表开放海域。')
line('3. 在关系任务的自然速度污染视图上，四骨干可信场相对动态普通场的 AP 差依次为 TCN +1.13、GRU +1.33、Transformer +0.84、GAT-TCN −0.18 个百分点，四项配对区间均跨零。后段复合污染下，Transformer 为 +2.00 个百分点，区间 [+0.14, +3.85]；其余骨干的对应区间跨零。')
line('4. 在关系任务自然速度污染下，三组共同有场且污染触及的 4,908 个时间步，动态普通场的平均污染前后速度变化为 0.864 m/s，可信场为 0.393 m/s，差值 −0.471 m/s，95% 配对区间 [−0.620, −0.357]；后段速度污染区间跨零，复合污染中可信场变化反而更大。')
line('5. 扩展邻船后，关系任务测试中新增周围船占候选船的时间步中位比例约为 89.3%；原四格干预只作用于指定船群。关系任务基础 AP 为 54.07%–61.67%，与此前只使用筛选后船群的实验不是相同邻域设定，这可能削弱群体干预信号，尚不能单独确定因果原因。')
line('6. 在可信场条件下，将一阶仿射拟合的门槛从 4 艘提高到 6 艘，GAT-TCN 四个视图的 AP 增加 2.89–4.63 个百分点，配对区间均大于零；其他骨干的结果并非一致。这是受当前样本和多次比较影响的探索性消融。')
line('')
line('范围：单主体采用原始报文重建的 1,680 个测试窗口；关系任务保留 220 个原有来源、880 个四格测试窗，并重新搜索满足时间、距离和方向条件的邻船。四个骨干各训练三个种子，主结果取固定第 25 轮。历史测试集此前使用过。')
line('')
line('关系任务原始来源当初经过可信邻船资格筛选；本次只移除了邻域重建与普通场计算中的这一筛选。新增周围船并不一定参与原四格干预，因此四格标签描述指定船群与目标之间的受控关系。低密度按测试窗口基础观测中的邻船数量划分，与港口和受限水道分组重叠，不等同于独立开放海域。目标船按身份隔离，部分背景船和报文跨集合复用。')
line('')
line('## 一、覆盖率与所选半径')
line('覆盖率为有可用背景参考的有效时间步占比，包含零阶回退；仿射覆盖率单独列出。')
line('')
line('|任务|视图|分组|配置|窗口/船|有参考|其中仿射|平均半径（海里）|3.5 / 5 / 8 海里占比|')
line('|---|---|---|---|---:|---:|---:|---:|---|')
for task,root in [('单主体',S),('关系',R)]:
 rr=[r for r in rows(root/'reports/test_field_coverage.csv') if r['task']==('single' if task=='单主体' else 'relation')]
 for view in V:
  for group in G:
   for arm in A:
    v=next((r for r in rr if r['view']==view and r['group']==group and r['arm']==arm),None)
    if not v:continue
    nr=int(v['target_steps']);aff=int(v['affine_steps'])/nr if nr else 0
    rad=f"{float(v['mean_radius_nm']):.2f}" if v['mean_radius_nm'] else '无'
    frac=' / '.join(pct(v[k]) if v[k] else '—' for k in ('r3p5_fraction','r5_fraction','r8_fraction'))
    line(f"|{task}|{V[view]}|{G[group]}|{D[arm]}|{v['windows']} / {v['ships']}|{pct(v['coverage'])}|{pct(aff)}|{rad}|{frac}|")
line('')
line('## 二、污染前后背景速度变化')
line('下表只比较同一窗口、同一时间步、三组方法在污染前后均有参考且污染实际触及邻船的共同样本。速度变化越小表示参考更稳定，不直接等于真实流场误差。单位为米/秒。')
line('')
line('|任务|污染视图|分组|比较|共同时间步/目标船|比较前→后|差值及95%配对区间|')
line('|---|---|---|---|---:|---:|---|')
for task,root,key in [('单主体',S,'single'),('关系',R,'relation')]:
 rr=rows(root/'reports/test_field_drift_pairs.csv')
 for view in list(V)[1:]:
  for group in G:
   for before,after in [('wide_plain','adaptive_plain_fallback'),('adaptive_plain_fallback','adaptive_credible_fallback')]:
    r=next((x for x in rr if x['task']==key and x['view']==view and x['group']==group and x['before']==before and x['after']==after and x['subset']=='common_three_hit'),None)
    if r and r.get('mean_difference_mps'):
     line(f"|{task}|{V[view]}|{G[group]}|{D[before]} → {D[after]}|{r['paired_steps']} / {r['ships']}|{float(r['mean_before_mps']):.3f} → {float(r['mean_after_mps']):.3f}|{float(r['mean_difference_mps']):+.3f} [{float(r['lower95']):+.3f}, {float(r['upper95']):+.3f}]|")
    else:line(f'|{task}|{V[view]}|{G[group]}|{D[before]} → {D[after]}|—|—|样本不足|')
line('')
line('## 三、检测 AP 与配对差异')
line('每格为三个种子测试 AP 的平均值；差值单位为百分点。置信区间按目标船编号聚类计算，反映现有三次训练的配对样本不确定性。')
line('')
line('|任务|视图|分组|骨干|固定小|固定大|动态普通＋回退|动态可信＋回退|固定大−固定小|动态普通−固定大|可信−动态普通|')
line('|---|---|---|---|---:|---:|---:|---:|---|---|---|')
for task,root,key in [('单主体',S,'single'),('关系',R,'relation')]:
 for name in N:
  rr,pp=data(key,root,name)
  for view in V:
   for group in G:
    vals=[]
    for arm in A:
     a=[float(r['AP']) for r in rr if r['checkpoint']=='last' and r['view']==view and r['group']==group and r['arm']==arm]
     vals.append(pct(statistics.mean(a)) if len(a)==3 else '—')
    ds=[]
    for before,after in zip(A,A[1:]):
     pr=next((r for r in pp if r['view']==view and r['group']==group and r['before']==before and r['after']==after),None);ds.append(delta(pr))
    line('|'+ '|'.join([task,V[view],G[group],N[name],*vals,*ds])+'|')
line('')
line('## 四、辅助消融')
line('动态普通无回退与有回退的比较用于衡量回退机制；可信资格筛选与可信加权分开比较；四船与六船仿射门槛的比较检验小样本斜率控制。差值单位为百分点。')
line('')
line('|任务|污染视图|骨干|增加回退|增加可信筛选|筛选后增加可信加权|可信场由4船改6船|')
line('|---|---|---|---|---|---|---|')
for task,root,key in [('单主体',S,'single'),('关系',R,'relation')]:
 for name in N:
  _,pp=data(key,root,name)
  for view in V:
   ds=[]
   for before,after in [('adaptive_plain_nofallback','adaptive_plain_fallback'),('adaptive_plain_fallback','adaptive_qualified_fallback'),('adaptive_qualified_fallback','adaptive_credible_fallback'),('adaptive_credible_fallback','adaptive_credible_deg6')]:
    pr=next((r for r in pp if r['view']==view and r['group']=='all' and r['before']==before and r['after']==after),None);ds.append(delta(pr))
   line('|'+ '|'.join([task,V[view],N[name],*ds])+'|')
line('')
line('注：自然时间分布污染、后段速度污染和复合污染均为受控构造；单主体与关系任务的注入对象及覆盖范围不同。参考船是否属于同一航道并未由海图标注直接验证。置信区间未作多重比较校正，不用于保证全部骨干均有收益。')
line('')
line('结果来源：单主体严格普通场实验 v2p1；关系任务原始报文重新搜索邻船实验 v3；保留每个模型的测试预测、模型权重、哈希清单、参数及逐组统计。')
outpath=R/'实验结果中文汇总_真实无可信筛选.md';outpath.write_text('\n'.join(out)+'\n',encoding='utf-8');print('REPORT',outpath,'lines',len(out))
