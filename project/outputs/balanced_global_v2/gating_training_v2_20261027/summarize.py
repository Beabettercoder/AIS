from pathlib import Path
import json,csv,statistics,hashlib,math
p=Path(__file__).resolve().parent
policies=('clean','all'); models=('tcn','gru'); arms=('raw9','old_gate9','comp_gate9','old_combo10','comp_combo10')
results=[]
for policy in policies:
 for model in models:
  for arm in arms:
   for seed in range(3):
    q=p/'runs'/policy/model/('seed_'+str(seed))/arm/'result.json'
    if not q.exists(): print('MISSING',q);continue
    r=json.loads(q.read_text());results.append((policy,model,arm,seed,r))
if len(results)!=60: raise SystemExit('Expected 60, found '+str(len(results)))
lines=['# 主基准门控与训练暴露对照实验','', '本次保留历史 raw7 字段，在同一版 8 区域数据和同一训练策略内比较原增量门控、端点平均速度补偿门控，以及相同流场与历史训练包的两个组合。每组 3 个种子，主指标为固定第 25 轮验证集区域等权 AP；验证集用于开发，本结果并非最终测试结果。', '', '|训练策略|骨干|输入组|区域等权 AP（均值 ± 样本标准差）|种子 0/1/2|每轮更新次数|', '|---|---|---|---:|---|---:|']
table=[]
for policy in policies:
 for model in models:
  for arm in arms:
   rr=[r for p1,m,a,s,r in results if (p1,m,a)==(policy,model,arm)]
   ap=[r['final_validation']['macro']['AP'] for r in rr]; n={r['optimizer_steps_per_epoch'] for r in rr}; assert len(n)==1
   lines.append(f'|{policy}|{model.upper()}|{arm}|{statistics.mean(ap):.4f} ± {statistics.stdev(ap):.4f}|'+ ' / '.join(f'{x:.4f}' for x in ap)+f'|{next(iter(n))}|')
   table.append(dict(policy=policy,model=model,arm=arm,mean_AP=statistics.mean(ap),sd_AP=statistics.stdev(ap),seed0=ap[0],seed1=ap[1],seed2=ap[2],optimizer_steps=next(iter(n))))
lines+=['','固定策略与骨干下的配对变化，单位为百分点：','','|训练策略|骨干|对照|均值|三个种子|','|---|---|---|---:|---|']
for policy in policies:
 for model in models:
  for a,b in [('raw9','old_gate9'),('old_gate9','comp_gate9'),('old_combo10','comp_combo10'),('raw9','comp_combo10')]:
   d=[]
   for seed in range(3):
    lookup={(p1,m,arm,s):r['final_validation']['macro']['AP'] for p1,m,arm,s,r in results}
    d.append(100*(lookup[(policy,model,b,seed)]-lookup[(policy,model,a,seed)]))
   lines.append(f'|{policy}|{model.upper()}|{a} → {b}|{statistics.mean(d):+.3f}|'+ ' / '.join(f'{x:+.3f}' for x in d)+'|')
lines+=['','## 对照与解释','', '训练集共 7840 窗。clean 只使用其中不带受控观测污染的 5224 窗；all 使用全部 7840 窗。各策略内的五组使用相同窗口和按种子配对的打乱次序，并各自仅以该策略的训练窗口拟合特征归一化。clean 与 all 改变了训练样本数、受控污染暴露和每轮更新次数，策略之间的差异不能单独归因于污染暴露。','', 'raw9 由历史 raw7 加两个恒为零的通道组成；原门控和补偿门控同为原有七通道加可信度及运动学有效性。补偿在可信度有效、存在真实相邻正时间间隔的步骤上使用前后报告速度平均值乘时间间隔；其他步骤保留原门控。两组的非位置通道逐点相同，但此补偿假设报告速度在相关步骤可用于重建位移。','', '两组组合均采用既有留一法标量流场残差和历史训练包（增广分类损失、类型分布一致性和隐状态平滑等），且不启用图分支。组合与 raw 之间同时改变了输入和训练损失；原组合与补偿组合之间只改变门控位置输入。该实验使用既有标量流场，不是新稳健聚合器。','', '三个种子描述训练随机性，不能替代独立区域或外部测试；历史门控本来就在位移增量上加权后累加，并非对累计坐标直接乘权重。']
(p/'主基准门控训练对照报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (p/'summary.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
print('\n'.join(lines))
