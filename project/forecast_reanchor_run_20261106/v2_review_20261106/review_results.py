"""Read completed v2 predictions and write numeric review only. No training or PDF."""
import sys,json,csv
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS/forecast_reanchor_run_20261106')
sys.path.insert(0,str(ROOT/'v2'));import report as R
WORK=ROOT/'work_v2';OUT=ROOT/'v2_review_20261106';OUT.mkdir(exist_ok=True)
MODELS=['tcn','gru','transformer','itransformer'];NAMES=['TCN','GRU','Transformer','iTransformer']
EXPS=['training_constant','data_small','main','capacity_h128','capacity_h256']
views=['base','target_jump','target_drift','target_speed','neighbor_stress']
labels=['基础观测','目标末点位置跳变','目标后段位置漂移','目标后段速度尖峰','邻船速度与位置污染']
summ={name:json.loads((WORK/name/'reports/test/summary.json').read_text()) for name in EXPS}
assert all(s['status']=='complete' and s['completed_models']==48 for s in summ.values())
def get(e,m,a,v='base'):
    return next(x for x in summ[e]['summary'] if (x['model'],x['arm'],x['view'],x['subset'])==(m,a,v,'all'))
def avg(e,a,v='base',metric='ADE_m_mean'):return float(np.mean([get(e,m,a,v)[metric] for m in MODELS]))
summary=[]
for e in EXPS:
    row=dict(experiment=e)
    for a in ['P0','P1','P2','P3']:
        for metric in ['ADE_m_mean','FDE_m_mean']:row[a+'_'+metric]=avg(e,a,metric=metric)
    summary.append(row)
R.csv_save(OUT/'五套实验基础误差.csv',summary)
records,index=R.discover(WORK/'main','test')
cv_pairs=[]
for model in MODELS:
    for view in views:
        for arm,baseline in [('P0','CV_raw'),('P3','CV_gate')]:
            aa=[index[(model,arm,s,view)] for s in [0,1,2]]
            bb=[index[(baseline,baseline,-1,view)]]*3
            for x in R.S.paired_bootstrap(aa,bb,'all',2000,20261106):cv_pairs.append(dict(model=model,view=view,contrast=arm+'-'+baseline,**x))
R.csv_save(OUT/'主配置相对恒速_配对区间.csv',cv_pairs)
R.save(OUT/'主配置相对恒速_配对区间.json',cv_pairs)
lines=['# v2运行结果核查与数值摘要','',
       '五套实验各48组，训练及预测测试输出齐全，共240组。全量独立审计通过：192组新训练协议确实选择基础验证ADE最优权重，48组旧机制对照使用末轮；3,600份预测文件重算误差与输出一致。以下全部为测试集结果，不是验证集最低分。主配置为隐藏64、扩充训练池、余弦调度＋基础验证ADE最优权重。','',
       '## 一、训练、数据、容量分别带来了什么','',
       '下表是四骨干等权、各骨干三种子平均的描述性误差，单位为米。完整可信配置包括重锚门控和可信背景场。',
       '','|设置|训练窗|隐藏维度|原始输入ADE/FDE|完整可信配置ADE/FDE|',
       '|---|---:|---:|---:|---:|']
names=['常量学习率＋末轮对照','余弦学习率＋最佳验证权重','改进训练＋扩充数据（主配置）','扩充数据＋容量128','扩充数据＋容量256']
for e,n in zip(EXPS,names):
    s=json.loads((WORK/e/'experiment.json').read_text());d=json.loads((Path(s['data_dir'])/'train_manifest.json').read_text())
    lines.append(f"|{n}|{d['windows']:,}|{s['hidden']}|{avg(e,'P0'):.2f} / {avg(e,'P0',metric='FDE_m_mean'):.2f}|{avg(e,'P3'):.2f} / {avg(e,'P3',metric='FDE_m_mean'):.2f}|")
lines += ['',
    '基础观测中，完整可信配置的平均ADE从311.17米，经训练机制升级降到304.67米，扩充数据后进一步到302.38米。扩大到128和256后分别为303.19和304.34米，因此容量扩展没有带来一致的基础精度收益。训练机制同时改变调度与选型，不能把差值单独归因于余弦调度。', '',
    '改善不覆盖所有观测条件：完整可信配置在速度尖峰下的ADE由旧机制1838.85米变为新机制小数据1952.35米，扩数据后为1927.16米。按基础验证误差选型获得的基础精度优势，没有同步转化为未见污染下的更优表现。','',
    '## 二、主配置的全部观测条件','',
    '每格为四骨干等权的三种子平均ADE / FDE（米）；恒速基线无需训练。','',
    '|条件|原始输入|重锚门控|门控＋普通场|门控＋可信场|原始恒速|重锚恒速|','|---|---:|---:|---:|---:|---:|---:|']
for v,label in zip(views,labels):
    cells=[f"{avg('main',a,v):.2f} / {avg('main',a,v,'FDE_m_mean'):.2f}" for a in ['P0','P1','P2','P3']]
    for a in ['CV_raw','CV_gate']:
        x=get('main',a,a,v);cells.append(f"{x['ADE_m_mean']:.2f} / {x['FDE_m_mean']:.2f}")
    lines.append('|'+label+'|'+'|'.join(cells)+'|')
lines += ['',
    '门控仍是目标观测污染下的主要收益来源：跳变ADE由1106.99降到396.83米；速度尖峰由3867.80降到1906.76米。背景场在此基础上的额外提升并不稳定。渐进漂移仍没有明确净改善。', '',
    '## 三、与恒速基线及同骨干原始输入比较','',
    '基础观测原始恒速ADE/FDE为303.44/650.56米。', '',
    '|骨干|原始神经输入ADE/FDE|完整可信配置ADE/FDE|原始神经输入−原始恒速ADE及95%区间|',
    '|---|---:|---:|---:|']
for m,name in zip(MODELS,NAMES):
    a=get('main',m,'P0');b=get('main',m,'P3')
    ci=next(x for x in cv_pairs if x['model']==m and x['view']=='base' and x['contrast']=='P0-CV_raw' and x['aggregation']=='pooled' and x['metric']=='ADE_m')
    lines.append(f"|{name}|{a['ADE_m_mean']:.2f} / {a['FDE_m_mean']:.2f}|{b['ADE_m_mean']:.2f} / {b['FDE_m_mean']:.2f}|{ci['delta']:+.2f} [{ci['ci95_low']:+.2f}, {ci['ci95_high']:+.2f}]|")
lines += ['',
    '四种原始神经模型的平均ADE均低于恒速基线，说明此前“神经网络一定不如恒速”的概括不成立。但同骨干比较，主配置的完整可信场仍高于原始神经输入；不能把训练和数据带来的改善当作背景场自身贡献。', '',
    '## 四、可信权重在邻船污染下的净作用','',
    '下表为可信场−普通场的测试ADE差及目标船聚类95%区间；负值表示改善。','',
    '|骨干|隐藏64主配置|隐藏128|隐藏256|','|---|---:|---:|---:|']
for m,name in zip(MODELS,NAMES):
    cells=[]
    for e in ['main','capacity_h128','capacity_h256']:
        x=next(x for x in summ[e]['paired'] if x['model']==m and x['view']=='neighbor_stress' and x['subset']=='all' and x['contrast']=='P3-P2' and x['metric']=='ADE_m' and x['aggregation']=='pooled')
        cells.append(f"{x['delta']:+.2f} [{x['ci95_low']:+.2f}, {x['ci95_high']:+.2f}]")
    lines.append('|'+name+'|'+'|'.join(cells)+'|')
lines += ['',
    '主配置的四条ADE区间均跨零；较大容量中的部分骨干有负向区间。需要分别报告这些条件，不能概括成所有骨干都获得稳定背景场增益。区间以已训练种子为条件，未作多重比较校正。', '',
    '## 五、核查范围与文件','',
    '基础配额5,600窗，扩充配额实际14,897窗，两者独立训练船均为1,378艘；验证1,200窗/296船，测试1,183窗/303船。上海未达到配额上限，未重复采样凑数。数据量对照保持相同验证与测试窗口、同一标准化及相同轮数，但增加样本也增加了优化更新次数，因此它是固定轮数下扩数据的综合效果。容量对照只改变隐藏维度。', '',
    '测试使用冻结后的所选模型。训练机制对照特意保留常量学习率＋最后一轮，其余四套使用基础验证ADE最优轮次；best_val与last_epoch列是验证记录，不能拿它们代替测试误差。该目标船留出测试来源已用于研究，不称全新外部盲测。', '',
    '本次只读取结果并核查，没有重训、修改权重或生成PDF。完整性与选型哈希核验见本目录独立审计文件，逐因素配对数据见work_v2/comparisons。','']
(OUT/'结果核查摘要.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote',OUT/'结果核查摘要.md')
print(json.dumps([x for x in cv_pairs if x['view']=='base' and x['aggregation']=='pooled' and x['metric']=='ADE_m'],indent=2))
