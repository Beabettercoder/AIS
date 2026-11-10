"""Combine the two independently verified task lines, preserving their scopes."""
from pathlib import Path
import argparse,json,hashlib,csv

ARMS={'raw_incremental_compat':'历史网络＋诊断增量7通道','raw_legacy_compat':'历史单船基线（原字段）',
      'kg_compat':'+ 原门控处理','kglf_compat':'+ 标量流场残差','kglf_psa_compat':'+ 历史训练包',
      'full35_compat':'+ 原图分支：主表完整执行版','raw_legacy_masked':'单船基线：有效长度控制',
      'full35_masked':'完整执行版：有效长度控制'}
METHODS={'target_only':'仅目标','neighbor_only':'仅邻船','relative_mean':'普通均值参照',
         'relative_median':'中位数参照','relative_reliable':'成员级可信加权参照'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);r=ap.parse_args().stage
    protocol=read(r/'protocol.json');hist=read(r/'audit/historical_final/historical_reproduction_summary.json')
    audits=['audit/inputs/independent_historical_input_audit.json','audit/historical_final/independent_historical_training_audit.json',
            'audit/independent_relation_construction_audit.json','audit/relation_full_base/independent_relation_score_audit.json',
            'audit/relation_speed_stress/independent_relation_score_audit.json','audit/independent_relation_bootstrap_audit.json',
            'audit/independent_report_weighted_followup_audit.json']
    for f in audits:assert read(r/f)['status']=='passed',f
    assert hist['completed_runs']==48
    for b in protocol['backbones']:assert read(r/'execution'/f'train_{b}.json')['exit_code']==0
    index={(x['backbone'],x['arm']):x for x in hist['all_configurations']}
    base=read(r/'relation/base_statistics_v2/statistics_summary.json');stress=read(r/'relation/speed_stress_statistics_v2/statistics_summary.json')
    follow=read(r/'relation/report_weighted_statistics_v1/statistics_summary.json');ci=read(r/'relation/bootstrap_v1/paired_mmsi_bootstrap.json')
    inputs=read(r/'data/input_manifest.json')
    lines=['# 历史方法迁移复现与群体关系统计：两条任务实验报告','',
           '日期：2026-10-26。两条任务均使用训练/验证数据，没有生成新的测试集成绩。','',
           '## 1. 本轮完成内容','',
           '- 历史主表执行版：TCN、GRU各8种设置×3种子，共48次完整训练，固定25轮主结果及验证最优轮次均保留。',
           '- 新基准输入桥接：恢复历史物理通道语义，重建真实因果邻船历史，独立核对全部9,520个训练/验证窗口。',
           '- 群体关系任务：385个来源窗口、1,540个四格变体，完成普通均值、中位数、成员可信加权和单侧负对照。',
           '- 同源额外邻船速度污染、2,000次目标MMSI聚类bootstrap，以及单独标明的报文级加权探索性跟进。','',
           '## 2. 复现的历史版本与保留的修复','',
           '历史Table 1由38_table1_final10.py汇总，完整方法行来自35_graph_ours_final.py。实际完整TCN有71,125个注册参数，GRU为65,237个；不是约45K的无图版本。后来60_graph_fix.py的物理双距离CDG是另一个版本，本轮没有混入。',
           '原网络为64宽三层kernel5 TCN或单层双向GRU，配二分类与五类头。完整训练包含四维物理条件控制器、统计扰动、类型JS一致性、增强分支BCE和隐藏序列二阶平滑。控制器与主网络一起最小化损失，旧代码没有内层对抗最大化；epsilon实际范围为0.05—0.25。',
           '本轮严格对齐这些网络层和训练方程，仍使用修复后的因果运动学权重、留一法/每船最新观测与同步污染输入。因此名称是“历史网络与训练方程迁移到修复新基准”，不是把旧有目标混入邻域、重复报文或测试集选阈值错误重新带回。','',
           '### 输入语义与训练协议','',
           '- 历史raw7：[相对首点x米、y米、vx m/s、vy m/s、SOG kn、sinCOG、cosCOG]。',
           '- 之前诊断raw7：[位置增量/半径、速度/尺度、时间间隔/窗口、sinCOG、cosCOG]。维数相同不代表物理量相同。',
           '- 逐组件比较恢复原门控处理和标量场残差；权重与场值来自新基准已核验观测。',
           '- 继承历史clean-only梯度训练：7,840训练窗中5,224个未加受控污染窗参与梯度；标准化仍使用全部训练有效步。所有本轮设置使用同一批来源。',
           '- 固定epoch25是主结果；验证集最佳轮次另列，不替代主结果。阈值只用验证集，不复用旧test分位数规则。',
           '- compat保留原20槽位处理；masked单独修正真实长度编码、末有效残差、ep平均、增强和隐藏平滑范围。图的5步历史仍按原结构左补零。','',
           '## 3. 历史模型在同一新基准的结果','',
           '区域等权验证AP，三种子均值±样本标准差。','',
           '| 设置 | TCN固定25轮 | GRU固定25轮 | TCN验证最优 | GRU验证最优 |','|---|---:|---:|---:|---:|']
    flat=[]
    for arm in protocol['arms']:
        cells=[ARMS[arm]]
        for tag,b in [('final','tcn'),('final','gru'),('best','tcn'),('best','gru')]:
            v=index[b,arm][tag]['AP'];cells.append(f"{v['mean']:.6f} ± {v['sample_std']:.6f}")
        lines.append('| '+' | '.join(cells)+' |')
        for b in protocol['backbones']:
            flat.append(dict(backbone=b,arm=arm,**{f'{tag}_{m}_{v}':index[b,arm][tag][m][v] for tag in ['final','best'] for m in ['AP','AUROC','F1'] for v in ['mean','sample_std']}))
    lines+=['','### 固定25轮的关键配对变化','',
            '| 对照 | TCN平均变化pp | GRU平均变化pp |','|---|---:|---:|']
    for fr,to in protocol['predeclared_contrasts']:
        ds=[100*(index[b,to]['final']['AP']['mean']-index[b,fr]['final']['AP']['mean']) for b in protocol['backbones']]
        lines.append(f"| {ARMS[fr]} → {ARMS[to]} | {ds[0]:+.3f} | {ds[1]:+.3f} |")
    lines+=['','恢复历史输入语义在同一历史网络中提高了TCN约2.99pp、GRU约4.69pp，说明此前输入定义差异确有实质影响。TCN在KG+LF上加入历史训练包提高约0.79pp，三个种子均为正；但完整执行版总体仍未稳定超过其单船基线。',
            '有效长度控制后，完整模型也未超过相应单船基线，不能将差异仅归结为padding。此前27K诊断TCN与本次64宽、双头、clean-only历史网络的分数不能作为单一结构因素的比较。','',
            '## 4. 数据变化能说明什么','',
            '本轮能够定量比较同一新数据下的输入表示、模块包及padding处理。新旧数据绝对分数的差，还同时包含地域构成、样本资格、平滑生成、类别比例、MMSI隔离、上下文修复和评估规则变化，不能把旧表与新表之差当作一个独立的“数据贡献百分点”。',
            '具体核实的数据管线差异包括：',
            '- 旧07构建对末步邻船污染只改NB_XY/NB_V/合成NB_W，NB_SEQ继续读取未污染历史；旧35完整前向使用nb_seq与nb_xy，不读取nb_v。旧LF也未随这项末步污染重新拟合。新版则在观测层扰动后同步进入邻船历史与场计算。',
            '- 旧邻船置信度可由已知注入强度直接构造；当前权重由实际位置—速度相容性计算。',
            '- 旧源数据构建存在目标混入与同MMSI重复报告，本轮保持目标留一、每船最新观测和身份隔离修复。',
            '- 旧4500训练窗中1116个不足20步；新7840训练窗中3963个不足20步，padding行为的适用范围发生变化。',
            '这些是确定的版本差异，但尚未逐项做独立数据干预归因；没有将它们直接换算成某个AP降幅。','',
            '## 5. 群体关系任务：先看简单统计','',
            '四格定义为目标变速开关a与固定船群变速开关b，标签是a XOR b。只看任一侧时，其输入签名对应平衡标签；模型或统计量必须使用两侧关系。标签表示受控干预因子的一致性，不是自然异常人工金标；原始未来自然变化仍保留，因此同因子不保证实际速度变化完全相等。',
            '456个锚点候选中385个通过整组验收，71个拒绝。训练273来源/169目标MMSI，验证112来源/58MMSI，四兄弟保持同一来源划分；不能把1540场景当作1540独立来源。',
            '固定统计只读取最终可见目标/邻船观测，按可从目标长度重算的过去锚点和前后速度比构造相对变化；不使用生成a/b、真实未扰动速度或注入onset作为预测特征。',
            '主“可信加权”先计算每邻船速度变化，再用前后平均运动学权重的几何平均给成员加权；它不是历史35神经模型或空间仿射场。','',
            '| 分数 | 原位置污染视图 pooled AP | 邻船速度压力 pooled AP | 原视图海域等权AP | 压力海域等权AP |','|---|---:|---:|---:|---:|']
    for key,name in METHODS.items():
        v=base['metrics'][key]['splits']['val'];w=stress['metrics'][key]['splits']['val']
        lines.append(f"| {name} | {v['pooled']['AP']:.6f} | {w['pooled']['AP']:.6f} | {v['macro']['AP']:.6f} | {w['macro']['AP']:.6f} |")
    lines+=['','额外速度污染按来源哈希选择ceil(0.15K)个邻船，anchor之后连续两报告乘3—5倍并截到102.1kn，四兄弟共用相同操作。由于每组只有4—8船，实际选择成员比例约22.8%，不是精确15%。验证112来源均有数值变化，39来源的固定后段统计实际采到了污染，全部样本都保留。',
            '阈值只由基础视图训练集选取，压力视图沿用，不重新校准。',
            '2,000次目标MMSI聚类配对bootstrap（pooled AP差）：',
            '- 基础视图，成员可信加权减普通均值：约+0.12pp，95%区间[-0.02,+0.45]pp。',
            '- 邻船速度污染，成员可信加权减普通均值：+2.77pp，95%区间[+1.18,+5.25]pp。',
            '- 同一污染下，成员可信加权减中位数：-8.18pp，95%区间[-12.83,-4.48]pp。',
            '置信区间条件于这58个验证目标MMSI；不覆盖未知海域分布，也没有完全建模不同目标共用背景船群的相关性。海域数量不均，Haugesund仅1个验证MMSI，因此不报告容易误读的macro跨域置信区间。','',
            '## 6. 报文级加权的探索性跟进','',
            '诊断发现成员级权重可能稀释单条坏报告的影响，故保留旧主结果后，另冻结一个探索性对照：仅在每邻船前后速度均值计算前使用报文权重，群体仍普通均值；目标、样本、时间、门槛均不改。零权重段回退原段均值，不删除成员或来源。该方案由本轮诊断启发，不能冒充原先预注册主比较。','',
            '| 视图 | 报文级加权 pooled AP | 海域等权AP |','|---|---:|---:|']
    for view,label in [('base','原位置污染视图'),('speed_stress','邻船速度压力')]:
        m=follow['views'][view]['splits']['val'];lines.append(f"| {label} | {m['pooled']['AP']:.6f} | {m['macro']['AP']:.6f} |")
    lines+=['','污染下相对普通均值提高约2.03pp，但仍未超过中位数，也没有证实优于原成员加权。加权位置不能单独解释全部差异。',
            '另有直接诊断：低速改速报告更容易在固定2.5m/s门槛下保留高权重；部分低报告权重在汇总成成员权重后又被平均稀释。完整计数保存在mechanism_diagnostics_v1，属于机制分析，不用于修改本轮分数或选取好看样本。','',
            '## 7. 本轮结论','',
            '1. 历史字段和网络版本确实不能与前几轮诊断载体混同；恢复字段语义带来可测变化，但历史完整执行版在修复后的新基准仍未取得普遍优势。',
            '2. 群体关系任务具有可观察的相对信息，普通相对统计明显高于单侧机会水平。',
            '3. 观测可信度在邻船速度污染下相对均值有帮助，但当前简单统计中稳健中位数更强；不能把该比较泛化为整套神经方法的胜负。',
            '4. 后续需要将“观测是否可信”与“行为参照是否合适”分开验证，并保留中位数这种强简单基线。当前报告没有宣布新的通用增益。','',
            '## 8. 文件与核验入口','',
            '- `historical/历史完整方法取证.md`、`audit/历史实现独立审查.md`：历史源代码和版本链。',
            '- `data/input_manifest.json`、`audit/inputs/independent_historical_input_audit.json`：新输入/邻船历史逐值回查。',
            '- `audit/historical_final/historical_reproduction_summary.json`：48组fixed-final/best全部指标和配对差。',
            '- `runs/`：每个种子的25轮预测、固定末轮/最优权重、类型预测、标准化与激活日志。',
            '- `relation/群体关系简单统计实验报告.md`：来源、普通/可信/中位数全部结果与区域计数。',
            '- `relation/report_weighted_statistics_v1/报文级加权探索性对照报告.md`：追加探索对照。',
            '- `audit/independent_relation_bootstrap_audit.json`、`audit/independent_report_weighted_followup_audit.json`：独立统计重算。',
            '- `figures/`及`relation/figures/`：真实结果图，英文Times New Roman。','',
            '全部48组神经训练、输入桥接、385来源四格、两种固定视图、MMSI聚类bootstrap与探索性报文对照均已独立核验。神经checkpoint重放最大差4.17×10^-7，在预设5×10^-6数值容差内；不能写成逐位误差零。旧代码、旧输入和system3未修改。','']
    with (r/'两条任务实验报告.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    with (r/'historical_configuration_summary.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    status={'status':'completed_and_independently_verified','protocol_sha256':sha(r/'protocol.json'),'historical_runs':48,
            'primary_epoch':25,'source_gradient_windows':inputs['totals']['train']['clean_gradient_windows'],
            'relation_source_windows':385,'relation_scenarios':1540,'validation_relation_mmsi':58,
            'exploratory_report_weighted_followup':True,'new_test_evaluation':False,
            'audit_sha256':{f:sha(r/f) for f in audits}}
    with (r/'stage_completion.json').open('x') as f:json.dump(status,f,indent=2)
    print(json.dumps(status))
if __name__=='__main__':main()
