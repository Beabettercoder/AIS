"""Research-facing report from completed independent audits."""
from pathlib import Path
import argparse,json,hashlib,csv

NAMES={'self_quality':'自身运动＋可信度旁路','mean_context':'普通均值邻域旁路',
       'affine_context':'无权仿射场旁路','reliable_context':'可信仿射场旁路'}
BN={'tcn':'TCN','gru':'GRU','transformer':'Transformer'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();r=args.stage
    p=read(r/'protocol.json');s=read(r/'audit/final/adapter_summary.json');a=read(r/'audit/final/independent_adapter_audit.json')
    n=read(r/'neighborhood/scan_final/neighborhood_scan_summary.json');q=read(r/'neighborhood/relation_smoke/relation_smoke_manifest.json')
    extra=read(r/'audit/independent_neighborhood_relation_audit.json')
    assert all(x['status']=='passed' for x in [s,a,n,q,extra])
    assert s['completed_runs']==36
    for b in p['backbones']:assert read(r/'execution'/f'{b}.json')['exit_code']==0
    index={(x['backbone'],x['arm']):x for x in s['all_configurations']}
    lines=['# 独立上下文旁路与群体关系任务：阶段实验报告','',
           '日期：2026-10-26。当前阶段使用训练集与验证集；没有产生新的测试集成绩。','',
           '本阶段完成36次冻结基线旁路训练、17套邻域条件扫描，以及30个来源窗口的四格群体关系任务构造验收。全部结果与独立核验保存在本目录。','',
           '## 1. 实验设置与等价性','',
           '- 保留上一阶段每个主干/种子的 `raw16` 最佳权重与原标准化；encoder与分类头全部冻结，并保持eval模式。',
           '- 主分支仍是原7个运动特征加9个零槽位，计算路径完全不变。',
           '- 旁路为16→32→64的点式MLP，共2,656个可训练参数；末层权重/偏置置零，alpha固定为1。',
           '- 先按共同有效步汇总旁路增量，再加入原模型的64维池化表示；不对目标运动或个体残差施乘法门控。',
           '- 四组仅在邻船残差与散度/旋度信息上不同，使用相同支持掩码、初始化和批次顺序。自身运动＋可信度对照也接收相同邻域可用性信息，不是完全无上下文信息的单船模型。',
           '- 8区全量7,840训练窗、1,680验证窗；3主干×4组×3种子，每次训练12轮。PSA、一致性训练、CDG关闭。',
           '- 原模型epoch0也进入验证AP择优；严格并列取最早轮次，全部12轮仍完整执行。',
           '- 初始化、旁路关闭、零支持窗口的预测分数与原基线逐样本一致，独立重放最大误差为0，冻结参数未改变。',
           '- 验证集781个窗口没有共同有效上下文；训练为3,784个。验证共同支持11,032步，训练49,298步。',
           '- 分数等价不代表重新选择F1阈值后的二分类决定也必然相同。','',
           '## 2. 全部旁路结果','',
           '数值为区域等权验证AP，三种子均值±样本标准差。','',
           '| 设置 | TCN | GRU | Transformer |','|---|---:|---:|---:|']
    baseline=['原冻结单船基线']
    for b in p['backbones']:
        v=index[b,'self_quality']['baseline']['AP'];baseline.append(f"{v['mean']:.6f} ± {v['sample_std']:.6f}")
    lines.append('| '+' | '.join(baseline)+' |')
    rows=[]
    for arm in p['arms']:
        cells=[NAMES[arm]]
        for b in p['backbones']:
            v=index[b,arm];apv=v['selected']['AP'];cells.append(f"{apv['mean']:.6f} ± {apv['sample_std']:.6f}")
            rows.append(dict(backbone=b,arm=arm,selected_AP=apv['mean'],selected_AP_std=apv['sample_std'],baseline_AP=v['baseline']['AP']['mean'],
                             final_epoch_AP=v['final_epoch']['AP']['mean'],selected_epoch0_count=v['selected_epoch0_count']))
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','可信场旁路相对于“自身运动＋可信度旁路”的净AP变化：','',
            '| 主干 | 均值变化（百分点） | 三个种子的配对变化（百分点） |','|---|---:|---|']
    deltas={}
    for b in p['backbones']:
        x=index[b,'reliable_context']['selected']['AP']['values'];y=index[b,'self_quality']['selected']['AP']['values']
        ds=[100*(a-bb) for a,bb in zip(x,y)];deltas[b]=ds
        lines.append(f"| {BN[b]} | {sum(ds)/len(ds):+.4f} | "+', '.join(f'{v:+.4f}' for v in ds)+' |')
    lines+=['',f"36个运行中，{s['epoch0_selections']}个选回原模型epoch0。选择结果不低于同一验证集的原AP，与这个选择规则直接相关，不是新数据上的性能保证。",'',
            '为保留完整训练表现，以下同时列出固定第12轮的AP均值：','',
            '| 设置 | TCN | GRU | Transformer |','|---|---:|---:|---:|']
    for arm in p['arms']:
        lines.append('| '+NAMES[arm]+' | '+' | '.join(f"{index[b,arm]['final_epoch']['AP']['mean']:.6f}" for b in p['backbones'])+' |')
    lines+=['','结论：冻结旁路实现了可验证的基线保留，但可信邻域相对同支持对照的额外收益接近零。该结果不支持把“直接拼接”认定为此前下降的唯一原因，也不把本轮与旧从头训练的差异单独归因于融合位置。','',
            '## 3. 邻域参数扫描','',
            f"共恢复并逐条核查 {n['neighbor_reports_recovered_and_checked']:,} 条邻船报告，在所有训练/验证有效目标步上检查17套固定条件。每个半径重新计算数据边界覆盖；没有依据模型AP选择半径。",'',
            '| 半径nm | 航向差° | 邻船最低SOG kn | 距离尺度nm | 训练场有效率 | 验证场有效率 |','|---:|---:|---:|---:|---:|---:|']
    for c in n['scan_configurations']:
        tr=n['splits']['train']['config_results'][c['name']];va=n['splits']['val']['config_results'][c['name']]
        lines.append(f"| {c['radius_nm']:g} | {c['heading_deg']} | {c['min_speed_kn']:g} | {c['sigma_nm']:g} | {100*tr['affine_valid_fraction']:.2f}% | {100*va['affine_valid_fraction']:.2f}% |")
    lines+=['','上述分母是有效目标时间步，属于全步合并比例；区域等权比例与每域计数另存JSON。仿射场仍要求至少4艘可信邻船以及空间扩展、条件数检查；达到3艘或Kish有效数3只作为诊断，不自动判为可靠场。',
            '收紧条件明显降低可用率。这个扫描检验覆盖与数值可拟合性，不能据此证明邻船在同一实际航道，也不能宣布某个半径提升了检测精度。',
            '诊断重放曾错误跳过一条NaN速度/航向源报文，已修正为保留全行、仅对有效行施加污染并重算门控。原数据没有修改；失败记录、回归验收和各域代码版本均保留。','',
            '## 4. 群体关系任务的四格构造验收','',
            '| 目标共同变速开关a | 固定船群变速开关b | 受控一致性标签 |','|---:|---:|---|',
            '| 0 | 0 | 0：一致 |','| 0 | 1 | 1：不一致 |','| 1 | 0 | 1：不一致 |','| 1 | 1 | 0：一致 |','',
            f"已生成 {q['independent_source_windows']} 个来源窗口的 {q['quartet_instances']} 个场景变体。四个兄弟样本共用原始来源及划分，不能计为四倍独立样本。",'',
            '- 在a固定时，目标船的全部输入轨迹在两种b下相同；在b固定时，固定船群的原始输入在两种a下相同。',
            '- 目标单独输入与船群单独输入的数值签名均对应平衡标签，排除仅凭某一侧是否发生变化即可识别标签的构造。',
            '- 船群在过去锚点固定，成员后段不满足验收时拒绝整个来源组，不事后删掉成员。',
            '- 同一来源四格使用相同的标签无关观测污染分配；原始位置—速度创新在污染前保持。',
            '- 标签定义为受控的相对一致性，不是自然异常人工金标，也没有替换旧个体异常标签。',
            '- 本阶段只验收构造，没有训练关系分类模型；不能把它当作本方法优于普通关系模型的证据。','',
            '各区过去锚点条件的候选数及实际小样例数：','',
            '| 区域 | 训练候选 | 验证候选 | 训练样例来源 | 验证样例来源 |','|---|---:|---:|---:|---:|']
    for region in p['regions']:
        rr=q['regions'][region]
        lines.append(f"| {region} | {n['relation_feasibility']['train'][region]} | {n['relation_feasibility']['val'][region]} | {rr['train']['accepted_sources']} | {rr['val']['accepted_sources']} |")
    lines+=['','这些候选数只是锚点条件可行性统计，后段观测和变换验收仍可能拒绝候选。Haugesund数量明显不足，当前样例不具备八域等规模正式关系基准的规模。','',
            '## 5. 当前可以确定的推进方向','',
            '1. 保留已验证的独立旁路和分数回退机制，暂不把它写成已取得稳定精度增益。',
            '2. 保留原邻域作为对照，不把紧邻阈值统一替换为默认值；后续要同时评估参照质量与覆盖损失。',
            '3. 用独立的四格关系任务检验何时确实需要上下文，并给普通均值/关系模型同样的信息。正式任务须解决稀疏区域可行样本不足，再做模型比较。',
            '4. 观测故障类型、通道选择性门控、PSA与CDG仍需独立实验，本轮没有同时加入。','',
            '## 6. 证据入口','',
            '- `protocol.json`：本阶段训练前冻结的设置。',
            '- `audit/final/adapter_summary.json`：全部选择结果、最后一轮结果及配对差值。',
            '- `audit/final/independent_adapter_audit.json`：逐模型等价性、权重与指标重放。',
            '- `neighborhood/scan_final/neighborhood_scan_summary.json`：17套条件和区域可行性。',
            '- `neighborhood/relation_smoke/relation_smoke_manifest.json`：四格样例及来源。',
            '- `audit/independent_neighborhood_relation_audit.json`：独立邻域/关系验收。',
            '- `figures/`：实际结果PNG/SVG，英文统一Times New Roman。',
            '- `runs/`：36组权重、0—12轮验证预测、阈值、标准化与来源校验。',
            '',
            '旧数据、旧模型、旧报告和system3保持原状。新包复现实验还需其记录的parent数据目录，或此前component_review_validation_20261026.zip中的对应输入。',
            '']
    with (r/'阶段实验报告.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    with (r/'adapter_configuration_summary.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    complete=dict(status='completed_and_independently_verified',protocol_sha256=sha(r/'protocol.json'),
                  adapter_runs=36,adapter_epochs=12,epoch0_selections=s['epoch0_selections'],
                  primary_scope='Validation-only frozen-baseline pointwise adapters; no new test evaluation',
                  trusted_context_minus_self_quality_AP_pp=deltas,
                  neighbor_reports_checked=n['neighbor_reports_recovered_and_checked'],neighbor_configurations=17,
                  relational_source_windows=q['independent_source_windows'],relational_variants=q['quartet_instances'],
                  audit_sha256={name:sha(r/name) for name in ['audit/final/independent_adapter_audit.json','audit/independent_neighborhood_relation_audit.json']})
    with (r/'stage_completion.json').open('x',encoding='utf-8') as f:json.dump(complete,f,indent=2)
    print(json.dumps(complete))
if __name__=='__main__':main()
