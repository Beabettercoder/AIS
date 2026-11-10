"""Assemble a review from independently audited results; never fit a model."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics

NAMES = {
    'raw16': '基础运动输入', 'raw_meta': '基础运动＋可信度信息',
    'gated_meta': '目标门控＋可信度信息', 'mean_meta': '普通均值邻域＋可信度信息',
    'affine_meta': '无权仿射场＋可信度信息', 'reliable_raw': '可信流场＋保留原始运动',
    'reliable_gate_target': '可信流场＋仅目标门控',
    'reliable_gate_residual': '可信流场＋仅残差门控',
    'legacy_full': '可信流场＋双门控（旧配置）',
    'reliable_raw_fallback': '可信流场＋原始运动＋均值回退',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    root = args.stage
    protocol = read(root / 'protocol.json')
    summary = read(root / 'audit/final/component_summary.json')
    audit = read(root / 'audit/final/independent_component_metric_audit.json')
    assert summary['status'] == audit['status'] == 'passed'
    assert summary['completed_runs'] == summary['expected_runs'] == 90
    assert summary['protocol_sha256'] == sha(root / 'protocol.json')
    for b in protocol['backbones']:
        assert read(root / 'execution' / f'{b}.json')['exit_code'] == 0
        assert read(root / 'runs' / b / 'worker_summary.json')['source_data_unchanged']
    table = {(x['backbone'], x['feature_group']): x for x in summary['all_configurations']}
    contrast = {(x['backbone'], x['from'], x['to']): x for x in summary['paired_contrasts']}
    feature_diagnostics = read(root / 'data/feature_diagnostics.json')
    field_diagnostics = read(root / 'data/field_availability_diagnosis.json')
    coverage = read(root / 'data/coverage_fallback_addendum.json')
    lines = [
        '可信度感知上下文：组件诊断实验报告（2026-10-26）', '',
        '本阶段已完成：3种主干×10种输入×3个随机种子，共90次完整训练及独立复核。',
        '本报告数值均为验证集开发结果。本轮没有读取测试特征、标签或预测，也没有生成新的测试结果。', '',
        '一、固定实验设置',
        '8区域，训练7840窗、验证1680窗；沿用已冻结标签和MMSI划分。各域训练980窗、验证210窗。',
        'TCN、GRU、Transformer沿用上一阶段实际实现；全部输入16通道、隐藏宽度32，种子0/1/2，每次24轮。',
        'AdamW，学习率0.002，weight decay 0.01，batch128，dropout0.1。PSA、一致性损失和CDG关闭。',
        '同一主干与种子下初始化、每轮样本顺序一致。各输入方案使用各自训练有效步的标准化参数。',
        '模型轮次按验证集区域等权AP最高值确定；严格并列取最早轮次。F1阈值来自验证集。',
        '16维同形输入保证名义参数数量相同；零槽位不提供有效信息，不能称严格有效容量匹配。', '',
        '二、全部结果（区域等权验证AP，均值±种子样本标准差）',
        '输入方案 | TCN | GRU | Transformer',
    ]
    metrics_rows = []
    for f in protocol['feature_names']:
        cells = [NAMES[f]]
        for b in protocol['backbones']:
            row = table[b, f]
            assert row['AP']['n'] == 3
            cells.append(f"{row['AP']['mean']:.4f} ± {row['AP']['sample_std']:.4f}")
            metrics_rows.append(dict(backbone=b, feature_group=f, feature_name=NAMES[f],
                                     **{f'{m}_{s}': row[m][s] for m in ['AP', 'AUROC', 'F1'] for s in ['mean', 'sample_std']}))
        lines.append(' | '.join(cells))
    lines += ['', '三、预先规定的逐种子配对比较',
              '以下正值表示后者AP较高，单位为百分点；逐种子值按seed0、seed1、seed2排列。']
    for p in protocol['predeclared_contrasts']:
        lines.append(f"{NAMES[p['from']]} → {NAMES[p['to']]}")
        for b in protocol['backbones']:
            d = contrast[b, p['from'], p['to']]['AP']
            values = ', '.join(f'{100*v:+.3f}' for v in d['values'])
            lines.append(f"  {b}: {100*d['mean']:+.3f} ± {100*d['sample_std']:.3f}；[{values}]")
    lines += ['', '四、门控2×2交互',
              '计算：双门控－仅目标门控－仅残差门控＋两者关闭。该比较包含各方案自身训练标准化与验证选轮。']
    for item in summary['gating_2x2_interaction']:
        d = item['AP_interaction']
        lines.append(f"{item['backbone']}: {100*d['mean']:+.3f} ± {100*d['sample_std']:.3f} pp")
    lines += ['', '五、输入和流场可用性诊断']
    for split, name in [('train', '训练'), ('val', '验证')]:
        d = feature_diagnostics['splits'][split]['all']
        lines.append(f"{name}：{d['valid_target_steps']}个有效目标步，低门控比例（可评估步为分母）{100*d['low_gate_fraction_of_assessable_steps']:.3f}%。")
        lines.append(f"低门控步中，{100*d['low_gate_reported_velocity_unchanged_by_controlled_corruption_fraction']:.2f}%的报告速度/方向在受控污染前后未变，但统一门控仍压低这些通道；这不等于确认原始报告完全准确。")
        lines.append(f"可信仿射场有效比例{100*d['trusted_affine_valid_fraction']:.3f}%；均值回退适用{d['trusted_constant_fallback_steps']}步，增加{100*d['trusted_constant_fallback_fraction_of_valid_steps']:.3f}个百分点。")
        fields = field_diagnostics['results'][split]['all_regions']
        lines.append('互斥无效原因/有效场计数：' + json.dumps(fields['category_counts'], ensure_ascii=False))
    lines += [
        '边界允许标志遵循原覆盖策略；覆盖未知时并不保证完整搜索半径。均值回退中训练0步、验证1步处于未知覆盖，详见coverage_fallback_addendum.json。',
        '邻域不足包含低速目标不建立航向邻域，不能全部解释为实际海面船舶稀疏。', '',
        '六、独立核验',
        '逐数组核查16份train/val数据和10种派生输入，源输入保持不变；独立重算训练标准化参数。',
        '逐次重算90个模型的24轮验证AP、最佳轮次、阈值、各区域AP/AUROC/F1，重新加载权重重放验证预测。',
        '核对初始化和批次顺序哈希、逐样本来源ID及标签、保存文件哈希。全部90个运行均保留。',
        '按实际污染、噪声类型/作用对象分层；单类子组的排序指标留空，并保留样本数量与参与区域数。',
        '额外邻域支持分层属于解释性分析，不参与主比较和模型选择。', '',
        '七、结论口径',
        '本阶段用于判断组件设计，数值不能直接写成最终测试或未见区域泛化成绩。',
        'affine_meta与reliable_raw比较的是邻船筛选、可信度权重和距离权重的组合处理，不能归因于单项权重。',
        '结果中的标准差来自3个训练种子，不代表独立数据集的置信区间。', '',
        '文件索引',
        'protocol.json：训练前固定的全部设置与比较。',
        'audit/final/component_summary.json：全部均值、标准差、逐种子配对差与交互。',
        'audit/final/independent_component_metric_audit.json：逐模型独立核查与分层指标。',
        'runs/：所有权重、标准化参数、每轮预测、最佳轮次逐样本预测与日志。',
        'figures/：Times New Roman英文图表，数值来自实际保存结果。',
    ]
    with (root / '组件诊断实验报告.txt').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    with (root / 'validation_configuration_summary.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    status = dict(status='completed_and_independently_verified', stage='validation_only_component_diagnostics',
                  protocol_sha256=sha(root / 'protocol.json'), runs=90,
                  train_windows=7840, validation_windows=1680, seeds=[0, 1, 2], epochs=24,
                  final_audit_sha256=sha(root / 'audit/final/independent_component_metric_audit.json'),
                  aggregate_sha256=sha(root / 'audit/final/component_summary.json'),
                  all_configurations=summary['all_configurations'], paired_contrasts=summary['paired_contrasts'],
                  input_diagnostics=feature_diagnostics['splits'],
                  source_preservation=read(root / 'data/feature_manifest.json')['source_preservation'])
    with (root / 'stage_completion.json').open('x', encoding='utf-8') as stream:
        json.dump(status, stream, indent=2, ensure_ascii=False)
    print(json.dumps(dict(status=status['status'], runs=90), ensure_ascii=False))


if __name__ == '__main__':
    main()
