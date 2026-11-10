"""Build a short Chinese reading guide from completed, audited prediction files."""
import argparse,json
from pathlib import Path
import numpy as np

MODEL={'tcn':'TCN','gru':'GRU','transformer':'Transformer','itransformer':'iTransformer'}
VIEW={'base':'基础真实观测','target_jump':'目标末点位置跳变','target_drift':'目标后段位置漂移',
      'target_speed':'目标后段速度尖峰','neighbor_stress':'邻船后段速度与位置污染'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));args=p.parse_args();root=args.root
    s=json.loads((root/'reports/summary.json').read_text());d=json.loads((root/'data/data_manifest.json').read_text())
    assert s['status']=='complete' and s['completed_training_runs']==48
    rows=s['summary'];pairs=s['paired']
    get=lambda m,a,v,sub='all':next(r for r in rows if (r['model'],r['arm'],r['view'],r['subset'])==(m,a,v,sub))
    fmt=lambda r:f"{r['ADE_m_mean']:.2f} / {r['FDE_m_mean']:.2f}"
    ref=get('tcn','P0','base');nat=get('tcn','P0','base','observed_maneuver')
    lines=['# 四骨干航迹预测小试：结果阅读版','',
        f"已完成48组训练（四骨干×四配置×三种子）。验证集{ref['windows']}窗、{ref['target_mmsi']}艘目标船，八区域各150窗；训练集八区域各700窗，共5600窗。",'',
        '这里比较的是未来真实 AIS 观测位置的预测误差。ADE 是八个预测点的平均位置误差，FDE 是最后一个预测点的位置误差，单位为米，越低越好。不是异常识别分数。','',
        '## 本轮结论','',
        '最明确的收益来自运动学门控对历史目标观测污染的抑制。四骨干等权平均ADE：末点位置跳变时，原始输入为1,098.38米、仅加门控为439.83米；速度尖峰时从3,762.76米降至1,655.10米。位置漂移的改善很小。', '',
        '基础真实观测下，原始输入平均ADE为315.20米，完整可信场配置为335.18米，恒速外推为308.62米。本轮没有支持“完整框架普遍提高常规预测精度”的结论。位置跳变下门控恒速外推本身已达438.89米，说明这一大幅改善主要由前端门控贡献。', '',
        '在邻船污染下，可信场相对普通场使TCN、Transformer、iTransformer的ADE分别降低5.34、5.31、18.70米，三项目标船聚类区间均低于零；GRU差值为+0.91米，区间跨零。这个结果说明可信加权能减轻背景污染，但没有抵消当前门控在基础航迹上的误差代价。', '',
        '本阶段止于已约定的四骨干预测小试，未扩展八骨干。后续方法取舍应同时查看基础观测误差、污染下误差和恒速基线；既有受控机动及群体关系检测成果保持原样。','',
        '## 先看四种配置','','|名称|输入与训练|','|---|---|',
        '|原始输入|原始历史运动，常规预测训练|',
        '|运动学门控|历史位置与报告速度相容性门控，常规预测训练|',
        '|门控＋普通背景场|运动学门控＋距离加权留一法背景场|',
        '|门控＋可信背景场|运动学门控＋距离×观测可信度加权留一法背景场|','',
        '四组都用同一骨干结构和SmoothL1损失，预测头学习恒速外推之上的位置修正；本小试未加入分类实验的增强训练包。TCN/GRU/Transformer各25轮，iTransformer 75轮，按固定最终轮次报告。','',
        '## 各观测条件的完整对照','',
        '每格为三种子平均 **ADE / FDE（米）**。恒速基线不需要训练；只读取当时可见的最后位置、航速与航向。']
    for view,title in VIEW.items():
        lines.extend(['',f'### {title}','',
            '|骨干|原始输入|运动学门控|门控＋普通背景场|门控＋可信背景场|',
            '|---|---:|---:|---:|---:|'])
        for m,name in MODEL.items():lines.append('|'+name+'|'+'|'.join(fmt(get(m,a,view)) for a in ['P0','P1','P2','P3'])+'|')
        lines.append(f"\n恒速外推：原始输入 **{fmt(get('CV_raw','CV_raw',view))}**；运动学门控输入 **{fmt(get('CV_gate','CV_gate',view))}**。")
    lines.extend(['','## 只改变可信权重，背景场有多少净收益','',
        '下表为“可信背景场－普通背景场”的 ADE 差值及目标船聚类95%区间（米）。负值代表改善；跨零代表本次数据不能确认稳定差异。',
        '','|骨干|基础观测|目标位置跳变|目标位置漂移|目标速度尖峰|邻船污染|',
        '|---|---:|---:|---:|---:|---:|'])
    for m,name in MODEL.items():
        cells=[]
        for view in VIEW:
            r=next(r for r in pairs if r['model']==m and r['view']==view and r['contrast']=='P3-P2'
                and r['subset']=='all' and r['aggregation']=='pooled' and r['metric']=='ADE_m')
            cells.append(f"{r['delta']:+.2f} [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]")
        lines.append('|'+name+'|'+'|'.join(cells)+'|')
    lines.extend(['','## 自然机动子集','',
        f"原始观测中有{nat['windows']}个窗口满足移动时航向变化至少30°或航速范围至少2节的分层规则。这只是机动分层，不是事故标签。以下均为基础观测下的ADE / FDE（米）。",'',
        '|骨干|原始输入|运动学门控|门控＋普通背景场|门控＋可信背景场|',
        '|---|---:|---:|---:|---:|'])
    for m,name in MODEL.items():lines.append('|'+name+'|'+'|'.join(fmt(get(m,a,'base','observed_maneuver')) for a in ['P0','P1','P2','P3'])+'|')
    lines.extend(['','## 数据与核验','',
        f"12个历史真实报告预测8个后续真实报告。最后预测点距历史末点的中位跨度为{s['time_horizon']['final_horizon_s_median']:.0f}秒，范围{s['time_horizon']['final_horizon_s_min']:.0f}–{s['time_horizon']['final_horizon_s_max']:.0f}秒；模型接收查询时间，但不接收未来位置、速度或邻船状态。",'',
        '目标船身份按既有划分隔离；训练历史拟合标准化和门控速度上限。未来参考未经人工机动注入，所有受控污染只施加在历史输入。所有预测在固定海区米制坐标系中比较，污染后不使用干净末点重新对齐。', '',
        f"基础视图训练／验证背景船重用数：{d['base_background_shared_mmsi']}。训练背景包含验证／测试目标数：{d['train_background_with_heldout_targets']}；验证背景包含测试目标数：{d['val_background_with_test_targets']}。本轮没有构造或读取测试窗口。",'',
        '10项独立实现检查已通过；中断后的轮次恢复与连续训练参数逐值一致。完整训练后的单船输入模型在邻船污染前后预测必须逐点相同，程序自动检查。逐项区间以已训练三个种子为条件，按目标船重采样2000次。','',
        '补充核验从原始parquet物理行号独立重算全部6,800窗的54,400个未来参考位置，与缓存逐值一致，最大坐标差为0米；训练1,378艘与验证296艘目标船零交集，12个“骨干×种子”组内的四配置初始权重、参数数量与标准化一致。详见“原始报文与完整训练核验.json”。','',
        '## 文件定位','',
        '- `summary.csv`：各骨干、配置、视图及自然机动子集的均值、种子标准差。',
        '- `paired_bootstrap.csv`：门控、背景场、可信权重的逐项配对差值与区间。',
        '- `per_region.csv` / `per_horizon.csv`：区域与预测跨度分层。',
        '- `independent_postflight.json`：实现及训练后不变量核验。',
        '- `parallel_runs/骨干/骨干/seed_编号/配置/`：逐轮history.csv、模型checkpoint.pt、各视图逐窗预测。',
        '- `data/window_provenance.json`：每个窗口对应原始文件、行号、时间和船舶身份。',''])
    (root/'reports/本轮结果阅读版.md').write_text('\n'.join(lines),encoding='utf-8')
    compact=[]
    for v in VIEW:
        item={'view':v}
        for a in ['P0','P1','P2','P3']:
            item[a+'_ADE']=float(np.mean([get(m,a,v)['ADE_m_mean'] for m in MODEL]))
            item[a+'_FDE']=float(np.mean([get(m,a,v)['FDE_m_mean'] for m in MODEL]))
        compact.append(item)
    print(json.dumps({'descriptive_equal_backbone_means':compact,'observed_maneuver_windows':nat['windows'],'val_target_mmsi':ref['target_mmsi']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
