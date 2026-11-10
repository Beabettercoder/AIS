from pathlib import Path
import argparse,json

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;f=r/'report_weighted_statistics_v1'
    new=read(f/'statistics_summary.json');base=read(r/'base_statistics_v2/statistics_summary.json');stress=read(r/'speed_stress_statistics_v2/statistics_summary.json');boot=read(f/'bootstrap/paired_bootstrap.json');hit=read(f/'hit_diagnostics.json')
    lines=['# 报文级可信聚合：探索性隔离对照','',
           '这是在本轮门控机制诊断后增加的探索性跟进，不属于此前预先固定的主比较。原v2所有分数、协议、数据和图表保持原样。','',
           '## 唯一改变','',
           '目标相对变化d_i、固定anchor、前后段、target-grid对齐及重复latest次数、可用成员、样本、e_kin公式、2.5/8门控参数全部不变。只把每个邻船前后段的报告速度均值改为对应报告w的加权均值，然后计算原来的ratio变化；邻船之间仍使用普通mean。',
           '若某段sum(w)=0，回退到这一段原普通均值，不删除该成员或来源。未叠加成员级权重、median或任何新阈值。',
           '新评分阈值只从base训练分数校准，再固定用于速度污染视图。旧方法沿用各自旧阈值。','',
           '## 全部验证比较：112来源、58目标MMSI、448场景','',
           '| 方法 | base pooled AP | speed-stress pooled AP | base macro AP | speed-stress macro AP |','|---|---:|---:|---:|---:|']
    for name,label in [('relative_mean','普通mean'),('relative_reliable','原成员级加权'),('relative_median','median')]:
        b=base['metrics'][name]['splits']['val'];s=stress['metrics'][name]['splits']['val'];lines.append(f"| {label} | {b['pooled']['AP']:.6f} | {s['pooled']['AP']:.6f} | {b['macro']['AP']:.6f} | {s['macro']['AP']:.6f} |")
    b=new['views']['base']['splits']['val'];s=new['views']['speed_stress']['splits']['val'];lines.append(f"| 新报文级加权（探索性） | {b['pooled']['AP']:.6f} | {s['pooled']['AP']:.6f} | {b['macro']['AP']:.6f} | {s['macro']['AP']:.6f} |")
    lines+=['','## 同一2000次目标MMSI抽样的成对比较','',
            '抽样序列与原bootstrap的hash完全一致。下表为新报文加权减去对照的pooled AP差，单位为百分点；区间是当前样本条件下的百分位区间。','',
            '| 视图 | 对照 | AP差 | 95%区间 |','|---|---|---:|---|']
    for c in boot['paired_contrasts']:
        d=c['AP'];lines.append(f"| {c['view']} | {c['comparator']} | {100*d['point']:+.3f} | [{100*d['lower_2_5']:+.3f}, {100*d['upper_97_5']:+.3f}] |")
    lines+=['','## 回退与实际命中','']
    for view in ['base','speed_stress']:
        for split in ['train','val']:
            d=new['views'][view]['splits'][split];ff=d['fallbacks_unique_neighbor_states'];lines.append(f"- {view}/{split}：{d['source_windows']}个来源、{d['target_mmsi']}个目标MMSI；前段零权重回退{ff['prefix_fallbacks_unique_neighbor_states']}个邻船状态，后段{ff['suffix_fallbacks_unique_neighbor_states']}个，涉及{d['sources_with_fallback']}来源。四格a副本已折叠。")
    for split,d in hit['splits'].items():
        lines.append(f"- 速度污染{split}：固定后段实际采到改变报告的来源{d['raw_suffix_hit_sources']}个、邻船状态{d['raw_suffix_hit_member_states']}个；这些状态中普通mean给改变报告的平均份额为{d['mean_ordinary_bad_report_share']:.4f}，报文加权后为{d['mean_report_weighted_bad_report_share']:.4f}，但有{d['member_states_bad_share_increased']}个状态的份额反而增大。该份额只描述直接聚合系数，不等于完整因果影响。")
    lines+=['','## 结论','',
            '新报文级加权在速度污染下优于普通mean，但没有证实优于原成员级加权，也没有接近median参照的表现。仅把可信度前移到报告聚合，仍不足以解释或消除此前的差距。',
            '门控的绝对尺度、首个与连续报错的相容残差差异、后继报告权重、以及原始船群自然变化都仍需要独立检查。本轮不据此调整门槛、选择更有利的样本或继续增加评分超参数。',
            '所有新旧结果均保留；探索性结果不混充预注册主结论。','']
    (f/'报文级加权探索性对照报告.md').write_text('\n'.join(lines),encoding='utf-8');print(str(f/'报文级加权探索性对照报告.md'))
if __name__=='__main__':main()
