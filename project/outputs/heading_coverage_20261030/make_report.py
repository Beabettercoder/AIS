from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/heading_coverage_20261030')
c=json.loads((ROOT/'coverage_full.json').read_text()); p=json.loads((ROOT/'kinematic_consistency.json').read_text())
with np.load(ROOT/'heading_train.npz',allow_pickle=False) as z: a={x:z[x] for x in z.files}
with np.load(ROOT/'heading_val.npz',allow_pickle=False) as z: b={x:z[x] for x in z.files}
intersection=set(a['target_mmsi'].astype(str))&set(b['target_mmsi'].astype(str));assert not intersection
def fmt(x):
 return '—' if not x['requests'] else f"{100*x['valid_rate']:.1f}% ({x['valid_heading']}/{x['requests']})"
def low(g):
 total=sum(a['low_requests'] for a in g);valid=sum(a['low_valid'] for a in g);return '—' if not total else f'{100*valid/total:.1f}% ({valid}/{total})'
regions=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
lines=['# 八区域 Heading 覆盖与四格生成一致性核查','',
 '## 范围与映射','',
 '覆盖全部已选放宽规则训练/开发验证来源（2,034 / 272），仅按 Base 原始报文中 MMSI 与 UTC 秒回查 canonical parquet，字段缺失或无效（Heading 不在 [0,360)）均保存为 NaN 与显式 mask。没有读取测试标签，不将原位置、干净速度或标签写入 Heading 缓存。缓存采用明示地区顺序 Houston、NY、LA、Puget、Great Belt、Haugesund、Kattegat、Shanghai，区内 window_id 排序，与其他结果文件数组顺序不保证一致，必须使用唯一 window_id 映射并逐条校验邻船 MMSI 与时间。',
 '','以下为报文请求级数值，含跨窗口和目标/邻船的重复；低速按源报文 SOG<2 kn 标记，仅用于覆盖描述，不是模型输入或故障标签。没有声称Heading独立传感器，也不能把其缺失解释为污染。','',
 '| 地区 | 训练目标 | 训练邻船 | 验证目标 | 验证邻船 | 全部低速 Heading 有效 |',
 '|---|---:|---:|---:|---:|---:|']
for r in regions:
 g=c[r]['groups'];lines.append(f"| {r} | {fmt(g['train_target'])} | {fmt(g['train_neighbor'])} | {fmt(g['val_target'])} | {fmt(g['val_neighbor'])} | {low([g[x] for x in ('train_target','train_neighbor','val_target','val_neighbor')])} |")
lines += ['', '每区请求的 MMSI+秒均成功匹配，筛选船舶的 canonical 同键重复行数均为0。LA验证只有1个来源、Haugesund验证0来源、Shanghai验证2个来源；不能由这些比例推断海域总体覆盖。训练与验证目标 MMSI 交集为0，但邻船原始报文有跨分区复用。',
 '', '## 四格与位置速度一致性', '',
 f"2,306 来源在 Base、自然速度及后段速度三视图的目标/邻船时间和 COG 四格均一致；附加 Heading 与有效mask的目标侧、邻船侧单独签名分别 {p['target_and_neighbor_only_with_heading_balanced_per_source_view']['target']} 组各自对应一正一负，未发现仅靠单侧 Heading 或其缺失模式直接编码四格标签。",
 f"对 {p['quartet_trajectories_checked']:,} 条目标或邻船四格轨迹，在加入位置噪声之前，受控减速前后位置差分与报告速度相容残差的最大绝对改变为 {p['max_abs_pre_noise_velocity_innovation_change_mps']:.3g} m/s。这验证积分生成保留原残差；不能证明源Heading与受控生成轨迹的物理关系或独立传感器来源已获真值验证。",
 f"实际源报文中，SOG<2 kn 且Heading/COG有效的 {p['source_heading_minus_cog_abs_deg_by_source_sog']['low']['n']:,} 条报文请求，其 Heading−COG 绝对角差中位为 {p['source_heading_minus_cog_abs_deg_by_source_sog']['low']['median']:.1f}°；SOG≥5 kn 的 {p['source_heading_minus_cog_abs_deg_by_source_sog']['moving']['n']:,} 条为 {p['source_heading_minus_cog_abs_deg_by_source_sog']['moving']['median']:.1f}°。低速时航迹方向与船首方向不一定重合，差角大不能直接视为污染。重复请求非独立。",
 '', '训练与验证场中时间、位置和 SOG 被受控修改时，Heading只作为按相同身份和时间从源报文恢复的共享字段，未由受控标签生成。未来一旦研究转向注入或其他受控操作，必须重新核查或重生成 Heading 的物理一致性；本核查不覆盖那类注入。',
 '', '## 可复核资产','', '- `heading_train.npz`、`heading_val.npz`：目标 [N,20] 和逐来源邻船报文展平序列，含各来源 offsets、船舶、时刻、Heading值和有效mask；不含标签。','- `coverage_full.json`：各区各分区目标/邻船/低速覆盖、匹配、重复键及背景报文重用。','- `kinematic_consistency.json`：逐源四格单侧标签平衡和注入前后运动学残差检查。','- `build_heading.py`、`kinematic_consistency.py`：可复算构造与检验。']
(ROOT/'Heading全量覆盖与生成一致性.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('report',ROOT/'Heading全量覆盖与生成一致性.md')
