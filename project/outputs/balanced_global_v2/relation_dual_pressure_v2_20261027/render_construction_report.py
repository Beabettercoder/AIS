"""Write pre-score dataset construction report from frozen manifests."""
from pathlib import Path
from collections import Counter
import json
OUT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
m=json.loads((OUT/'dataset_manifest_combined.json').read_text());main=json.loads((OUT/'dataset_manifest.json').read_text());supp=json.loads((OUT/'supplement/dataset_manifest.json').read_text());effect=json.loads((OUT/'effective_pressure_audit.json').read_text())['counts'];regions=['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai'];splits=['train','val','test']
text=['# 八区域关系四格扩充与双压力构造报告','','日期：2026-10-27。本报告只记录来源、资格、污染视图和身份重用；模型训练/评估结果另报。数据为受控 XOR 关系构造，成员后段可用性筛选使其属于离线实验。','','## 1. 时间顺序与来源','','首次 v1 的无模型分数候选清点先发生；v2 的三种训练暴露、候选规则与源码锁定后对相同规则重放，候选池 SHA-256 完全一致。根据 v2 的来源资格数量，LA/Haugesund 在 test 无合格来源，随后冻结这两区释放每船24窗上限的补充版本，加入 cap 剔除的窗口；该修订发生在查看本轮模型分数之前，但已经看过来源可行性与此前 pilot/test 结果，不应称完整预注册盲测。原 v2、补充版和合并池均保留。','','所有来源使用已登记原始标准化候选文件，目标船身份沿既有 global MMSI split；除原 release 目标外，仅将未作为 release 目标且未在主基准 train/val 邻域清单出现的 test 组身份列为候选。受限于既有数据源和旧test背景曝光，不能据此声称全新独立身份。','', '## 2. 通过与拒绝：放宽规则','','半径≤5 nm、航向差≤30°、速度差≤4.5 kn、同向可信邻船至少3艘；额外需要完整前后报文和至少两次对齐。每格为 `候选 / 接受来源 / 接受目标MMSI / 邻船不足 / 目标前段不合格 / 邻船未来不足 / 其他拒绝`；候选按原身份/区最多24窗，LA/Hau 的超限有效窗在补充版全部扫描。','','| 区域 | Train | Val | Test |','|---|---|---|---|']
for r in regions:
 cells=[]
 for s in splits:
  a=main['counts'].get(r,{}).get('relaxed',{}).get(s,{});b=supp['counts'].get(r,{}).get('relaxed',{}).get(s,{});c=Counter(a);c.update(b);rows=[x for x in m['samples'] if x['version']=='relaxed' and x['region']==r and x['split']==s];acc=len(rows);short=c.get('rejected_eligible_neighbor_count_below_version_minimum',0);pre=c.get('rejected_target_prefix_or_duration_ineligible',0);future=c.get('rejected_cohort_insufficient_observed_future',0);other=c.get('candidates',0)-acc-short-pre-future
  cells.append(f"{c.get('candidates',0)} / {acc} / {len({x['mmsi'] for x in rows})} / {short} / {pre} / {future} / {other}")
 text.append('| '+r+' | '+' | '.join(cells)+' |')
text += ['', '汇总：放宽版 train/val/test 为 2,034/272/220 来源、584/86/83 个目标 MMSI；test 每区来源 Houston 86、NY 74、LA 1、Puget 24、Great Belt 26、Haugesund 1、Kattegat 6、Shanghai 2。LA/Hau 各一个来源，不代表这两个区域具有足够的统计支持。原规则与放宽规则采用各自合法邻船成员，不能把方法差异归因于唯一门槛。原规则 train/val/test 为 555/71/37 来源。','', '## 3. 范围与身份重用','','选择/干预之前，目标 MMSI 在三划分之间互斥；每个划分的邻船排除其他划分候选目标以及旧 release 的其他划分目标。不同划分可共享其他背景船/报文，实际交集如下。覆盖完整指有登记的当日地理覆盖框可容纳整个所用半径，并不证明接收站完整观测。','','| 版本 | 对照分区 | 共享背景船 | 共享背景原始报文（MMSI、时间） |','|---|---|---:|---:|']
for v in ['original','relaxed']:
 for pair,d in m['background_overlap'][v].items():text.append(f"| {v} | {pair} | {d['shared_background_mmsi']} | {d['shared_background_reports']} |")
text += ['', '放宽版 train/val/test 的完整覆盖来源分别为 1,932/2,034、264/272、212/220。test 的 220 个来源中 209 个目标身份曾在旧 test 上下文清单出现；220 个来源目标均未出现在所核查的旧 train/val 普通/可信邻域身份清单。test 底层来源曾参与早期单船 pilot，数据没有完全未曝光的盲测地位。目标时间窗口存在来自同一船的多个来源，群体背景可跨分区重用；目标身份隔离不等于场景独立。','', '## 4. 自然与后段定向压力','','同一版本/来源的 Base、natural 和 suffix 共用目标、四格、固定成员与后段选中成员；选中成员的随机种子、3–5倍倍率相同，各视图仅改变选择的两个相邻航速报文。natural 在锚点后选择，suffix 强制在目标后段最后三个时刻的最新邻船报文中选择。所有变体共享相同操作；不依据标签、模型分数或期望的统计结论选择。','','以下是事后遍历目标后段对齐报文的实际数值变化检查，所有来源均保留；该检查与中间步骤可能受到扰动是两个问题。','','| 版本 / 分区 | 来源 | natural 后段实际变值来源 | suffix 后段实际变值来源 | 选中邻船总数 | suffix 后段实际变值邻船次数 |','|---|---:|---:|---:|---:|---:|']
for v in ['original','relaxed']:
 for s in splits:
  x=effect[v+'_'+s];text.append(f"| {v} / {s} | {x['sources']} | {x['natural_sources_numeric_hit']} | {x['suffix_sources_numeric_hit']} | {x['suffix_members_selected']} | {x['suffix_members_suffix_numeric_change']} |")
text += ['', '放宽版 test 的 220 个来源中 130 个仅有 3 艘邻船；`ceil(K/2)` 导致选中邻船 501/840（59.64%），162 个来源中选中超过半数成员，构成高强度条件压力。所谓“选中”与每条实际变值不同；即使全部后段来源至少一条变值，仍有5个选中成员无末段数值变化。两视图均重算由观测报文推导的运动学权重。','', '逐窗口资格见 `construction_attempts.jsonl` 和 `supplement/construction_attempts.jsonl`；候选选择剔除见各区域 `*_candidate_rejections.jsonl`；来源和同一成员压力详细记录见 `dataset_manifest_combined.json`，实际值检查见 `effective_pressure_audit.json`。']
(OUT/'八区域关系扩充构造清点.md').write_text('\n'.join(text)+'\n',encoding='utf8')
print('DATASET_REPORT_DONE',flush=True)
