"""Render fixed observed-statistic results, including all controls and views."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

NAMES=['target_only','neighbor_only','relative_mean','relative_median','relative_reliable']
CN={'target_only':'仅目标变化','neighbor_only':'仅船群变化','relative_mean':'相对变化：均值参照','relative_median':'相对变化：中位数参照','relative_reliable':'相对变化：可信度加权参照'}
EN=['Target only','Neighbors only','Relative mean','Relative median','Reliability weighted']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--fonts',type=Path,required=True);a=ap.parse_args();r=a.root
    base=read(r/'base_statistics_v2/statistics_summary.json');stress=read(r/'speed_stress_statistics_v2/statistics_summary.json');data=read(r/'all_sources_v1/relation_manifest.json');contam=read(r/'neighbor_speed_stress_v1/stress_manifest.json');boot=read(r/'bootstrap_v1/paired_mmsi_bootstrap.json')
    for p in a.fonts.glob('times*.ttf'):font_manager.fontManager.addfont(str(p))
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8.5,'axes.titlesize':10,'svg.fonttype':'path','axes.unicode_minus':False})
    out=r/'figures';out.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(180/25.4,91/25.4),sharey=True)
    y=np.arange(len(NAMES));colors=['#297E9E','#CB7B3E']
    for k,(d,view,label,color) in enumerate([(base,'base','Position-only source view',colors[0]),(stress,'speed_stress','Added neighbor-speed errors',colors[1])]):
        for j,n in enumerate(NAMES):
            row=boot['metric_intervals'][view][n]['AP'];m=row['point'];lo=max(0,m-row['lower_2_5']);hi=max(0,row['upper_97_5']-m)
            axes[0].errorbar(m,j+(k-.5)*.18,xerr=np.array([[lo],[hi]]),fmt='o',color=color,markersize=4,capsize=3,label=label if j==0 else None)
            axes[1].plot(d['metrics'][n]['splits']['val']['macro']['AP'],j+(k-.5)*.18,'o',color=color,markersize=4)
    for ax in axes:
        ax.axvline(.5,color='#9AA4A2',linestyle='--',lw=.8);ax.set_xlim(.45,1);ax.set_xlabel('Average precision');ax.grid(axis='x',alpha=.18);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(y,EN);axes[0].invert_yaxis();axes[0].set_title('Pooled validation AP\n95% target-MMSI bootstrap interval')
    axes[1].set_title('Equal-region macro AP\nPoint estimates; unequal domain counts')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=8)
    fig.suptitle('Observed relative-motion statistics on controlled source quartets',x=.025,ha='left',fontsize=10)
    fig.tight_layout(rect=(0,.13,1,.9))
    for ext in ['png','svg']:fig.savefig(out/f'relational_statistics.{ext}',dpi=450)
    plt.close(fig)
    lines=['# 群体关系任务：简单统计与可信参照对照','',
           '本阶段检验可观测轨迹中的关系信息，不训练神经网络，不读取测试轨迹/标签。四格任务是独立的受控一致性任务，不替换既有个体异常标签。','',
           '## 数据与任务','',
           f"尝试此前固定锚点条件的456个候选，273个训练来源和112个验证来源通过完整后段验收，共385个来源窗口、1540个四格变体。训练169个目标MMSI，验证58个，身份交集为0。71个候选整组拒绝，拒绝记录完整保留。",'',
           '| 目标变化a | 船群变化b | 标签a XOR b |','|---:|---:|---:|','|0|0|0|','|0|1|1|','|1|0|1|','|1|1|0|','',
           '固定a时目标输入逐点相同，固定b时船群输入逐点相同；每个单侧签名对应平衡标签。成员在过去锚点选择，后段只用于整组验收。四兄弟不作为四个独立样本。','',
           '标签表示人工施加的共同乘法因子是否一致。原始各船未来的自然变化仍保留，因此同因子不保证实际前后速度变化完全相等。此标签不是自然异常人工金标。','',
           '## 固定统计规则','',
           '从最终可见轨迹长度重算锚点 max(2,n//2−1)。目标前缀平均报告速度与后段最后3个报告速度的比值变化记为d_i。邻船按目标时间网格取当时及之前300秒内最新观测，分别归一到自身前缀平均速度，得到d_j。前后至少2个有效target-grid值才使用该成员；这不等同至少2条不同邻船报告。',
           '主要分数为 |d_i−mean(d_j)|，并比较median(d_j)和运动学可信度加权mean(d_j)。可信权重仅根据最终可见位置/速度的相容误差计算；目标变化不乘可信度门控。全零可信权重时回退到同一普通均值并记录。这里是固定船群的简单相对变速参照，没有拟合空间仿射场，也不是已训练的PG-DRGN模型。',
           '不读取生成onset、a/b开关、原始未扰动真值或样例编号作为预测输入。标签只在分数计算后进入评测。阈值仅用base训练分数选择，然后原样用于验证及速度污染视图。','',
           'v2加入AIS有效范围检查（SOG<102.2、COG<360等）。原30来源均无越界值，v1与v2全部120个分数及分量完全一致；v1原结果保留。','',
           '## 验证结果：所有448个场景，112个来源，58个目标MMSI','',
           '| 方法 | base pooled AP | base pooled AUROC | speed-stress pooled AP | speed-stress pooled AUROC |','|---|---:|---:|---:|---:|']
    for n in NAMES:
        b=base['metrics'][n]['splits']['val']['pooled'];s=stress['metrics'][n]['splits']['val']['pooled'];lines.append(f"| {CN[n]} | {b['AP']:.4f} | {b['AUROC']:.4f} | {s['AP']:.4f} | {s['AUROC']:.4f} |")
    lines+=['','| 方法 | base区域等权AP | speed-stress区域等权AP |','|---|---:|---:|']
    for n in NAMES:lines.append(f"| {CN[n]} | {base['metrics'][n]['splits']['val']['macro']['AP']:.4f} | {stress['metrics'][n]['splits']['val']['macro']['AP']:.4f} |")
    lines+=['','两个单侧负对照均为AP/AUROC=0.5。F1不是机会水平检验：平衡数据上全报正类也能得到F1=2/3，故不据此宣称负对照有预测能力。',
            'base视图只施加位置跳变，SOG统计自然不受这类附加污染直接影响；它不能证明速度污染鲁棒性。','',
            '## 独立邻船速度污染视图','',
            '对每个来源按哈希选择ceil(0.15×邻船数)成员，在锚点后的候选报告中随机选连续2条，SOG乘U[3,5]并封顶102.1kn，复用已有观测污染算子；target保持不变。四兄弟完全共用成员、索引、因子，操作与标签无关。',
            '每组仅4—8名邻船，ceil(0.15K)会产生整数上取整：训练请求305/1340个成员实例（22.76%），验证请求138/605（22.81%），不能称精确15%污染。',
            f"验证112个来源全部有数值速度污染，其中{contam['counts']['val']['suffix_overlap_sources']}个来源的固定后段SOG统计直接采到受污染报告。所有来源都保留，未按命中、分数或结果挑选。",'',
            '## 成对MMSI聚类不确定性','',
            '使用58个验证目标MMSI整组抽样2000次，所有窗口和四兄弟随MMSI一起保留，同一抽样用于两视图与全部方法。下表为pooled AP配对差，单位为百分点。','',
            '| 视图 | 对比（后者−前者） | 点估计 | 95%百分位区间 |','|---|---|---:|---|']
    for c in boot['contrasts']:
        if c['view'] not in ['base','speed_stress']:continue
        v=c['AP'];lines.append(f"| {c['view']} | {CN[c['to']]} − {CN[c['from']]} | {100*v['point']:+.3f} | [{100*v['lower_2_5']:+.3f}, {100*v['upper_97_5']:+.3f}] |")
    lines+=['','区间条件于当前目标MMSI和固定分数；不能覆盖未见海域的不确定性，共享背景场景还可能关联不同目标船。区域等权仅报告点估计和样本量，Haugesund只有1个验证MMSI，不制造虚假的跨域稳健结论。','',
            '## 结论与边界','',
            '1. 四格任务中，相对运动统计确实提供了单侧观察没有的信息，说明值得进一步开展关系任务研究。',
            '2. 速度污染下，物理可信度加权比普通均值更稳健，但中位数参照仍明显更强。不能据此宣称可信方法击败了强稳健基线。',
            '3. base中位数优于均值而可信加权几乎不变，提示“观测相容”与“行为参照合适”需要区分；这是待继续检验的机制解释，不是唯一原因的证明。',
            '4. pooled与区域等权AP差异较大，不能只挑较高的macro数值。全部分数、阈值、混淆矩阵、分域数量和拒绝来源均保留。','',
            '## 各区验收规模','',
            '| 区域 | 训练来源 | 训练MMSI | 验证来源 | 验证MMSI |','|---|---:|---:|---:|---:|']
    for reg,rs in data['regions'].items():lines.append(f"| {reg} | {rs['train']['accepted_sources']} | {rs['train']['target_mmsi']} | {rs['val']['accepted_sources']} | {rs['val']['target_mmsi']} |")
    lines+=['','## 文件索引','',
            '- `statistic_protocol_v2.json`、`observed_statistics_v2.py`：固定观察统计与合法性检查。',
            '- `all_sources_v1/relation_manifest.json`：所有接受/拒绝来源及源文件哈希。',
            '- `base_statistics_v2/`：主分数、完整分量与指标。',
            '- `speed_stress_protocol_v1.json`、`neighbor_speed_stress_v1/stress_manifest.json`：独立速度污染规则与实际生效记录。',
            '- `speed_stress_statistics_v2/`：沿用base训练阈值的全部污染分数。',
            '- `bootstrap_v1/paired_mmsi_bootstrap.json`：成对聚类区间。',
            '- `v1_v2_bridge.json`：30源桥接分数完全一致。',
            '- `figures/relational_statistics.png`：实际指标与区间图，英文Times New Roman。','']
    (r/'群体关系简单统计实验报告.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'figure_provenance.json').write_text(json.dumps({'source_hashes':{x:sha(r/x) for x in ['base_statistics_v2/statistics_summary.json','speed_stress_statistics_v2/statistics_summary.json','bootstrap_v1/paired_mmsi_bootstrap.json']},'font':'Times New Roman','outputs':{p.name:sha(p) for p in out.iterdir()}},indent=2),encoding='utf-8')
    print(json.dumps({'status':'written','report':str(r/'群体关系简单统计实验报告.md')}),flush=True)
if __name__=='__main__':main()
