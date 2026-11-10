"""Verify and summarize 48 paired recent-late runs, best and fixed last."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from recent_late_provenance import sha
from summarize_eight_matched import paired_bootstrap

METHODS=('tft_gcn','simad','dmrad','carla')
ARMS=('raw','causal_bundle','f0_bundle','f1_bundle')
PAIRS=(('raw','causal_bundle'),('causal_bundle','f0_bundle'),('f0_bundle','f1_bundle'),('raw','f1_bundle'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107/recent_original_late');p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
    root=Path(a.out);manifest=json.loads((root/'frozen_manifest.json').read_text());rows=[];pairs=[];ref=None
    if len(manifest['runs'])!=48 or manifest['seeds']!=[0,1,2]:raise ValueError('expected frozen 48 runs and three seeds')
    for method in METHODS:
      for checkpoint in ('best_val','last'):
        data={}
        for arm in ARMS:
          for seed in range(3):
            key=f'{method}/{arm}/seed{seed}';folder=root/'runs'/key
            if sha(folder/(checkpoint+'.pt'))!=manifest['runs'][key][checkpoint]['sha256']:raise ValueError('checkpoint mismatch '+key)
            result=json.loads((folder/f'test_{checkpoint}_metrics.json').read_text());metric=result[f'test_{checkpoint}']
            with np.load(folder/f'test_{checkpoint}_scores.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
            if len(d['y'])!=1680:raise ValueError('expected all 1680 windows')
            if ref is None:ref=d
            for k in ('window_id','mmsi','region','y'):
                if not np.array_equal(ref[k],d[k]):raise ValueError(f'unpaired identity {key}/{k}')
            if result['checkpoint_sha256']!=manifest['runs'][key][checkpoint]['sha256']:raise ValueError('result checkpoint mismatch '+key)
            data[(arm,seed)]=d;rows.append({'method':method,'checkpoint':checkpoint,'arm':arm,'seed':seed,'epoch':result['epoch'],'phase':result['phase'],**metric})
        for before,after in PAIRS:
            aa=[data[(after,s)]['score'] for s in range(3)];bb=[data[(before,s)]['score'] for s in range(3)]
            ci=paired_bootstrap(ref['y'],aa,bb,ref['mmsi'],a.bootstrap,1729)
            pairs.append({'method':method,'checkpoint':checkpoint,'comparison':after+' minus '+before,**ci})
    dest=root/'summary';dest.mkdir(exist_ok=True)
    for name,items in (('per_seed.csv',rows),('paired_95ci.csv',pairs)):
        with (dest/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    (dest/'summary.json').write_text(json.dumps({'rows':rows,'pairs':pairs,'scope':'same 1680 previously exposed test windows; clustered by target MMSI, intervals conditional on 3 seeds','bootstrap':a.bootstrap,'manifest_sha256':sha(root/'frozen_manifest.json')},indent=2),encoding='utf8')
    names={'tft_gcn':'TFT-GCN','simad':'SimAD','dmrad':'DMRAD*','carla':'CARLA'}
    labels={'raw':'原始输入','causal_bundle':'门控＋四维训练包','f0_bundle':'普通场＋门控＋训练包','f1_bundle':'可信场＋门控＋训练包'}
    md=['# 近期方法接入原版晚期旁路：单主体三种子结果','',
        '本表同时保留验证集最优权重和固定第25轮权重；测试集均为此前研究已接触过的1,680个窗口。每个方法四组采用相同目标船划分、种子、训练预算和原方法辅助损失。AP、AUROC为三种子均值±标准差，百分数。','']
    for checkpoint,title in (('best_val','验证集最优权重'),('last','固定第25轮权重')):
        md += ['## '+title,'','|方法|配置|AP|AUROC|名义1%阈值的实际误报率|对应召回率|','|---|---|---:|---:|---:|---:|']
        for method in METHODS:
            for arm in ARMS:
                rs=[r for r in rows if r['method']==method and r['checkpoint']==checkpoint and r['arm']==arm]
                if len(rs)!=3:raise ValueError('missing three-seed result')
                def pm(k):
                    values=np.array([r[k] for r in rs])*100
                    return f'{values.mean():.2f}% ± {values.std(ddof=1):.2f}%'
                md.append(f"|{names[method]}|{labels[arm]}|{pm('pooled_AP')}|{pm('AUROC')}|{np.mean([r['FAR@1%'] for r in rs])*100:.2f}%|{np.mean([r['recall@1%'] for r in rs])*100:.2f}%|")
        md += ['', '配对差值为三种子平均 AP 差，单位百分点；区间为 2,000 次目标 MMSI 聚类 Bootstrap。','',
               '|方法|对照变化|AP差值和95%区间（百分点）|','|---|---|---:|']
        for method in METHODS:
            for before,after in PAIRS:
                item=next(r for r in pairs if r['method']==method and r['checkpoint']==checkpoint and r['comparison']==after+' minus '+before)
                md.append(f"|{names[method]}|{labels[before]} → {labels[after]}|{item['delta_AP']*100:+.2f} [{item['lower95']*100:+.2f}, {item['upper95']*100:+.2f}]|")
        md.append('')
    md += ['## 方法与结果的解释','',
        '- 原始输入到门控＋训练包同时改变目标门控和训练损失；普通场与可信场比较才隔离背景场加权。没有场的训练包仍会读取四维背景统计。',
        '- 四种近期模型使用目标船7维主流与ρ加权4维场旁路，旁路系数初始化为0；同时保留各模型原有辅助损失。CARLA包含对比预训练和聚类两个阶段；验证最优权重可能位于任一阶段，逐种子文件保留 phase。',
        '- 输入保留原始采样点顺序和历史前缀掩码；但近期方法原生网络及其辅助损失不消费不规则时间或内部有效掩码，仍会看到右侧补零，不能称为与真正支持不规则采样的骨干等价。',
        '- 平滑损失使用 TFT-GCN 原生时域隐藏、CARLA 卷积隐藏、DMRAD 趋势/残差解码时间分量，SimAD 使用 patch 隐藏重复映射到时间步的代理；不可将四种模型的内部平滑效果视为完全相同。',
        '- 95%区间按目标MMSI聚类，表示给定三种子模型下的船舶抽样不确定性，不覆盖种子总体不确定性；多重比较未校正。',
        '- DMRAD 同名公开仓库与论文作者正式实现的关系仍待确认；近期模型均为 AIS 监督适配，不等于原论文原样复现。',
        '', '## 对上一轮结果的两处更正','',
        '- 上一轮群体关系任务中，SimAD 后段速度污染的可信场−普通场为 +1.78 个百分点，95%区间 [−0.05, +3.78]，区间跨零。',
        '- 上一轮 CARLA 的监督适配第一阶段已经使用二分类标签损失；现有证据不能将其下降归因于第一阶段没有标签。','',
        '逐种子完整指标和固定名义1%/5%阈值的实际误报率与召回率见 `per_seed.csv`；全部配对区间见 `paired_95ci.csv`。']
    (dest/'中文总结.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    print(f'SUMMARIZED 48 runs, best+last, {a.bootstrap} target-vessel bootstrap draws',flush=True)
if __name__=='__main__':main()
