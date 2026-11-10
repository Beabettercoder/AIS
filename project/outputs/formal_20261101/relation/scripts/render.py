from pathlib import Path
import json,hashlib
import numpy as np
O=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/relation');o=json.loads((O/'summary.json').read_text());names=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];views={'base':'Base','natural':'自然速度','suffix':'后段速度','combined_l100_m600':'后段速度＋位置'}
L=['# 群体关系任务：八骨干三种子完整对照','', '本表为预先授权执行的35维原阶段累计关系方案；11个方向投影是目标COG计算，并非AIS独立Heading字段。C0/F0/F1同维/同seed初始化；F0为距离加权、F1为距离及运动学可信度加权，报文聚合及支持度也不同，比较为整体配置。','', '训练仅Base，iTransformer与PatchTST来源四格成批，其他随机批次；iTransformer 75轮、其他25轮，固定末轮。所有测试来源此前已曝光、背景可能重用，非新盲测；标签为受控四格而非真实故障。','', 'AP为3种子平均±样本标准差。差值区间为2000次目标MMSI聚类重抽样，固定模型下的样本不确定性，不包括种子总体不确定性/共享背景依赖/多重比较。阈值由每个模型验证集Base负类确定，测试必须报告实际FAR。','']
for split in ('val','test'):
 L+=['## '+('开发验证集' if split=='val' else '已曝光测试集'),'']
 for view,vtitle in views.items():
  L+=['### '+vtitle,'','|骨干|C0 AP|F0 AP|F1 AP|F1−F0（pp）[95% CI]|','|---|---:|---:|---:|---|']
  for n in names:
   z=o['results'][split][n]['views'][view];a=z['arms'];d=z['paired']['F1_minus_F0'];fmt=lambda x:f"{x['pooled_AP_mean']:.4f} ± {x['pooled_AP_sd']:.4f}"
   L.append(f"|{n}|{fmt(a['C0'])}|{fmt(a['F0'])}|{fmt(a['F1'])}|{d['mean_AP_diff']*100:+.2f} [{d['mmsi_cluster95'][0]*100:+.2f}, {d['mmsi_cluster95'][1]*100:+.2f}]|")
  L+=['']
  if split=='test':
   L+=['#### F1：固定验证集阈值的实际误报与召回','','|骨干|1%目标：实际FAR|1%目标：召回|5%目标：实际FAR|5%目标：召回|','|---|---:|---:|---:|---:|']
   for n in names:
    z=o['results'][split][n]['views'][view]['arms']['F1']['seeds'];means=lambda budget,key:float(np.mean([a['FAR_'+budget][key] for a in z]));L.append(f"|{n}|{means('0.01','actual_FAR'):.4f}|{means('0.01','recall'):.4f}|{means('0.05','actual_FAR'):.4f}|{means('0.05','recall'):.4f}|")
   L+=['']
L+=['## 各区域测试样本支持','','每个原始来源有4格且同源正负配对；同一目标MMSI/邻船可出现在多个来源，不能视作独立事件。','','|区域|窗口|来源|目标船|正例|','|---|---:|---:|---:|---:|']
z=o['results']['test']['tcn']['per_region']
for r,s in sorted(z.items()):L.append(f"|{r}|{s['windows']}|{s['sources']}|{s['mmsi']}|{s['positives']}|")
L+=['','详细逐seed AP、AUROC、实际FAR/召回及分组数据见 `summary.json` 和各run的结果/预测。']
out=O/'关系任务正式结果.md'
if out.exists():raise FileExistsError(out)
out.write_text('\n'.join(L)+'\n',encoding='utf8');print(str(out))
