from pathlib import Path
import json,datetime,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties,findfont
D=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/cases/v5')
a=json.loads((D/'case_scores.json').read_text());audit=json.loads((D/'realcase_input_audit.json').read_text());models=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];labels=['TCN','GRU','Transformer','iTransformer','PatchTST','Mamba','Neural-CDE','GAT-TCN']; colors=['#246a9d','#d87938','#548b56','#b45860','#7b70a8','#a78846','#348e8a','#5d6680']
font='/home/ps/.local/share/fonts/TimesNewRoman/TimesNewRoman.ttf';assert Path(font).exists();matplotlib.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(2,2,figsize=(15,8),gridspec_kw={'height_ratios':[2.2,1.0]},layout='constrained')
for col,name in enumerate(('MATSON','SARA')):
 rows=a['cases'][name];when=[datetime.datetime.fromisoformat(r['end_utc']) for r in rows];sog=[r['median_sog_kn'] for r in rows];support=[r['field_fraction'] for r in rows]
 for i,(key,label,color) in enumerate(zip(models,labels,colors)):
  y=[r['models'][key]['mean'] for r in rows];ax[0,col].plot(when,y,'o-',lw=1.3,ms=3,color=color,label=label)
 ax[0,col].set_ylim(0,1);ax[0,col].set_ylabel('Anomaly score (seed mean)');ax[0,col].grid(alpha=.18)
 ax[1,col].plot(when,sog,'o-',color='#2d7472',lw=1.5,ms=3);ax[1,col].set_ylabel('Median SOG (kn)');ax[1,col].grid(alpha=.16)
 twin=ax[1,col].twinx();twin.bar(when,support,width=0.009,color='#a2acb1',alpha=.45,label='Field-valid fraction');twin.set_ylim(0,1);twin.set_ylabel('Valid field fraction')
 for row in (0,1):ax[row,col].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=datetime.timezone.utc));ax[row,col].tick_params(axis='x',rotation=0)
 ax[1,col].set_xlabel('Window end time (UTC)')
 if name=='MATSON':
  ax[0,col].set_title('(a) MATSON NAVIGATOR — 2017-01-13');ax[1,col].set_title('(c) Motion and fitted-field availability')
  whenevent=datetime.datetime.fromisoformat('2017-01-13T18:00:00+00:00')
  ax[0,col].axvline(whenevent,color='#9d5a55',ls='--',lw=1.2);ax[1,col].axvline(whenevent,color='#9d5a55',ls='--',lw=1.2)
 else:
  ax[0,col].set_title('(b) SARA REGINA — 2024-07-16');ax[1,col].set_title('(d) Motion and fitted-field availability')
ax[0,0].legend(ncol=4,loc='lower center',bbox_to_anchor=(1.04,1.01),frameon=False,fontsize=9)
fig.suptitle('Single-vessel F1 event-associated replay (out-of-distribution)',fontsize=15)
fig.text(.5,-.015,'Window-level model scores are not calibrated probabilities. MATSON dashed line: event-table reported time; SARA contact time is unknown. Receiver coverage is unverified.',ha='center',fontsize=9)
for ext in ('png','pdf','svg'):
 path=D/('Fig8_single_vessel_event_replay.'+ext);assert not path.exists();fig.savefig(path,dpi=300,bbox_inches='tight')
plt.close(fig)
header='''# 单主体 F1：两起真实事件的域外回放\n\n## 实验范围\n\n- 模型：完整单主体实验的 8 个骨干，每个 3 个固定种子，仅使用已冻结第25轮 F1 权重；同一目标窗口显示各骨干三个种子的平均分，全部逐 seed 数值保存在 `case_scores.json`。\n- 时间：所有时间轴为 UTC；23 分钟非重叠窗口，保留至少 8 条报告、最多20条。MATSON 复用此前冻结的 5 个事件关联窗口；SARA 覆盖 2024-07-16 12:00–22:00 UTC 的原始数据可运行窗口。\n- MATSON 数据来自 Sandia HawaiiCoast_GT 事件表与对应 AIS，事件表提供 18:00 UTC 报告时刻；关联时间段是弱标签，不能将每个报文都当故障。SARA 仅有同船、同日、同区域事件报道；碰撞确切时刻未知，12:00–22:00 是审查窗口，不是事故时段或标签。\n- 每时步使用该时刻以前的最近邻船（最多300秒），原 8 海里、同向、可信与仿射几何条件计算；未通过时 `rho=0` 回到同一模型的目标运动流。SARA 邻船来自既有 Miami enriched 数据、目标来自 NOAA 原始数据；两个区域的接收覆盖均未获独立证明。\n- F1 连续输出是模型分数，不能保证实际异常概率；旧验证集负样本阈值仅用于描述，不能称真实事件召回、漏检率或现实误报率。本数据为事件案例，不是全部类别代表性测试。\n- 已知构造器首版因 pandas 微秒时间戳除错导致空窗口，在任何评分前被断言阻止；v2显式转秒。v3修正padding时间至0；v4将可信权重cast float32以精确匹配训练缓存。四版本均保留，v5对迈阿密位置使用源文件记载的区域中心投影。唯一用于评分的是v5。训练集 Houston 980 窗自船因果门控 parity 检查中，v4权重/位置/其他通道最大差均为0。\n\n## 窗口结果（每骨干三种子均值）\n\n'''
lines=[header]
for name in ('MATSON','SARA'):
 lines.append(f'### {name}\n\n| 窗口结束 UTC | 报文数 | 中位 SOG kn | 有效场步 | '+ ' | '.join(labels)+' |\n')
 lines.append('|---|---:|---:|---:|'+'---:|'*len(labels)+'\n')
 for r in a['cases'][name]:lines.append('| '+r['end_utc']+' | '+str(r['n'])+' | '+format(r['median_sog_kn'],'.2f')+' | '+str(r['context_valid_steps'])+'/'+str(r['n'])+' | '+' | '.join(f"{r['models'][key]['mean']:.3f}" for key in models)+' |\n')
 lines.append('\n')
lines.append('## 可复核资产\n\n`case_protocol_prescore.json` 保存数据和8×3模型选择，`case_models_prescore_lock.json` 保存已冻结权重与验证阈值散列，`realcase_inputs.npz` / `realcase_input_audit.json` 保存物理输入与邻船支持，`case_scores.npz/json` 保存全部得分与阈值比较，`Fig8_single_vessel_event_replay.png/pdf/svg` 为预览图。收集器只读取观测，不读取人为注入标签或合成前真值。\n')
(D/'真实事件单主体F1回放报告.md').write_text(''.join(lines),encoding='utf8');print('PLOT_AND_REPORT_SAVED',flush=True)
