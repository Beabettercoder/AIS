from pathlib import Path
import json,datetime,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties, fontManager, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
D=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/cases/v6'); a=json.loads((D/'rolling_scores.json').read_text()); models=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];labels=['TCN','GRU','Transformer','iTransformer','PatchTST','Mamba','Neural-CDE','GAT-TCN'];colors=['#246a9d','#d87938','#548b56','#b45860','#7b70a8','#a78846','#348e8a','#5d6680']
font='/home/ps/.local/share/fonts/TimesNewRoman/TimesNewRoman.ttf';assert Path(font).exists();fontManager.addfont(font);assert Path(findfont(FontProperties(family='Times New Roman'),fallback_to_default=False)).name==Path(font).name;matplotlib.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(16,15),layout='constrained');gs=fig.add_gridspec(6,4,height_ratios=[1,1,.60,1,1,.60]); axes={}
for case,rowstart in [('MATSON',0),('SARA',3)]:
 rows=a['cases'][case];tt=[datetime.datetime.fromisoformat(r['end_utc']) for r in rows]
 for k,(name,label,color) in enumerate(zip(models,labels,colors)):
  r=rowstart+k//4;c=k%4;ax=fig.add_subplot(gs[r,c]);axes[(case,name)]=ax
  scores=np.asarray([r['models'][name]['scores'] for r in rows]);mean=scores.mean(1);low=scores.min(1);high=scores.max(1)
  ax.fill_between(tt,low,high,color=color,alpha=.15,lw=0);ax.plot(tt,mean,'o-',color=color,ms=3,lw=1.5);ax.set_ylim(0,1);ax.set_title(label,fontsize=10);ax.grid(alpha=.16)
  if c==0:ax.set_ylabel('Anomaly score')
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=datetime.timezone.utc));ax.tick_params(axis='x',labelsize=8)
  if case=='MATSON':ax.axvline(datetime.datetime.fromisoformat('2017-01-13T18:00:00+00:00'),color='#9d5a55',ls='--',lw=.9)
 ax=fig.add_subplot(gs[rowstart+2,:]);sog=[r['median_sog_kn'] for r in rows];support=[r['field_fraction'] for r in rows];ax.plot(tt,sog,'o-',color='#2d7472',lw=1.5,ms=3);ax.set_ylabel('Median SOG (kn)');ax.grid(alpha=.16)
 twin=ax.twinx();twin.bar(tt,support,width=.009,color='#9faeb1',alpha=.35);twin.set_ylim(0,1);twin.set_ylabel('Valid field fraction');ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=datetime.timezone.utc));ax.set_xlabel('Window end time (UTC)')
 if case=='MATSON':ax.axvline(datetime.datetime.fromisoformat('2017-01-13T18:00:00+00:00'),color='#9d5a55',ls='--',lw=.9)
 label='MATSON NAVIGATOR — 2017-01-13' if case=='MATSON' else 'SARA REGINA — 2024-07-16'
 axes[(case,models[0])].text(0,1.38,label,transform=axes[(case,models[0])].transAxes,ha='left',va='bottom',fontsize=13,fontweight='bold',clip_on=False)
fig.suptitle('Single-vessel F1 event-associated replay (out-of-distribution)',fontsize=15)
fig.legend(handles=[Line2D([0],[0],color='#466f8a',lw=1.5,marker='o',ms=3,label='Three-seed mean'),Patch(facecolor='#466f8a',alpha=.15,label='Three-seed min–max'),Patch(facecolor='#9faeb1',alpha=.35,label='Valid field fraction'),Line2D([0],[0],color='#2d7472',label='Window median SOG'),Line2D([0],[0],color='#9d5a55',ls='--',label='Reported event time')],loc='upper center',bbox_to_anchor=(.5,.965),ncol=5,frameon=False,fontsize=9)
fig.text(.5,-.005,'Overlapping windows; scores are not calibrated probabilities; bands are seed ranges, not confidence intervals. MATSON: event-table reported time. SARA: exact contact time unknown. Receiver coverage unverified.',ha='center',fontsize=9)
for ext in ('png','pdf','svg'):
 p=D/('Fig8_single_vessel_event_replay_rolling_TNR.'+ext);assert not p.exists();fig.savefig(p,dpi=300,bbox_inches='tight')
plt.close(fig)
p=D/'rolling_scores_all_backbones_seeds_TNR.csv';assert not p.exists()
with p.open('x',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['case','window_index','start_utc','end_utc','message_count','median_sog_kn','valid_field_steps','field_fraction','backbone','seed','score','val_threshold_1pct','val_threshold_5pct','exceeds_val_1pct','exceeds_val_5pct'])
 for case in ('MATSON','SARA'):
  for r in a['cases'][case]:
   for model in models:
    q=r['models'][model]
    for seed in range(3):w.writerow([case,r['index'],r['start_utc'],r['end_utc'],r['n'],r['median_sog_kn'],r['context_valid_steps'],r['field_fraction'],model,seed,q['scores'][seed],q['threshold_1pct'][seed],q['threshold_5pct'][seed],q['over_1pct'][seed],q['over_5pct'][seed]])
print('SEED_RANGE_FIGURE_AND_CSV_SAVED',flush=True)

notes=["# 单主体 F1：真实案例滚动窗口时间轴\n\n", "本补充分析在读取任何事件分数之前冻结：在每个案例固定时段内，每隔至少5分钟以一个已到达报文为决策点，向后回看23分钟，逆序按最小60秒间隔取最近最多20条有效报文，至少8条才运行。窗口高度重叠，不是独立事件或独立样本；这是离线案例描述，不计算真实召回/FAR。MATSON 18:00 UTC是事件表报告时间，不保证精确起点；SARA碰撞精确时间未知。\n\n", "每个骨干显示3个种子的均值；图中阴影是3个种子的最小–最大范围，不是置信区间。目标及邻船的现场覆盖未知，场不可用回退到该模型自船运动流。各模型阈值来源于合成开发负样本，输出分数不是校准风险或概率。\n\n"]
for case in ('MATSON','SARA'):
 rows=a['cases'][case];notes.append(f'## {case}（{len(rows)}个重叠窗口）\n\n| 窗口结束 UTC | 报文 | 中位 SOG kn | 场有效步 | '+ ' | '.join(labels)+' |\n')
 notes.append('|---|---:|---:|---:|'+'---:|'*len(labels)+'\n')
 for r in rows:notes.append('| '+r['end_utc']+' | '+str(r['n'])+' | '+format(r['median_sog_kn'],'.2f')+' | '+str(r['context_valid_steps'])+'/'+str(r['n'])+' | '+' | '.join(f"{r['models'][key]['mean']:.3f}" for key in models)+' |\n')
 notes.append('\n')
notes.append('\n逐模型、逐种子的完整数值和开发集阈值见 `rolling_scores_all_backbones_seeds_TNR.csv`；输入审计见 `rolling_input_audit.json`、`rolling_independent_field_check.json` 和 `rolling_protocol_prescore.json`。非重叠主窗口与模型得分独立保留。\n')
p=D/'真实事件滚动窗口回放说明_TNR.md';assert not p.exists();p.write_text(''.join(notes),encoding='utf8')
