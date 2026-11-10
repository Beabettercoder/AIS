"""Optional static figures; requires matplotlib only, never starts training."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));args=p.parse_args();root=args.root
    summary=json.loads((root/'reports/summary.json').read_text());assert summary['status']=='complete'
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':['DejaVu Serif','Noto Sans CJK JP','Noto Sans CJK SC'],
                         'axes.unicode_minus':False,'font.size':10,'savefig.dpi':180})
    rows=summary['summary'];models=['tcn','gru','transformer','itransformer'];names=['TCN','GRU','Transformer','iTransformer']
    views=['base','target_jump','target_drift','target_speed','neighbor_stress']
    titles=['基础真实观测','目标末点位置跳变','目标后段位置漂移','目标后段速度尖峰','邻船后段速度与位置污染']
    colors=['#9CAFB6','#508BA5','#D99550','#167B6B'];arms=['P0','P1','P2','P3']
    labels=['原始输入','运动学门控','门控＋普通背景场','门控＋可信背景场']
    get=lambda m,a,v:next(r for r in rows if (r['model'],r['arm'],r['view'],r['subset'])==(m,a,v,'all'))
    fig,axes=plt.subplots(2,3,figsize=(16,9));handles=[]
    for ax,v,title in zip(axes.flat,views,titles):
        x=np.arange(4);width=.19
        for j,a in enumerate(arms):
            y=[get(m,a,v)['ADE_m_mean'] for m in models]
            bars=ax.bar(x+(j-1.5)*width,y,width,color=colors[j],label=labels[j])
            if v=='base':handles.append(bars)
        ax.axhline(get('CV_raw','CV_raw',v)['ADE_m_mean'],color='#333333',ls='--',lw=1.3,label='原始恒速外推')
        ax.axhline(get('CV_gate','CV_gate',v)['ADE_m_mean'],color='#167B6B',ls=':',lw=1.3,label='门控恒速外推')
        ax.set_title(title);ax.set_xticks(x,names,rotation=16);ax.set_ylabel('ADE（米，越低越好）');ax.set_ylim(bottom=0)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes.flat[5].axis('off')
    axes.flat[5].text(.03,.85,'四骨干 × 四配置 × 三种子\n\n验证集：八海区，1,200个窗口\n\n柱高：三种子平均ADE\n虚线：原始恒速外推\n点线：门控恒速外推\n\n每个面板独立纵轴，均从零起始。\n各配置与骨干的完整配对区间见报告。',va='top',linespacing=1.7)
    fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.01));fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(root/'reports/预测误差对照.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,m,name in zip(axes.flat,models,names):
        for a,c,label in zip(arms,colors,labels):
            curves=[]
            for s in range(3):
                path=root/f'parallel_runs/{m}/{m}/seed_{s}/{a}/history.csv'
                with path.open() as f:curves.append([float(r['val_ADE_m']) for r in csv.DictReader(f)])
            curves=np.array(curves);x=np.arange(1,curves.shape[1]+1)
            ax.plot(x,curves.mean(0),color=c,label=label)
            ax.fill_between(x,curves.min(0),curves.max(0),color=c,alpha=.12)
        ax.set_title(name);ax.set_xlabel('训练轮次');ax.set_ylabel('基础观测验证ADE（米）');ax.grid(alpha=.18)
    axes.flat[0].legend(fontsize=8);fig.suptitle('逐轮验证误差：曲线为三种子均值，色带为最小值到最大值',y=1.01)
    fig.tight_layout();fig.savefig(root/'reports/训练过程曲线.png',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
