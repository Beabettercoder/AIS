"""Figures from all frozen-adapter validation runs; no interpolated results."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

NAMES={'self_quality':'Self + quality control','mean_context':'Ordinary mean context',
       'affine_context':'Uniform affine context','reliable_context':'Reliable affine context'}
BN={'tcn':'TCN','gru':'GRU','transformer':'Transformer'}
COLORS=['#B88743','#6B93B8','#8073AB','#0F6C5C']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--fonts',type=Path,required=True)
    a=ap.parse_args();r=a.stage;p=json.loads((r/'protocol.json').read_text());bb=p['backbones'];arms=p['arms'];seeds=p['training']['seeds']
    for f in a.fonts.glob('times*.ttf'):font_manager.fontManager.addfont(str(f))
    font=font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8.5,'axes.titlesize':10,'svg.fonttype':'path','axes.unicode_minus':False})
    d={};sources={}
    for b in bb:
        for s in seeds:
            for arm in arms:
                path=r/'runs'/b/f'seed_{s}'/arm/'result.json';v=json.loads(path.read_text())
                assert v['status']=='completed_validation_only' and v['protocol_sha256']==sha(r/'protocol.json')
                assert len(v['history'])==13;d[b,s,arm]=v;sources[str(path.relative_to(r))]=sha(path)
    dest=r/'figures';dest.mkdir(exist_ok=False)
    save=lambda fig,name:[fig.savefig(dest/f'{name}.{ext}',dpi=450) for ext in ['png','svg']]
    fig,axes=plt.subplots(1,3,figsize=(180/25.4,82/25.4),sharey=True)
    rows=['Unchanged baseline']+[NAMES[x] for x in arms]
    for ax,b in zip(axes,bb):
        vals=[np.array([d[b,s,arms[0]]['baseline_validation']['macro']['AP'] for s in seeds])]
        vals += [np.array([d[b,s,arm]['validation']['macro']['AP'] for s in seeds]) for arm in arms]
        for i,(v,color) in enumerate(zip(vals,['#7F8A8F']+COLORS)):
            ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='o',color=color,capsize=3,markersize=4.5)
            for j,x in enumerate(v):ax.scatter(x,i+(j-1)*.09,color=color,s=10,alpha=.4)
        ax.set_title(BN[b]);ax.set_xlabel('Selected validation macro AP');ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(len(rows)),rows);axes[0].invert_yaxis()
    fig.suptitle('Frozen-baseline adapters: actual seeds, mean and sample SD',x=.02,ha='left',fontsize=10)
    fig.tight_layout(rect=(0,.03,1,.94));save(fig,'adapter_validation_comparison');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(180/25.4,78/25.4),sharex=True,sharey=True)
    contrast_arms=arms[1:];deltas=[]
    for ax,b in zip(axes,bb):
        for i,arm in enumerate(contrast_arms):
            v=100*np.array([d[b,s,arm]['validation']['macro']['AP']-d[b,s,'self_quality']['validation']['macro']['AP'] for s in seeds]);deltas.extend(v)
            ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='o',color=COLORS[i+1],capsize=3,markersize=4.5)
            for j,x in enumerate(v):ax.scatter(x,i+(j-1)*.08,color=COLORS[i+1],alpha=.5,s=12)
        ax.axvline(0,color='#7F8A8F',linewidth=.8);ax.set_title(BN[b]);ax.set_xlabel('Validation AP difference (pp)')
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    limit=max(.5,max(abs(float(v)) for v in deltas)*1.3)
    axes[0].set_xlim(-limit,limit);axes[0].set_yticks(range(3),[NAMES[x] for x in contrast_arms]);axes[0].invert_yaxis()
    fig.suptitle('Context benefit relative to the same-support self + quality adapter',x=.02,ha='left',fontsize=10)
    fig.tight_layout(rect=(0,.03,1,.94));save(fig,'adapter_paired_context_gain');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(180/25.4,88/25.4),sharey=True)
    for ax,b in zip(axes,bb):
        baseline=np.mean([d[b,s,arms[0]]['baseline_validation']['macro']['AP'] for s in seeds])
        ax.axhline(baseline,color='#7F8A8F',linestyle='--',linewidth=.9,label='Unchanged baseline')
        for arm,color in zip(arms,COLORS):
            values=np.array([[v['validation_macro_AP'] for v in d[b,s,arm]['history']] for s in seeds])
            x=np.arange(13);m=values.mean(0);sd=values.std(0,ddof=1)
            ax.plot(x,m,color=color,linewidth=1.2,label=NAMES[arm]);ax.fill_between(x,m-sd,m+sd,color=color,alpha=.1,linewidth=0)
        ax.set_title(BN[b]);ax.set_xlabel('Adapter epoch');ax.set_xticks([0,4,8,12]);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Validation macro AP')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',frameon=False,fontsize=7.5)
    fig.suptitle('All 12 adaptation epochs; epoch 0 remains eligible for selection',x=.07,ha='left',fontsize=10)
    fig.tight_layout(rect=(0,.2,1,.94));save(fig,'adapter_validation_learning_curves');plt.close(fig)
    (dest/'figure_provenance.json').write_text(json.dumps({'scope':'Validation development, all 36 runs, actual three-seed values; selected metrics may use epoch0 baseline.',
        'font':'Times New Roman','font_path':font,'protocol_sha256':sha(r/'protocol.json'),'source_results':sources,
        'outputs':{f.name:sha(f) for f in dest.iterdir()}},indent=2))
    print(json.dumps({'status':'rendered','runs':len(d),'directory':str(dest)}))

if __name__=='__main__':main()
