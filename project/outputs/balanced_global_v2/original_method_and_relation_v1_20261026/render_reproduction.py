"""Actual audited historical-model reproduction results, fixed epoch and best separate."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

NAMES={'raw_incremental_compat':'Incremental input bridge','raw_legacy_compat':'Historical raw input',
       'kg_compat':'+ Kinematic gate','kglf_compat':'+ Local field','kglf_psa_compat':'+ Historical training bundle',
       'full35_compat':'+ Graph: full Table-1 model','raw_legacy_masked':'Raw input, valid-length control',
       'full35_masked':'Full model, valid-length control'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--fonts',type=Path,required=True);a=ap.parse_args();r=a.stage
    path=r/'audit/historical_final/historical_reproduction_summary.json';s=json.loads(path.read_text());assert s['status']=='passed' and s['completed_runs']==48
    p=json.loads((r/'protocol.json').read_text());arms=p['arms'];bb=p['backbones'];index={(x['backbone'],x['arm']):x for x in s['all_configurations']}
    for f in a.fonts.glob('times*.ttf'):font_manager.fontManager.addfont(str(f))
    fp=font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8.5,'axes.titlesize':10,'svg.fonttype':'path','axes.unicode_minus':False})
    out=r/'figures';out.mkdir(exist_ok=True)
    means=np.array([[index[b,arm][tag]['AP']['mean'] for b in bb for tag in ['final','best']] for arm in arms])
    stds=np.array([[index[b,arm][tag]['AP']['sample_std'] for b in bb for tag in ['final','best']] for arm in arms])
    fig,ax=plt.subplots(figsize=(180/25.4,112/25.4));cmap=LinearSegmentedColormap.from_list('results',['#F5F8F7','#BFD8D1','#357D70'])
    ax.imshow(means,aspect='auto',cmap=cmap,vmin=.7,vmax=.8)
    ax.set_xticks(range(4),['TCN\nEpoch 25','TCN\nBest val.','GRU\nEpoch 25','GRU\nBest val.']);ax.xaxis.tick_top()
    ax.set_yticks(range(8),[NAMES[x] for x in arms]);ax.tick_params(length=0)
    for i in range(8):
        for j in range(4):ax.text(j,i,f'{means[i,j]:.4f}\n+/- {stds[i,j]:.4f}',ha='center',va='center',fontsize=8.1,color='#142D27')
    ax.axvline(1.5,color='white',linewidth=3);ax.axhline(5.5,color='white',linewidth=3)
    ax.set_xticks(np.arange(-.5,4),minor=True);ax.set_yticks(np.arange(-.5,8),minor=True);ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    fig.suptitle('Historical architectures on the new benchmark: validation macro AP',x=.02,ha='left',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ['png','svg']:fig.savefig(out/f'historical_reproduction_ap.{ext}',dpi=450)
    plt.close(fig)
    selected=[('raw_incremental_compat','raw_legacy_compat','Restore historical input semantics'),
              ('raw_legacy_compat','kg_compat','Add original gating operation'),
              ('kg_compat','kglf_compat','Add scalar local-field residual'),
              ('kglf_compat','kglf_psa_compat','Add historical training bundle'),
              ('kglf_psa_compat','full35_compat','Add historical graph'),
              ('raw_legacy_masked','full35_masked','Full versus raw, valid length')]
    fig,axes=plt.subplots(1,2,figsize=(180/25.4,104/25.4),sharex=True,sharey=True);maxdelta=0
    for ax,b,color in zip(axes,bb,['#247C9C','#C38643']):
        for i,(fr,to,label) in enumerate(selected):
            values=100*(np.array(index[b,to]['final']['AP']['values'])-np.array(index[b,fr]['final']['AP']['values']))
            maxdelta=max(maxdelta,float(np.max(np.abs(values))))
            ax.errorbar(values.mean(),i,xerr=values.std(ddof=1),fmt='o',color=color,capsize=3,markersize=4.5)
            for j,x in enumerate(values):ax.scatter(x,i+(j-1)*.1,color=color,s=12,alpha=.45)
        ax.axvline(0,color='#7E8A85',linewidth=.8);ax.set_title(b.upper());ax.set_xlabel('Epoch-25 validation AP change (pp)');ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(len(selected)),[x[2] for x in selected]);axes[0].invert_yaxis();axes[0].set_xlim(-maxdelta*1.25,maxdelta*1.25)
    fig.suptitle('Paired contrasts: every seed, mean and sample SD',x=.02,ha='left',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ['png','svg']:fig.savefig(out/f'historical_paired_contrasts.{ext}',dpi=450)
    plt.close(fig)
    proof={'scope':'All48 runs, source-audited Table-1 architectures transferred to repaired inputs; fixed epoch25 and validation-best shown separately.',
           'summary_sha256':sha(path),'font':'Times New Roman','font_file':fp,'arms':arms,'backbones':bb,
           'AP_means':means.tolist(),'AP_sample_stds':stds.tolist(),'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.suffix in ['.png','.svg']}}
    (out/'reproduction_figures_provenance.json').write_text(json.dumps(proof,indent=2));print(json.dumps({'status':'rendered','directory':str(out)}))
if __name__=='__main__':main()
