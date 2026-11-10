"""Render only actually recomputed study metrics; English in Times New Roman."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).resolve().parent
ARCH=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph_temporal']
NAMES=['TCN','GRU','Transformer','iTransformer*','PatchTST*','Mamba-ref*','Neural-CDE*','GAT-TCN*']
COLORS={'tcn':'#087f73','gru':'#326ba8','transformer':'#c97825'}
def setstyle():
    fonts=R.parent/'balanced_global_v2'/'visuals'/'v2'/'fonts_runtime'
    for p in fonts.glob('*.ttf'):font_manager.fontManager.addfont(str(p))
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.22,'axes.axisbelow':True,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
def save(fig,name):
    out=R/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def gains():
    data=json.loads((R/'analysis'/'paired_bootstrap.json').read_text());fig,axes=plt.subplots(2,2,figsize=(10.4,7.3),layout='constrained')
    specs=[('individual','C3-C0','(a) Individual: framework vs target only'),('individual','C3-C2','(b) Individual: reliability vs median'),('relation','C3-C0','(c) Relational: framework vs target only'),('relation','C3-C2','(d) Relational: reliability vs median')]
    for ax,(task,comparison,title) in zip(axes.flat,specs):
        rr=[next(r for r in data if (r['task'],r['backbone'],r['view'],r['comparison'])==(task,a,'base',comparison)) for a in ARCH]
        y=np.arange(8);m=np.array([r['mean_delta_pp'] for r in rr]);lo=np.array([r['ci95_pp'][0] for r in rr]);hi=np.array([r['ci95_pp'][1] for r in rr])
        # A percentile interval need not straddle the point estimator; draw endpoints directly.
        for j in range(8):ax.plot([lo[j],hi[j]],[j,j],color='#397f75',lw=1.4);ax.scatter(m[j],j,color='#087f73',s=29,zorder=3)
        ax.axvline(0,color='#48545b',ls='--',lw=.9);ax.set_yticks(y,NAMES);ax.invert_yaxis();ax.set_xlabel('Paired AP difference (percentage points)');ax.set_title(title,loc='left',fontsize=10.5);ax.grid(axis='y',visible=False)
    fig.suptitle('One shared context operator across eight encoder adaptations',fontsize=14)
    fig.supxlabel('Three fixed seeds; pointwise target-MMSI cluster intervals. *Documented classification/reference adaptations.',fontsize=9)
    save(fig,'fig3_cross_backbone_gain')
def operating():
    data=json.loads((R/'analysis'/'aggregate_metrics.json').read_text());fig,ax=plt.subplots(figsize=(6.4,4.2),layout='constrained')
    for arch in ['tcn','gru','transformer']:
      for arm,ls,mark in [('C2','--','o'),('C3','-','s')]:
        rr=next(r for r in data if (r['task'],r['backbone'],r['view'],r['arm'])==('individual',arch,'base',arm))
        keys=['0.001','0.002','0.005','0.01','0.02','0.05'];x=[rr['FAR_'+k]['mean']*100 for k in keys];y=[rr['Recall_'+k]['mean'] for k in keys]
        ax.plot(x,y,color=COLORS[arch],ls=ls,marker=mark,ms=4,label=arch.upper()+' '+('median' if arm=='C2' else 'reliable'))
    ax.set_xlabel('Actual test false-alarm rate (%)');ax.set_ylabel('Recall');ax.set_ylim(0,1);ax.legend(fontsize=8,ncol=2);ax.set_title('Validation-frozen thresholds; actual test operating points')
    save(fig,'fig4_low_far_detail')
def stress_curves():
    from summarize_results import ap_matrix
    core=json.loads((R/'analysis'/'aggregate_metrics.json').read_text());d=np.load(R/'data'/'relation'/'test.npz',allow_pickle=False);ships=d['mmsi'].astype(str);y=d['y'];unique,inv=np.unique(ships,return_inverse=True)
    boot=np.load(R/'analysis'/'bootstrap_relation.npz',allow_pickle=False);assert np.array_equal(unique,boot['unique_mmsi']);counts=boot['cluster_counts']
    folder=R/'stress_results'/'tcn'/'seed_0'/'C3';names=sorted(p.stem for p in folder.glob('*.npz'))
    groups={g:[n for n in names if n.startswith(g+'_')] for g in ['speed','position','combined']}
    assert all(len(v)==5 for v in groups.values()),groups
    fig,axes=plt.subplots(2,2,figsize=(10.4,7.2),layout='constrained');records=[]
    for ax,g,title in zip(axes.flat[:3],['position','speed','combined'],['(a) Added position stress','(b) Added reported-speed stress','(c) Combined observation stress']):
        xs=np.array([0,50,150,300,600]) if g=='position' else np.array([0,.25,.5,.75,1.])
        for arch in ['tcn','gru','transformer']:
            means=[];los=[];his=[]
            for name in groups[g]:
                probs={arm:[] for arm in ['C2','C3']}
                for arm in probs:
                    for seed in [0,1,2]:
                        p=np.load(R/'stress_results'/arch/f'seed_{seed}'/arm/(name+'.npz'),allow_pickle=False);assert np.array_equal(p['window_id'],d['window_id']);probs[arm].append(p['score'])
                diffs=[]
                for start in range(0,len(counts),128):
                    w=counts[start:start+128,inv].astype(float)
                    diff=np.mean([ap_matrix(y,p,w) for p in probs['C3']],axis=0)-np.mean([ap_matrix(y,p,w) for p in probs['C2']],axis=0);diffs.extend(diff)
                baseline=np.ones((1,len(y)));m=(np.mean([ap_matrix(y,p,baseline)[0] for p in probs['C3']])-np.mean([ap_matrix(y,p,baseline)[0] for p in probs['C2']]))*100
                lo,hi=np.quantile(np.asarray(diffs)*100,[.025,.975]);means.append(m);los.append(lo);his.append(hi);records.append({'view':name,'backbone':arch,'delta_pp':m,'CI95':[lo,hi]})
            ax.plot(xs,means,color=COLORS[arch],marker='o',ms=4,label=arch.upper());ax.fill_between(xs,los,his,color=COLORS[arch],alpha=.10)
        ax.axhline(0,color='#59666d',ls='--',lw=.8);ax.set_title(title,loc='left');ax.set_ylabel('Reliable − ordinary median AP (pp)');ax.set_xlabel('Added displacement (m)' if g=='position' else 'Fraction of frozen suffix-speed perturbation');ax.legend(fontsize=8)
    ax=axes.flat[3]
    for arch in ['tcn','gru','transformer']:
      for arm,ls,marker in [('C2','--','o'),('C3','-','s')]:
        rr=next(r for r in core if (r['task'],r['backbone'],r['view'],r['arm'])==('individual',arch,'base',arm))
        keys=['0.001','0.002','0.005','0.01','0.02','0.05'];ax.plot([rr['FAR_'+k]['mean']*100 for k in keys],[rr['Recall_'+k]['mean'] for k in keys],ls=ls,marker=marker,ms=3.5,color=COLORS[arch],label=arch.upper()+' '+arm)
    ax.set_title('(d) Low-false-alarm operation (individual)',loc='left');ax.set_xlabel('Actual test FAR (%)');ax.set_ylabel('Recall');ax.set_ylim(0,1);ax.legend(fontsize=7.5,ncol=2)
    fig.suptitle('Observation stress and validation-frozen operating points',fontsize=14);save(fig,'fig4_robustness_and_far')
    (R/'analysis'/'stress_curve_pairs.json').write_text(json.dumps(records,indent=2),encoding='utf8')
if __name__=='__main__':
    setstyle();gains();operating()
    if (R/'stress_results'/'transformer'/'seed_2'/'C4'/'result.json').exists():stress_curves()
