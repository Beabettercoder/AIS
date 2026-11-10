from pathlib import Path
import json,gzip
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_results import setstyle,save,ARCH,NAMES,COLORS
from summarize_results import table,fmt
R=Path(__file__).resolve().parent;O=R/'field_extension'
def main():
    setstyle();p=json.loads((O/'analysis'/'paired_bootstrap.json').read_text());ag=json.loads((O/'analysis'/'aggregate_metrics.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10.4,7.3),layout='constrained')
    specs=[('F1-C0','(a) Reliable LOO field vs raw target'),('F1-F0','(b) Reliability within the same LOO field'),('F2-C0','(c) Target gating + reliable field vs raw'),('F2-F1','(d) Added target-gating input package')]
    for ax,(comp,title) in zip(axes.flat,specs):
        for i,arch in enumerate(ARCH):
            r=next(x for x in p if x['backbone']==arch and x['comparison']==comp);ax.plot(r['ci95_pp'],[i,i],color='#397f75');ax.scatter(r['mean_delta_pp'],i,color='#087f73',s=28,zorder=3)
        ax.axvline(0,color='#506169',lw=.9,ls='--');ax.set_yticks(np.arange(8),NAMES);ax.invert_yaxis();ax.set_title(title,loc='left',fontsize=10.3);ax.set_xlabel('Region-macro AP difference (pp)');ax.grid(axis='y',visible=False)
    fig.suptitle('Kinematic reliability and a target-excluded local affine field',fontsize=14)
    fig.supxlabel('Three fixed seeds; pointwise target-MMSI cluster intervals. Previously exposed individual-task test.',fontsize=9)
    save(fig,'fig3_kinematic_loo_field_gain')
    data=np.load(R/'extra_data'/'field_stress_test.npz',allow_pickle=False);y=data['y'];region=data['region'].astype(str);eligible=np.zeros(len(y),bool)
    with gzip.open(R/'extra_data'/'field_stress_selection.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);eligible[int(row['global_row'])]=row['eligible_member_count']>0
    assert eligible.sum()==800
    views=sorted({k.rsplit('__',1)[0] for k in data.files if k.endswith('__F0')});groups={k:[v for v in views if v.startswith(k+'_')] for k in ['speed','position','combined']}
    rec=[]
    for arch in ARCH:
     for view in views:
      for arm in ['F0','F1','F2']:
       vals=[];eligiblevals=[]
       for seed in [0,1,2]:
        q=np.load(R/'field_stress_results'/arch/f'seed_{seed}'/arm/(view+'.npz'),allow_pickle=False);assert np.array_equal(q['window_id'],data['window_id']);score=q['score']
        vals.append(float(np.mean([average_precision_score(y[region==r],score[region==r]) for r in np.unique(region)])))
        eligiblevals.append(float(average_precision_score(y[eligible],score[eligible])))
       rec.append({'backbone':arch,'view':view,'arm':arm,'macro_AP_seeds':vals,'macro_AP_mean':float(np.mean(vals)),'macro_AP_std':float(np.std(vals,ddof=1)),'eligible_pooled_AP_seeds':eligiblevals,'eligible_pooled_AP_mean':float(np.mean(eligiblevals)),'eligible_windows':800,'all_windows':1680})
    (O/'analysis'/'stress_metrics.json').write_text(json.dumps(rec,indent=2),encoding='utf8')
    def get(b,v,a):return next(x for x in rec if (x['backbone'],x['view'],x['arm'])==(b,v,a))
    fig,axes=plt.subplots(2,2,figsize=(10.4,7.2),layout='constrained')
    for ax,g,title in zip(axes.flat[:3],['position','speed','combined'],['(a) Neighbor-position stress','(b) Neighbor-speed stress','(c) Combined neighbor stress']):
        xs=[0,50,150,300,600] if g=='position' else [0,.25,.5,.75,1]
        for b in ['tcn','gru','transformer']:
            d=np.array([np.array(get(b,v,'F1')['macro_AP_seeds'])-np.array(get(b,v,'F0')['macro_AP_seeds']) for v in groups[g]])*100;m=d.mean(1);s=d.std(1,ddof=1)
            ax.plot(xs,m,color=COLORS[b],marker='o',ms=4,label=b.upper());ax.fill_between(xs,m-s,m+s,color=COLORS[b],alpha=.12)
        ax.axhline(0,color='#59666d',ls='--',lw=.8);ax.set_title(title,loc='left');ax.set_ylabel('Reliable − distance-only LOO field AP (pp)');ax.set_xlabel('Added position offset (m)' if g=='position' else 'Speed-perturbation strength multiplier');ax.legend(fontsize=8)
    ax=axes.flat[3]
    for b in ['tcn','gru','transformer']:
     for arm,ls,mark in [('F0','--','o'),('F1','-','s')]:
      a=next(x for x in ag if x['backbone']==b and x['arm']==arm);ks=['0.001','0.002','0.005','0.01','0.02','0.05'];ax.plot([a['FAR_'+k]['mean']*100 for k in ks],[a['Recall_'+k]['mean'] for k in ks],color=COLORS[b],ls=ls,marker=mark,ms=3.5,label=b.upper()+' '+arm)
    ax.set_title('(d) Validation-frozen operating points',loc='left');ax.set_xlabel('Actual test FAR (%)');ax.set_ylabel('Recall');ax.set_ylim(0,1);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Reliability in the local affine field under observation stress',fontsize=14)
    fig.supxlabel('All 1,680 windows retained; 800 eligible for added stress. Bands: SD of three paired seed differences.',fontsize=9)
    save(fig,'fig4_kinematic_loo_field_robustness')
    rows=[]
    for b in ARCH:
        cells=[]
        for g in ['position','speed','combined']:
            v=groups[g][-1];d=np.array(get(b,v,'F1')['macro_AP_seeds'])-np.array(get(b,v,'F0')['macro_AP_seeds']);cells.append(f'{d.mean()*100:+.3f} ± {d.std(ddof=1)*100:.3f}')
        rows.append([b]+cells)
    (R/'tables'/'Table4_field_stress_summary.md').write_text('# 场模型邻船压力配对\n\n每格为F1−F0区域等权AP百分点差的三种子均值±SD。全部1680窗保留；800具备预定扰动条件。正负结果全部列出。\n\n'+table(['骨干','位置600m','最高速度压力','最高复合压力'],rows)+'\n\neligible来源条件下pooled AP见field_extension/analysis/stress_metrics.json，与全体区域等权AP分别记录。',encoding='utf8')
if __name__=='__main__':main()
