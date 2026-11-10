"""Render validation-only fixed-checkpoint sensitivity, Fig. 5 and Table S5.

Requires the existing AIS .venv matplotlib; does not import torch or read test.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ('fig1_framework.font_regular.ttf','fig1_framework.font_bold.ttf'):
        font_manager.fontManager.addfont(str(FIG/name))
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.labelsize':11,'axes.titlesize':12,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.color':'#e4e7e9','grid.linewidth':.6,'legend.fontsize':10})
    path=ROOT/'sensitivity_results/results.json';results=json.loads(path.read_text());assert results['status']=='complete'
    manifest=json.loads((ROOT/'sensitivity_results/evaluation_manifest.json').read_text())
    assert manifest['no_hyperparameter_selection'] and manifest['no_test_data_or_scores'] and manifest['data_split']=='val'
    rows=results['rows'];aggregates=[]
    keys=sorted({(r['family'],r['view'],r['architecture']) for r in rows})
    for family,view,arch in keys:
        group=[r for r in rows if (r['family'],r['view'],r['architecture'])==(family,view,arch)]
        assert sorted(r['seed'] for r in group)==[0,1,2]
        aggregates.append({'family':family,'view':view,'architecture':arch,'value':group[0]['value'],
                           'macro_AP_mean':float(np.mean([r['macro_AP'] for r in group])),
                           'macro_AP_sample_sd':float(np.std([r['macro_AP'] for r in group],ddof=1)),
                           'reference_valid_fraction':group[0]['reference_valid_fraction'],
                           'candidate_count_mean_valid_target_steps':group[0]['candidate_count_mean_valid_target_steps']})
    colors={'tcn':'#356c9b','gru':'#8e5a97','transformer':'#238576','graph_temporal':'#384955'}
    plotted=[r for r in aggregates if r['family']!='baseline']
    low=max(0.,np.floor((min(r['macro_AP_mean']-r['macro_AP_sample_sd'] for r in plotted)-.02)*100)/100)
    high=min(1.,np.ceil((max(r['macro_AP_mean']+r['macro_AP_sample_sd'] for r in plotted)+.02)*100)/100)
    labels={'tcn':'TCN','gru':'GRU','transformer':'Transformer','graph_temporal':'Graph temporal (two-node GAT)'}
    panels=[('temperature','Graph attention temperature','Temperature τ',1.),('topk','Neighbor budget','Top K',None),
            ('radius','Neighborhood radius','Radius (nm)',8.),('heading','Heading difference','Maximum difference (degrees)',45.),
            ('e0','Compatibility threshold','e₀ (m/s)',2.5),('sigma','Reliability decay scale','σₑ (m/s)',2.)]
    fig,axes=plt.subplots(2,3,figsize=(13.5,8.2));fig.subplots_adjust(left=.075,right=.98,bottom=.18,top=.86,hspace=.48,wspace=.29)
    for j,(family,title,xlabel,default) in enumerate(panels):
        ax=axes.flat[j];arches=['graph_temporal'] if family=='temperature' else ['tcn','gru','transformer']
        for arch in arches:
            group=sorted([r for r in aggregates if r['family']==family and r['architecture']==arch],key=lambda r:float(r['value']))
            assert group
            xx=[float(r['value']) for r in group];yy=[r['macro_AP_mean'] for r in group];ee=[r['macro_AP_sample_sd'] for r in group]
            ax.errorbar(xx,yy,yerr=ee,color=colors[arch],lw=1.4,marker='o',ms=4,capsize=3,label=labels[arch])
            if family=='topk':
                baseline=next(r for r in aggregates if r['family']=='baseline' and r['architecture']==arch)
                ax.axhline(baseline['macro_AP_mean'],color=colors[arch],ls=':',lw=1,alpha=.7)
        if default is not None:ax.axvline(default,color='#8b9299',ls='--',lw=.9,zorder=0)
        else:ax.text(.03,.03,'Dotted: frozen all-member baseline',transform=ax.transAxes,fontsize=8.6,color='#515b64')
        if family=='temperature':
            ax.set_xscale('log');ax.set_xticks([.05,.1,.2,.5,1,2],labels=['0.05','0.1','0.2','0.5','1','2'])
            ax.text(.03,.03,'GAT logits only; not median aggregation',transform=ax.transAxes,fontsize=8.4,color='#515b64')
        elif family=='heading':ax.set_xticks([30,45]);ax.set_xlim(28,47)
        elif family=='topk':ax.set_xticks([4,8,12,16,24,32])
        ax.set_title(f'({chr(97+j)}) {title}',pad=9);ax.set_xlabel(xlabel)
        if j%3==0:ax.set_ylabel('Validation region-macro AP')
        ax.set_ylim(low,high)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=colors[a],marker='o',lw=1.4,label=labels[a]) for a in ('tcn','gru','transformer','graph_temporal')]
    handles.append(Line2D([],[],color='#8b9299',ls='--',label='Frozen default'))
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.52,.945),frameon=False)
    fig.suptitle('Fixed-checkpoint validation sensitivity',fontsize=18,y=.985)
    fig.text(.075,.100,'Points: mean of three trained seeds; error bars: sample standard deviation. Weights and training normalization remain fixed; no retraining or selection.',fontsize=10)
    fig.text(.075,.071,'Frozen defaults: τ = 1 (graph module only), all members, R = 8 nm, heading = 45°, e₀ = 2.5 m/s, σₑ = 2 m/s. Top K has no finite frozen default.',fontsize=10)
    fig.text(.075,.042,'Radius and heading only filter the archived candidate pool (≤ 8 nm and ≤ 45°). No test scores enter this figure; these are inference interventions.',fontsize=10)
    fig.text(.075,.015,f'All six AP axes share the displayed truncated range [{low:.2f}, {high:.2f}]. Exact per-seed values and context coverage are retained in Table S5.',fontsize=9.5)
    for ext in ('pdf','png','svg'):fig.savefig(FIG/f'fig5_sensitivity.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    table=ROOT/'tables/table_s5_sensitivity_aggregate.csv'
    with table.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregates[0]));writer.writeheader();writer.writerows(aggregates)
    meta={'results_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'data_split':'validation only','fixed_checkpoint':True,'no_selection_or_retraining':True,'panels':[p[0] for p in panels],
          'shared_displayed_AP_axis':[float(low),float(high)],'axis_truncation_disclosed':True}
    (FIG/'fig5_sensitivity.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print('FIG5_RENDERED',flush=True)


if __name__=='__main__':main()
