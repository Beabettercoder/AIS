"""Plot actual pilot outputs; no interpolation or invented uncertainty bars."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;REL=BASE/'labeled_v1'
for p in (ROOT/'visuals/v2/fonts_runtime').glob('times*.ttf'):font_manager.fontManager.addfont(str(p))
font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8.5,'axes.titlesize':10,'svg.fonttype':'path','axes.unicode_minus':False})
PILOT=REL/'pilot_scene_seed0';summary=json.loads((PILOT/'pilot_summary.json').read_text());assert summary['status']=='completed_single_seed_pilot'
features=['raw','plain_context','plain_affine_context','context'];names=['Raw trajectory','Ordinary context','Unweighted affine','Reliable context'];colors=['#657A86','#DFA157','#9A8EBD','#0F6C5C'];backbones=['tcn','gru','transformer']
dest=REL/'figures';dest.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(180/25.4,78/25.4));x=np.arange(3);w=.18
for j,f in enumerate(features):
 y=[next(r['test']['macro']['AP'] for r in summary['results'] if r['backbone']==b and r['feature_group']==f) for b in backbones]
 bars=ax.bar(x+(j-1.5)*w,y,width=w*.93,label=names[j],color=colors[j],edgecolor='white',linewidth=.4)
 for bar,value in zip(bars,y):ax.text(bar.get_x()+bar.get_width()/2,value+.012,f'{value:.3f}',ha='center',fontsize=7.2)
ax.set_xticks(x,['TCN','GRU','Transformer']);ax.set_ylim(0,1);ax.set_ylabel('Region-macro AP');ax.set_title('Paired pipeline pilot: seed 0, 8 epochs',loc='left')
ax.grid(axis='y',alpha=.2,zorder=0);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
ax.legend(ncol=2,fontsize=7.7,loc='upper right',frameon=False);fig.tight_layout()
for ext in ['png','svg']:fig.savefig(dest/f'pilot_backbone_comparison_seed0.{ext}',dpi=400)
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(180/25.4,68/25.4),sharey=True)
for ax,b in zip(axes,backbones):
 for f,label,color in zip(features,names,colors):
  r=json.loads((PILOT/f'{b}__{f}/metrics.json').read_text());h=r['history']
  ax.plot([v['epoch'] for v in h],[v['validation_macro_AP'] for v in h],color=color,lw=1.15,label=label)
 ax.set_title(b.upper() if b!='transformer' else 'Transformer');ax.set_xlabel('Epoch');ax.set_xticks([1,4,8]);ax.set_ylim(.35,.85);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Validation macro AP');fig.legend(names,ncol=4,loc='lower center',frameon=False,fontsize=7.5)
fig.suptitle('Validation learning curves (pipeline pilot)',x=.08,ha='left',fontsize=10.5)
fig.tight_layout(rect=(0,.13,1,.92))
for ext in ['png','svg']:fig.savefig(dest/f'pilot_validation_curves_seed0.{ext}',dpi=400)
plt.close(fig)
proof={'pilot_summary_sha256':hashlib.sha256((PILOT/'pilot_summary.json').read_bytes()).hexdigest(),'source':'Actual saved metrics and validation histories','scope':'One fixed seed / 8-epoch paired pipeline pilot; no error bars inferred','font':'Times New Roman','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('pilot_*') if p.is_file()}}
(dest/'pilot_figures_provenance.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2),flush=True)
