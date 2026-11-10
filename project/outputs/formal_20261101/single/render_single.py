import json,matplotlib
matplotlib.use('Agg')
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
font_path='/home/ps/.local/share/fonts/TimesNewRoman/TimesNewRoman.ttf'
font_manager.fontManager.addfont(font_path)
font_name=font_manager.FontProperties(fname=font_path).get_name()
assert font_name == 'Times New Roman', font_name
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');j=json.loads((P/'single_formal_summary.json').read_text())['models'];order=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];labels=['TCN','GRU','Transformer','iTransformer','PatchTST','Mamba','Neural-CDE','GAT–TCN*']
plt.rcParams.update({'font.family':font_name,'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(1,3,figsize=(15,5.25),gridspec_kw={'width_ratios':[1.12,1,1]},layout='constrained')
y=np.arange(len(order));macro=lambda arm:np.array([j[n]['test']['performance'][arm]['macro_AP']['mean'] for n in order]);sd=lambda arm:np.array([j[n]['test']['performance'][arm]['macro_AP']['sd'] for n in order]);a=axs[0]
a.barh(y-.18,macro('raw'),.35,xerr=sd('raw'),color='#BAC6C3',edgecolor='white',error_kw={'capsize':2,'lw':.8},label='Raw');a.barh(y+.18,macro('f1_bundle'),.35,xerr=sd('f1_bundle'),color='#177462',edgecolor='white',error_kw={'capsize':2,'lw':.8},label='Full configuration');a.set_yticks(y,labels);a.invert_yaxis();a.set_xlim(0,1);a.set_xlabel('Region-macro AP (mean ± seed SD)');a.set_title('(a) Absolute performance',loc='left');a.legend(frameon=False,loc='lower right');a.grid(axis='x',alpha=.2);a.set_axisbelow(True)
for ax,other,title in [(axs[1],'raw','(b) Full − Raw'),(axs[2],'f0_bundle','(c) Full − ordinary field')]:
 d=[j[n]['test']['paired_pooled_AP']['f1_minus_'+other] for n in order];mu=np.array([x['mean']*100 for x in d]);lo=np.array([x['CI95'][0]*100 for x in d]);hi=np.array([x['CI95'][1]*100 for x in d]);ax.axvline(0,color='#8b9693',lw=.8);ax.errorbar(mu,y,xerr=np.vstack((mu-lo,hi-mu)),fmt='o',capsize=2.5,color='#177462',ecolor='#177462',lw=1.1,ms=4);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Pooled AP difference (percentage points)');ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
for ax in axs:
 for sp in ('top','right'):ax.spines[sp].set_visible(False)
fig.suptitle('Single-agent benchmark: paired backbone comparisons',fontsize=15,fontweight='bold')
fig.text(.5,-.035,'Previously exposed test · 3 fixed seeds · 95% target-MMSI bootstrap · *target-only graph adaptation',ha='center',fontsize=9,color='#555555')
(P/'figures').mkdir(exist_ok=True)
for ext in ('png','pdf','svg'):fig.savefig(P/'figures'/('single_backbone_paired.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
print('rendered',flush=True)
