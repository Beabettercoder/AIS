import json,matplotlib
matplotlib.use('Agg')
from pathlib import Path
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');j=json.loads((P/'single_formal_summary.json').read_text())['models'];pairs=json.loads((P/'single_pair_decomposition.json').read_text())['paired'];order=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];names=['TCN','GRU','Transformer','iTransformer','PatchTST','Mamba','Neural-CDE','GAT–TCN*'];arms=['raw','causal_bundle','f0_bundle','f1_bundle'];label=['Raw','Causal + bundle','Ordinary field (F0)','Trusted field (F1)'];colors=['#BAC6C3','#779EA5','#3B8DA1','#0F6C5C'];font='/home/ps/.local/share/fonts/TimesNewRoman/TimesNewRoman.ttf';font_manager.fontManager.addfont(font);family=font_manager.FontProperties(fname=font).get_name();assert family=='Times New Roman';plt.rcParams.update({'font.family':family,'font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False})
fig,ax=plt.subplots(1,4,figsize=(17.5,6.3),sharey=True,gridspec_kw={'width_ratios':[1.5,1.08,1.08,1.08]});fig.subplots_adjust(left=.095,right=.985,top=.84,bottom=.19,wspace=.19);y=np.arange(8);height=.17
for i,(a,lab,c) in enumerate(zip(arms,label,colors)):
 v=np.array([j[n]['test']['performance'][a]['macro_AP']['mean'] for n in order]);sd=np.array([j[n]['test']['performance'][a]['macro_AP']['sd'] for n in order]);ax[0].barh(y+(i-1.5)*height,v,height,xerr=sd,color=c,edgecolor='white',linewidth=.3,label=lab,error_kw={'capsize':1.7,'lw':.65})
ax[0].set_yticks(y,names);ax[0].set_ylim(7.6,-.6);ax[0].set_xlim(0,1);ax[0].set_xlabel('Region-macro AP (mean ± seed SD)');ax[0].set_title('(a) Four configurations',loc='left')
for k,key,title in [(1,'causal_minus_raw','(b) (Causal + bundle) − Raw'),(2,'f0_minus_causal','(c) F0 − (Causal + bundle)'),(3,'f1_minus_f0','(d) F1 − F0')]:
 d=[pairs[n][key]['pooled'] for n in order];mu=np.array([z['mean']*100 for z in d]);lo=np.array([z['CI95'][0]*100 for z in d]);hi=np.array([z['CI95'][1]*100 for z in d]);ax[k].axvline(0,color='#899693',lw=.9);ax[k].errorbar(mu,y,xerr=np.vstack((mu-lo,hi-mu)),fmt='o',color='#0F6C5C',ecolor='#0F6C5C',capsize=2.5,lw=1.1,ms=4);ax[k].set_xlabel('Pooled AP difference (pp)');ax[k].set_title(title,loc='left');ax[k].tick_params(axis='y',length=0)
for a in ax:
 a.grid(axis='x',alpha=.17);a.set_axisbelow(True)
 for side in ('top','right'):a.spines[side].set_visible(False)
handles,labels=ax[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,.94),ncol=4,frameon=False,fontsize=10)
fig.suptitle('Single-agent benchmark: four-arm paired comparisons',fontsize=15,fontweight='bold',y=.99)
fig.text(.5,.055,'Previously exposed test · 3 fixed seeds · 95% target-MMSI bootstrap · *target-only graph adaptation',ha='center',fontsize=9,color='#555555')
(P/'figures').mkdir(exist_ok=True)
for ext in ('png','pdf','svg'):fig.savefig(P/'figures'/('single_four_arm_paired.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
