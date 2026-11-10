from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,findfont, fontManager
fontManager.addfont("/home/ps/.local/share/fonts/TimesNewRoman/TimesNewRoman.ttf")
R=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/relation');d=json.loads((R/'summary.json').read_text())['results']['test'];names=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];labels=['TCN','GRU','Transformer','iTransformer','PatchTST','Mamba','Neural-CDE','GAT–TCN'];views=[('base','(a) Base'),('natural','(b) Natural speed stress'),('suffix','(c) Suffix speed stress'),('combined_l100_m600','(d) Speed + position stress')]
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':8,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12.2,7.0),layout='constrained');x=np.arange(len(names));w=.36
for ax,(view,title) in zip(axes.flat,views):
 for k,arm,col,hatch in [(-.5,'F0','#376b87','//'),(.5,'F1','#c7793c','\\\\')]:
  z=[d[n]['views'][view]['arms'][arm] for n in names];m=[i['pooled_AP_mean'] for i in z];sd=[i['pooled_AP_sd'] for i in z]
  ax.bar(x+k*w,m,w,color=col,edgecolor='#26363a',linewidth=.6,hatch=hatch,label='Ordinary field' if arm=='F0' else 'Reliability-aware field',yerr=sd,error_kw={'ecolor':'#26363a','lw':.7,'capsize':2},zorder=3)
 ax.set_title(title,loc='left');ax.set_ylim(0,1.02);ax.set_ylabel('Average precision');ax.set_xticks(x,labels,rotation=32,ha='right');ax.grid(axis='y',alpha=.25,linestyle=':',zorder=0)
handles,labels_=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels_,loc='outside upper center',ncol=2,frameon=False)
for ext in ('pdf','png','svg'):fig.savefig(R/f'fig_relation_backbones_tnr.{ext}',dpi=450,bbox_inches='tight')
print('Font:',findfont(FontProperties(family='Times New Roman')))
