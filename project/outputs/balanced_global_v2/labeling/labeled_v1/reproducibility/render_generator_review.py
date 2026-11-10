"""Fixed-hash qualitative examples from verified source/counterfactual pairs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;REL=BASE/'labeled_v1'
sys.path.insert(0,str(BASE/'generator'))
from maneuver_generator import generate_maneuver,PhysicalLimits
for p in (ROOT/'visuals/v2/fonts_runtime').glob('times*.ttf'):font_manager.fontManager.addfont(str(p))
font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':7.7,'axes.titlesize':8.5,'svg.fonttype':'path','axes.unicode_minus':False})
labels=pd.read_parquet(REL/'label_manifest.parquet');kinds=['turn_back','deviation','stall','sinuous','speed_drop'];pretty=['Turn-back','Deviation','Stall','Sinuous','Speed drop']
fig,axes=plt.subplots(5,3,figsize=(180/25.4,210/25.4));records=[];cache={}
for rownum,kind in enumerate(kinds):
 pool=labels[(labels.y==1)&(labels.split=='val')&(labels.latent_kind==kind)]
 order=pool.window_id.map(lambda w:hashlib.sha256(('example_v1:'+w).encode()).hexdigest());row=pool.loc[order.idxmin()]
 region=row.region
 if region not in cache:
  folder=BASE/'counterfactuals_v1'/region
  with np.load(folder/'counterfactual_observations.npz',allow_pickle=False) as z:cache[region]={k:z[k] for k in z.files}
  cache[region]['limits']=PhysicalLimits(**json.loads((folder/'counterfactual_report.json').read_text())['physical_limits'])
 z=cache[region];j=int(row.counterfactual_row);n=int(z['mask'][j].sum());t=z['t'][j,:n];p=z['original_pos'][j,:n];s=z['original_sog'][j,:n];c=z['original_cog'][j,:n]
 smooth={'pos':z['smooth_pos'][j,:n],'sog_kn':z['smooth_sog'][j,:n],'cog_deg':z['smooth_cog'][j,:n]}
 hard=generate_maneuver(t,p,s,c,kind=kind,onset_s=row.onset_utc_s,params=json.loads(row.counterfactual_applied_params),profile='hard_step',adapt=False,limits=z['limits']);assert hard['accepted']
 elapsed=(t-t[0])/60;onset=(row.onset_utc_s-t[0])/60
 for dat,label,color,style in [({'pos':p,'sog_kn':s,'cog_deg':c},'Source','#526D82','-'),(smooth,'Smooth','#0F6C5C','-'),(hard,'Matched step','#D7792C','--')]:
  xy=(dat['pos']-p[0])/1852;axes[rownum,0].plot(xy[:,0],xy[:,1],style,color=color,lw=.85,marker='o',ms=1.3,label=label)
  axes[rownum,1].plot(elapsed,dat['sog_kn'],style,color=color,lw=.85,marker='o',ms=1.2)
  offset=np.rad2deg(np.unwrap(np.deg2rad((dat['cog_deg']-c+180)%360-180)))
  axes[rownum,2].plot(elapsed,offset,style,color=color,lw=.85,marker='o',ms=1.2)
 for col,a in enumerate(axes[rownum]):
  a.grid(alpha=.18);a.spines[['top','right']].set_visible(False);a.tick_params(labelsize=7,length=2)
  if col>0:a.axvline(onset,color='#7C8E96',lw=.6,ls=':');a.set_xlabel('Elapsed time (min)')
 axes[rownum,0].set_aspect('equal',adjustable='datalim');axes[rownum,0].set_xlabel('East displacement (nm)');axes[rownum,0].set_ylabel('North (nm)')
 axes[rownum,0].set_title(f'({chr(97+3*rownum)}) {pretty[rownum]} | {region}',loc='left')
 axes[rownum,1].set_title(f'({chr(98+3*rownum)}) Reported speed',loc='left');axes[rownum,1].set_ylabel('SOG (kn)')
 axes[rownum,2].set_title(f'({chr(99+3*rownum)}) Course offset',loc='left');axes[rownum,2].set_ylabel('COG offset (deg)')
 records.append({'kind':kind,'region':region,'base_window_id':row.window_id,'mmsi':int(row.mmsi),'onset_s':row.onset_utc_s,'source_split':'val','selection':'minimum SHA256(example_v1:window_id), no model score used'})
axes[0,0].legend(loc='best',fontsize=6.6,frameon=False)
fig.suptitle('Controlled maneuvers preserving source observation innovations',x=.07,ha='left',fontsize=11)
fig.tight_layout(rect=(0,.025,1,.975),h_pad=.85,w_pad=.8)
fig.text(.07,.01,'Pre-noise examples. Points are actual AIS report times; dotted lines mark the prescribed onset.',fontsize=7.4,color='#526D82')
out=REL/'figures';out.mkdir(exist_ok=True)
for ext in ['png','svg']:fig.savefig(out/f'generator_matched_examples.{ext}',dpi=400)
plt.close(fig)
(out/'generator_examples_provenance.json').write_text(json.dumps({'examples':records,'label_manifest_sha256':hashlib.sha256((REL/'label_manifest.parquet').read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256((BASE/'generator/maneuver_generator.py').read_bytes()).hexdigest()},indent=2))
print('Saved 5 deterministic matched generator examples',flush=True)
