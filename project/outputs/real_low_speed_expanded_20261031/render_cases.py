#!/usr/bin/env python3
"""Predetermined hash-selected source-only candidate examples for inspection."""
from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path('/mnt/sda/WangZijia/AIS');OUT=ROOT/'outputs/real_low_speed_expanded_20261031'
df=pd.read_csv(OUT/'review_manifest.csv',dtype={'mmsi':str});df=df[df.heading_usable].copy();df['selection']=df.key.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
choices=pd.concat([df[df['group']==g].sort_values('selection').head(2) for g in ['turn_rule','straight_control']]);assert len(choices)==4
font_manager.fontManager.addfont('/home/ps/.local/share/fonts/TimesNewRoman/TimesNewRoman.ttf');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/home/ps/.local/share/fonts/TimesNewRoman/TimesNewRoman.ttf').get_name(),'font.size':9})
fig,axes=plt.subplots(4,3,figsize=(12.2,11))
ids=[]
for j,(_,r) in enumerate(choices.iterrows()):
 times=json.loads(r.timestamps);z=pd.read_parquet(r.source_file,columns=['mmsi','timestamp','lon','lat','sog','cog','heading'],filters=[('mmsi','==',int(r.mmsi))]);z['order']=np.arange(len(z));z=z.sort_values(['timestamp','order'],kind='stable').drop_duplicates('timestamp',keep='last').set_index('timestamp');z=z.loc[times]
 x=(z.lon.to_numpy()-z.lon.iloc[0])*111195*np.cos(np.deg2rad(z.lat.mean()));y=(z.lat.to_numpy()-z.lat.iloc[0])*111195;t=(np.array(times)-times[0])/60;h=z.heading.to_numpy(float).copy();h[(h<0)|(h>=360)]=np.nan
 ax=axes[j,0];ax.plot(x,y,color='#0F6C5C',lw=1.4);pts=ax.scatter(x,y,c=t,cmap='viridis',s=15);ax.scatter(x[0],y[0],c='#D88742',marker='s',s=30);ax.set(xlabel='East (m)',ylabel='North (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.18);ax.set_title(f'{r.region}  ·  {r.group}',fontsize=10)
 ax=axes[j,1];ax.scatter(t,z.cog,s=11,label='COG',color='#1C6D90');ax.scatter(t,h,s=10,label='Heading',color='#D08043');ax.set(xlabel='Time (min)',ylabel='Direction (deg)',ylim=(-8,368));ax.grid(alpha=.18)
 if j==0:ax.legend(frameon=False,fontsize=8,loc='upper right')
 ax=axes[j,2];ax.plot(t,z.sog,color='#5E6D9E',lw=1.5);ax.axhline(2,color='grey',ls='--',lw=.7);ax.set(xlabel='Time (min)',ylabel='SOG (kn)');ax.grid(alpha=.18)
 ids.append({'window_id':r.window_id,'mmsi':str(r.mmsi),'region':r.region,'group':r['group'],'selection_sha256':r.selection})
fig.suptitle('Observed training-set trajectories selected by fixed COG rules',fontsize=13,y=.995);fig.tight_layout();fig.savefig(OUT/'real_low_speed_cases.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'real_low_speed_cases.pdf',bbox_inches='tight');plt.close(fig)
(OUT/'case_selection.json').write_text(json.dumps({'selection':'Two smallest SHA256(key) per group among rule-selected windows with usable Heading; all original AIS reports, train MMSI only','cases':ids},indent=2),encoding='utf8');print(json.dumps(ids))
