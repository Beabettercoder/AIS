"""Render scientific benchmark geography strictly from verified selection assets.

Normal mode requires a verified_build report plus selected_windows.parquet. The
optional --preview mode labels its output as a layout check and never names that
output a final benchmark map. No coordinates or sample counts are invented here.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import FuncFormatter, MaxNLocator

COLORS={'North America':'#247BA0','Europe':'#B45F32','Asia':'#5D659C'}
CONTINENTS=('North America','Europe','Asia')
SEA='#EDF5F8';LAND='#D7DCDE';COAST='#ABB6BB';INK='#253441'

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def polygon_rings(obj):
 if obj.get('type')=='FeatureCollection':
  for f in obj['features']:yield from polygon_rings(f['geometry'])
 elif obj.get('type')=='Feature':yield from polygon_rings(obj['geometry'])
 elif obj.get('type')=='GeometryCollection':
  for g in obj['geometries']:yield from polygon_rings(g)
 elif obj.get('type')=='Polygon':yield np.array(obj['coordinates'][0])[:,:2]
 elif obj.get('type')=='MultiPolygon':
  for p in obj['coordinates']:yield np.array(p[0])[:,:2]
 else:raise ValueError('Unsupported land geometry: '+str(obj.get('type')))

def degree(value,axis):
 if abs(value)<1e-8:return '0°'
 direction=('E' if value>0 else 'W') if axis=='lon' else ('N' if value>0 else 'S')
 magnitude=abs(value)
 text=f'{magnitude:g}' if abs(magnitude-round(magnitude))>1e-8 else str(int(round(magnitude)))
 return text+'°'+direction

def draw_land(ax,rings,bounds,world=False):
 ax.set_facecolor(SEA)
 relevant=[r for r in rings if r[:,0].max()>=bounds[0] and r[:,0].min()<=bounds[2] and r[:,1].max()>=bounds[1] and r[:,1].min()<=bounds[3]]
 patches=[Polygon(r,closed=True) for r in relevant]
 collection=PatchCollection(patches,facecolor=LAND,edgecolor=COAST,linewidth=.35,zorder=1)
 ax.add_collection(collection)
 ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3])
 ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:degree(x,'lon')))
 ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:degree(y,'lat')))
 ax.xaxis.set_major_locator(MaxNLocator(5 if world else 4))
 ax.yaxis.set_major_locator(MaxNLocator(4))
 ax.grid(color='#FFFFFF',lw=.7,alpha=.8,zorder=0)
 ax.tick_params(labelsize=8,length=2,color='#A0ACB3',labelcolor='#576771')
 for spine in ax.spines.values():spine.set_color('#C4CFD4');spine.set_linewidth(.65)
 if world:ax.set_aspect('equal',adjustable='box')
 else:ax.set_aspect(1/max(.2,np.cos(np.deg2rad((bounds[1]+bounds[3])/2))),adjustable='box')

def nice_extent(regions):
 boxes=np.array([r['core_bbox'] for r in regions])
 lo=boxes[:,:2].min(0);hi=boxes[:,2:].max(0)
 if len(regions)==1:
  pad=np.maximum((hi-lo)*.65,[.45,.4])
 else:pad=np.maximum((hi-lo)*.17,[.7,.65])
 return [float(lo[0]-pad[0]),float(lo[1]-pad[1]),float(hi[0]+pad[0]),float(hi[1]+pad[1])]

def draw_region(ax,region,number,world=False,endpoints=None):
 x0,y0,x1,y1=region['core_bbox'];color=COLORS[region['continent']]
 ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=color,alpha=.18,zorder=3))
 ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='none',edgecolor=color,lw=.9 if world else 1.25,zorder=4))
 x,y=(x0+x1)/2,(y0+y1)/2
 if world:
  ax.scatter([x],[y],s=27,facecolors=color,edgecolors='white',linewidth=.65,zorder=6)
 else:
  if endpoints is not None:
   ax.scatter(endpoints[:,0],endpoints[:,1],s=1.1,color=color,alpha=.25,edgecolors='none',rasterized=True,zorder=2)
  ax.scatter([x],[y],s=21,color=color,edgecolors='white',lw=.5,zorder=6)
  # Only label placement is customized; geometry and counts always come from assets.
  placements={'houston':(9,-14,'left'),'ny':(9,9,'left'),'la':(9,-14,'left'),'puget':(9,9,'left'),
              'haugesund':(9,11,'left'),'great_belt':(-10,-17,'right'),'oresund':(9,-6,'left'),'kattegat':(10,9,'left'),'shanghai':(0,20,'center')}
  dx,dy,ha=placements.get(region['key'],(8,9,'left'))
  ax.annotate(f'{number}  {region["display_name"]}',xy=(x,y),xytext=(dx,dy),textcoords='offset points',
              ha=ha,va='center',fontsize=8.3,color=INK,zorder=8,
              bbox=dict(boxstyle='round,pad=.24',facecolor='white',edgecolor='none',alpha=.94),
              arrowprops=dict(arrowstyle='-',color=color,lw=.7,shrinkA=2,shrinkB=4))

def load_assets(args):
 report_path=args.selection/'selection_report.json' if args.selection.is_dir() else args.selection
 report=json.loads(report_path.read_text()); registry=json.loads(args.registry.read_text())
 specs=registry.get('regions',registry);manifest=report_path.parent/'selected_windows.parquet'
 expected_status='verified_dryrun' if args.preview else 'verified_build'
 if args.preview:
  assert report.get('status') in ('verified_dryrun','verified_build'),'Preview requires an actual verified capacity/selection report.'
 else:
  assert report.get('status')==expected_status,'Final map requires a verified_build selection report.'
  assert manifest.exists(),'Final map requires selected_windows.parquet.'
  h=report.get('output_hashes',{}).get('selected_windows.parquet')
  if h:assert digest(manifest)==h,'Selected manifest hash differs from the verified report.'
 names=report['regions'];assert len(names)==len(set(names)) and all(r in specs for r in names)
 entries=[];endpoints={}
 selected=pd.read_parquet(manifest) if not args.preview else None
 if selected is not None:
  assert set(selected.region)==set(names)
  assert not selected.window_id.duplicated().any()
 for key in names:
  meta=dict(specs[key]);meta['key']=key
  assert meta['continent'] in COLORS and len(meta['core_bbox'])==4
  x0,y0,x1,y1=map(float,meta['core_bbox']);assert -180<=x0<x1<=180 and -90<=y0<y1<=90
  if selected is None:
   counts=report['validation']['regions'][key]
   ships=sum(v['ships'] for v in counts.values());windows=sum(v['windows'] for v in counts.values())
  else:
   g=selected[selected.region==key]
   ships=int(g.mmsi.nunique());windows=len(g)
   assert ships==report['requested_ships_per_region'] and windows==report['requested_windows_per_region']
   source=report['source_files'][key]
   assert source['canonical_sha256']==meta['sha256'],'Registry and selection use different canonical sources: '+key
   canonical=Path(meta['canonical_file'])
   assert digest(canonical)==meta['sha256'],'Canonical file has changed: '+key
   raw=pd.read_parquet(canonical,columns=['mmsi','timestamp','lon','lat'])
   ix=np.array([json.loads(v)[-1] for v in g.row_indices],dtype=np.int64)
   lon=raw.lon.to_numpy()[ix];lat=raw.lat.to_numpy()[ix]
   assert np.array_equal(raw.mmsi.to_numpy()[ix],g.mmsi.to_numpy())
   assert np.array_equal(raw.timestamp.to_numpy()[ix],g.end_utc_s.to_numpy())
   assert np.all((lon>=x0-1e-7)&(lon<=x1+1e-7)&(lat>=y0-1e-7)&(lat<=y1+1e-7)), 'Selected target endpoint outside registry core: '+key
   endpoints[key]=np.column_stack((lon,lat))
  meta.update(selected_ships=int(ships),selected_windows=int(windows));entries.append(meta)
 # Keep the formal selection's order inside each geographic panel.
 entries=sorted(entries,key=lambda r:(CONTINENTS.index(r['continent']),names.index(r['key'])))
 for i,r in enumerate(entries,1):r['number']=i
 proof={'mode':'layout_preview' if args.preview else 'verified_manifest_map','selection_report':str(report_path),'selection_report_sha256':digest(report_path),
        'selected_windows_file':str(manifest) if not args.preview else None,'selected_windows_sha256':digest(manifest) if not args.preview else None,
        'source_registry':str(args.registry),'source_registry_sha256':digest(args.registry),'land_file':str(args.land),'land_sha256':digest(args.land),
        'core_endpoint_check':'all selected target endpoints verified against core bbox and canonical source hash' if not args.preview else 'not performed in layout preview',
        'regions':[{k:r[k] for k in ('key','number','display_name','continent','core_bbox','selected_ships','selected_windows')} for r in entries]}
 return entries,endpoints,proof

def render(args):
 entries,endpoints,proof=load_assets(args)
 rings=list(polygon_rings(json.loads(args.land.read_text())))
 detail_rings=list(polygon_rings(json.loads(args.detail_land.read_text()))) if args.detail_land else rings
 if args.detail_land:proof.update(detail_land_file=str(args.detail_land),detail_land_sha256=digest(args.detail_land))
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'svg.fonttype':'none','savefig.facecolor':'white','axes.unicode_minus':False})
 fig=plt.figure(figsize=(17.2,10.0),facecolor='white')
 gs=fig.add_gridspec(2,12,left=.055,right=.975,bottom=.105,top=.875,hspace=.34,wspace=.65,height_ratios=[1.12,1.0])
 world=fig.add_subplot(gs[0,:9]);keyax=fig.add_subplot(gs[0,9:]);keyax.axis('off')
 draw_land(world,rings,[-180,-60,180,85],world=True)
 world.set_title('(a) Geographic coverage',loc='left',fontweight='normal',pad=9)
 for r in entries:draw_region(world,r,r['number'],world=True)
 for continent in CONTINENTS:
  group=[r for r in entries if r['continent']==continent]
  if not group:continue
  center=np.mean([[(r['core_bbox'][0]+r['core_bbox'][2])/2,(r['core_bbox'][1]+r['core_bbox'][3])/2] for r in group],axis=0)
  pos={'North America':(-145,7),'Europe':(28,73),'Asia':(139,9)}[continent]
  nums=str(group[0]['number']) if len(group)==1 else f'{group[0]["number"]}–{group[-1]["number"]}'
  world.annotate(f'{continent} ({nums})',xy=center,xytext=pos,color=COLORS[continent],fontsize=9.5,ha='center',va='center',
                 bbox=dict(boxstyle='round,pad=.28',facecolor='white',edgecolor='none',alpha=.95),
                 arrowprops=dict(arrowstyle='-',color=COLORS[continent],lw=.85),zorder=10)
 y=.98
 keyax.text(0,y,'Selected study regions',fontsize=11,fontweight='normal',va='top',color=INK,transform=keyax.transAxes);y-=.105
 for continent in CONTINENTS:
  group=[r for r in entries if r['continent']==continent]
  if not group:continue
  keyax.text(0,y,continent.upper(),fontsize=8.5,fontweight='bold',color=COLORS[continent],va='top',transform=keyax.transAxes);y-=.06
  for r in group:
   keyax.text(.015,y,f'{r["number"]:02d}  {r["display_name"]}',fontsize=9.3,color=INK,va='top',transform=keyax.transAxes);y-=.064
  y-=.025
 keyax.add_patch(Rectangle((0,max(.035,y-.03)),.065,.023,facecolor='#627987',alpha=.25,edgecolor='#627987',transform=keyax.transAxes))
 keyax.text(.09,max(.035,y-.03)+.011,'Core study boundary',fontsize=8.5,va='center',color='#526572',transform=keyax.transAxes)
 panels=[]
 for j,continent in enumerate(CONTINENTS):
  group=[r for r in entries if r['continent']==continent]
  ax=fig.add_subplot(gs[1,j*4:(j+1)*4]);panels.append(ax)
  if not group:
   ax.axis('off');ax.text(.5,.5,'No selected region',ha='center',va='center',transform=ax.transAxes,color='#687983');continue
  bounds=nice_extent(group);draw_land(ax,detail_rings,bounds)
  ax.set_title(f'({chr(98+j)}) {continent}',loc='left',fontweight='normal',pad=10)
  for r in group:draw_region(ax,r,r['number'],endpoints=endpoints.get(r['key']))
  ax.set_xlabel('Longitude',fontsize=8.5,labelpad=6,color='#526572');ax.set_ylabel('Latitude',fontsize=8.5,labelpad=4,color='#526572')
 ships=sorted({r['selected_ships'] for r in entries});windows=sorted({r['selected_windows'] for r in entries})
 shiptext=f'{ships[0]:,}' if len(ships)==1 else f'{ships[0]:,}–{ships[-1]:,}'
 wintext=f'{windows[0]:,}' if len(windows)==1 else f'{windows[0]:,}–{windows[-1]:,}'
 title='Balanced AIS source-window benchmark' if not args.preview else 'Geography layout preview — verified capacity only'
 fig.text(.055,.962,title,fontsize=19,fontweight='normal',ha='left',va='top',color=INK)
 subtitle=f'{len(entries)} regions  |  {shiptext} target vessels and {wintext} independent source windows per region'
 if args.preview:subtitle+='  |  Final selection manifest required'
 fig.text(.055,.924,subtitle,fontsize=10.4,ha='left',va='top',color='#526572')
 foot='Boxes show registered core boundaries. Dots in detailed maps show selected window endpoints. Vessel counts use unique MMSI.'
 if args.preview:foot='LAYOUT CHECK ONLY. Region geometry comes from the source registry; no final selection is claimed.'
 fig.text(.055,.045,foot,fontsize=8.3,color='#576771',ha='left',va='center')
 land_note='Natural Earth, 1:110m overview / 1:10m detailed panels' if args.detail_land else 'Natural Earth, 1:110m'
 fig.text(.055,.024,f'Coordinates: WGS84. Land: {land_note}. Each detailed panel has its own geographic scale.',fontsize=8.0,color='#6A7A84',ha='left',va='center')
 args.out.mkdir(parents=True,exist_ok=True)
 prefix='layout_preview' if args.preview else args.prefix
 png=args.out/(prefix+'.png');svg=args.out/(prefix+'.svg')
 fig.savefig(png,dpi=300);fig.savefig(svg);plt.close(fig)
 proof.update(output_png=str(png),output_svg=str(svg),dpi=300,figsize_inches=[17.2,10.0],title=title,subtitle=subtitle)
 (args.out/(prefix+'_metadata.json')).write_text(json.dumps(proof,indent=2,ensure_ascii=False))
 print(json.dumps({'mode':proof['mode'],'regions':len(entries),'png':str(png),'svg':str(svg),'subtitle':subtitle},ensure_ascii=False),flush=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--selection',type=Path,required=True,help='Selection directory or selection_report.json')
 ap.add_argument('--registry',type=Path,required=True)
 ap.add_argument('--land',type=Path,default=Path('/mnt/sda/WangZijia/AIS/data/geo/ne_110m_land.geojson'))
 ap.add_argument('--detail-land',type=Path,help='Optional Natural Earth 10m land GeoJSON for the three detailed maps')
 ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prefix',default='benchmark_geography')
 ap.add_argument('--preview',action='store_true',help='Explicitly labelled smoke-test image from an actual verified dryrun report')
 args=ap.parse_args();render(args)

if __name__=='__main__':main()
