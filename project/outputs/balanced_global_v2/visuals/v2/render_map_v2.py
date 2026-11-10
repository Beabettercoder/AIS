"""Publication-sized Robinson locator and eight verified AIS track panels."""
from pathlib import Path
import sys,json,hashlib,argparse
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'vendor'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection,PatchCollection
from matplotlib.patches import Polygon,Rectangle,PathPatch
from matplotlib.path import Path as MPath
from matplotlib.ticker import FixedLocator,FuncFormatter
from pyproj import Proj,__version__ as pyproj_version

ROOT=Path('/mnt/sda/WangZijia/AIS')
REL=ROOT/'outputs/balanced_global_v2/release_20261026'
SEA='#DDEEF7';LAND='#F5F1E6';COAST='#7C929F';INK='#233746';GRID='#B6CCD7'
COLORS={'North America':'#2563A6','Europe':'#D7792C','Asia':'#16877A'}
ORDER=['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai']
PROJECTION=Proj('+proj=robin +lon_0=0 +R=6371008.8')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def polygons(obj):
 typ=obj.get('type')
 if typ=='FeatureCollection':
  for f in obj['features']:yield from polygons(f)
 elif typ=='Feature':yield from polygons(obj['geometry'])
 elif typ=='GeometryCollection':
  for g in obj['geometries']:yield from polygons(g)
 elif typ=='Polygon':yield [np.asarray(r,dtype=float)[:,:2] for r in obj['coordinates']]
 elif typ=='MultiPolygon':
  for p in obj['coordinates']:yield [np.asarray(r,dtype=float)[:,:2] for r in p]

def polygon_patch(rings,project=False):
 verts=[];codes=[]
 for j,ring in enumerate(rings):
  r=ring.copy()
  area=np.sum(r[:-1,0]*r[1:,1]-r[1:,0]*r[:-1,1])
  if (j==0 and area<0) or (j>0 and area>0):r=r[::-1]
  if project:r=np.column_stack(PROJECTION(r[:,0],r[:,1]))
  verts.extend(r);codes.extend([MPath.MOVETO]+[MPath.LINETO]*(len(r)-2)+[MPath.CLOSEPOLY])
 return PathPatch(MPath(np.asarray(verts),np.array(codes,dtype=np.uint8)))

def fmt(v,axis,decimals=1):
 if abs(v)<1e-7:return '0°'
 s=f'{abs(v):.{decimals}f}'.rstrip('0').rstrip('.') if decimals else f'{abs(v):.0f}'
 return s+'°'+(('E' if v>0 else 'W') if axis=='lon' else ('N' if v>0 else 'S'))

def world(ax,land,entries,standalone=False):
 lats=np.linspace(-60,85,360)
 left=np.column_stack(PROJECTION(np.full_like(lats,-180),lats))
 right=np.column_stack(PROJECTION(np.full_like(lats,180),lats))[::-1]
 boundary=np.vstack([left,right,left[:1]])
 ocean=Polygon(boundary,closed=True,facecolor=SEA,edgecolor='#A4BDCA',lw=.6,zorder=0)
 ax.add_patch(ocean)
 patches=[polygon_patch(r,True) for r in land]
 collection=PatchCollection(patches,facecolor=LAND,edgecolor=COAST,linewidth=.28,zorder=2)
 collection.set_clip_path(ocean);ax.add_collection(collection)
 for lat in [-40,0,40,80]:
  lon=np.linspace(-180,180,720);xx,yy=PROJECTION(lon,np.full_like(lon,lat));line,=ax.plot(xx,yy,color=GRID,lw=.38,alpha=.65,zorder=1);line.set_clip_path(ocean)
 for lon in [-120,-60,0,60,120]:
  xx,yy=PROJECTION(np.full_like(lats,lon),lats);line,=ax.plot(xx,yy,color=GRID,lw=.38,alpha=.65,zorder=1);line.set_clip_path(ocean)
 for ent in entries:
  box=ent['core_bbox'];lon=(box[0]+box[2])/2;lat=(box[1]+box[3])/2
  xx,yy=PROJECTION(lon,lat)
  ax.scatter([xx],[yy],s=22 if standalone else 18,facecolors=COLORS[ent['continent']],edgecolors='white',lw=.65,zorder=5)
 for continent in ['North America','Europe','Asia']:
  ee=[e for e in entries if e['continent']==continent];n=len(ee);nw=sum(e['windows'] for e in ee)
  pos={'North America':(-126,5),'Europe':(85,65),'Asia':(132,-3)}[continent]
  anchor={'North America':(-100,38),'Europe':(10.7,57.7),'Asia':(122,31)}[continent]
  point=PROJECTION(*anchor);label=PROJECTION(*pos)
  s='region' if n==1 else 'regions'
  ax.annotate(f'{continent}\n{n} {s} | {nw:,} windows',xy=point,xytext=label,ha='center',va='center',fontsize=9 if standalone else 8.5,
   color=COLORS[continent],linespacing=1.12,bbox=dict(boxstyle='round,pad=.30',facecolor='white',edgecolor='none',alpha=.96),
   arrowprops=dict(arrowstyle='-',color=COLORS[continent],lw=.7,shrinkA=3,shrinkB=4),zorder=10)
 ax.set_xlim(boundary[:,0].min()-1.3e6,boundary[:,0].max()+1.3e6)
 ax.set_ylim(boundary[:,1].min()-.55e6,boundary[:,1].max()+.6e6)
 ax.set_aspect('equal',adjustable='box');ax.axis('off')
 return ocean

def detailed(ax,land,e,asset,letter):
 c=COLORS[e['continent']];x0,y0,x1,y1=e['core_bbox']
 padx=max(.035,(x1-x0)*.12);pady=max(.03,(y1-y0)*.12)
 box=[x0-padx,y0-pady,x1+padx,y1+pady]
 ax.set_facecolor(SEA)
 rel=[r for r in land if r[0][:,0].max()>=box[0] and r[0][:,0].min()<=box[2] and r[0][:,1].max()>=box[1] and r[0][:,1].min()<=box[3]]
 pc=PatchCollection([polygon_patch(r) for r in rel],facecolor=LAND,edgecolor=COAST,linewidth=.27,zorder=1);ax.add_collection(pc)
 xy=asset['coordinates'];off=asset['offsets'];times=asset['timestamps']
 keep=np.ones(len(xy)-1,dtype=bool);keep[off[1:-1]-1]=False
 dt=np.diff(times);keep&=(dt>0)&(dt<=300)
 segments=np.stack((xy[:-1][keep],xy[1:][keep]),axis=1)
 line=LineCollection(segments,colors=c,linewidths=.27,alpha=.16,zorder=3,rasterized=False)
 ax.add_collection(line)
 ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='none',edgecolor=c,linewidth=.8,linestyle=(0,(3,2)),zorder=5))
 # Compute a matching geographic extent once, preserving physical aspect and
 # the shared frame without repeated autoscaling during PNG/PDF/SVG export.
 pos=ax.get_position();fw,fh=ax.figure.get_size_inches()
 frame_ratio=pos.width*fw/(pos.height*fh);coslat=np.cos(np.deg2rad((y0+y1)/2))
 desired=frame_ratio/coslat;cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
 sx=box[2]-box[0];sy=box[3]-box[1]
 if sx/sy>desired:sy=sx/desired
 else:sx=sy*desired
 ax.set_xlim(cx-sx/2,cx+sx/2);ax.set_ylim(cy-sy/2,cy+sy/2)
 ax.set_aspect(1/coslat,adjustable='box')
 # Three readable ticks. Round their ACTUAL locations before formatting, so
 # labels never imply a coordinate different from the plotted grid line.
 ticksx=np.unique(np.round(np.linspace(x0,x1,3),1))
 ticksy=np.unique(np.round(np.linspace(y0,y1,3),1))
 ax.xaxis.set_major_locator(FixedLocator(ticksx))
 ax.yaxis.set_major_locator(FixedLocator(ticksy))
 ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:fmt(v,'lon',1)))
 ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:fmt(v,'lat',1)))
 ax.tick_params(axis='both',labelsize=7.5,length=2,pad=2.2,color=COAST,labelcolor='#435A68')
 ax.grid(color=GRID,lw=.3,alpha=.55,zorder=0)
 for spine in ax.spines.values():spine.set_color('#A4BDCA');spine.set_linewidth(.55)
 name=e['display_name'].replace('–','-').replace('—','-')
 ax.set_title(f'({letter}) {name}',loc='left',fontsize=9.1,fontweight='normal',pad=5,color=INK)
 return {'rendered_consecutive_segments':int(keep.sum()),'break_rule':'Do not join different windows; break report gaps over 300 seconds','actual_display_extent':None}

def register_fonts(folder):
 files=[folder/n for n in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']]
 for p in files:assert p.exists();font_manager.fontManager.addfont(str(p))
 resolved=font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
 assert Path(resolved).resolve()==files[0].resolve()
 plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.titlesize':10,'axes.titlecolor':INK,'text.color':INK,
  'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','axes.unicode_minus':False,'savefig.facecolor':'white',
  'path.simplify':True,'path.simplify_threshold':.05})
 return {'family':'Times New Roman','resolved_regular_file':resolved,'font_sha256':{p.name:sha(p) for p in files},'pdf_embedding':'TrueType subset, fonttype42','svg_text':'glyph paths preserving exact Times New Roman appearance'}

def save(fig,out,prefix):
 out.mkdir(parents=True,exist_ok=True)
 for fmt_ in ['png','pdf','svg']:
  fig.savefig(out/f'{prefix}.{fmt_}',dpi=600,metadata={'Creator':'Verified AIS geography renderer v2'} if fmt_=='pdf' else None)
 return {f'{prefix}.{ext}':{'sha256':sha(out/f'{prefix}.{ext}'),'bytes':(out/f'{prefix}.{ext}').stat().st_size} for ext in ['png','pdf','svg']}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=BASE/'output');args=ap.parse_args()
 fontproof=register_fonts(BASE/'fonts_runtime')
 registry=json.loads((REL/'source_registry.json').read_text())['regions']
 selection=json.loads((REL/'selection_report.json').read_text())
 tm=json.loads((BASE/'track_assets/track_assets_manifest.json').read_text())
 assert selection['status']=='verified_build'
 entries=[];assets={};proof={}
 for i,key in enumerate(ORDER):
  m=registry[key];a=BASE/'track_assets'/f'{key}.npz';z=np.load(a,allow_pickle=False)
  assert len(z['window_id'])==2700 and len(set(z['mmsi']))==300 and len(z['offsets'])==2701
  assert z['offsets'][0]==0 and z['offsets'][-1]==len(z['coordinates'])
  assert sha(Path(m['canonical_file']))==m['sha256']==selection['source_files'][key]['canonical_sha256']
  entries.append(dict(key=key,number=i+1,display_name=m['display_name'],continent=m['continent'],core_bbox=m['core_bbox'],windows=2700,vessels=300))
  assets[key]={k:z[k] for k in z.files};proof[key]={'asset_sha256':sha(a),'points':len(z['coordinates']),'windows':2700,'vessels':300}
 assert sum(v['points'] for v in proof.values())==372893
 coarse=list(polygons(json.loads((ROOT/'data/geo/ne_110m_land.geojson').read_text())))
 fine=list(polygons(json.loads((BASE.parent/'map_assets/ne_10m_land.geojson').read_text())))
 # 180 mm width, intended for a two-column journal page, with real 8-10 pt text.
 fig=plt.figure(figsize=(180/25.4,205/25.4),facecolor='white')
 fig.text(.055,.975,'Geographic coverage and selected AIS trajectories',ha='left',va='top',fontsize=12.6,color=INK)
 fig.text(.055,.947,'8 regions | 300 target vessels and 2,700 source windows per region',ha='left',va='top',fontsize=8.9,color='#4A6574')
 ax=fig.add_axes([.04,.614,.92,.310]);world(ax,coarse,entries)
 fig.text(.055,.918,'(a) Global distribution',fontsize=10,color=INK,va='top')
 gs=fig.add_gridspec(2,4,left=.085,right=.985,bottom=.085,top=.568,wspace=.52,hspace=.32)
 detail_axes=[]
 for i,e in enumerate(entries):
  a=fig.add_subplot(gs[i//4,i%4]);detail_axes.append(a)
  proof[e['key']].update(detailed(a,fine,e,assets[e['key']],chr(ord('b')+i)))
 fig.text(.055,.035,'Colored lines: selected source windows. Dashed outlines: study boundaries.',fontsize=8,color='#4A6574')
 fig.text(.055,.017,'Robinson overview; local maps in WGS84. Land: Natural Earth (110m / 10m).',fontsize=7.7,color='#4A6574')
 fig.canvas.draw()
 for e,a in zip(entries,detail_axes):proof[e['key']]['actual_display_extent']=list(a.get_xlim())+list(a.get_ylim())
 outputs=save(fig,args.out,'benchmark_geography_v2');plt.close(fig)
 standalone=plt.figure(figsize=(180/25.4,88/25.4),facecolor='white');wa=standalone.add_axes([.015,.015,.97,.91]);world(wa,coarse,entries,True)
 standalone.text(.04,.97,'Global distribution of the eight study regions',fontsize=11,va='top',color=INK)
 outputs.update(save(standalone,args.out,'world_overview_v2'));plt.close(standalone)
 meta={'version':'map_design_v2','selection_report_sha256':sha(REL/'selection_report.json'),'selected_windows_sha256':sha(REL/'selected_windows.parquet'),'source_registry_sha256':sha(REL/'source_registry.json'),'renderer_sha256':sha(Path(__file__)),'track_manifest_sha256':sha(BASE/'track_assets/track_assets_manifest.json'),'projection':'+proj=robin +lon_0=0 +R=6371008.8','projection_documentation':'https://proj.org/en/stable/operations/projections/robin.html','pyproj_version':pyproj_version,'figure_width_mm':180,'figure_height_mm':205,'fonts':fontproof,'palette':{'ocean':SEA,'land':LAND,'coast':COAST,'regions':COLORS},'regions':entries,'tracks':proof,'independent_windows':21600,'observations':372893,'output_files':outputs,'old_map_or_dataset_modified':False}
 (args.out/'map_design_v2_metadata.json').write_text(json.dumps(meta,indent=2))
 print(json.dumps({'output':str(args.out),'regions':8,'windows':21600,'observations':372893,'files':list(outputs)},indent=2),flush=True)

if __name__=='__main__':main()
