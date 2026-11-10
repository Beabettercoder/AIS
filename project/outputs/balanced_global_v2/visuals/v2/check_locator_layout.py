"""Check continent cards against real site markers and the map outline."""
from pathlib import Path
import json
import numpy as np
import render_map_v2 as m

m.register_fonts(m.BASE/'fonts_runtime')
meta=json.loads((m.BASE/'output/map_design_v2_metadata.json').read_text())
land=list(m.polygons(json.loads((m.ROOT/'data/geo/ne_110m_land.geojson').read_text())))
result={}
for name,size,rect,standalone in [('complete_figure',[180,205],[.04,.614,.92,.310],False),('standalone',[180,88],[.015,.015,.97,.91],True)]:
 f=m.plt.figure(figsize=np.array(size)/25.4);a=f.add_axes(rect);ocean=m.world(a,land,meta['regions'],standalone);f.canvas.draw();renderer=f.canvas.get_renderer()
 points=[]
 for e in meta['regions']:
  x0,y0,x1,y1=e['core_bbox'];p=a.transData.transform(m.PROJECTION((x0+x1)/2,(y0+y1)/2));points.append((e['key'],p))
 cards=[]
 for text in a.texts:
  b=text.get_bbox_patch().get_window_extent(renderer)
  covered=[key for key,p in points if b.contains(*p)]
  corners=[[b.x0,b.y0],[b.x0,b.y1],[b.x1,b.y0],[b.x1,b.y1]]
  inside=[bool(ocean.contains_point(p)) for p in corners]
  assert not covered,(name,text.get_text(),covered)
  assert all(inside),(name,text.get_text(),'card overlaps map outline')
  cards.append({'text':text.get_text(),'covered_site_markers':covered,'all_corners_inside_outline':True})
 result[name]={'status':'passed','site_markers_checked':len(points),'cards':cards};m.plt.close(f)
(m.BASE/'output/locator_layout_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
