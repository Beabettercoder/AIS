"""Fetch the small official Natural Earth 10m land archive for detailed panels."""
from pathlib import Path
import hashlib,json,urllib.request,zipfile,struct
import numpy as np

OUT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/visuals/map_assets')
OUT.mkdir(parents=True,exist_ok=True)
url='https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip'
archive=OUT/'ne_10m_land.zip'
if not archive.exists():
 with urllib.request.urlopen(url,timeout=60) as response:
  data=response.read()
 assert data[:2]==b'PK'
 archive.write_bytes(data)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 data=z.read('ne_10m_land.shp')
assert struct.unpack_from('>i',data,0)[0]==9994
pos=100;features=[]
while pos<len(data):
 rec_id,length=struct.unpack_from('>2i',data,pos);record=data[pos+8:pos+8+2*length];pos+=8+2*length
 kind=struct.unpack_from('<i',record,0)[0]
 if kind==0:continue
 assert kind==5,kind
 nparts,npoints=struct.unpack_from('<2i',record,36)
 offsets=np.frombuffer(record,dtype='<i4',count=nparts,offset=44)
 points=np.frombuffer(record,dtype='<f8',count=2*npoints,offset=44+4*nparts).reshape(npoints,2)
 rings=[points[a:b] for a,b in zip(offsets,np.r_[offsets[1:],npoints])]
 # Shapefile land outer rings use clockwise order. Interior lake rings are not
 # exported as land polygons. The figure is a location map, not navigation data.
 outers=[]
 for r in rings:
  signed=np.sum(r[:-1,0]*r[1:,1]-r[1:,0]*r[:-1,1])/2
  if signed<0:outers.append([r.tolist()])
 assert outers
 features.append({'type':'Feature','properties':{'source_record':int(rec_id)},'geometry':{'type':'MultiPolygon','coordinates':outers}})
path=OUT/'ne_10m_land.geojson'
path.write_text(json.dumps({'type':'FeatureCollection','features':features},separators=(',',':')))
meta={'source_page':'https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/','download_url':url,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'geojson_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'geojson_bytes':path.stat().st_size,'features':len(features),'geometry_note':'Land exterior rings; lake holes omitted, same location-map drawing convention as the 110m overview.'}
(OUT/'ne_10m_land_source.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta),flush=True)
