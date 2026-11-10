"""Build observed Heading side evidence from actual aligned reports; train/val only."""
import argparse,hashlib,json,importlib.util
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
OLD=ROOT/'balanced_global_v2/relation_dual_pressure_v2_20261027'
FIELDS=ROOT/'relation_repair_20261030'
COVER=ROOT/'heading_coverage_20261030'
OUT=ROOT/'heading_evidence_pilot_20261030'
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
builder=module(FIELDS/'builder.py','original_builder')

def heading_features(o,case,heading,hvalid,target_id,fieldmask):
 t=o['target_t'][case]; p=o['target_pos'][case]; n=len(t)
 values=np.zeros((2,n,5),np.float32)
 reports=[]
 for member,lo,hi in zip(o['cohort_mmsi'],o['member_offsets'][:-1],o['member_offsets'][1:]):
  lo,hi=int(lo),int(hi); nt=o['neighbor_t'][case,lo:hi];ns=o['neighbor_sog_kn'][case,lo:hi];nc=o['neighbor_cog_deg'][case,lo:hi];np_=o['neighbor_pos'][case,lo:hi]
  h=heading[lo:hi]; ok=hvalid[lo:hi]
  assert np.all(np.diff(nt)>0) and str(member)!=str(target_id)
  w,known=builder.core.absolute_weights(nt,np_,ns,nc);ix=np.searchsorted(nt,t,side='right')-1
  yaw=np.zeros(len(nt));yv=np.zeros(len(nt),bool);dt=np.diff(nt);good=(dt>0)&(dt<=300)&ok[1:]&ok[:-1]
  yaw[1:][good]=np.deg2rad(((h[1:][good]-h[:-1][good]+180)%360)-180)/dt[good]
  yv[1:]=good
  reports.append((str(member),nt,np_,nc,w,ix,h,ok,yaw,yv))
 for j in range(n):
  if not fieldmask[j]:continue
  rec=[];phase_start=t[0] if j<n//2 else t[n//2]
  for member,nt,pos,cog,w,ix,h,ok,yaw,yv in reports:
   q=int(ix[j]);
   if q<0 or not 0<=t[j]-nt[q]<=300:continue
   selected=np.flatnonzero((nt>=phase_start)&(nt<=t[j]));
   if len(selected)==0:selected=np.array([q])
   d=pos[q]-p[j];dist=np.exp(-.5*np.sum(d*d)/builder.field.SIGMA**2)
   delta=np.deg2rad((h[q]-cog[q]+180)%360-180) if ok[q] and np.isfinite(cog[q]) else 0.
   rec.append((dist,dist*float(w[selected].mean()),bool(ok[q]) and np.isfinite(cog[q]),np.sin(delta),np.cos(delta),bool(yv[q]),yaw[q]))
  if not rec:continue
  z=np.asarray(rec,float)
  for arm in range(2):
   wt=z[:,arm];total=wt.sum()
   if total<=1e-12:continue
   a=wt*z[:,2];den=a.sum();b=wt*z[:,5];dy=b.sum()
   if den>1e-12:values[arm,j,0]=(a*z[:,3]).sum()/den;values[arm,j,1]=(a*z[:,4]).sum()/den
   values[arm,j,2]=den/total
   if dy>1e-12:values[arm,j,3]=(b*z[:,6]).sum()/dy
   values[arm,j,4]=dy/total
 return values

def main(split,limit=0):
 manifest=json.loads((OLD/'dataset_manifest_combined.json').read_text())
 rows=sorted([r for r in manifest['samples'] if r['version']=='relaxed' and r['split']==split],key=lambda r:(r['region'],r['window_id']))
 with np.load(COVER/f'heading_{split}.npz',allow_pickle=False) as q:h={k:q[k] for k in q.files}
 with np.load(FIELDS/f'{split}_relation_field.npz',allow_pickle=False) as q:d={k:q[k] for k in q.files}
 assert len(rows)==len(h['window_id']) and len(np.unique(h['window_id']))==len(rows)
 hindex={str(w):i for i,w in enumerate(h['window_id'])}
 assert set(hindex)==set(r['window_id'] for r in rows)
 if limit: rows=rows[:limit]
 views=('base','natural','suffix','combined_l100_m600');extra={v:[] for v in views}
 for ri,row in enumerate(rows):
  hi=hindex[row['window_id']];assert str(h['region'][hi])==row['region']
  nlo,nhi=h['sample_neighbor_offsets'][hi:hi+2];hng=h['neighbor_heading'][nlo:nhi];hv=h['neighbor_valid'][nlo:nhi]
  data={}
  for view in ('base','natural','suffix'):
   entry=row['files'][view];path=OLD/entry['path'];assert builder.field.sha(path)==entry['sha256']
   with np.load(path,allow_pickle=False) as z:data[view]={k:z[k] for k in ('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','y')}
  o=data['base'];assert np.array_equal(o['neighbor_t'][0],h['neighbor_t'][nlo:nhi]);assert np.array_equal(o['neighbor_mmsi'].astype(str),h['neighbor_mmsi'][nlo:nhi].astype(str))
  for v in ('natural','suffix'):
   for key in ('target_t','neighbor_t','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','y'):
    assert np.array_equal(data[v][key],o[key]),(row['window_id'],v,key)
  combined={k:v.copy() for k,v in o.items()};combined['neighbor_sog_kn']=data['suffix']['neighbor_sog_kn'].copy()
  for m in row['stress']['members']:
   lo=int(combined['member_offsets'][m['member_index']]);ix=lo+int(m['indices']['suffix']['report_index'])
   for q in (ix,ix+1):combined['neighbor_pos'][:,q]+=600*builder.direction(row['window_id'],m['mmsi'])
  data['combined_l100_m600']=combined
  for case in range(4):
   j=4*ri+case;assert d['source'][j]==row['window_id'] and int(d['case'][j])==case and d['y'][j]==o['y'][case]
   for view in views:
    n=len(o['target_t'][case]);fieldmask=d[view+'__field'][j,0,:n,7]>0
    arr=np.zeros((2,20,5),np.float32);arr[:,:n]=heading_features(data[view],case,hng,hv,row['mmsi'],fieldmask)
    extra[view].append(arr)
  if (ri+1)%100==0:print('PROGRESS',split,ri+1,len(rows),flush=True)
 out={k:d[k][:4*len(rows)] for k in ('source','case','y','mmsi','region','length')}
 for view in views:
  a=np.asarray(extra[view]);out[view+'__F0']=a[:,0];out[view+'__F1']=a[:,1]
 path=OUT/f'{split}{"_smoke"+str(limit) if limit else ""}_heading_features.npz';assert not path.exists();np.savez_compressed(path,**out)
 print('DONE',split,len(rows),builder.field.sha(path),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val'],required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args();main(a.split,a.limit)
