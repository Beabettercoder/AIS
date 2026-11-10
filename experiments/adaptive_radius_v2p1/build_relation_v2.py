"""Rebuild original relation quartets with unfiltered and credible radii."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
from radius_core import choose,ARMS,NM

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');OLD=ROOT/'relation_repair_20261030';FROZEN=ROOT/'formal_20261101/relation'
COHORT=ROOT/'balanced_global_v2/relation_dual_pressure_v2_20261027';STAGE=ROOT/'framework_validation_v1_20261028'
OUT=Path('/home/ps/AIS_adaptive_radius_relation_v2_20261109');VIEWS=('base','natural','suffix','combined_l100_m600')
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
core=module(STAGE/'data_builder.py','new_relation_core')
prior=module(OLD/'builder.py','new_relation_prior')
def velocity(s,c):
 a=np.deg2rad(c);return np.column_stack((s*np.sin(a),s*np.cos(a)))*NM/3600.
def calc(o,base,case,target):
 t=o['target_t'][case];p=o['target_pos'][case];tc=o['target_cog_deg'][case];tv=velocity(o['target_sog_kn'][case],tc);n=len(t)
 series=[]
 for member,lo,hi in zip(o['cohort_mmsi'],o['member_offsets'][:-1],o['member_offsets'][1:]):
  lo,hi=int(lo),int(hi);nt=o['neighbor_t'][case,lo:hi];ns=o['neighbor_sog_kn'][case,lo:hi];nc=o['neighbor_cog_deg'][case,lo:hi];pos=o['neighbor_pos'][case,lo:hi]
  ww,known=core.absolute_weights(nt,pos,ns,nc);ix=np.searchsorted(nt,t,side='right')-1
  if str(member)==str(target):raise ValueError('target appears in cohort')
  bs=base['neighbor_sog_kn'][case,lo:hi];bp=base['neighbor_pos'][case,lo:hi]
  series.append((str(member),nt,ns,nc,pos,velocity(ns,nc),ww,known,ix,bs,bp))
 out={arm:np.zeros((n,8),np.float32) for arm in ARMS};counts={arm:np.zeros(n,np.int16) for arm in ARMS};radii={arm:np.zeros(n,np.float32) for arm in ARMS};all_count=np.zeros(n,np.int16);hit=np.zeros(n,bool)
 for j in range(n):
  positions=[];v0=[];v1=[];weights=[];valid=[];ids=[];phase=t[0] if j<n//2 else t[n//2]
  for mid,nt,ns,nc,pos,vel,w,known,ix,bs,bp in series:
   q=int(ix[j])
   if q<0 or not 0<=t[j]-nt[q]<=300:continue
   if not (np.isfinite(pos[q]).all() and np.isfinite(ns[q]) and np.isfinite(nc[q]) and np.isfinite(tc[j]) and ns[q]>=0 and 0<=nc[q]<360 and 0<=tc[j]<360):continue
   angle=abs((float(nc[q])-float(tc[j])+180)%360-180)
   if angle>45 or np.linalg.norm(pos[q]-p[j])>8*NM:continue
   selected=np.flatnonzero((nt>=phase)&(nt<=t[j]));selected=selected if len(selected) else np.array([q]);vv=vel[selected];ww=w[selected]
   if not np.isfinite(vv).all():raise ValueError('nonfinite historical speed')
   avg=vv.mean(0);weighted=np.average(vv,axis=0,weights=ww) if float(ww.sum())>1e-12 else avg
   positions.append(pos[q]-p[j]);v0.append(avg);v1.append(weighted);weights.append(float(w[q]));valid.append(bool(known[q]));ids.append(mid)
   if np.any(ns[selected]!=bs[selected]) or np.any(pos[selected]!=bp[selected]):hit[j]=True
  x,diag=choose(positions,v0,v1,weights,valid,ids,target);all_count[j]=diag['plain_count_8nm']
  for arm,q in x.items():
   counts[arm][j]=q['count'];radii[arm][j]=q['radius_nm']
   if q['mode']:
    bg=q['bg'];J=q['grad'];out[arm][j]=[*bg,*(tv[j]-bg),np.trace(J),J[1,0]-J[0,1],q['rho'],q['mode']]
 return out,counts,radii,all_count,hit
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=['train','val','test']);p.add_argument('--region',required=True,choices=REGIONS);p.add_argument('--limit',type=int,default=0);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
 dest=a.out/('smoke_relation' if a.limit else 'relation')/f'{a.split}_{a.region}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():raise FileExistsError(dest)
 manifest=COHORT/'dataset_manifest_combined.json';entries=sorted((r for r in json.loads(manifest.read_text())['samples'] if r['version']=='relaxed' and r['split']==a.split and r['region']==a.region),key=lambda r:(r['region'],r['window_id']))
 refpath=(FROZEN/'test_relation_field.npz') if a.split=='test' else (OLD/f'{a.split}_relation_field.npz')
 with np.load(refpath,allow_pickle=False) as z:ref={k:z[k] for k in ('source','case','y','length','mmsi','region','raw7','causal7','self_w','time_s')}
 subset=np.flatnonzero(ref['region'].astype(str)==a.region);subset=subset[:a.limit*4] if a.limit else subset;count=len(subset)
 ref={k:v[subset] for k,v in ref.items()};entries=entries[:a.limit] if a.limit else entries
 if count!=len(entries)*4:raise ValueError('source quartet mismatch')
 out={k:v.copy() for k,v in ref.items()};out['sparse_source']=np.zeros(count,bool);out['base_count8']=np.zeros((count,20),np.int16)
 for v in VIEWS:
  out[v+'__hit']=np.zeros((count,20),bool)
  for arm in ARMS:
   out[v+'__'+arm+'__field']=np.zeros((count,20,8),np.float32);out[v+'__'+arm+'__count']=np.zeros((count,20),np.int16);out[v+'__'+arm+'__radius']=np.zeros((count,20),np.float32)
 start=time.time();keys=('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','y')
 for i,row in enumerate(entries):
  obs={}
  for view in ('base','natural','suffix'):
   file=COHORT/row['files'][view]['path']
   if sha(file)!=row['files'][view]['sha256']:raise ValueError('source hash mismatch')
   with np.load(file,allow_pickle=False) as z:obs[view]={k:z[k] for k in keys}
  for key in keys:
   if key!='neighbor_sog_kn' and not np.array_equal(obs['base'][key],obs['suffix'][key]):raise ValueError('suffix changed '+key)
  composite={k:v.copy() for k,v in obs['base'].items()};composite['neighbor_sog_kn']=obs['suffix']['neighbor_sog_kn'].copy()
  for m in row['stress']['members']:
   lo=int(composite['member_offsets'][m['member_index']]);hi=int(composite['member_offsets'][m['member_index']+1]);ix=lo+int(m['indices']['suffix']['report_index'])
   if not lo<=ix<ix+1<hi:raise ValueError('invalid stress index')
   composite['neighbor_pos'][:,ix:ix+2]+=600*prior.direction(row['window_id'],m['mmsi'])
  obs['combined_l100_m600']=composite
  for case in range(4):
   at=4*i+case;length=int(ref['length'][at]);o=obs['base']
   if str(ref['source'][at])!=str(row['window_id']) or int(ref['case'][at])!=case or str(ref['mmsi'][at])!=str(row['mmsi']) or int(ref['y'][at])!=int(o['y'][case]):raise ValueError('quartet metadata mismatch')
   if not np.array_equal(ref['time_s'][at,:length],o['target_t'][case]):raise ValueError('target timestamps mismatch')
   for view in VIEWS:
    fields,counts,radii,n_all,hit=calc(obs[view],o,case,row['mmsi']);out[view+'__hit'][at,:length]=hit
    if view=='base':out['base_count8'][at,:length]=n_all
    for arm in ARMS:
     out[view+'__'+arm+'__field'][at,:length]=fields[arm];out[view+'__'+arm+'__count'][at,:length]=counts[arm];out[view+'__'+arm+'__radius'][at,:length]=radii[arm]
  # Sparse/source classification derives from the no-intervention case only;
  # it is fixed across all four variants and independent of scored labels.
  n=int(ref['length'][4*i]);sparse=float(np.median(out['base_count8'][4*i,:n]))<4.;out['sparse_source'][4*i:4*i+4]=sparse
  if (i+1)%100==0:print('RELATION',a.split,a.region,i+1,'/',len(entries),flush=True)
 np.savez_compressed(dest,**out)
 audit={'split':a.split,'region':a.region,'source_count':len(entries),'windows':count,'sha256':sha(dest),'protocol_sha256':sha(Path(__file__).with_name('protocol.json')),'builder_sha256':sha(Path(__file__)),'operator_sha256':sha(Path(__file__).with_name('radius_core.py')),'reference_sha256':sha(refpath),'manifest_sha256':sha(manifest),'stress':'same source/target/MMSI/cohort; re-evaluate recorded cohort eligibility at current time, no external-cohort requery','sparse':'no-intervention case base unfiltered 8nm count; same label for all four cases','elapsed_seconds':time.time()-start}
 dest.with_suffix('.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print('RELATION_COMPLETE',json.dumps(audit),flush=True)
if __name__=='__main__':main()
