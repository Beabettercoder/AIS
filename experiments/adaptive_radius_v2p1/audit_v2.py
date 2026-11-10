"""Identity, pairing, eligibility and stress-source audit before training."""
from pathlib import Path
import json
import numpy as np
from radius_core import ARMS
import train_single_v2 as single
import train_relation_v2 as relation

def check_arm(mode,radius,count,field,rho,arm,mask):
 if not np.isin(mode,[0,1,2]).all() or np.any(mode[~mask]!=0) or not np.isfinite(field).all() or not np.isfinite(rho).all():raise ValueError('invalid feature '+arm)
 if np.any((mode==0)&(rho!=0)) or np.any(field[mode==0]!=0):raise ValueError('invalid missing reference '+arm)
 if np.any((mode==2)&(count<4)) or np.any((mode==1)&(count<1)):raise ValueError('invalid neighbor count '+arm)
 if 'deg6' in arm and np.any((mode==2)&(count<6)):raise ValueError('degree6 slope below6')
 if arm in ('target_only','tight_plain','wide_plain','adaptive_plain_nofallback') and np.any(mode==1):raise ValueError('unexpected fallback '+arm)
 if np.any(mode>0) and not np.isin(radius[mode>0],[3.5,5.,8.]).all():raise ValueError('unknown radius')
 if arm in ('target_only','tight_plain','wide_plain','adaptive_plain_nofallback','adaptive_plain_fallback','adaptive_plain_deg6'):
  expected=np.minimum(count/4.,1.)
  if not np.allclose(rho[mode>0],expected[mode>0],atol=1e-7):raise ValueError('ordinary uses non-geometric rho '+arm)

def main():
 root=Path('/home/ps/AIS_adaptive_radius_relation_v2p1_20261109');results={};hashes={};sets={}
 source_audit=json.loads((root/'reports/relation_source_views_audit.json').read_text())
 if source_audit['status']!='passed':raise ValueError('relation source audit missing')
 for task,loader,expected in (('single',single.data,{'train':7840,'val':1680,'test':1680}),('relation',relation.data,{'train':8136,'val':1088,'test':880})):
  sets[task]={};results[task]={}
  for split in ('train','val','test'):
   d,h=loader(root,split);hashes.update(h);sets[task][split]=d
   if len(d['y'])!=expected[split]:raise ValueError('window count '+task+split)
   if task=='single':
    for arm in ARMS:check_arm(d[arm+'__mode'],d[arm+'__radius_nm'],d[arm+'__count'],d[arm+'__field'],d[arm+'__rho'],arm,d['mask'])
    if split=='test':
     for view in ('natural','suffix','combined_l100_m600'):
      s,hh=single.data(root,'test',view);hashes.update(hh)
      for k in ('window_id','mmsi','region','y','mask','time_s','target','raw','behavior_type'):
       if not np.array_equal(d[k],s[k]):raise ValueError('single view mismatch '+view+k)
      for arm in ARMS:check_arm(s[arm+'__mode'],s[arm+'__radius_nm'],s[arm+'__count'],s[arm+'__field'],s[arm+'__rho'],arm,s['mask'])
   else:
    src,inv,counts=np.unique(d['source'].astype(str),return_inverse=True,return_counts=True)
    if np.any(counts!=4):raise ValueError('invalid quartet size '+split)
    for i in range(len(src)):
     ix=np.flatnonzero(inv==i)
     if sorted(d['case'][ix].tolist())!=[0,1,2,3] or int(d['y'][ix].sum())!=2 or len(set(d['sparse_source'][ix]))!=1:raise ValueError('quartet inconsistent '+split)
    for view in relation.VIEWS:
     for arm in ARMS:
      field=d[view+'__'+arm+'__field'];mode=field[:,:,7];rho=field[:,:,6];count=d[view+'__'+arm+'__count'];rad=d[view+'__'+arm+'__radius']
      check_arm(mode,rad,count,field,rho,arm,d['mask'])
   results[task][split]={'windows':len(d['y']),'ships':len(np.unique(d['mmsi'])),'steps':int(d['mask'].sum())}
  for a,b in (('train','val'),('train','test'),('val','test')):
   if set(sets[task][a]['mmsi'].astype(str))&set(sets[task][b]['mmsi'].astype(str)):raise ValueError('target vessel overlap '+task+a+b)
 out=root/'reports/pretrain_audit.json';result={'status':'passed','datasets':results,'source_view_audit':source_audit,'all_target_MMSI_disjoint':True,'ordinary_rho':'count only; no reliability weights','input_sha256':hashes,'note':'existing cohort limits context; geographic sparse group is density-defined and overlaps port/strait'}
 out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2),flush=True)
if __name__=='__main__':main()
