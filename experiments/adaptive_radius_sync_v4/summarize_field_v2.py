"""Field availability and paired base-to-stress velocity drift; no truth claim."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from radius_core import ARMS
import train_single_v2 as single
import train_relation_v2 as relation
VIEWS=('base','natural','suffix','combined_l100_m600');PORTS=('houston','ny','la','shanghai')
PAIRS=(('wide_plain','adaptive_plain_fallback'),('adaptive_plain_fallback','adaptive_qualified_fallback'),('adaptive_qualified_fallback','adaptive_credible_fallback'),('adaptive_plain_fallback','adaptive_credible_fallback'))
def groups(d,task):
 p=np.isin(d['region'],PORTS);sparse=np.array([np.median(c[m])<4 for c,m in zip(d['eligible_count_8nm'],d['mask'])]) if task=='single' else d['sparse_source']
 return {'all':np.ones(len(p),bool),'ports':p,'restricted_channels':~p,'sparse_overlapping':sparse}
def get(d,task,view,arm):
 if task=='single':return d[arm+'__bg'],d[arm+'__mode'],d[arm+'__radius_nm']
 f=d[view+'__'+arm+'__field'];return f[:,:,:2],f[:,:,7],d[view+'__'+arm+'__radius']
def ci(a,b,mask,ships,draws):
 wi,ti=np.where(mask);units,inv=np.unique(ships[wi].astype(str),return_inverse=True)
 if len(units)<2:return None
 delta=(b-a)[mask];counts=np.bincount(inv,minlength=len(units));sums=np.bincount(inv,weights=delta,minlength=len(units));rng=np.random.default_rng(1729);q=[]
 for _ in range(draws):
  mult=np.bincount(rng.integers(0,len(units),len(units)),minlength=len(units));den=(mult*counts).sum()
  if den:q.append(float((mult*sums).sum()/den))
 lo,hi=np.quantile(q,[.025,.975]);return {'paired_steps':int(mask.sum()),'ships':len(units),'mean_before_mps':float(a[mask].mean()),'mean_after_mps':float(b[mask].mean()),'mean_difference_mps':float(delta.mean()),'lower95':float(lo),'upper95':float(hi)}
def write(p,rows):
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=('val','test'),default='val');p.add_argument('--root',type=Path,default=Path('/home/ps/AIS_adaptive_sync_20261110'));p.add_argument('--draws',type=int,default=2000);a=p.parse_args();root=a.root
 if a.split=='test' and not (root/'frozen_manifest.json').exists():raise ValueError('freeze all models before heldout field analysis')
 coverage=[];drifts=[];pairs=[]
 for task,loader in (('relation',relation.data),):
  base,_=loader(root,a.split);gr=groups(base,task);views={'base':base}
  if task=='relation':views.update({v:base for v in VIEWS[1:]})
  elif a.split=='test':
   for view in VIEWS[1:]:
    d,_=single.data(root,'test',view)
    for k in ('window_id','mmsi','region','y','mask','time_s'):
     if not np.array_equal(base[k],d[k]):raise ValueError('view not paired '+k)
    views[view]=d
  mask=base['mask'];ships=base['mmsi'].astype(str)
  for view,d in views.items():
   for group,sel in gr.items():
    in_scope=mask&sel[:,None];den=int(in_scope.sum());vals={};support={}
    for arm in ARMS:
     bg,mode,radius=get(d,task,view,arm);valid=in_scope&(mode>0);rr=radius[valid];n=len(rr)
     coverage.append({'task':task,'view':view,'group':group,'arm':arm,'windows':int(sel.sum()),'ships':len(np.unique(ships[sel])),'target_steps':den,'field_steps':n,'affine_steps':int((in_scope&(mode==2)).sum()),'fallback_steps':int((in_scope&(mode==1)).sum()),'coverage':float(n/den) if den else None,'fallback_fraction':float((in_scope&(mode==1)).sum()/den) if den else None,'mean_radius_nm':float(rr.mean()) if n else None,'r3p5_fraction':float((rr==3.5).mean()) if n else None,'r5_fraction':float((rr==5.).mean()) if n else None,'r8_fraction':float((rr==8.).mean()) if n else None})
     if view=='base':continue
     prior,pm,_,=get(base,task,'base',arm);support[arm]=in_scope&(pm>0)&(mode>0);vals[arm]=np.linalg.norm(bg-prior,axis=-1)
     hit=d['stress_hit'] if task=='single' else d[view+'__hit']
     for subset,xx in (('all',support[arm]),('hit',support[arm]&hit)):
      vv=vals[arm][xx];drifts.append({'task':task,'view':view,'group':group,'arm':arm,'subset':subset,'valid_steps':int(len(vv)),'hit_steps_in_scope':int((in_scope&hit).sum()),'total_steps':den,'mean_mps':float(vv.mean()) if len(vv) else None,'median_mps':float(np.median(vv)) if len(vv) else None})
    if view=='base':continue
    hit=d['stress_hit'] if task=='single' else d[view+'__hit']
    common=support['wide_plain']&support['adaptive_plain_fallback']&support['adaptive_credible_fallback']
    for before,after in PAIRS:
     for subset,extra in (('pair_support',np.ones_like(mask,bool)),('pair_hit',hit),('common_three',common),('common_three_hit',common&hit)):
      valid=support[before]&support[after]&extra;result=ci(vals[before],vals[after],valid,ships,a.draws)
      pairs.append({'task':task,'view':view,'group':group,'before':before,'after':after,'subset':subset,**(result or {'paired_steps':int(valid.sum()),'reason':'fewer than two vessels'})})
 out=root/'reports';out.mkdir(exist_ok=True);write(out/f'{a.split}_field_coverage.csv',coverage);write(out/f'{a.split}_field_drift.csv',drifts);write(out/f'{a.split}_field_drift_pairs.csv',pairs)
 (out/f'{a.split}_field_summary.json').write_text(json.dumps({'split':a.split,'coverage_rows':len(coverage),'drift_rows':len(drifts),'pair_rows':len(pairs),'definition':'||V_stress - V_base|| on same window/case/time; base may contain pre-existing noise; not true-field error','pairing':'only steps where both methods are available in both base and stress; availability/no-field reported separately','direct_hit':'single current used report, relation any used phase report; effects through next reliability interval may outlast direct hit','sparse':'overlaps geographic groups and is not verified open ocean','uncertainty':'target vessel cluster bootstrap, 2000 draws; not seed uncertainty; uncorrected comparisons'},indent=2))
 print('FIELD_SUMMARY_COMPLETE',a.split,len(coverage),len(drifts),len(pairs),flush=True)
if __name__=='__main__':main()
