"""Construct paired offline quartet sources and two label-blind speed stresses."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib,json,math,sys
import numpy as np
import pandas as pd

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');OUT=ROOT/'relation_dual_pressure_v2_20261027'
OLD=ROOT/'original_method_and_relation_v1_20261026/relation';sys.path.insert(0,str(OLD/'vendor'));sys.path.insert(0,str(OUT))
from audit_neighborhood import Source,REGIONS,wrap,NM,sha
from quartet_builder_v2 import build_one
def seed(wid,mid):return int.from_bytes(hashlib.sha256(('speed-stress:'+wid+':'+mid).encode()).digest()[:8],'little')
def rank(wid,mid):return hashlib.sha256(('speed-stress-member:'+wid+':'+mid).encode()).digest()
def jsonwrite(path,value):
 with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def coverage(meta,p,when,radius_nm):
 date=datetime.fromtimestamp(float(when),timezone.utc).strftime('%Y-%m-%d');box=meta.get('coverage_bbox_by_utc_day',{}).get(date)
 if not box:return False
 core=meta['core_bbox'];midlon=(core[0]+core[2])/2;midlat=(core[1]+core[3])/2
 lon=midlon+p[0]/(111195*np.cos(np.deg2rad(midlat)));lat=midlat+p[1]/111195
 rx=radius_nm*NM/(111195*np.cos(np.deg2rad(midlat)));ry=radius_nm*NM/111195
 return bool(box[0]<=lon-rx and lon+rx<=box[2] and box[1]<=lat-ry and lat+ry<=box[3])

def align_info(a):
 t=a['target_t'][0];anchor=max(2,len(t)//2-1);prefix=np.arange(anchor+1);suffix=np.arange(anchor+1,len(t))[-3:];out=[]
 for j,(lo,hi) in enumerate(zip(a['member_offsets'][:-1],a['member_offsets'][1:])):
  lo=int(lo);hi=int(hi);nt=a['neighbor_t'][0,lo:hi];ix=np.searchsorted(nt,t,side='right')-1;safe=np.maximum(ix,0)
  ok=(ix>=0)&(t>=nt[safe])&(t-nt[safe]<=300)
  before=int(ok[prefix].sum());after=int(ok[suffix].sum())
  directed=sorted(set(int(i) for i in ix[suffix][ok[suffix]] if nt[int(i)]>t[anchor] and int(i)+1<len(nt)))
  natural=[int(i) for i in np.flatnonzero(nt>t[anchor]) if int(i)+1<len(nt)]
  out.append({'member_index':j,'pre_grid_count':before,'suffix_grid_count':after,'directed_indices':directed,'natural_indices':natural})
 return anchor,suffix,out

def stress_views(a,wid):
 anchor,suffix,alignment=align_info(a);members=a['cohort_mmsi'].astype(str);eligible=[x for x in alignment if x['directed_indices'] and x['natural_indices']]
 if not eligible:return None,'no_member_with_target_aligned_suffix_and_two_reports'
 eligible=sorted(eligible,key=lambda x:rank(wid,members[x['member_index']]));chosen=eligible[:min(math.ceil(len(members)/2),len(eligible))]
 out={v:{k:np.asarray(x).copy() for k,x in a.items() if k in ['target_t','target_pos','target_sog_kn','target_cog_deg','target_mask','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','neighbor_mask','neighbor_mmsi','member_offsets','cohort_mmsi','target_mmsi','base_window_id','source_window_id','split','region','y']} for v in ['natural','suffix']}
 proof=[]
 for e in chosen:
  j=e['member_index'];lo=int(a['member_offsets'][j]);hi=int(a['member_offsets'][j+1]);tt=a['neighbor_t'][0,lo:hi];rng=np.random.default_rng(seed(wid,members[j]));factor=float(rng.uniform(3.,5.))
  i_nat=int(rng.choice(e['natural_indices']));i_hit=int(rng.choice(e['directed_indices']));item={'mmsi':str(members[j]),'member_index':j,'seed':seed(wid,members[j]),'factor':factor,'indices':{}}
  for view,ix in [('natural',i_nat),('suffix',i_hit)]:
   old=out[view]['neighbor_sog_kn'][:,lo+ix:lo+ix+2].copy();out[view]['neighbor_sog_kn'][:,lo+ix:lo+ix+2]=np.minimum(old*factor,102.1)
   latest=np.searchsorted(tt,a['target_t'][0,suffix],side='right')-1;safe=np.maximum(latest,0);hit=(latest>=ix)&(latest<ix+2)&(latest>=0)&(a['target_t'][0,suffix]-tt[safe]<=300)
   item['indices'][view]={'report_index':ix,'times_s':tt[ix:ix+2].tolist(),'suffix_grid_hits':int(hit.sum()),'numeric_changed':bool(np.any(out[view]['neighbor_sog_kn'][:,lo+ix:lo+ix+2]!=old))}
  proof.append(item)
 for v in out:
  assert np.array_equal(out[v]['neighbor_sog_kn'][0],out[v]['neighbor_sog_kn'][2]) and np.array_equal(out[v]['neighbor_sog_kn'][1],out[v]['neighbor_sog_kn'][3])
 assert all(x['indices']['suffix']['suffix_grid_hits']>0 for x in proof)
 return (out,{'members':proof,'selected_count':len(proof),'cohort_count':len(members),'anchor_index':anchor}),None

def main():
 lock=json.loads((OUT/'construction_lock.json').read_text());assert all(sha(OUT/k)==v for k,v in lock['local_files_sha256'].items())
 manifest=json.loads((OUT/'candidate_pool_manifest.json').read_text());assert sha(OUT/'candidate_pool.jsonl')==manifest['candidate_pool_sha256']
 pool=[json.loads(x) for x in (OUT/'candidate_pool.jsonl').read_text(encoding='utf8').splitlines()];targets={s:set(manifest['target_ids'][s]) for s in ('train','val','test')}
 selected=pd.read_parquet(ROOT/'release_20261026/selected_windows.parquet');orig={s:set(selected.loc[selected.split==s,'mmsi'].astype(str)) for s in targets}
 groups=pd.read_parquet(ROOT/'release_20261026/global_all_mmsi_split.parquet');allids=set(groups.mmsi.astype(str))
 allowed={s:allids-set().union(*(targets[q]|orig[q] for q in targets if q!=s)) for s in targets}
 registry=json.loads((ROOT/'release_20261026/source_registry.json').read_text())['regions'];config=json.loads((OUT/'protocol.json').read_text());records=[];attempts=[];bg={v:{s:set() for s in targets} for v in config['versions']};reports={v:{s:set() for s in targets} for v in config['versions']};counts={}
 for region in REGIONS:
  src=Source(registry[region]);counts[region]={};part=[x for x in pool if x['region']==region]
  for entry in part:
   t=np.array(entry['source_timestamps'],float);p,s,c=src.source_target(entry['mmsi'],t);anchor=max(2,len(t)//2-1);at=float(t[anchor]);ids=src.latest(at,entry['mmsi'],allowed[entry['split']]);dist=np.linalg.norm(src.p[ids]-p[anchor],axis=1);angle=np.abs(wrap(src.c[ids]-c[anchor]));ds=np.abs(src.s[ids]-s[anchor]);base=src.good[ids]&src.valid[ids]&(src.w[ids]>=.3)&(src.s[ids]>=3)
   prefix=bool(s[anchor]>=3 and np.median(s[:anchor+1])>=3 and anchor>=2 and t[-1]-at>=180)
   for ver,cfg in config['versions'].items():
    bucket=counts[region].setdefault(ver,{}).setdefault(entry['split'],Counter());bucket['candidates']+=1
    ok=base&(dist<=cfg['radius_nm']*NM)&(angle<=cfg['heading_deg'])&(ds<=cfg['delta_speed_kn']);cohort=ids[ok];cohort=cohort[np.argsort(src.ids[cohort],kind='stable')][:cfg['max_members']]
    cs=len(cohort);covered=coverage(registry[region],p[anchor],at,cfg['radius_nm']);bucket['coverage_limited']+=int(not covered)
    reason=None;got=None
    if not prefix:reason='target_prefix_or_duration_ineligible'
    elif cs<cfg['min_members']:reason='eligible_neighbor_count_below_version_minimum'
    else:
     row={**entry,'anchor_step':anchor,'anchor_time_s':at,'source_target_speed_kn':float(s[anchor]),'source_target_course_deg':float(c[anchor]),'source_target_position_m':p[anchor].tolist(),'cohort_mmsi':src.ids[cohort].tolist(),'cohort_report_time_s':src.t[cohort].tolist(),'cohort_count':cs,'prefix_ok':prefix,'accepted':True}
     got,reason=build_one(src,row,allowed[entry['split']])
     if got is not None:
      arrays,proof=got;_,_,align=align_info(arrays)
      if any(x['pre_grid_count']<2 or x['suffix_grid_count']<2 for x in align):reason='member_insufficient_target_grid_alignment';got=None
      else:
       stress,reason=stress_views(arrays,entry['window_id'])
       if stress is None:got=None
    audit={'region':region,'split':entry['split'],'window_id':entry['window_id'],'mmsi':entry['mmsi'],'version':ver,'anchor_cohort_count':cs,'coverage_complete':covered,'reason':reason}
    attempts.append(audit)
    if got is None:bucket['rejected_'+str(reason)]+=1;continue
    views,sp=stress;bucket['accepted']+=1;case_dir=OUT/'cases'/ver/region/entry['split'];case_dir.mkdir(parents=True,exist_ok=True);files={}
    for view,data in [('base',arrays),('natural',views['natural']),('suffix',views['suffix'])]:
     path=case_dir/(entry['window_id']+'_'+view+'.npz');assert not path.exists();np.savez_compressed(path,**data);files[view]={'path':str(path.relative_to(OUT)),'sha256':sha(path)}
    mids=list(map(str,arrays['cohort_mmsi']));bg[ver][entry['split']].update(mids)
    reports[ver][entry['split']].update((str(m),float(t0)) for m,t0 in zip(arrays['neighbor_mmsi'],arrays['source_neighbor_t']))
    for v in ('natural','suffix'):
     bucket[v+'_hit_sources']+=int(any(x['indices'][v]['suffix_grid_hits']>0 for x in sp['members']));bucket[v+'_numeric_sources']+=int(any(x['indices'][v]['numeric_changed'] for x in sp['members']))
    records.append({**entry,'version':ver,'coverage_complete':covered,'cohort_mmsi':mids,'cohort_anchor_time_s':row['cohort_report_time_s'],'files':files,'stress':sp,'quartet_innovation_error_mps':proof['max_innovation_error_mps']})
  print('REGION',region,json.dumps(counts[region],ensure_ascii=False),flush=True)
 # Actual target vs other split background and target identity isolation.
 for ver in config['versions']:
  for s in targets:
   other=set().union(*(targets[x] for x in targets if x!=s));assert not bg[ver][s]&other,(ver,s)
 overlaps={}
 for ver in config['versions']:
  overlaps[ver]={}
  for a,b in [('train','val'),('train','test'),('val','test')]:overlaps[ver][a+'_'+b]={'shared_background_mmsi':len(bg[ver][a]&bg[ver][b]),'shared_background_reports':len(reports[ver][a]&reports[ver][b])}
 with (OUT/'construction_attempts.jsonl').open('x',encoding='utf8') as f:
  for x in attempts:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 jsonwrite(OUT/'dataset_manifest.json',{'status':'constructed_before_model_training_or_scores','protocol_sha256':sha(OUT/'protocol.json'),'selection_manifest_sha256':sha(OUT/'candidate_pool_manifest.json'),'construction_lock_sha256':sha(OUT/'construction_lock.json'),'counts':counts,'background_overlap':overlaps,'samples':records,'attempts_file':'construction_attempts.jsonl','attempts_sha256':sha(OUT/'construction_attempts.jsonl')})
 print('BUILD_COMPLETE','sources',len(records),'overlaps',json.dumps(overlaps),flush=True)
if __name__=='__main__':main()
