"""Independent replay of quartet marginal pairing, controlled stresses and identity overlap."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,math
import numpy as np
import pandas as pd

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=ROOT/'relation_independent_audit_20261027'
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();RUN=args.run
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def seed(wid,mid):return int.from_bytes(hashlib.sha256(('speed-stress:'+wid+':'+mid).encode()).digest()[:8],'little')
def rank(wid,mid):return hashlib.sha256(('speed-stress-member:'+wid+':'+mid).encode()).digest()

lock=json.loads((RUN/'construction_lock.json').read_text(encoding='utf8'))
for path,h in lock['local_files_sha256'].items():assert sha(RUN/path)==h,path
for path,h in lock['external_dependencies_sha256'].items():assert sha(Path(path))==h,path
manifest=json.loads((RUN/'dataset_manifest.json').read_text(encoding='utf8'))
assert manifest['protocol_sha256']==sha(RUN/'protocol.json') and manifest['construction_lock_sha256']==sha(RUN/'construction_lock.json')
assert sha(RUN/manifest['attempts_file'])==manifest['attempts_sha256']
cp=json.loads((RUN/'candidate_pool_manifest.json').read_text(encoding='utf8'))
selected=pd.read_parquet(ROOT/'release_20261026/selected_windows.parquet',columns=['mmsi','split'])
orig={s:set(selected.loc[selected.split==s,'mmsi'].astype(str)) for s in ('train','val','test')}
targets={s:set(cp['target_ids'][s]) for s in ('train','val','test')}
assert all(not targets[a]&targets[b] for a,b in [('train','val'),('train','test'),('val','test')])
pool={}
for line in (RUN/'candidate_pool.jsonl').read_text(encoding='utf8').splitlines():
 d=json.loads(line);k=(d['region'],d['window_id']);assert k not in pool;pool[k]=d
attempts={};attempt_count=Counter()
with (RUN/manifest['attempts_file']).open(encoding='utf8') as f:
 for line in f:
  d=json.loads(line);k=(d['version'],d['region'],d['window_id']);assert k not in attempts;attempts[k]=d
  attempt_count[(d['version'],d['region'],d['split'],'accepted' if d['reason'] is None else 'rejected_'+str(d['reason']))]+=1
assert len(attempts)==2*len(pool)
for region,versions in manifest['counts'].items():
 for version,splits in versions.items():
  for split,counts in splits.items():
   assert counts['candidates']==sum(v for (V,R,S,_),v in attempt_count.items() if (V,R,S)==(version,region,split))
   for key,value in counts.items():
    if key=='accepted' or key.startswith('rejected_'):assert value==attempt_count[(version,region,split,key)],(region,version,split,key)

bg=defaultdict(set);reports=defaultdict(set);source_time=defaultdict(list);sample_seen=set();actual=Counter();counts=Counter();max_innovation_error=0.
STATIC=('target_t','target_pos','target_sog_kn','target_cog_deg','target_mask','neighbor_t','neighbor_pos','neighbor_cog_deg','neighbor_mask','neighbor_mmsi','member_offsets')
for entry in manifest['samples']:
 ver,region,split,wid,mid=entry['version'],entry['region'],entry['split'],entry['window_id'],entry['mmsi']
 key=(ver,region,wid);assert key not in sample_seen;sample_seen.add(key);assert key in attempts and attempts[key]['reason'] is None
 source=pool[(region,wid)];assert (mid,split,entry['source_timestamps'])==(source['mmsi'],source['split'],source['source_timestamps'])
 views={}
 for view in ('base','natural','suffix'):
  loc=RUN/entry['files'][view]['path'];assert sha(loc)==entry['files'][view]['sha256']
  with np.load(loc,allow_pickle=False) as z:views[view]={k:z[k] for k in z.files}
 b=views['base'];assert np.array_equal(b['y'],[0,1,1,0]);assert str(b['target_mmsi'])==mid and str(b['source_window_id'])==wid
 assert str(b['region'])==region and str(b['split'])==split
 if 'meta_target_factor' in b:assert np.array_equal(b['meta_target_factor'],[0,0,1,1])
 if 'meta_cohort_factor' in b:assert np.array_equal(b['meta_cohort_factor'],[0,1,0,1])
 assert np.array_equal(b['source_target_t'],np.array(source['source_timestamps'],float))
 assert np.array_equal(b['target_t'][0],b['target_t'][1]) and all(np.array_equal(b['target_t'][0],b['target_t'][j]) for j in (2,3))
 for k in ('target_pos','target_sog_kn','target_cog_deg','target_mask'):
  assert np.array_equal(b[k][0],b[k][1]) and np.array_equal(b[k][2],b[k][3]),(key,k)
 for k in ('neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','neighbor_mask'):
  assert np.array_equal(b[k][0],b[k][2]) and np.array_equal(b[k][1],b[k][3]),(key,k)
 members=b['cohort_mmsi'].astype(str).tolist();assert members==entry['cohort_mmsi'] and mid not in members and len(set(members))==len(members)
 assert members==sorted(members) and (4 if ver=='original' else 3)<=len(members)<=8
 other=set().union(*(targets[s]|orig[s] for s in targets if s!=split));assert not set(members)&other
 offs=b['member_offsets'];assert len(offs)==len(members)+1 and int(offs[0])==0 and int(offs[-1])==len(b['neighbor_mmsi'])
 assert np.all(np.diff(offs)>0)
 for j,m in enumerate(members):
  lo,hi=int(offs[j]),int(offs[j+1]);assert set(b['neighbor_mmsi'][lo:hi].astype(str))=={m}
  assert np.all(np.diff(b['neighbor_t'][0,lo:hi])>0)
 bg[(ver,split)].update(members);reports[(ver,split)].update(zip(b['neighbor_mmsi'].astype(str),b['source_neighbor_t'].astype(float)))
 source_time[(ver,split,region)].append((entry['source_start'],entry['source_end']))
 for view in ('natural','suffix'):
  v=views[view]
  for field in STATIC:assert np.array_equal(v[field],b[field]),(key,view,field)
  for field in ('y','cohort_mmsi','target_mmsi','source_window_id','split','region'):assert np.array_equal(v[field],b[field])
 t=b['target_t'][0];anchor=max(2,len(t)//2-1);assert entry['stress']['anchor_index']==anchor
 prefix=np.arange(anchor+1);suffix=np.arange(anchor+1,len(t))[-3:];elig=[];align={}
 for j,m in enumerate(members):
  lo,hi=int(offs[j]),int(offs[j+1]);nt=b['neighbor_t'][0,lo:hi];ix=np.searchsorted(nt,t,side='right')-1;safe=np.maximum(ix,0);good=(ix>=0)&(t>=nt[safe])&(t-nt[safe]<=300)
  assert int(good[prefix].sum())>=2 and int(good[suffix].sum())>=2
  directed=sorted(set(int(i) for i in ix[suffix][good[suffix]] if nt[int(i)]>t[anchor] and int(i)+1<len(nt)))
  natural=[int(i) for i in np.flatnonzero(nt>t[anchor]) if int(i)+1<len(nt)]
  align[j]=(nt,ix,good,directed,natural)
  if directed and natural:elig.append(j)
 elig.sort(key=lambda j:rank(wid,members[j]));chosen=elig[:min(math.ceil(len(members)/2),len(elig))]
 proof=entry['stress']['members'];assert [x['member_index'] for x in proof]==chosen and entry['stress']['selected_count']==len(proof)
 expected={'natural':b['neighbor_sog_kn'].copy(),'suffix':b['neighbor_sog_kn'].copy()};strict_hit={'natural':False,'suffix':False}
 for rec in proof:
  j=rec['member_index'];lo=int(offs[j]);nt,ix,good,directed,natural=align[j]
  assert rec['mmsi']==members[j] and rec['seed']==seed(wid,members[j]);rng=np.random.default_rng(seed(wid,members[j]));factor=float(rng.uniform(3,5))
  assert factor==rec['factor'];choices={'natural':int(rng.choice(natural)),'suffix':int(rng.choice(directed))}
  for view,k in choices.items():
   info=rec['indices'][view];assert info['report_index']==k and np.array_equal(info['times_s'],nt[k:k+2])
   old=b['neighbor_sog_kn'][:,lo+k:lo+k+2];expected[view][:,lo+k:lo+k+2]=np.minimum(old*factor,102.1)
   hit=good[suffix]&(ix[suffix]>=k)&(ix[suffix]<k+2);assert info['suffix_grid_hits']==int(hit.sum())
   changed=bool(np.any(expected[view][:,lo+k:lo+k+2]!=old));assert info['numeric_changed']==changed
   for grid in suffix[hit]:
    used=int(ix[grid]);strict_hit[view]|=bool(np.any(expected[view][:,lo+used]!=b['neighbor_sog_kn'][:,lo+used]))
 for view in ('natural','suffix'):
  assert np.array_equal(views[view]['neighbor_sog_kn'],expected[view]),(key,view)
  actual[(ver,split,view,'effective_suffix_sources')]+=int(strict_hit[view])
  actual[(ver,split,view,'suffix_index_hit_sources')]+=int(any(x['indices'][view]['suffix_grid_hits']>0 for x in proof))
  actual[(ver,split,view,'numerically_changed_sources')]+=int(any(x['indices'][view]['numeric_changed'] for x in proof))
 counts[(ver,split,region)]+=1;max_innovation_error=max(max_innovation_error,float(entry['quartet_innovation_error_mps']))

assert len(sample_seen)==sum(a['reason'] is None for a in attempts.values())==len(manifest['samples'])
overlaps={}
for ver in ('original','relaxed'):
 for a,b in [('train','val'),('train','test'),('val','test')]:
  bb=len(bg[(ver,a)]&bg[(ver,b)]);rr=len(reports[(ver,a)]&reports[(ver,b)]);x=manifest['background_overlap'][ver][a+'_'+b]
  assert bb==x['shared_background_mmsi'] and rr==x['shared_background_reports'];overlaps[ver+'_'+a+'_'+b]={'background_mmsi':bb,'background_reports':rr}
out={'status':'passed','dataset_sha256':sha(RUN/'dataset_manifest.json'),'candidate_sha256':sha(RUN/'candidate_pool.jsonl'),
     'construction_lock_sha256':sha(RUN/'construction_lock.json'),'attempts':len(attempts),'accepted_sources':len(sample_seen),
     'views_per_source':3,'quartet_cases_per_source':4,'source_counts':{'|'.join(k):v for k,v in counts.items()},
     'strict_effective_suffix_counts':{'|'.join(k):v for k,v in actual.items()},'background_overlap':overlaps,
     'max_saved_innovation_error_mps':max_innovation_error,
     'scope':'all accepted saved four-cell observations, source identity, hash-locks, member chronology, natural and directed stress operations; no independent rerun of original source candidate-cohort proximity or report injection generator',
     'script_sha256':sha(Path(__file__))}
OUT.mkdir(parents=True,exist_ok=True);dest=OUT/(RUN.name+'_quartet_audit.json')
with dest.open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k not in ('source_counts','strict_effective_suffix_counts','background_overlap')},indent=2),flush=True)
