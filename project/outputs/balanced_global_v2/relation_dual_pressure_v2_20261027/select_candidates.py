"""Select frozen source windows without labels or model scores."""
from pathlib import Path
from collections import Counter
import gzip, hashlib, json
import pandas as pd

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=ROOT/'relation_dual_pressure_v2_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 reg=json.loads((ROOT/'release_20261026/source_registry.json').read_text(encoding='utf8'))['regions']
 prev=pd.read_parquet(ROOT/'release_20261026/selected_windows.parquet');selected=set(prev.mmsi.astype(str))
 prior_selected_windows=set(prev.window_id.astype(str));prior_report=set()
 for row in prev.itertuples(index=False):
  for t in json.loads(row.timestamps):prior_report.add((str(row.mmsi),int(t)))
 groups=pd.read_parquet(ROOT/'release_20261026/global_all_mmsi_split.parquet');mapping={str(a):str(b) for a,b in zip(groups.mmsi,groups.split)}
 exposure={s:set() for s in ['train','val','test']};inputs=[]
 for r in REGIONS:
  for s in ['train','val','test']:
   path=ROOT/'labeling/labeled_v1/features_scene'/r/(s+'_context_audit.jsonl.gz')
   with gzip.open(path,'rt',encoding='utf8') as f:
    for line in f:
     o=json.loads(line);exposure[s].update(map(str,o['plain_context_exposure_mmsi']));exposure[s].update(map(str,o['context_exposure_mmsi']))
   inputs.append({'file':str(path),'sha256':sha(path)})
 trainval=exposure['train']|exposure['val'];all_results=[];counts={};source_ids={s:set() for s in ['train','val','test']};source_paths=[]
 for r in REGIONS:
  path=ROOT/'candidates'/f'{r}.parquet';df=pd.read_parquet(path);source_paths.append({'path':str(path),'sha256':sha(path),'canonical_source':reg[r]['canonical_file'],'canonical_sha256':reg[r]['sha256']})
  df['mmsi']=df.mmsi.astype(str);df['split']=df.mmsi.map(mapping);df['known_release_target']=df.mmsi.isin(selected)
  df['prior_trainval_background']=df.mmsi.isin(trainval);df['prior_test_background']=df.mmsi.isin(exposure['test'])
  df['old_window']=df.window_id.astype(str).isin(prior_selected_windows)
  qualified=df.all_points_core.astype(bool)&(df.duration_s>=660)&(df.max_gap_s<=300)&df.split.notna()
  allowed_identity=df.known_release_target|((df.split=='test')&~df.prior_trainval_background)
  df['reject_reason']='';df.loc[~qualified,'reject_reason']='invalid_source_window_or_unmapped_identity';df.loc[qualified&~allowed_identity,'reject_reason']='identity_not_in_fixed_target_pool'
  df['rank']=df.window_id.astype(str).map(lambda x:hashlib.sha256(('rel-v1:'+r+':'+x).encode()).hexdigest())
  eligible=df[qualified&allowed_identity].sort_values(['mmsi','rank','window_id'],kind='stable').copy();eligible['within_ship_rank']=eligible.groupby('mmsi').cumcount();chosen=eligible[eligible.within_ship_rank<24]
  chosen_idx=set(chosen.index);df.loc[qualified&allowed_identity&~df.index.isin(chosen_idx),'reject_reason']='per_target_region_hash_cap_24'
  df['chosen']=df.index.isin(chosen_idx);cc=Counter(df.reject_reason);counts[r]={'all_candidates':len(df),'selected':len(chosen),'reject_reasons':dict(cc),'splits':{}}
  for s in ['train','val','test']:
   part=chosen[chosen.split==s];source_ids[s].update(part.mmsi);counts[r]['splits'][s]={'windows':len(part),'target_mmsi':int(part.mmsi.nunique()),'release_target_windows':int(part.known_release_target.sum()),'prior_trainval_context_windows':int(part.prior_trainval_background.sum()),'prior_test_context_windows':int(part.prior_test_background.sum()),'old_release_windows':int(part.old_window.sum())}
  for row in chosen.itertuples(index=False):
   ts=json.loads(row.timestamps);all_results.append({'window_id':str(row.window_id),'base_window_id':str(row.window_id),'region':r,'split':str(row.split),'mmsi':str(row.mmsi),'source_timestamps':ts,'source_start':int(ts[0]),'source_end':int(ts[-1]),'old_release_window':bool(row.old_window),'known_release_target':bool(row.known_release_target),'prior_trainval_background':bool(row.prior_trainval_background),'prior_test_background':bool(row.prior_test_background),'old_release_target_report_overlap_count':sum((str(row.mmsi),int(t)) in prior_report for t in ts),'rank':str(row.rank)})
  rejection=OUT/f'{r}_candidate_rejections.jsonl'
  with rejection.open('x',encoding='utf8') as f:
   for row in df.loc[~df.chosen,['window_id','mmsi','split','reject_reason']].itertuples(index=False):f.write(json.dumps({'region':r,'window_id':str(row.window_id),'mmsi':str(row.mmsi),'split':str(row.split),'reason':str(row.reject_reason)},ensure_ascii=False)+'\n')
  print('SELECT',r,counts[r]['splits'],flush=True)
 for a,b in [('train','val'),('train','test'),('val','test')]:assert not source_ids[a]&source_ids[b],(a,b)
 assert not set(x['window_id'] for x in all_results if x['split']=='test')&set(x['window_id'] for x in all_results if x['split']!='test')
 rows=OUT/'candidate_pool.jsonl'
 with rows.open('x',encoding='utf8') as f:
  for x in sorted(all_results,key=lambda x:(x['region'],x['split'],x['rank'])):f.write(json.dumps(x,ensure_ascii=False)+'\n')
 result={'status':'selected_without_model_scores','protocol_sha256':sha(OUT/'protocol.json'),'candidate_code_sha256':sha(Path(__file__)),
  'release_selected_sha256':sha(ROOT/'release_20261026/selected_windows.parquet'),'global_groups_sha256':sha(ROOT/'release_20261026/global_all_mmsi_split.parquet'),
  'prior_context_audits':inputs,'prior_background_counts':{s:len(v) for s,v in exposure.items()},'prior_trainval_background_union':len(trainval),'sources':source_paths,'counts':counts,'target_ids':{s:sorted(v) for s,v in source_ids.items()},'candidate_pool_sha256':sha(rows)}
 (OUT/'candidate_pool_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 print('CANDIDATE_POOL_COMPLETE',len(all_results),{s:len(v) for s,v in source_ids.items()},flush=True)
if __name__=='__main__':main()
