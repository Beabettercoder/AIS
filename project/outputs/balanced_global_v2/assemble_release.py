"""Freeze the verified release, provenance bundle and review package."""
from pathlib import Path
import hashlib,json,shutil,zipfile
import pandas as pd

BASE=Path(__file__).resolve().parent
REL=BASE/'release_20261026'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())

sel=pd.read_parquet(REL/'selected_windows.parquet');groups=pd.read_parquet(REL/'global_all_mmsi_split.parquet')
selection=read(REL/'selection_report.json');registry=read(REL/'source_registry.json')
strict=read(REL/'independent_release_audit.json');scene=read(REL/'independent_scene_release_audit.json')
assert strict['status']==scene['status']=='passed'
assert strict['verified_windows']==scene['verified_windows']==len(sel)==21600
assert scene['same_source_windows_as_strict_verified']
for a in strict['cross_split_checks'].values():assert not any(a.values())
assert not any(strict['repeated_target_observation_events'].values())
assert not any(scene['repeated_target_observation_events'].values())
assert not groups.mmsi.duplicated().any()
preservation=read(BASE/'source_preservation_verification.json')
assert preservation['nine_original_files_unchanged'] and preservation['original_npz_hashes_unchanged']
assert preservation['original_npz_files_verified']==39
mapmeta=read(REL/'figures/benchmark_geography_metadata.json')
assert mapmeta['mode']=='verified_manifest_map' and mapmeta['selected_windows_sha256']==sha(REL/'selected_windows.parquet')
assert set(m['key'] for m in mapmeta['regions'])==set(sel.region)

evidence=REL/'evidence';evidence.mkdir(exist_ok=True)
paths=['source_preservation_verification.json','unit_quality_audit.json','west/raw_extraction_manifest.json','east/existing_raw_manifest.json','east/source_protocol.json','east/dma_normalization.json','east/dma_capacity_verified.json','east/dma_output_validation.json','east/norway_api_acquisition.json','east/norway_expansion_validation.json','visuals/map_assets/ne_10m_land_source.json']
for rel in paths:
 p=BASE/rel;dest=evidence/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
code=REL/'reproducibility';code.mkdir(exist_ok=True)
for rel in ['build_candidates.py','build_source_registry.py','build_features.py','build_features_scene.py','audit_units.py','assemble_release.py','sampling/balanced_sampler.py','sampling/test_balanced_sampler.py','context/context_features.py','context/test_context_features.py','validation/verify_release.py','validation/verify_scene_release.py','validation/test_release_verifier.py','validation/test_scene_verifier.py','visuals/render_benchmark_map.py']:
 p=BASE/rel
 if p.exists():
  dest=code/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)

regions=[];contexts={}
for name,sub in sel.groupby('region',sort=True):
 m=registry['regions'][name];assert sub.mmsi.nunique()==300 and len(sub)==2700
 assert sub.groupby('mmsi').size().max()<=20
 assert sha(Path(m['canonical_file']))==m['sha256']
 for split,nv,nw in [('train',210,1890),('val',45,405),('test',45,405)]:
  v=sub[sub.split==split];assert v.mmsi.nunique()==nv and len(v)==nw
 regions.append({'region':name,'display_name':m['display_name'],'continent':m['continent'],'core_bbox':m['core_bbox'],'source_days_utc':sorted(m['coverage_bbox_by_utc_day']),'raw_source_rows':selection['source_files'][name]['raw_rows'],'target_vessels':300,'source_windows':2700})
 contexts[name]={}
 for protocol,directory in [('strict_all_vessels','features'),('target_disjoint_causal_scene','scene_features')]:
  r=read(REL/directory/name/'feature_report.json')
  for split,hashes in r['output_hashes'].items():
   assert sha(REL/directory/name/f'{split}.npz')==hashes['npz_sha256']
   assert sha(REL/directory/name/f'{split}_context_audit.jsonl.gz')==hashes['audit_sha256']
  contexts[name][protocol]=r['splits']
hemis={'west':sum((r['core_bbox'][0]+r['core_bbox'][2])/2<0 for r in regions),'east':sum((r['core_bbox'][0]+r['core_bbox'][2])/2>0 for r in regions)}
assert hemis=={'west':4,'east':4}
summary={'version':'balanced_global_v2_release_20261026','status':'verified_observed_source_foundation','regions':regions,'hemisphere_region_counts':hemis,'source_windows':len(sel),'unique_target_mmsi':int(sel.mmsi.nunique()),'region_vessel_memberships':2400,'unique_observed_background_and_target_ids':len(groups),'raw_source_rows_by_region_sum':sum(r['raw_source_rows'] for r in regions),'selected_target_observations':strict['verified_raw_steps'],'split_windows':sel.groupby('split').size().to_dict(),'split_unique_target_mmsi':sel.groupby('split').mmsi.nunique().to_dict(),'max_source_windows_per_vessel_per_region':20,'cross_region_max_per_vessel':strict['maximum_windows_per_global_mmsi'],'feature_views':{'features':'Strict all-vessel inductive isolation, including every accessed background identity','scene_features':'Same source windows; globally disjoint targets; train excludes held-out targets, val excludes test targets, test retains full causal scene. Background identities may recur.'},'labels_and_experiments':'Unlabeled observed-source foundation. No synthetic anomaly injection, model fitting or performance replacement occurred in this release. The two feature views do not increase the independent sample count.','raw_sampling':registry['sampling_policy'],'context_density_and_validity':contexts,'independent_audits':{'strict':{'file':'independent_release_audit.json','neighbor_records_checked':strict['verified_neighbor_edges'],'sha256':sha(REL/'independent_release_audit.json')},'scene':{'file':'independent_scene_release_audit.json','neighbor_records_checked':scene['verified_neighbor_edges'],'sha256':sha(REL/'independent_scene_release_audit.json')}},'preserved_historical_files':48,'server_release_path':str(REL),'review_package_scope':'Metadata, figures, NumPy feature arrays, source-window pointers, reports and code. Large per-neighbor audit streams and raw sources remain at the server paths recorded here.'}
(REL/'release_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
lines=['AIS区域均衡数据基础版本 2026-10-26','',f'正式入选8个区域，每域300艘目标船、2700个不重叠原始窗口。合计21600窗、{summary["unique_target_mmsi"]}个不同目标MMSI。','北美4区：Houston–Galveston、New York、Los Angeles、Puget Sound。','欧洲3区：Great Belt、Haugesund、Kattegat。亚洲1区：Shanghai。','','各域训练/验证/测试分别为210/45/45艘船、1890/405/405窗。','同一MMSI全局只属于一个划分；每域每船最多20窗。','原始窗口跨度11—23分钟、8—20个观测，相邻保留点至少60秒，不插值。','跨域有15艘船出现于两个区域，仍处于同一划分；没有重复使用目标原始观测。','','features/：严格船舶隔离版，目标和所有读取的背景身份跨集合零交集。','scene_features/：同一窗口的场景上下文版，训练排除验证/测试目标，验证排除测试目标，测试保留完整可观测现场；背景身份允许跨集合复用。','两种视图使用相同目标窗口，不能当成两倍独立样本。','','所有窗口保留原始文件、MMSI、绝对时间和Parquet物理行索引。','独立核验逐条检查所有目标观测、原始7维特征、时间、填充、邻船身份、最新因果报告与哈希。','48个旧原始文件及实验NPZ的哈希与历史记录一致。','','本版本完成数据基础、划分、可信上下文和地图；异常标签注入与模型重训尚未在此版本执行。','服务器完整版本：'+str(REL),'本地核验包包含清单、特征数组、报告、代码和地图。大型逐邻船审计流及原始数据保留在服务器。']
(REL/'数据版本说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
files={str(p.relative_to(REL)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(REL.rglob('*')) if p.is_file() and p.name not in ['release_file_manifest.json']}
(REL/'release_file_manifest.json').write_text(json.dumps({'files':files,'scope':'All release files except this manifest; source canonical files are hashed in source_registry.json'},indent=2,ensure_ascii=False))
package=BASE/'balanced_global_v2_review_20261026.zip'
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for p in sorted(REL.rglob('*')):
  if p.is_file() and not p.name.endswith('_context_audit.jsonl.gz'):
   z.write(p,str(p.relative_to(REL)),compress_type=zipfile.ZIP_STORED if p.suffix in ['.npz','.png','.parquet'] else zipfile.ZIP_DEFLATED)
package_report={'path':str(package),'sha256':sha(package),'bytes':package.stat().st_size,'zip_crc_passed':False,'files':0}
with zipfile.ZipFile(package) as z:
 assert z.testzip() is None;package_report.update(zip_crc_passed=True,files=len(z.namelist()))
(BASE/'review_package_manifest.json').write_text(json.dumps(package_report,indent=2))
print(json.dumps({'summary':str(REL/'release_summary.json'),'package':package_report},indent=2),flush=True)
