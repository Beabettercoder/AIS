from pathlib import Path
import json
from build_candidates import OUT, REGIONS, sha

west=json.loads((OUT/'west/raw_extraction_manifest.json').read_text())
regions={}
names={'houston':'Houston–Galveston','ny':'New York','la':'Los Angeles','puget':'Puget Sound','shanghai':'Shanghai','haugesund':'Haugesund','great_belt':'Great Belt','oresund':'Oresund','kattegat':'Kattegat'}
for r,(rel,core) in REGIONS.items():
 p=OUT/rel
 meta={'display_name':names[r],'canonical_file':str(p),'sha256':sha(p),'core_bbox':core}
 if r in west['regions']:
  w=west['regions'][r];cov=w['coverage_bbox'];bbox=[cov['lon'][0],cov['lat'][0],cov['lon'][1],cov['lat'][1]]
  meta.update(coverage_bbox_by_utc_day={f'2024-07-{d:02}':bbox for d in [15,16,17]},provider='NOAA / BOEM MarineCadastre',source_url='https://marinecadastre.gov/ais/',continent='North America',source_evidence=str(OUT/'west/raw_extraction_manifest.json'))
 elif r=='haugesund':
  meta.update(coverage_bbox_by_utc_day={f'2024-01-{d:02}':core for d in range(3,8)},provider='Norwegian Coastal Administration',source_url='https://kystdatahuset.kystverket.no/ws/swagger/index.html',continent='Europe',source_evidence=str(OUT/'east/norway_expansion_validation.json'))
 elif r=='shanghai':
  meta.update(coverage_bbox_by_utc_day={f'2023-09-{d}':core for d in [22,23]},coverage_evidence='Published data extent, not a documented receiver footprint. Conservative rectangular boundary treatment.',provider='Wang / Dalian Maritime University via Mendeley Data',source_url='https://data.mendeley.com/datasets/drntjjtcv8/1',license='CC BY 4.0',continent='Asia',source_evidence=str(OUT.parent.parent/'data/raw/external/mendeley_drntjjtcv8_v1_shanghai/Shanghai_manifest.json'))
 elif r=='great_belt':
  meta.update(coverage_bbox_by_utc_day={f'2024-08-{d}':core for d in [15,16]},provider='Danish Maritime Authority',source_url='https://www.dma.dk/safety-at-sea/navigational-information/ais-data',continent='Europe',source_evidence=str(OUT/'east/existing_raw_manifest.json'))
 regions[r]=meta
dma=OUT/'east/dma_normalization.json'
if dma.exists():
 for m in json.loads(dma.read_text()):
  if 'region' not in m:continue
  r=m['region'];regions[r]={'display_name':names[r],'canonical_file':m['path'],'sha256':m['sha256'],'core_bbox':m['core_bbox'],'coverage_bbox_by_utc_day':m['coverage_bbox_by_utc_day'],'provider':'Danish Maritime Authority','source_url':'https://www.dma.dk/safety-at-sea/navigational-information/ais-data','continent':'Europe','source_evidence':str(dma)}
obj={'schema':'balanced_global_v2_source_registry','regions':regions,'coordinate_system':'WGS84 geographic; features use region-centered local equirectangular meters','raw_data_mutated':False,'sampling_policy':{'window_seconds':1380,'minimum_duration_seconds':660,'minimum_observations':8,'maximum_observations':20,'minimum_report_spacing_seconds':60,'moving_observations_required':4,'moving_threshold_kn':2,'minimum_max_sog_kn':3,'target_region_assignment':'last observation within core ROI','segment_gap_seconds':1800,'source_cells':'Nonoverlapping 23-minute cells per contiguous vessel segment','target_mmsi_pattern':'9 digits starting 2–7','speed_valid_range_kn':'[0,102.2)','course_valid_range_deg':'[0,360)','interpolation':False,'anomaly_labels':'Not assigned in this observed-source foundation; original source windows are not asserted to be manually verified normal'}}
(OUT/'source_registry.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False))
print('Registry:',len(regions),'real data regions',flush=True)
