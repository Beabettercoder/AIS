"""Continue this concrete acquisition job, then normalize and audit it."""
from pathlib import Path
import os,time,subprocess,sys,json
ROOT=Path('/mnt/sda/WangZijia/AIS')
TOOLS=ROOT/'outputs/balanced_global_v2/tools/east'
RAW=ROOT/'data/raw/external/dma_2024'
OUT=ROOT/'outputs/balanced_global_v2/east'
(OUT/'finish_dma.pid').write_text(str(os.getpid()))
pid=int((RAW/'aisdk-2024-08-17.ranges/download.pid').read_text())
started=time.monotonic()
while True:
 try: os.kill(pid,0)
 except ProcessLookupError: break
 if time.monotonic()-started>7200:raise RuntimeError('acquisition still active after bounded 2h wait; no restart attempted')
 time.sleep(10)
if not (RAW/'aisdk-2024-08-17.zip').exists():raise RuntimeError('Downloader exited without complete ZIP; inspect range_download.log')
subprocess.run([sys.executable,str(TOOLS/'normalize_dma_archives.py')],check=True)
sys.path.insert(0,str(ROOT/'outputs/balanced_global_v2'))
from build_candidates import candidates,capacity
import pandas as pd
report={}
meta=json.loads((OUT/'dma_normalization.json').read_text())
for entry in meta:
 if 'region' not in entry:continue
 name=entry['region'];p=Path(entry['path']);d=pd.read_parquet(p).reset_index(drop=True)
 assert not d.duplicated(['mmsi','timestamp']).any()
 assert d.timestamp.between(1723680000,1723939199).all()
 c=candidates(d,name,entry['core_bbox']);c=c[c.duration_s>=660].reset_index(drop=True)
 c.to_parquet(OUT/f'{name}_capacity_candidates.parquet',index=False)
 report[name]={'file':str(p),'core_bbox':entry['core_bbox'],'coverage_bbox_by_utc_day':entry['coverage_bbox_by_utc_day'],'capacity':capacity(c),'rows':len(d),'unique_mmsi':int(d.mmsi.nunique()),'time_start':int(d.timestamp.min()),'time_end':int(d.timestamp.max())}
 print(name,json.dumps(report[name]),flush=True)
(OUT/'dma_capacity_verified.json').write_text(json.dumps(report,indent=2))
print('DMA_NORMALIZATION_AND_CAPACITY_COMPLETE',flush=True)
