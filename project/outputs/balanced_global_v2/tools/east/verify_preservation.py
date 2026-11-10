from pathlib import Path
import hashlib,json,time
ROOT=Path('/mnt/sda/WangZijia/AIS');OUT=ROOT/'outputs/balanced_global_v2'
expected={
 '/mnt/sda/WangZijia/houston_galveston_2024_07/data/raw/noaa_ais/AIS_2024_07_15.zip':'4dd81f5b69e5b1c8ce041f2fe14b7db1443a7a3407d276e6cedfd33dc6ec9d53',
 '/mnt/sda/WangZijia/houston_galveston_2024_07/data/raw/noaa_ais/AIS_2024_07_16.zip':'bda91a05c829ace2cf8f14592262de2891ba72633e24d7a81c340888c5d86fff',
 '/mnt/sda/WangZijia/houston_galveston_2024_07/data/raw/noaa_ais/AIS_2024_07_17.zip':'65f90e0277734ae3d62d88bee5c9a1fef5f93198f6ca2460b962e12351a9ee1d',
 '/mnt/sda/WangZijia/AIS/system3/current/storage/intake/276d8046-ce10-414e-9f5b-e8f4162133e3.bin':'c4ff89b89c69a36d66977ee9ef7d43d439552eb0ba4144a0da114b7ed8371a54',
 '/mnt/sda/WangZijia/AIS/system3/current/storage/intake/93ac07ae-8968-4ff2-9d2f-5760a00bfed4.bin':'21411cee3355fde0e5b961d7e2e9e8c58429f811ea2189ed33ef295fbc22c58c',
 '/mnt/sda/WangZijia/AIS/system3/current/storage/intake/f03b0d9c-b9fd-4866-9f9b-e90f3496ef03.bin':'81db60f713b01f55486ea4e03362336f990308eeb460484bfe0e565e5da9fedb',
 '/mnt/sda/WangZijia/AIS/system3/current/storage/intake/304b3a1f-74ac-405a-8ce9-a6bea83d61dd.bin':'97c464ba2de79b6a5ba9ebd81e7bee8cbc4076d94ed86637f2ed59851240978b',
 '/mnt/sda/WangZijia/AIS/system3/current/storage/intake/4fe7e5ff-a7fb-402a-9eb7-b1429babd903.bin':'1970a5462cd43e526cd92bf74ad4137deeb5b1a0192cc6ecd13906ff41c8766f',
 '/mnt/sda/WangZijia/AIS/system3/current/storage/intake/cb3d089c-c6c8-489b-82a3-1e9a97e819e1.bin':'c4f508c322c9949d458fc6f2c9569b705b77dd93a420848b391d2075120c8775'}
results=[]
for path,want in expected.items():
 h=hashlib.sha256();p=Path(path)
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 result={'path':path,'expected_sha256':want,'actual_sha256':h.hexdigest(),'unchanged':h.hexdigest()==want,'bytes':p.stat().st_size}
 assert result['unchanged'],path
 results.append(result)
historical=ROOT/'outputs/split_label_provenance_20261025/global_mmsi_v1/validation_report.json'
npz_expected=json.loads(historical.read_text())['original_files']
assert len(npz_expected)==39
npz_results=[]
for path,want in npz_expected.items():
 h=hashlib.sha256();p=Path(path)
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 result={'path':path,'expected_sha256':want,'actual_sha256':h.hexdigest(),'unchanged':h.hexdigest()==want,'bytes':p.stat().st_size}
 assert result['unchanged'],path
 npz_results.append(result)
running=[]
names={'download_dma.py','download_norway.py','download_dma_ranges.py','finish_dma.py'}
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:args=proc.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
 except (OSError,PermissionError):continue
 if any(Path(a).name in names for a in args):running.append({'pid':int(proc.name),'arguments':args})
parts=[{'path':str(p),'bytes':p.stat().st_size} for p in (ROOT/'data/raw/external/dma_2024').rglob('*.part')]
report={'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'baseline':'Original 2026-10-25 west/raw_extraction_manifest.json source_days and east/existing_raw_manifest.json snapshots; expected hashes embedded in verifier','nine_original_files_unchanged':all(x['unchanged'] for x in results),'files':results,'running_acquisition_processes':running,'retained_part_file_count':len(parts),'retained_part_total_bytes':sum(x['bytes'] for x in parts),'retained_parts':parts,'sqlite_note':'These acquisition/normalization/verifier programs do not open production SQLite. No claim is made that the live application has not updated its own database.','cleanup':'No original files, complete archives, or reusable partial chunks deleted.'}
report.update({'original_npz_baseline':str(historical),'original_npz_files_verified':len(npz_results),'original_npz_hashes_unchanged':all(x['unchanged'] for x in npz_results),'original_npz_files':npz_results})
(OUT/'source_preservation_verification.json').write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k not in ['files','retained_parts','original_npz_files']},indent=2),flush=True)
