"""Train-source controlled report diagnostics; injection metadata is evaluation only."""
import sys,json,collections
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
OLD=ROOT/'balanced_global_v2/relation_dual_pressure_v2_20261027'
OUT=ROOT/'heading_evidence_pilot_20261030'
H=ROOT/'heading_coverage_20261030'
manifest=json.loads((OLD/'dataset_manifest_combined.json').read_text())
rows=sorted([r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='train'],key=lambda r:(r['region'],r['window_id']))
with np.load(H/'heading_train.npz',allow_pickle=False) as f: h={k:f[k] for k in f.files}
hi={str(w):i for i,w in enumerate(h['window_id'])};assert set(hi)==set(r['window_id'] for r in rows)
groups=collections.defaultdict(lambda:{'n':0,'target':set(),'neighbor':set(),'heading_valid':0,'yaw_valid':0,'angle':[],'yaw':[]})
paired={'n_changed':0,'n_heading_pair_valid':0,'heading_same_by_source_cache_construction':True,'cog_maxdiff_deg':0.}
def observe(g,row,member,base,hd,valid):
 g['n']+=1;g['target'].add(row['mmsi']);g['neighbor'].add(str(member))
 if valid:
  g['heading_valid']+=1;g['angle'].append(abs((hd[0]-base[0]+180)%360-180))
 if valid and np.isfinite(hd[1]):g['yaw_valid']+=1;g['yaw'].append(abs((hd[0]-hd[1]+180)%360-180)/base[1])
for ri,row in enumerate(rows):
 ix=hi[row['window_id']];lo,up=h['sample_neighbor_offsets'][ix:ix+2];hs=h['neighbor_heading'][lo:up];hv=h['neighbor_valid'][lo:up]
 with np.load(OLD/row['files']['base']['path'],allow_pickle=False) as f:b={k:f[k] for k in ('neighbor_t','neighbor_sog_kn','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi')}
 with np.load(OLD/row['files']['suffix']['path'],allow_pickle=False) as f:s={k:f[k] for k in ('neighbor_t','neighbor_sog_kn','neighbor_cog_deg')}
 assert np.array_equal(b['neighbor_t'],s['neighbor_t']) and np.array_equal(b['neighbor_cog_deg'],s['neighbor_cog_deg'])
 assert np.array_equal(b['neighbor_t'][0],h['neighbor_t'][lo:up]);assert np.array_equal(b['neighbor_mmsi'].astype(str),h['neighbor_mmsi'][lo:up].astype(str))
 for case in range(4):
  for member,start,end in zip(b['cohort_mmsi'],b['member_offsets'][:-1],b['member_offsets'][1:]):
   start,end=int(start),int(end);tt=b['neighbor_t'][case,start:end];v=b['neighbor_sog_kn'][case,start:end];v1=s['neighbor_sog_kn'][case,start:end];c=b['neighbor_cog_deg'][case,start:end];hd=hs[start:end];ok=hv[start:end];change=np.abs(v1-v)>1e-9
   for q in range(1,len(tt)):
    dt=tt[q]-tt[q-1]
    if not 0<dt<=300:continue
    good=bool(ok[q] and np.isfinite(c[q]));anglebase=(float(c[q]),float(dt));hdpair=(float(hd[q]) if ok[q] else np.nan,float(hd[q-1]) if ok[q-1] else np.nan)
    if change[q]:
     paired['n_changed']+=1
     if ok[q]:paired['n_heading_pair_valid']+=1
     paired['cog_maxdiff_deg']=max(paired['cog_maxdiff_deg'],abs(float(c[q]-s['neighbor_cog_deg'][case,start+q])))
     if v[q]<2:observe(groups['changed_current_baseline_under_2kn'],row,member,anglebase,hdpair,good)
    if not change[q] and not change[q-1]:
     if v[q]<2:observe(groups['unchanged_current_baseline_under_2kn'],row,member,anglebase,hdpair,good)
     if q>=3 and np.median(v[q-3:q])<2 and abs(v[q]-v[q-1])>=1:observe(groups['unchanged_history_under_2kn_speed_change_ge1kn'],row,member,anglebase,hdpair,good)
def finalize(d):
 for g in d.values():
  g['target']=len(g['target']);g['neighbor']=len(g['neighbor']);
  for k in ('angle','yaw'):
   x=np.asarray(g.pop(k));g[k+'_median']=float(np.median(x)) if len(x) else None;g[k+'_p90']=float(np.quantile(x,.9)) if len(x) else None
 return d
out={'scope':'train-only controlled report diagnostics; report records may repeat; no clean/true fault labels; original heading reused identically across four cells and stress views','paired_invariance':paired,'groups':finalize(groups)}
(OUT/'train_report_probe.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
