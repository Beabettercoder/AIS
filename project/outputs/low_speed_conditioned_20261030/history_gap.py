import json,hashlib
from pathlib import Path
import numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/relation_dual_pressure_v2_20261027')
O=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_conditioned_20261030')
m=json.loads((R/'dataset_manifest_combined.json').read_text()); rows=sorted([r for r in m['samples'] if r['version']=='relaxed' and r['split']=='train'],key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())[:400]
values=[]
for row in rows:
 with np.load(R/row['files']['base']['path'],allow_pickle=False) as z:
  t=z['neighbor_t'];s=z['neighbor_sog_kn'];off=z['member_offsets'];
  for case in range(4):
   for lo,hi in zip(off[:-1],off[1:]):
    lo,hi=int(lo),int(hi);tt=t[case,lo:hi];ss=s[case,lo:hi]
    if len(tt)<4:continue
    ix=np.arange(3,len(tt)); dt=tt[ix]-tt[ix-1];h=np.median(np.column_stack((ss[ix-3],ss[ix-2],ss[ix-1])),axis=1);valid=(dt>0)&(dt<=300)&(h<2)&np.isfinite(h)
    for span in (tt[ix]-tt[ix-3])[valid]:values.append((str(row['mmsi']),float(span)))
spans=np.array([x[1] for x in values]); out={'scope':'400 train sources, Base four cases; correlated report rows, h is median prior3 SOG under 2kn; current dt in (0,300s]','n':len(spans),'targets':len(set(x[0] for x in values)),'p50_seconds':float(np.median(spans)),'p90_seconds':float(np.quantile(spans,.9)),'fraction_span_over_300_seconds':float(np.mean(spans>300)),'fraction_span_over_900_seconds':float(np.mean(spans>900))}
(O/'history_gap.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
