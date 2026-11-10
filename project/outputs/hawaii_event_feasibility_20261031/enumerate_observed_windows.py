import csv,json,datetime,statistics,hashlib
from pathlib import Path
root=Path(r'E:\AIS\outputs\hawaii_event_feasibility_20261031');f=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8-sig'));src=Path(r'E:\AIS\outputs\external_event_evidence_20261031\hawaii_case2\Hawaii_2017_01.csv');ids={x['mmsi'] for x in f['events']};rows={k:[] for k in ids}
with src.open(encoding='utf-8-sig',newline='') as h:
 for r in csv.DictReader(h):
  if r['MMSI'] in ids:
   try:
    t=int(datetime.datetime.fromisoformat(r['datetime_utc']).timestamp());s=float(r['speed_over_ground_knots']);c=float(r['course_over_ground_deg']);lat=float(r['lat']);lon=float(r['lon'])
    if 0<=s<102.2 and 0<=c<360 and -90<=lat<=90 and -180<=lon<=180:rows[r['MMSI']].append((t,s))
   except (ValueError,TypeError):pass
out=[]
for e in f['events']:
 v=sorted(rows[e['mmsi']]);v=list({k:next(x for x in v if x[0]==k) for k in sorted(set(x[0] for x in v))}.values());st=int(datetime.datetime.fromisoformat(e['start_utc']).timestamp());en=int(datetime.datetime.fromisoformat(e['end_utc']).timestamp()); seg=[];a=[]
 for x in v:
  if a and x[0]-a[-1][0]>1800:seg.append(a);a=[]
  a.append(x)
 if a:seg.append(a)
 for g in seg:
  begin=g[0][0]
  while begin<=g[-1][0]:
   subset=[x for x in g if begin<=x[0]<begin+1380];begin+=1380;take=[];last=-10**15
   for x in subset:
    if x[0]-last>=60:take.append(x);last=x[0]
   take=take[-20:]
   if not take or not st<=take[-1][0]<=en or len(take)<8:continue
   t=[x[0] for x in take];s=[x[1] for x in take];anchor=max(2,len(t)//2-1);prefix=s[anchor]>=3 and statistics.median(s[:anchor+1])>=3 and t[-1]-t[anchor]>=180
   out.append({'incident':e['incident'],'name':e['name'],'mmsi':e['mmsi'],'start_utc':datetime.datetime.fromtimestamp(t[0],datetime.timezone.utc).isoformat(),'end_utc':datetime.datetime.fromtimestamp(t[-1],datetime.timezone.utc).isoformat(),'n':len(t),'max_sog':max(s),'median_sog':statistics.median(s),'n_moving':sum(q>=2 for q in s),'timestamps':t,'source_foundation_ok':sum(q>=2 for q in s)>=4 and max(s)>=3,'duration_gap_ok':t[-1]-t[0]>=660 and max(y-x for x,y in zip(t,t[1:]))<=300,'target_anchor_ok':prefix})
(root/'observed_event_windows.json').write_text(json.dumps({'prior_protocol_sha256':hashlib.sha256((root/'frozen_protocol.json').read_bytes()).hexdigest(),'scope':'pre-score OOD diagnostic all Sandia-associated event cells with >=8 raw observations; no score used; window-level outcome/contamination labels not assigned','windows':out},indent=2),encoding='utf8')
from collections import Counter
for name in sorted({x['name'] for x in out}):
 w=[x for x in out if x['name']==name];print(name,len(w),'full_target_domain',sum(z['source_foundation_ok'] and z['duration_gap_ok'] and z['target_anchor_ok'] for z in w),'minmedmax_heading?',flush=True)
