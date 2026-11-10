import csv,json,hashlib,datetime,statistics,math
from pathlib import Path
from collections import Counter
root=Path(r'E:\AIS\outputs\hawaii_event_feasibility_20261031')
p=root/'frozen_protocol.json';f=json.loads(p.read_text(encoding='utf-8-sig'))
src=Path(r'E:\AIS\outputs\external_event_evidence_20261031\hawaii_case2\Hawaii_2017_01.csv')
h=hashlib.sha256()
with src.open('rb') as a:
 for b in iter(lambda:a.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==f['input_file_sha256']
events=f['events']; ids={e['mmsi'] for e in events}; rows={x:[] for x in ids}
with src.open(encoding='utf-8-sig',newline='') as a:
 for row in csv.DictReader(a):
  if row['MMSI'] in ids:
   try:
    t=datetime.datetime.fromisoformat(row['datetime_utc']);s=float(row['speed_over_ground_knots']);c=float(row['course_over_ground_deg']);lat=float(row['lat']);lon=float(row['lon'])
    if 0<=s<102.2 and 0<=c<360 and -90<=lat<=90 and -180<=lon<=180:
     rows[row['MMSI']].append((int(t.timestamp()),s,c,lat,lon,row.get('heading_deg','')))
   except (ValueError,TypeError):pass
out=[];wins=[]
for e in events:
 data=sorted(rows[e['mmsi']],key=lambda x:x[0]);ded=[];seen=set()
 for x in data:
  if x[0] not in seen:seen.add(x[0]);ded.append(x)
 rows[e['mmsi']]=ded
 st=int(datetime.datetime.fromisoformat(e['start_utc']).timestamp()); en=int(datetime.datetime.fromisoformat(e['end_utc']).timestamp());ev=[x for x in ded if st<=x[0]<=en]
 seg=[];acc=[]
 for x in ded:
  if acc and x[0]-acc[-1][0]>1800:seg.append(acc);acc=[]
  acc.append(x)
 if acc:seg.append(acc)
 eligible=[];anchor=[];cells=0
 for g in seg:
  begin=g[0][0]
  while begin<=g[-1][0]:
   subset=[x for x in g if begin<=x[0]<begin+1380];begin+=1380
   take=[];last=-10**15
   for x in subset:
    if x[0]-last>=60:take.append(x);last=x[0]
   take=take[-20:]
   if not take or not st<=take[-1][0]<=en:continue
   cells+=1
   if len(take)<8 or sum(x[1]>=2 for x in take)<4 or max(x[1] for x in take)<3:continue
   row={'incident':e['incident'],'name':e['name'],'mmsi':e['mmsi'],'start_utc':datetime.datetime.fromtimestamp(take[0][0],datetime.timezone.utc).isoformat(),'end_utc':datetime.datetime.fromtimestamp(take[-1][0],datetime.timezone.utc).isoformat(),'n':len(take),'max_sog':max(x[1] for x in take),'n_moving':sum(x[1]>=2 for x in take),'timestamps':[x[0] for x in take]};eligible.append(row);wins.append(row)
   a=max(2,len(take)//2-1); condition=take[a][1]>=3 and statistics.median([x[1] for x in take[:a+1]])>=3 and take[-1][0]-take[a][0]>=180
   if condition:anchor.append(row)
 validh=0
 for x in ev:
  try: v=float(x[5]);validh+=int(math.isfinite(v) and 0<=v<360)
  except (ValueError,TypeError):pass
 out.append(dict(incident=e['incident'],name=e['name'],mmsi=e['mmsi'],event_points=len(ev),valid_heading_points=validh,event_min_sog=min((x[1] for x in ev),default=None),event_median_sog=statistics.median([x[1] for x in ev]) if ev else None,event_max_sog=max((x[1] for x in ev),default=None),event_ge2=sum(x[1]>=2 for x in ev),event_ge3=sum(x[1]>=3 for x in ev),event_cells=cells,foundation_eligible_windows=len(eligible),target_anchor_pass_windows=len(anchor)))
 print(out[-1],flush=True)
(root/'eligibility.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':h.hexdigest(),'cases':out,'foundation_eligible_windows':wins,'warning':'Targets only. No neighbor or affine qualification has yet been evaluated. No model scores used.'},indent=2),encoding='utf8')
