import json,datetime,hashlib,gzip,csv
from pathlib import Path
root=Path(r'E:\AIS\outputs\hawaii_event_feasibility_20261031'); data=json.loads((root/'eligibility.json').read_text()); events=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8-sig'))['events']; wins=[]
for w in data['foundation_eligible_windows']:
 t=w['timestamps']; a=max(2,len(t)//2-1)
 if t[-1]-t[0]<660 or max(y-x for x,y in zip(t,t[1:]))>300: continue
 wins.append(w)
(root/'candidate_targets.json').write_text(json.dumps({'protocol_addendum_sha256':hashlib.sha256((root/'frozen_protocol_addendum.json').read_bytes()).hexdigest(),'windows':wins},indent=2),encoding='utf8')
ints=[]
for e in events:
 a=datetime.datetime.fromisoformat(e['start_utc'])-datetime.timedelta(seconds=1800);b=datetime.datetime.fromisoformat(e['end_utc'])+datetime.timedelta(seconds=1800)
 ints.append((a.strftime('%Y-%m-%d %H:%M:%S+00:00'),b.strftime('%Y-%m-%d %H:%M:%S+00:00')))
src=Path(r'E:\AIS\outputs\external_event_evidence_20261031\hawaii_case2\Hawaii_2017_01.csv');out=root/'external_event_context.csv.gz';keep=['MMSI','datetime_utc','lat','lon','speed_over_ground_knots','course_over_ground_deg','heading_deg']; n=0
with src.open(encoding='utf-8-sig',newline='') as a,gzip.open(out,'wt',encoding='utf8',newline='') as b:
 reader=csv.DictReader(a);writer=csv.DictWriter(b,fieldnames=keep);writer.writeheader()
 for row in reader:
  t=row['datetime_utc']
  if any(x<=t<=y for x,y in ints): writer.writerow({k:row[k] for k in keep});n+=1
print('rows',n,'windows',len(wins),'gz_bytes',out.stat().st_size)
for e in events:
 print(e['name'],'candidate',sum(w['mmsi']==e['mmsi'] for w in wins))
