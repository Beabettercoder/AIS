import csv,datetime,math,json,hashlib,bisect,collections
from pathlib import Path
src=Path(r'E:\AIS\outputs\external_event_evidence_20261031\hawaii_case2\Hawaii_2017_01.csv')
out=Path(r'E:\AIS\outputs\event_crosscheck_20261031')
assert src.stat().st_size==380885602
start=datetime.datetime.fromisoformat('2017-01-13T14:00:00+00:00')
end=datetime.datetime.fromisoformat('2017-01-13T22:00:00+00:00')
lo=datetime.datetime.fromisoformat('2017-01-13T06:01:56-10:00').astimezone(datetime.timezone.utc)
hi=datetime.datetime.fromisoformat('2017-01-13T09:58:54-10:00').astimezone(datetime.timezone.utc)
assert lo.isoformat()=='2017-01-13T16:01:56+00:00' and hi.isoformat()=='2017-01-13T19:58:54+00:00'
rows=0; target=[]; groups=collections.defaultdict(list)
def num(s):
 try:return float(s)
 except (TypeError,ValueError):return float('nan')
with src.open(encoding='utf8',newline='') as f:
 for r in csv.DictReader(f):
  rows+=1;s=r['datetime_utc'];mid=r['MMSI']
  if not s.startswith('2017-01-13'):continue
  t=datetime.datetime.fromisoformat(s)
  if t.tzinfo is None:raise ValueError('timestamp not timezone aware')
  if mid=='366792000' and lo<=t<=hi:target.append(r)
  if not start<=t<=end:continue
  lat,lon=num(r['lat']),num(r['lon'])
  if not (20.5<=lat<=21.8 and -158.8<=lon<=-157.2):continue
  if mid=='366792000':continue
  groups[mid].append((t.timestamp(),lat,lon,num(r['course_over_ground_deg']),num(r['speed_over_ground_knots'])))
target.sort(key=lambda r:r['datetime_utc'])
for mid in groups:groups[mid].sort(key=lambda x:x[0])
times={mid:[x[0] for x in rs] for mid,rs in groups.items()}
ncounts=[]
for r in target:
 t=datetime.datetime.fromisoformat(r['datetime_utc']).timestamp();lat=num(r['lat']);lon=num(r['lon']);cog=num(r['course_over_ground_deg']); n=0;same=0;moving=0
 for mid,rs in groups.items():
  k=bisect.bisect_right(times[mid],t)-1
  if k<0 or t-rs[k][0]>300:continue
  _,b,c,c2,nsog=rs[k]; distance=math.hypot((b-lat)*111195,(c-lon)*111195*math.cos(math.radians(lat)))
  if distance>3.5*1852:continue
  n+=1
  moving+=int(math.isfinite(nsog) and nsog>=2)
  if 0<=cog<360 and 0<=c2<360 and abs((c2-cog+180)%360-180)<45:same+=1
 ncounts.append({'time':r['datetime_utc'],'neighbors':n,'same_dir':same,'moving':moving})
headvalid=sum(0<=num(r['heading_deg'])<360 for r in target)
vals=sorted(num(r['speed_over_ground_knots']) for r in target if math.isfinite(num(r['speed_over_ground_knots'])))
def median(x):return x[len(x)//2] if len(x)%2 else (x[len(x)//2-1]+x[len(x)//2])/2
result={'source_local':str(src),'source_size':src.stat().st_size,'source_sha256':hashlib.file_digest(src.open('rb'),'sha256').hexdigest(),'total_month_rows':rows,'target_mmsi':'366792000','official_event_id':'2','event_report_time_utc':'2017-01-13T18:00:00+00:00','start_utc':lo.isoformat(),'end_utc':hi.isoformat(),'target_points':len(target),'first_target_utc':target[0]['datetime_utc'] if target else None,'last_target_utc':target[-1]['datetime_utc'] if target else None,'target_names':sorted(set(r['vessel_name'] for r in target)),'incident_num_values':sorted(set(r['incident_num'] for r in target)),'target_heading_valid':headvalid,'target_sog_lt2':sum(x<2 for x in vals),'target_sog_median_kn':median(vals) if vals else None,'context_unique_other_vessels':len(groups),'target_points_any_neighbor':sum(x['neighbors']>=1 for x in ncounts),'target_points_at_least4_neighbors':sum(x['neighbors']>=4 for x in ncounts),'target_points_at_least4_same_direction':sum(x['same_dir']>=4 for x in ncounts),'target_points_at_least4_moving_neighbors':sum(x['moving']>=4 for x in ncounts),'target_points_at_least1_moving_neighbor':sum(x['moving']>=1 for x in ncounts),'neighbor_rule':'last report at/before target, age <=300 s, <=3.5 nm; COG direction <45 degrees','context_bbox':[20.5,21.8,-158.8,-157.2],'source_event_table':'https://github.com/sandialabs/HawaiiCoast_GT_Code_Generation/blob/main/hawaii_primary_trajectories_of_interest_2017_2020.csv','warnings':['事件关联区间不是逐点故障标签','仅查看公开事件数据，未经原USCG IIR证实','邻船数量评估不代表同向数据经过运动学质量审计或场几何有效','该事件不等同于低速机动真值或观测污染']}
(out/'hawaii_independent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
(out/'hawaii_independent_counts.csv').write_text('time,neighbors,same_dir\n'+''.join(f"{x['time']},{x['neighbors']},{x['same_dir']}\n" for x in ncounts),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))


