"""Deterministic weather CSV intake, normalization, quality and storage."""
from __future__ import annotations

import csv, gzip, hashlib, io, json, math, sqlite3, zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .alignment_rules import rules_for
from .database import _initialize
from .datetime_parser import parse_utc

MAX_BYTES=200*1024*1024; MAX_EXPANDED=1024*1024*1024; MAX_ROWS=2_000_000
_LEGACY_ALIASES={
 "valid_time_utc":("valid_time_utc","timestamp_utc","datetime","timestamp","time","observation_time","时间"),
 "issue_time_utc":("issue_time_utc","forecast_time","run_time","init_time","发布时间"),
 "location_id":("location_id","station_id","station","buoy_id","grid_id","站点编号"),
 "location_name":("location_name","station_name","name","站点名称"),
 "lon":("lon","longitude","lng","经度"),"lat":("lat","latitude","纬度"),
 "wind_speed_mps":("wind_speed_mps","wind_speed","windspeed","wind_speed_knots","wind_speed_kts","风速"),
 "wind_direction_deg":("wind_direction_deg","wind_direction","wind_dir","风向"),
 "wind_gust_mps":("wind_gust_mps","wind_gust","gust","gust_knots","阵风"),
 "air_pressure_hpa":("air_pressure_hpa","pressure_hpa","pressure_mb","pressure","pressure_pa","气压"),
 "air_temperature_c":("air_temperature_c","temperature_c","temp_c","temperature","temp_f","气温"),
 "precipitation_mm":("precipitation_mm","precip_mm","precipitation","rain_mm","降水"),
 "visibility_m":("visibility_m","visibility","visibility_km","能见度"),
 "relative_humidity_pct":("relative_humidity_pct","humidity","rh","相对湿度"),
 "cloud_cover_pct":("cloud_cover_pct","cloud_cover","cloud","云量"),
 "wave_height_m":("wave_height_m","significant_wave_height_m","wave_height","浪高"),
 "wave_period_s":("wave_period_s","wave_period","浪周期"),
 "wave_direction_deg":("wave_direction_deg","wave_direction","wave_dir","浪向"),
 "water_temperature_c":("water_temperature_c","water_temperature","water_temp","水温"),
 "water_level_m":("water_level_m","water_level","水位")}
WEATHER_RULES=rules_for("weather_csv")
ALIASES={field:tuple(aliases) for field,aliases in WEATHER_RULES["aliases"].items()}
METRICS=tuple(k for k in ALIASES if k not in {"valid_time_utc","issue_time_utc","location_id","location_name","lon","lat"})
MISSING_VALUES={str(value).casefold() for value in WEATHER_RULES.get("missing_values", [])}

def _norm(v:str)->str:return "".join(c for c in v.casefold() if c.isalnum())
def mapping(columns:list[str], explicit:dict[str,str]|None=None)->dict[str,str]:
 by={_norm(c):c for c in columns}; result={k:next((by[_norm(a)] for a in aliases if _norm(a) in by),"") for k,aliases in ALIASES.items()}
 result.update({k:v for k,v in (explicit or {}).items() if k in ALIASES and v in columns}); return result
def _utc(v:Any, assume:bool)->str:
 return parse_utc(v,assume)
def _num(v:Any,lo:float,hi:float,name:str)->float|None:
 if v in (None,""):return None
 try:n=float(v)
 except (TypeError,ValueError) as e:raise ValueError(f"{name}不是数字") from e
 if not math.isfinite(n) or not lo<=n<=hi:raise ValueError(f"{name}超出合理范围")
 return n
def _payloads(data:bytes,name:str):
 if name.casefold().endswith(".zip") or data[:4]==b"PK\x03\x04":
  with zipfile.ZipFile(io.BytesIO(data)) as z:
   members=[x for x in z.infolist() if not x.is_dir() and x.filename.casefold().endswith((".csv",".csv.gz"))]
   if not members:raise ValueError("ZIP 中没有气象 CSV/CSV.GZ")
   if sum(x.file_size for x in members)>MAX_EXPANDED:raise ValueError("ZIP 解压后超过 512 MB")
   return [(Path(x.filename).name,z.read(x)) for x in sorted(members,key=lambda x:x.filename)]
 return [(Path(name).name,data)]
def _text(raw:bytes,name:str)->str:
 if name.casefold().endswith(".gz") or raw[:2]==b"\x1f\x8b":raw=gzip.decompress(raw)
 for enc in ("utf-8-sig","gb18030","utf-16","latin1"):
  try:return raw.decode(enc)
  except (UnicodeDecodeError,UnicodeError):continue
 raise ValueError(f"无法识别气象 CSV 编码（已尝试 UTF-8、GB18030、UTF-16）: {name}")
def schema_sample(data:bytes,source_name:str,limit:int=10)->tuple[list[str],list[dict[str,Any]]]:
 name,raw=_payloads(data,source_name)[0]
 reader=csv.DictReader(io.StringIO(_text(raw,name)))
 columns=reader.fieldnames or []
 return columns,[row for _,row in zip(range(limit),reader)]
def _issue(code,severity,dimension,message,count,total,samples=None,blocking=False):
 return {"code":code,"severity":severity,"dimension":dimension,"message":message,"affected_count":count,"affected_ratio":round(count/total,6) if total else 0,"samples":(samples or [])[:5],"blocking":blocking}

def _missing(value:Any)->bool:
 return value is None or str(value).strip().casefold() in MISSING_VALUES

def _source_unit(column:str,field:str)->str:
 name=_norm(column)
 if field in {"wind_speed_mps","wind_gust_mps"}:
  if any(token in name for token in ("knot","kts")):return "knots"
  if any(token in name for token in ("kmh","kph")):return "km/h"
  if "mph" in name:return "mph"
  return "m/s"
 if field=="air_pressure_hpa":
  if "kpa" in name:return "kPa"
  if "hpa" in name or "mb" in name:return "hPa"
  if name.endswith("pa"):return "Pa"
  return "hPa"
 if field in {"air_temperature_c","water_temperature_c"}:
  return "degF" if any(token in name for token in ("tempf","temperaturef","fahrenheit")) else "degC"
 if field=="precipitation_mm":
  return "inch" if any(token in name for token in ("precipin","rainin","inch")) else "mm"
 if field=="visibility_m":
  if any(token in name for token in ("visibilitynm","nauticalmile","nmi")):return "nautical_mile"
  if any(token in name for token in ("visibilitykm","kilometer","kilometre")):return "km"
  if any(token in name for token in ("visibilitymi","statutemile")):return "mile"
  return "m"
 if field in {"wave_height_m","water_level_m"}:
  return "ft" if any(token in name for token in ("feet","foot","ft")) else "m"
 if field=="cloud_cover_pct" and "okta" in name:return "okta"
 return WEATHER_RULES.get("canonical_units",{}).get(field,"")

def _normalize_metric(value:Any,column:str,field:str)->tuple[float|None,dict[str,Any]|None]:
 if _missing(value):return None,None
 try:number=float(value)
 except (TypeError,ValueError) as exc:raise ValueError(f"{field}不是数字") from exc
 if not math.isfinite(number):raise ValueError(f"{field}不是有限数值")
 unit=_source_unit(column,field); converted=number
 if unit=="knots":converted*=0.514444
 elif unit=="km/h":converted/=3.6
 elif unit=="mph":converted*=0.44704
 elif unit=="Pa":converted/=100
 elif unit=="kPa":converted*=10
 elif unit=="degF":converted=(converted-32)*5/9
 elif unit=="inch":converted*=25.4
 elif unit=="km":converted*=1000
 elif unit=="nautical_mile":converted*=1852
 elif unit=="mile":converted*=1609.344
 elif unit=="ft":converted*=0.3048
 elif unit=="okta":converted*=12.5
 target=WEATHER_RULES.get("canonical_units",{}).get(field,unit)
 audit=None if unit==target else {"field":field,"source_column":column,"source_unit":unit,
                                  "target_unit":target,"rule":f"weather_{field}_{unit}_to_{target}"}
 return converted,audit

NOAA_STATIONS={"8770613":("Morgans Point, Barbours Cut, TX",-94.98500,29.68169)}
def _noaa_coops_csv(data:bytes, source_name:str, location_id:str, location_name:str,
                    default_lon:float|None, default_lat:float|None)->tuple[bytes,str,str,float,float]:
 import re
 match=re.search(r"(?<!\d)(\d{7})(?!\d)",source_name)
 station=location_id or (match.group(1) if match else "")
 known=NOAA_STATIONS.get(station)
 name=location_name or (known[0] if known else "")
 lon=default_lon if default_lon is not None else (known[1] if known else None)
 lat=default_lat if default_lat is not None else (known[2] if known else None)
 if not station or lon is None or lat is None:raise ValueError("NOAA CO-OPS 文件需要站点 ID 和站点经纬度")
 text=_text(data,source_name); rows=list(csv.reader(io.StringIO(text)))
 if len(rows)<2:raise ValueError("NOAA CO-OPS 文件没有数据")
 lower=source_name.casefold(); fields=["valid_time_utc","location_id","location_name","lon","lat"]
 if "wind" in lower: fields += ["wind_speed_mps","wind_direction_deg","wind_gust_mps"]
 elif "air_pressure" in lower: fields += ["air_pressure_hpa"]
 elif "air_temperature" in lower: fields += ["air_temperature_c"]
 elif "water_temperature" in lower: fields += ["water_temperature_c"]
 elif "water_level" in lower: fields += ["water_level_m"]
 else: raise ValueError("无法从 NOAA CO-OPS 文件名识别产品类型")
 out=io.StringIO(); writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
 for row in rows[1:]:
  if len(row)<2 or not row[0].strip():continue
  item={"valid_time_utc":row[0].strip().replace(" ","T")+":00Z","location_id":station,"location_name":name,"lon":lon,"lat":lat}
  if "wind" in lower:
   if len(row)<5:continue
   item.update(wind_speed_mps=row[1].strip(),wind_direction_deg=row[2].strip(),wind_gust_mps=row[4].strip())
  else:item[fields[-1]]=row[1].strip()
  writer.writerow(item)
 return out.getvalue().encode(),station,name,float(lon),float(lat)

def parse_weather(data:bytes, source_name:str, data_mode:str, assume_utc:bool=False,
                  explicit_mapping:dict[str,str]|None=None, actions:set[str]|None=None,
                  source_type:str="generic", location_id:str="", location_name:str="",
                  default_lon:float|None=None, default_lat:float|None=None):
 actions=set(actions or ())
 if source_type=="noaa_coops":
  data,location_id,location_name,default_lon,default_lat=_noaa_coops_csv(data,source_name,location_id,location_name,default_lon,default_lat)
  source_name="normalized_"+Path(source_name).name; assume_utc=True
 payloads=_payloads(data,source_name); decoded=[(n,_text(r,n),hashlib.sha256(r).hexdigest()) for n,r in payloads]
 first=csv.DictReader(io.StringIO(decoded[0][1])); columns=first.fieldnames or []
 if not columns or len(columns)!=len(set(columns)):raise ValueError("气象 CSV 表头为空或重复")
 mp=mapping(columns,explicit_mapping); missing=[x for x in ("valid_time_utc",) if not mp[x]]
 if not mp["lon"] and default_lon is None:missing.append("lon")
 if not mp["lat"] and default_lat is None:missing.append("lat")
 if missing:raise ValueError("缺少必要字段映射: "+", ".join(missing))
 if not any(mp[x] for x in METRICS):raise ValueError("至少需要一个可识别的气象指标")
 rows=[]; seen=set(); duplicates=invalid=raw_count=0; errors=[]; files=[]
 normalizations=[]; issue_after_valid=gust_below_speed=station_drift=forecast_missing_issue=0
 issue_samples=[]; gust_samples=[]; drift_samples=[]; location_positions={}
 for fi,(name,text,sha) in enumerate(decoded,1):
  reader=csv.DictReader(io.StringIO(text));
  if reader.fieldnames!=columns:raise ValueError(f"批次字段不一致: {name}")
  accepted=0; first_time=last_time=None
  for rn,row in enumerate(reader,2):
   raw_count+=1
   try:
    valid=_utc(row[mp["valid_time_utc"]],assume_utc); issue=_utc(row[mp["issue_time_utc"]],assume_utc) if mp["issue_time_utc"] and not _missing(row.get(mp["issue_time_utc"])) else None
    lon=_num(row[mp["lon"]],-180,180,"经度") if mp["lon"] else default_lon; lat=_num(row[mp["lat"]],-90,90,"纬度") if mp["lat"] else default_lat
    loc=str(row.get(mp["location_id"],"") or location_id or f"{lon:.5f},{lat:.5f}").strip()[:120]; namev=str(row.get(mp["location_name"],"") or location_name or "").strip()[:200]
    vals={}
    ranges={"wind_speed_mps":(0,120),"wind_direction_deg":(0,360),"wind_gust_mps":(0,150),"air_pressure_hpa":(800,1100),"air_temperature_c":(-90,60),"precipitation_mm":(0,5000),"visibility_m":(0,200000),"relative_humidity_pct":(0,100),"cloud_cover_pct":(0,100),"wave_height_m":(0,40),"wave_period_s":(0,60),"wave_direction_deg":(0,360),"water_temperature_c":(-5,50),"water_level_m":(-20,30)}
    for field in METRICS:
     value=row.get(mp[field],"") if mp[field] else ""
     normalized,audit=_normalize_metric(value,mp[field],field) if mp[field] else (None,None)
     vals[field]=_num(normalized,*ranges[field],field) if normalized is not None else None
     if audit is not None and audit not in normalizations:normalizations.append(audit)
    if not any(value is not None for value in vals.values()):raise ValueError("该行没有气象指标")
    if issue and issue>valid:
     issue_after_valid+=1
     if len(issue_samples)<5:issue_samples.append({"file":name,"row":rn,"issue_time_utc":issue,"valid_time_utc":valid})
    if data_mode=="forecast" and issue is None:forecast_missing_issue+=1
    if vals["wind_speed_mps"] is not None and vals["wind_gust_mps"] is not None and vals["wind_gust_mps"]<vals["wind_speed_mps"]:
     gust_below_speed+=1
     if len(gust_samples)<5:gust_samples.append({"file":name,"row":rn,"wind_speed_mps":vals["wind_speed_mps"],"wind_gust_mps":vals["wind_gust_mps"]})
    previous_position=location_positions.get(loc)
    if previous_position and (abs(previous_position[0]-lon)>1e-6 or abs(previous_position[1]-lat)>1e-6):
     station_drift+=1
     if len(drift_samples)<5:drift_samples.append({"file":name,"row":rn,"location_id":loc,"previous":previous_position,"current":[lon,lat]})
    else:location_positions[loc]=(lon,lat)
    key=(valid,issue if data_mode=="forecast" else None,loc,lon,lat)
    if key in seen:duplicates+=1;continue
    canonical={"valid_time_utc":valid,"issue_time_utc":issue,"location_id":loc,
               "location_name":namev or None,"lon":lon,"lat":lat,
               **{field:value for field,value in vals.items() if value is not None}}
    stored={**row,**canonical,"_raw_properties":row}
    seen.add(key); rows.append((raw_count,valid,issue,loc,namev or None,lon,lat,*[vals[x] for x in METRICS],json.dumps(stored,ensure_ascii=False,separators=(",",":"))))
    accepted+=1;first_time=valid if first_time is None else min(first_time,valid);last_time=valid if last_time is None else max(last_time,valid)
   except (ValueError,TypeError) as e:
    invalid+=1
    if len(errors)<20:errors.append({"file":name,"row":rn,"error":str(e)})
  files.append({"file_index":fi,"source_name":name,"source_sha256":sha,"row_count":accepted,"first_utc":first_time,"last_utc":last_time})
 if not rows:raise ValueError("没有可用气象记录: "+json.dumps(errors,ensure_ascii=False))
 issues=[]; repairs=[]
 if duplicates:issues.append(_issue("WEATHER_EXACT_DUPLICATE","warning","uniqueness","发现完全重复的气象记录，预览中已去重。",duplicates,raw_count))
 dropping="drop_invalid_weather_rows" in actions
 if invalid:issues.append(_issue("WEATHER_INVALID_ROW","warning" if dropping else "error","validity","存在时间、位置或指标无效的记录。",invalid,raw_count,errors,not dropping));repairs.append({"action_id":"drop_invalid_weather_rows","label":"拒绝无效气象记录","category":"needs_confirmation","affected_count":invalid,"selected":dropping,"description":"不写入无法解析或超出合理范围的记录。"})
 if normalizations:issues.append(_issue("WEATHER_UNIT_NORMALIZED","info","units","已根据明确的字段单位转换为标准单位。",raw_count,raw_count,normalizations))
 if issue_after_valid:issues.append(_issue("WEATHER_ISSUE_AFTER_VALID","warning","temporal","部分预报发布时间晚于有效时间。",issue_after_valid,raw_count,issue_samples))
 if forecast_missing_issue:issues.append(_issue("WEATHER_FORECAST_MISSING_ISSUE_TIME","warning","completeness","部分预报记录缺少发布时间。",forecast_missing_issue,raw_count))
 if gust_below_speed:issues.append(_issue("WEATHER_GUST_BELOW_MEAN","warning","consistency","部分阵风小于平均风速。",gust_below_speed,raw_count,gust_samples))
 if station_drift:issues.append(_issue("WEATHER_STATION_POSITION_CHANGED","warning","spatial","同一位置编号出现不同坐标。",station_drift,raw_count,drift_samples))
 blockers=sum(1 for x in issues if x["blocking"]); summary={"raw_rows":raw_count,"accepted_rows":len(rows),"rejected_rows":invalid,"duplicates_removed":duplicates,"blockers":blockers,**{s:sum(1 for x in issues if x["severity"]==s) for s in ("error","warning","info")}}
 times=[r[1] for r in rows]; bbox=[min(r[5] for r in rows),min(r[6] for r in rows),max(r[5] for r in rows),max(r[6] for r in rows)]
 sample=[{"valid_time_utc":r[1],"location_id":r[3],"lon":r[5],"lat":r[6]} for r in rows[:5]]
 points=[{"type":"Feature","properties":{"location_id":r[3],"location_name":r[4]},"geometry":{"type":"Point","coordinates":[r[5],r[6]]}} for r in rows[:100]]
 mapped_columns={column for column in mp.values() if column}
 report={"schema_version":WEATHER_RULES["schema_version"],"ruleset_version":WEATHER_RULES["ruleset_version"],"data_mode":data_mode,"source_type":source_type,"columns":columns,"mapping":mp,"extra_columns":[column for column in columns if column not in mapped_columns],"normalizations":normalizations,"unit_mapping":{item["field"]:{"source_unit":item["source_unit"],"target_unit":item["target_unit"]} for item in normalizations},"row_count":len(rows),"raw_row_count":raw_count,"first_utc":min(times),"last_utc":max(times),"bbox_wgs84":bbox,"file_count":len(files),"files":files,"duplicates_removed":duplicates,"rejected_rows":invalid,"sample":sample,"quality_note":"已检查时间、位置、自动单位换算、指标范围、缺测值、重复记录、预报时间关系、阵风一致性和站点位置稳定性。","quality_report":{"version":2,"status":"blocked" if blockers else "ready","summary":summary,"issues":issues},"repair_plan":repairs,"repair_preview":{"before":{"rows":raw_count},"after":{"rows":len(rows),"rejected_rows":invalid,"duplicates_removed":duplicates},"selected_actions":sorted(actions)},"map_preview":{"type":"FeatureCollection","features":points}}
 return rows,report

def store_weather(db_path:Path,data:bytes,source_name:str,region_label:str,data_mode:str,assume_utc:bool,mapping_value,actions,source_type="generic",location_id="",location_name="",default_lon=None,default_lat=None):
 rows,report=parse_weather(data,source_name,data_mode,assume_utc,mapping_value,set(actions or ()),source_type,location_id,location_name,default_lon,default_lat)
 if report["quality_report"]["summary"]["blockers"]:raise ValueError("气象质量报告仍有阻断问题")
 digest=hashlib.sha256(data).hexdigest(); source_id=str(__import__('uuid').uuid4()); now=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
 conn=sqlite3.connect(db_path)
 try:
  _initialize(conn); conn.execute("PRAGMA foreign_keys=ON")
  old=conn.execute("SELECT source_id FROM weather_sources WHERE source_sha256=? AND region_label=?",(digest,region_label)).fetchone()
  if old:return {"status":"already_imported","source_id":old[0],**report}
  with conn:
   conn.execute("INSERT INTO weather_sources VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",(source_id,source_name,region_label,data_mode,source_type,digest,report["first_utc"],report["last_utc"],json.dumps(report["bbox_wgs84"]),json.dumps(report["mapping"],ensure_ascii=False),json.dumps(report.get("unit_mapping",{}),ensure_ascii=False),now,report["row_count"]))
   columns=("source_id,source_row_number,valid_time_utc,issue_time_utc,location_id,location_name,lon,lat,"
            "wind_speed_mps,wind_direction_deg,wind_gust_mps,air_pressure_hpa,air_temperature_c,precipitation_mm,"
            "visibility_m,relative_humidity_pct,cloud_cover_pct,wave_height_m,wave_period_s,wave_direction_deg,"
            "water_temperature_c,water_level_m,properties_json")
   conn.executemany(f"INSERT INTO weather_observations ({columns}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",[(source_id,*r) for r in rows])
   locations={r[3]:(r[4],r[5],r[6]) for r in rows}
   location_column=_norm(report["mapping"].get("location_id", ""))
   location_type="grid" if "grid" in location_column else ("buoy" if "buoy" in location_column else "station")
   conn.executemany("INSERT INTO weather_locations VALUES (?,?,?,?,?,?)",[(source_id,loc,name,lon,lat,location_type) for loc,(name,lon,lat) in locations.items()])
   measurements=[]
   for r in rows:
    for offset,variable in enumerate(METRICS,7):
     if r[offset] is not None:measurements.append((source_id,r[0],r[1],r[2],r[3],variable,r[offset],"accepted"))
   conn.executemany("INSERT INTO weather_measurements VALUES (?,?,?,?,?,?,?,?)",measurements)
   q=report["quality_report"]; conn.execute("INSERT INTO weather_intake_quality_audits VALUES (?,?,?,?,?,?,?,?,?,?)",(source_id,q["version"],json.dumps(q,ensure_ascii=False),json.dumps(report["repair_plan"],ensure_ascii=False),json.dumps(report["repair_preview"]["selected_actions"]),report["raw_row_count"],report["row_count"],report["rejected_rows"],report["duplicates_removed"],now))
  return {"status":"imported","source_id":source_id,"kind":"weather_csv","region_label":region_label,**report}
 finally:conn.close()
