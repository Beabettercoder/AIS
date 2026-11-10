"""Read the host's actual data catalog and AIS points. The host DB is read-only."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import math
import os
import sqlite3
import time
from functools import lru_cache

import numpy as np

RADIUS_NM=3.5
DISTANCE_SIGMA_NM=1.5
MIN_NEIGHBORS=3
RADIUS=RADIUS_NM*1852.0
LEGACY_RADIUS=8*1852.0
EARTH=6371008.8
_catalog_cache={}
_metadata_cache={}


def settings_path():
    explicit=os.environ.get('ANOMALY_SQLITE_PATH')
    if explicit:return Path(explicit).expanduser().resolve()
    from maritime_agent.config import Settings
    settings=Settings.from_env()
    if settings.storage_backend!='sqlite' or settings.db_path is None:
        raise ValueError('异常核查需要 system3 已配置的 SQLite 数据库')
    return settings.db_path.resolve()


@contextmanager
def connect(path=None):
    p=Path(path or settings_path()).resolve()
    db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True,timeout=30)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    try:yield db
    finally:db.close()


def utc(t):return datetime.fromtimestamp(float(t),timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
def epoch(s):
    d=datetime.fromisoformat(str(s).replace('Z','+00:00'))
    if d.tzinfo is None:d=d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def _scope(entry):
    if entry['table']=='ingested_ais_positions':
        ids=entry['source_ids'];return 'source_id IN ('+','.join('?' for _ in ids)+')',ids
    where='dataset_id=?';args=[entry['host_dataset_id']]
    if entry.get('region_bbox'):
        a,b,c,d=entry['region_bbox'];where+=' AND lon>=? AND lon<=? AND lat>=? AND lat<=?';args += [a,c,b,d]
    return where,args


def catalog(path=None,refresh=False):
    p=Path(path or settings_path()).resolve()
    signature=(str(p),p.stat().st_mtime_ns,p.stat().st_size)
    wal=Path(str(p)+'-wal')
    if wal.exists():signature += (wal.stat().st_mtime_ns,wal.stat().st_size)
    cached=_catalog_cache.get(signature)
    if cached and time.monotonic()-cached[0]<20 and not refresh:return cached[1]
    with connect(p) as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        entries=[];groups={}
        if 'ingested_sources' in tables:
            # Same registration, source kind and canonical-region binding as
            # SQLiteRepository.search_ingested_data / data-management.
            query="""SELECT s.*,b.region_id,r.canonical_name FROM ingested_sources s
                LEFT JOIN source_region_bindings b ON b.source_id=s.source_id AND b.source_kind='ais_csv'
                LEFT JOIN regions r ON r.region_id=b.region_id AND r.dataset_id=b.dataset_id
                WHERE s.source_kind='ais_csv' ORDER BY s.created_at_utc""" if 'source_region_bindings' in tables else "SELECT s.*,NULL region_id,NULL canonical_name FROM ingested_sources s WHERE source_kind='ais_csv'"
            seen=set()
            for row in db.execute(query):
                row=dict(row)
                if row['source_id'] in seen:continue
                seen.add(row['source_id'])
                name=row.get('canonical_name') or row['region_label'] or row['source_name']
                group=groups.setdefault(name,{'id':row.get('region_id') or row['source_id'],'name':name,'source_ids':[],
                      'table':'ingested_ais_positions','sources':[],'source_kind':'sqlite','has_mmsi':True,'has_absolute_time':True,
                      'row_count':0,'start_utc':row['first_utc'],'end_utc':row['last_utc'],'bbox':None,'unique_mmsi':None})
                group['source_ids'].append(row['source_id'])
                group['sources'].append({'id':row['source_id'],'name':row['source_name'],'rows':row['row_count']})
                group['row_count']+=row['row_count'];group['start_utc']=min(group['start_utc'],row['first_utc']);group['end_utc']=max(group['end_utc'],row['last_utc'])
                profile=json.loads(row.get('profile_json') or '{}');box=profile.get('observed_bbox_wgs84')
                if box and len(box)==4:
                    old=group['bbox'];group['bbox']=box if old is None else [min(old[0],box[0]),min(old[1],box[1]),max(old[2],box[2]),max(old[3],box[3])]
        entries.extend(groups.values())
        if 'ais_messages' in tables:
            for row in db.execute('SELECT * FROM datasets'):
                entry={'id':'base-'+row['dataset_id'],'name':row['label'],'table':'ais_messages',
                       'host_dataset_id':row['dataset_id'],'source_ids':[],'source_kind':'sqlite','has_mmsi':True,'has_absolute_time':True}
                entries.append(entry)
                if 'regions' in tables:
                    for region in db.execute('SELECT * FROM regions WHERE dataset_id=?',(row['dataset_id'],)):
                        if region['canonical_name'] in groups or not region['bbox_json']:continue
                        box=json.loads(region['bbox_json'])
                        if len(box)==4:
                            entries.append({**entry,'id':region['region_id'],'name':region['canonical_name'],
                                            'region_bbox':box})
        result=[]
        for entry in entries:
            where,args=_scope(entry)
            if entry['table']=='ais_messages':
                first=db.execute(f"SELECT timestamp_utc FROM ais_messages WHERE {where} ORDER BY timestamp_utc LIMIT 1",args).fetchone()
                if not first:continue
                last=db.execute(f"SELECT timestamp_utc FROM ais_messages WHERE {where} ORDER BY timestamp_utc DESC LIMIT 1",args).fetchone()
                entry.update(start_utc=first[0],end_utc=last[0],row_count=None,unique_mmsi=None,bbox=entry.get('region_bbox'))
            first,last=entry['start_utc'],entry['end_utc']
            if not first or not last:continue
            entry.update(default_start_utc=first,default_end_utc=utc(min(epoch(first)+1200,epoch(last))))
            count=f"{entry['row_count']:,}" if entry['row_count'] is not None else '逐报文'
            entry['label']=f"{entry['name']} · {count} 条 AIS · {first[:10]}—{last[:10]}"
            result.append(entry)
    value={'datasets':result,'database':'system3 SQLite','default_dataset':result[0]['id'] if result else None,
           'catalog_source':'ingested_sources + source_region_bindings + ais_messages',
           'notices':['直接读取 system3 已接入数据；新导入 AIS 来源在刷新目录后自动出现。']}
    _catalog_cache.clear();_catalog_cache[signature]=(time.monotonic(),value)
    return value


def dataset_metadata(dataset_id,path=None):
    entry=resolve(dataset_id,path);p=Path(path or settings_path()).resolve()
    wal=Path(str(p)+'-wal')
    signature=(str(p),entry['id'],p.stat().st_mtime_ns,tuple(entry.get('source_ids',[])),
               entry.get('row_count'),entry.get('end_utc'),
               wal.stat().st_mtime_ns if wal.exists() else 0,wal.stat().st_size if wal.exists() else 0)
    cached=_metadata_cache.get(signature)
    if cached:return cached
    where,args=_scope(entry)
    with connect(p) as db:
        r=db.execute(f"SELECT COUNT(*) n,COUNT(DISTINCT mmsi) vessels,MIN(timestamp_utc) first,MAX(timestamp_utc) last,MIN(lon) west,MIN(lat) south,MAX(lon) east,MAX(lat) north FROM {entry['table']} WHERE {where}",args).fetchone()
    out={**entry,'row_count':r['n'],'unique_mmsi':r['vessels'],'start_utc':r['first'],'end_utc':r['last'],
         'bbox':[r['west'],r['south'],r['east'],r['north']]}
    _metadata_cache[signature]=out
    return out


def resolve(dataset_id,path=None):
    ds=catalog(path)['datasets'];target=str(dataset_id).casefold()
    match=next((d for d in ds if d['id'].casefold()==target or d['name'].casefold()==target or target in [s['id'].casefold() for s in d.get('sources',[])]),None)
    if match is None:raise ValueError('已接入目录中没有这个 AIS 数据集，请刷新目录后重新选择')
    return match


def read_points(entry,start,end,path=None,preserve_report_fields=True):
    where,args=_scope(entry)
    table=entry['table'];sog='sog_knots' if table=='ingested_ais_positions' else 'sog_kn'
    cog='cog_degrees' if table=='ingested_ais_positions' else 'cog_deg'
    extra='extra_attributes_json' if table=='ingested_ais_positions' else "'{}'"
    sql=f"SELECT mmsi,timestamp_utc,lon,lat,{sog} sog,{cog} cog,{extra} extra FROM {table} WHERE {where} AND timestamp_utc>=? AND timestamp_utc<=? ORDER BY timestamp_utc,mmsi"
    with connect(path) as db:rows=db.execute(sql,[*args,utc(start),utc(end)]).fetchall()
    if not rows:return None
    ts=np.array([epoch(r['timestamp_utc']) for r in rows])
    mmsi=np.array([str(r['mmsi']) for r in rows]);lon=np.array([r['lon'] for r in rows]);lat=np.array([r['lat'] for r in rows])
    sog=np.array([r['sog'] if r['sog'] is not None else np.nan for r in rows],float)
    cog=np.array([r['cog'] if r['cog'] is not None else np.nan for r in rows],float)
    heading=np.full(len(rows),np.nan)
    for i,r in enumerate(rows):
        if r['extra'] and r['extra']!='{}':
            try:
                ex=json.loads(r['extra']);v=ex.get('heading_degrees',ex.get('heading'));heading[i]=float(v) if v is not None else np.nan
            except (ValueError,TypeError):pass
    lon0=float(np.mean(lon));lat0=float(np.mean(lat));cos=max(.001,math.cos(math.radians(lat0)))
    xy=np.column_stack((np.deg2rad((lon-lon0+180)%360-180)*EARTH*cos,np.deg2rad(lat-lat0)*EARTH))
    # Preserve each independently valid observation. Missing COG must never
    # overwrite an observed zero SOG with velocity derived from GPS noise.
    sog_reported=np.isfinite(sog)&(sog>=0)&(sog<102.2)
    cog_reported=np.isfinite(cog)&(cog>=0)&(cog<360)
    source_sog=np.where(sog_reported,sog,np.nan)
    source_cog=np.where(cog_reported,cog,np.nan)
    reported=sog_reported&cog_reported
    groups={};w=np.ones(len(ts));kinematic_weight=np.ones(len(ts));ekin=np.zeros(len(ts));velocity=np.zeros((len(ts),2))
    for v in np.unique(mmsi):
        idx=np.flatnonzero(mmsi==v);groups[v]=idx
        dt=np.diff(ts[idx],prepend=ts[idx][0]);dp=np.diff(xy[idx],axis=0,prepend=xy[idx][:1])
        estimate=dp/np.maximum(dt[:,None],1)
        bad=~reported[idx]
        bad_sog=~sog_reported[idx] if preserve_report_fields else bad
        bad_cog=~cog_reported[idx] if preserve_report_fields else bad
        sog[idx[bad_sog]]=np.linalg.norm(estimate[bad_sog],axis=1)/.514444
        cog[idx[bad_cog]]=(np.rad2deg(np.arctan2(estimate[bad_cog,0],estimate[bad_cog,1]))+360)%360
        vv=np.column_stack((sog[idx]*.514444*np.sin(np.deg2rad(cog[idx])),sog[idx]*.514444*np.cos(np.deg2rad(cog[idx]))))
        velocity[idx]=vv
        previous=np.vstack((vv[:1],vv[:-1]));report_ok=reported[idx]
        valid=(dt>=5)&(dt<=600)&report_ok&np.r_[False,report_ok[:-1]]
        err=np.linalg.norm(estimate-(vv+previous)/2,axis=1);err[~valid]=np.nan
        ekin[idx]=err;w[idx]=np.exp(-np.maximum(0,np.nan_to_num(err,nan=0)-2.5)**2/8)
        kinematic_weight[idx]=w[idx]
        w[idx[bad]]=np.minimum(w[idx[bad]],.25)
    return dict(timestamp=ts,mmsi=mmsi,lon=lon,lat=lat,sog=sog,cog=cog,heading=heading,
                xy=xy,velocity=velocity,w=w,e_kin=ekin,kinematic_weight=kinematic_weight,
                source_sog=source_sog,source_cog=source_cog,
                observed_sog=source_sog,observed_cog=source_cog,
                sog_reported=sog_reported,cog_reported=cog_reported,
                reported=reported,groups=groups,
                origin=[lon0,lat0],source_rows=len(rows))


def unproject(xy,origin):
    lon0,lat0=origin;cos=max(.001,math.cos(math.radians(lat0)))
    return np.column_stack((lat0+np.rad2deg(xy[:,1]/EARTH),lon0+np.rad2deg(xy[:,0]/(EARTH*cos))))


def _motion_observations(d,rows):
    """All original, distinct-time observations, before twenty-step sampling."""
    source_sog=np.asarray(d.get('source_sog',d['sog']))[rows]
    source_cog=np.asarray(d.get('source_cog',d['cog']))[rows]
    sm=np.asarray(d.get('sog_reported',np.isfinite(d['sog'])&(d['sog']>=0)&(d['sog']<102.2)),bool)[rows]
    cm=np.asarray(d.get('cog_reported',np.isfinite(d['cog'])&(d['cog']>=0)&(d['cog']<360)),bool)[rows]
    physical_w=np.asarray(d.get('kinematic_weight',d['w']),float)[rows]
    xy=d['xy'][rows]-d['xy'][rows[0]]
    gated=np.cumsum(np.diff(xy,axis=0,prepend=xy[:1])*physical_w[:,None],axis=0)
    finite=lambda values,valid:[float(value) if ok and np.isfinite(value) else None for value,ok in zip(values,valid)]
    return {'sog_kn':finite(source_sog,sm),'cog_deg':finite(source_cog,cm),
            'source_sog':finite(source_sog,sm),'source_cog':finite(source_cog,cm),
            'sog_reported_mask':sm.tolist(),'cog_reported_mask':cm.tolist(),
            'reported_mask':(sm&cm).tolist(),'xy_m':xy.tolist(),'gated_xy_m':gated.tolist(),
            'time_seconds':(d['timestamp'][rows]-d['timestamp'][rows[0]]).tolist(),
            'gate_weights':physical_w.tolist(),'e_kin_mps':[float(v) if np.isfinite(v) else None for v in d['e_kin'][rows]],
            'position_basis':'原始实际相对米制位置；gated_xy_m仅使用可计算的运动学置信度，不使用字段缺失惩罚'}


def build_window(entry,start,end,path=None,neighbor_revision=3):
    """20 actual steps per vessel; cKDTree snapshots and weighted LOO fits."""
    from scipy.spatial import cKDTree
    d=read_points(entry,start-600,end,path,preserve_report_fields=neighbor_revision>=3)
    if d is None:return None
    ts=d['timestamp'];included=(ts>=start)&(ts<end)
    radius=RADIUS if neighbor_revision>=3 else LEGACY_RADIUS
    sigma_nm=DISTANCE_SIGMA_NM if neighbor_revision>=3 else 4.0
    minimum=MIN_NEIGHBORS if neighbor_revision>=3 else 4
    targets=[];full_motion={}
    for vessel,rows in d['groups'].items():
        rows=rows[(ts[rows]>=start)&(ts[rows]<end)]
        source_rows=rows.copy()
        # Keep distinct timestamps. Do not invent intermediate AIS reports.
        if len(rows):rows=rows[np.r_[True,np.diff(ts[rows])>0]]
        if len(rows)<2:continue
        full_motion[vessel]=_motion_observations(d,source_rows)
        if len(rows)>20:
            edges=np.linspace(0,len(rows),21).astype(int);selected=[]
            for a,b in zip(edges[:-1],edges[1:]):
                segment=rows[a:b]
                selected.append(int(segment[np.argmax(np.nan_to_num(d['e_kin'][segment],nan=-1))]) if len(segment) else int(rows[min(a,len(rows)-1)]))
            selected[0]=int(rows[0]);selected[-1]=int(rows[-1]);rows=np.array(sorted(set(selected)))
        targets.append((vessel,rows))
    n=len(targets)
    if not n:return {'empty':True,'point_count':int(included.sum()),'vessels':int(len(np.unique(d['mmsi'][included]))),
                     'vessel_ids':np.unique(d['mmsi'][included]).tolist()}
    raw=np.zeros((n,20,7),np.float32);w=np.ones((n,20),np.float32);mask=np.zeros((n,20),bool)
    t=np.zeros((n,20),np.float32);rf=np.zeros((n,20),np.float32);rv=np.zeros((n,20,2),np.float32)
    ep=np.zeros((n,20,4),np.float32);ok=np.zeros((n,20),bool);ek=np.zeros((n,20),np.float32)
    nbxy=np.zeros((n,32,2),np.float32);nbv=np.zeros_like(nbxy);nbm=np.zeros((n,32),bool);nbw=np.zeros((n,32),np.float32);nbs=np.zeros((n,32,5,4),np.float32)
    metadata=[];snapshots={}
    if neighbor_revision>=2:
        # Sweep actual target observation times once. Each snapshot contains the
        # latest report of every distinct vessel at or before that exact UTC.
        latest={};cursor=0
        moments=sorted({float(ts[row]) for _,rows in targets for row in rows})
        for at in moments:
            stop=int(np.searchsorted(ts,at,'right'))
            for row in range(cursor,stop):latest[str(d['mmsi'][row])]=row
            cursor=stop
            active=np.array([row for row in latest.values() if at-ts[row]<=300],dtype=int)
            snapshots[at]=(active,cKDTree(d['xy'][active]) if len(active) else None)
    def snapshot(at):
        key=float(at) if neighbor_revision>=2 else int(at//60)*60
        if key in snapshots:return snapshots[key]
        rows=[]
        for indices in d['groups'].values():
            k=np.searchsorted(ts[indices],key,'right')-1
            if k>=0 and key-ts[indices[k]]<=300:rows.append(indices[k])
        rows=np.array(rows,dtype=int)
        tree=cKDTree(d['xy'][rows]) if len(rows) else None
        snapshots[key]=(rows,tree);return rows,tree
    for i,(vessel,rows) in enumerate(targets):
        m=len(rows);pos=d['xy'][rows];v=d['velocity'][rows];c=np.deg2rad(d['cog'][rows])
        raw[i,:m]=np.column_stack((pos-pos[0],v,d['sog'][rows],np.sin(c),np.cos(c)));w[i,:m]=d['w'][rows]
        mask[i,:m]=True;t[i,:m]=ts[rows]-ts[rows[0]];ek[i,:m]=d['e_kin'][rows]
        if m<20:raw[i,m:]=raw[i,m-1];w[i,m:]=w[i,m-1];t[i,m:]=t[i,m-1]
        neighbors_meta=[];neighbor_mean=[];neighbor_low=[];neighbor_high=[];field_vectors=[];neighbor_frames=[];seen_neighbors={}
        for k,row in enumerate(rows):
            candidates,tree=snapshot(ts[row]);near=np.array([],int)
            direction=float(c[k]);direction_basis='当前报告航向';direction_time=utc(ts[row])
            if neighbor_revision>=2 and not (d['reported'][row] and d['sog'][row]>=2 and d['w'][row]>=.3):
                own_history=d['groups'][vessel]
                reliable=own_history[(ts[own_history]<=ts[row])&(ts[own_history]>=ts[row]-300)&d['reported'][own_history]&(d['sog'][own_history]>=2)&(d['w'][own_history]>=.3)]
                if len(reliable):
                    reference=int(reliable[-1]);direction=float(np.deg2rad(d['cog'][reference]));direction_time=utc(ts[reference]);direction_basis='最近5分钟可靠在航航向'
                else:direction=None;direction_time=None;direction_basis='缺少可靠在航方向，同向关系不可判定'
            nearby_count=0
            if tree is not None:
                near=candidates[np.asarray(tree.query_ball_point(pos[k],radius),int)]
                near=near[d['mmsi'][near]!=vessel];nearby_count=len(near)
                if direction is None:near=np.array([],dtype=int)
                else:
                    angle=(np.deg2rad(d['cog'][near])-direction+np.pi)%(2*np.pi)-np.pi
                    near=near[(np.abs(angle)<np.pi/4)&(d['w'][near]>=.3)&(d['sog'][near]>=2)&d['reported'][near]]
                near=near[np.argsort(np.linalg.norm(d['xy'][near]-pos[k],axis=1))][:32]
            neighbor_mean.append(float(np.mean(d['sog'][near])) if len(near) else None)
            neighbor_low.append(float(np.quantile(d['sog'][near],.25)) if len(near) else None)
            neighbor_high.append(float(np.quantile(d['sog'][near],.75)) if len(near) else None)
            field_vectors.append(None)
            if len(near)>=minimum:
                delta=d['xy'][near]-pos[k];distance=np.linalg.norm(delta,axis=1)
                weights=d['w'][near]*np.exp(-.5*(distance/(sigma_nm*1852.))**2) if neighbor_revision>=3 else d['w'][near]*np.exp(-.5*(distance/(.5*radius))**2)
                A=np.column_stack((np.ones(len(near)),delta/radius));Aw=A*np.sqrt(weights[:,None])
                if np.linalg.matrix_rank(Aw)==3 and np.linalg.cond(Aw)<1000:
                    theta=np.linalg.lstsq(Aw,d['velocity'][near]*np.sqrt(weights[:,None]),rcond=None)[0]
                    rv[i,k]=v[k]-theta[0];rf[i,k]=np.linalg.norm(rv[i,k]);ok[i,k]=True;field_vectors[k]=theta[0].tolist()
                    ep[i,k]=[np.linalg.norm(theta[0]),0,(theta[1,0]+theta[2,1])/radius,(theta[1,1]-theta[2,0])/radius]
            members=[]
            for nrow in near:
                nm=str(d['mmsi'][nrow]);distance=float(np.linalg.norm(d['xy'][nrow]-pos[k])/1852.)
                distance_weight=float(np.exp(-.5*(distance/sigma_nm)**2))
                members.append({'mmsi':nm,'position':[float(d['lat'][nrow]),float(d['lon'][nrow])],
                    'velocity':d['velocity'][nrow].tolist(),'sog_kn':float(d['sog'][nrow]),'cog_deg':float(d['cog'][nrow]),
                    'weight':float(d['w'][nrow]),'distance_weight':distance_weight,
                    'combined_weight':float(d['w'][nrow])*distance_weight,
                    'distance_nm':distance,'report_utc':utc(ts[nrow]),
                    'report_epoch':float(ts[nrow]),'age_seconds':float(ts[row]-ts[nrow])})
                if nm not in seen_neighbors:
                    history=d['groups'][nm];track=history[(ts[history]>=start)&(ts[history]<end)]
                    if len(track)>80:track=track[np.unique(np.linspace(0,len(track)-1,80).astype(int))]
                    seen_neighbors[nm]={'mmsi':nm,'latlon':np.column_stack((d['lat'][track],d['lon'][track])).tolist(),
                                        'timestamps':ts[track].tolist()}
            neighbor_frames.append({'step':k,'time_utc':utc(ts[row]),'time_epoch':float(ts[row]),
                  'snapshot_utc':utc(float(ts[row]) if neighbor_revision>=2 else int(ts[row]//60)*60),
                  'neighbors':members,'count':len(members),'nearby_count':nearby_count,'field_valid':bool(ok[i,k]),
                  'radius_nm':radius/1852.,'distance_sigma_nm':sigma_nm,'minimum_neighbors':minimum,
                  'direction_basis':direction_basis,'direction_reference_utc':direction_time})
            if k==m-1:
                for j,nrow in enumerate(near):
                    nbxy[i,j]=d['xy'][nrow]-pos[k];nbv[i,j]=d['velocity'][nrow];nbm[i,j]=True
                    distance_nm=float(np.linalg.norm(d['xy'][nrow]-pos[k])/1852.)
                    nbw[i,j]=d['w'][nrow]*np.exp(-.5*(distance_nm/sigma_nm)**2) if neighbor_revision>=3 else d['w'][nrow]
                    history=d['groups'][d['mmsi'][nrow]];stop=np.searchsorted(ts[history],ts[nrow],'right');hist=history[max(0,stop-5):stop]
                    nbs[i,j,5-len(hist):]=np.column_stack((d['xy'][hist]-d['xy'][nrow],d['velocity'][hist]))
                    track=history[(ts[history]>=start)&(ts[history]<=ts[row])]
                    if len(track)>40:track=track[np.linspace(0,len(track)-1,40).astype(int)]
                    neighbors_meta.append({'mmsi':str(d['mmsi'][nrow]),'latlon':np.column_stack((d['lat'][track],d['lon'][track])).tolist(),
                      'position':[float(d['lat'][nrow]),float(d['lon'][nrow])],'velocity':d['velocity'][nrow].tolist(),'weight':float(d['w'][nrow])})
        rid=f"AIS-{vessel}-{int(end)}"
        motion=full_motion[vessel];source_values=[v for v in motion['sog_kn'] if v is not None]
        source_min=float(min(source_values)) if source_values else None
        source_max=float(max(source_values)) if source_values else None
        source_median=float(np.median(source_values)) if source_values else None
        motion_state='stationary_source_max_below_2kn' if source_max is not None and source_max<2 else 'observed_underway' if source_max is not None else 'unknown_missing_reported_sog'
        metadata.append({'mmsi':vessel,'window_id':rid,'start_utc':utc(ts[rows[0]]),'end_utc':utc(ts[rows[-1]]),
            'window_start_utc':utc(start),'window_end_utc':utc(end),'valid_steps':m,'source_index':i,
            'latlon':np.column_stack((d['lat'][rows],d['lon'][rows])).tolist(),'origin_position':pos[0].tolist(),
            'projection_origin':d['origin'],'neighbor_tracks':neighbors_meta,'neighbor_mean_sog':neighbor_mean,
            'neighbor_frames':neighbor_frames,'neighbor_track_catalog':list(seen_neighbors.values()),
            'neighbor_context_version':2,'neighbor_revision':neighbor_revision,
            'neighbor_rules':{'radius_nm':radius/1852.,'gaussian_sigma_nm':sigma_nm,'min_neighbors':minimum},
            'motion_observations':motion,'motion_state':motion_state,
            'source_sog_min_kn':source_min,'source_sog_max_kn':source_max,'source_sog_median_kn':source_median,
            'source_sog_valid_count':len(source_values),
            'neighbor_sog_low':neighbor_low,'neighbor_sog_high':neighbor_high,
            'field_vectors':field_vectors,'sog_kn':d['sog'][rows].tolist(),'cog_deg':d['cog'][rows].tolist(),
            'heading_deg':[float(x) if np.isfinite(x) else None for x in d['heading'][rows]],
            'report_fields_complete':bool(d['reported'][rows].all()),'valid_report_steps':int(d['reported'][rows].sum()),'source_kind':'sqlite'})
    arrays=dict(raw=raw,w_self=w,mask=mask,t=t,r_field=rf,r_vec=rv,e_phys=ep,field_valid=ok,e_kin=ek,
                nb_xy=nbxy,nb_v=nbv,nb_mask=nbm,nb_w=nbw,nb_seq=nbs)
    source_sog=np.asarray(d.get('source_sog',d['sog']))
    source_valid=np.isfinite(source_sog)&(source_sog>=0)&(source_sog<102.2)
    missing_sog_count=int((included&~source_valid).sum())
    if neighbor_revision>=3:
        actual=source_sog[included&source_valid]
        moving=int(len(np.unique(d['mmsi'][included&source_valid&(source_sog>=2)])))
    else:
        # Preserve the older scan's arithmetic, including its historical
        # feature-derived speed convention, when rehydrating saved evidence.
        actual=d['sog'][included];moving=int(len(np.unique(d['mmsi'][included&(d['sog']>=2)])))
    return {'arrays':arrays,'metadata':metadata,'dataset':entry,'point_count':int(included.sum()),
            'vessels':int(len(np.unique(d['mmsi'][included]))),'moving_vessels':moving,
            'mean_sog_kn':float(np.mean(actual)) if len(actual) else None,
            'source_sog_missing_count':missing_sog_count,
            'speed_statistics_basis':'original_valid_reported_sog' if neighbor_revision>=3 else 'legacy_model_feature_sog',
            'vessel_ids':np.unique(d['mmsi'][included]).tolist()}
