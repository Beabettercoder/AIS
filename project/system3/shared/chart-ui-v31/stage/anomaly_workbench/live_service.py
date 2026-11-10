"""Live SQLite workbench API and bounded background scan jobs."""
from __future__ import annotations
from collections import OrderedDict
from datetime import datetime,timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4
import gzip,hashlib,json,os,subprocess,threading,time,re
from fastapi import APIRouter,HTTPException,Request,Query
from fastapi.responses import FileResponse,Response
from pydantic import BaseModel,Field,field_validator
from . import sqlite_stream
from .reviews import ReviewStore
from .prompts import generate_brief
from .chart_explanations import generate_chart_explanations
from .queue_views import build_queue
from .evidence_snapshot import evidence_signature,unavailable_diagnostics
from .maneuver_intent import assess_maneuver_intent

ROOT=Path(__file__).resolve().parent
RUNTIME=Path(os.environ.get('ANOMALY_WORKBENCH_RUNTIME',ROOT/'runtime')).resolve()
STORE=ReviewStore(RUNTIME/'reviews.sqlite3')
router=APIRouter(prefix='/anomaly-workbench',tags=['Live SQLite AIS screening'])
JOBS=OrderedDict();LOCK=threading.RLock();RUNNING=threading.Semaphore(1)

class ScanRequest(BaseModel):
    dataset_id:str=Field(min_length=1,max_length=150)
    start_utc:str=Field(default='',max_length=40)
    end_utc:str=Field(default='',max_length=40)
    full_period:bool=False
    exclude_stationary:bool=True
    window_minutes:Literal[20]=20
    far_budget:float=.01
    top_k:int=Field(default=20,ge=1,le=100)
    @field_validator('far_budget')
    @classmethod
    def far(cls,v):
        if v not in [.001,.005,.01,.05]:raise ValueError('请选择有效的筛查严格程度')
        return v

class ReviewRequest(BaseModel):
    scan_id:str
    record_id:str
    status:Literal['priority','normal','signal']
    elapsed_seconds:float=Field(default=0,ge=0,le=86400,allow_inf_nan=False)
    note:str=Field(default='',max_length=2000)

class BriefRequest(BaseModel):use_llm:bool=False

def same_origin(request):
    origin=request.headers.get('origin','')
    if origin and origin.rstrip('/')!=f"{request.url.scheme}://{request.headers.get('host','')}":raise HTTPException(403,'仅接受同源的核查操作')

def python_worker():
    import importlib.util,sys
    configured=os.environ.get('PGDRGN_PYTHON')
    if configured:return configured
    if importlib.util.find_spec('torch') and importlib.util.find_spec('scipy'):return sys.executable
    p='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
    if Path(p).is_file():return p
    raise ValueError('请把 PGDRGN_PYTHON 指向含 torch、numpy 和 scipy 的独立环境')

def environment():
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT.parent)+os.pathsep+env.get('PYTHONPATH','')
    env['ANOMALY_SQLITE_PATH']=str(sqlite_stream.settings_path())
    env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2';env['CUDA_VISIBLE_DEVICES']=''
    return env

def job_by_id(job_id):
    with LOCK:job=JOBS.get(job_id)
    if job is None and re.fullmatch(r'[0-9a-f]{32}',job_id):
        folder=RUNTIME/'jobs'/job_id;stored=folder/'result.json.gz'
        if stored.is_file() and not folder.is_symlink() and not stored.is_symlink():
            with gzip.open(stored,'rt',encoding='utf-8') as f:scan=json.load(f)
            scan['scan_id']=job_id
            job={'id':job_id,'scan_id':job_id,'status':'complete','message':'已恢复已完成的真实扫描',
                 'params':scan['params'],'dataset':scan['dataset'],'summary':scan['summary'],
                 'completed':len(scan.get('timeline',[])),'total':len(scan.get('timeline',[])),
                 '_folder':folder,'_result':scan}
            with LOCK:
                JOBS[job_id]=job
                while len(JOBS)>12:
                    expired=next((k for k,v in JOBS.items() if k!=job_id and v['status'] not in ['queued','running']),None)
                    if expired is None:break
                    JOBS.pop(expired)
    if job is None:raise HTTPException(404,'扫描任务不存在或已过期，请重新执行')
    return job

def public_job(job):return {k:v for k,v in job.items() if not k.startswith('_')}

def run_job(job):
    with RUNNING:
        if job.get('status')=='cancelled':return
        job.update(status='running',message='初始化 SQLite 物理检测管线')
        folder=job['_folder'];folder.mkdir(parents=True,exist_ok=True);err=(folder/'worker.stderr.log').open('wb')
        try:
            p=subprocess.Popen([python_worker(),'-m','anomaly_workbench.live_worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,env=environment(),text=True,encoding='utf-8')
            job['_process']=p;p.stdin.write(json.dumps({'params':job['params'],'folder':str(folder),'dataset':job['dataset']},ensure_ascii=False));p.stdin.close()
            for line in p.stdout:
                if job.get('status')=='cancelled':break
                try:value=json.loads(line)
                except json.JSONDecodeError:continue
                if value.get('event')=='error':job.update(status='error',message=value.get('message','计算未完成'))
                else:job.update({k:v for k,v in value.items() if k!='event'})
            p.wait()
            if job.get('status')!='cancelled':
                if p.returncode==0 and (folder/'result.json.gz').is_file():job.update(status='complete',message='全部选定时段分析完成')
                elif job.get('status')!='error':job.update(status='error',message='检测进程退出，请检查独立环境和模型日志')
        except Exception as exc:job.update(status='error',message=str(exc))
        finally:err.close();job.pop('_process',None)

def result(job_id):
    job=job_by_id(job_id)
    if job['status']!='complete':raise HTTPException(409,'扫描尚未完成，请等待进度或检查任务错误')
    if '_result' not in job:
        with gzip.open(job['_folder']/'result.json.gz','rt',encoding='utf-8') as f:job['_result']=json.load(f)
        job['_result']['scan_id']=job_id
    return job,job['_result']

def load_window(job,index):
    p=job['_folder']/f'window-{index}.json.gz'
    if not p.is_file():return {'records':[],'summary':{}}
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)

def record(job_id,record_id,diagnostic=False):
    job,scan=result(job_id);index=scan['record_index'].get(record_id)
    if index is None:raise HTTPException(404,'该记录不属于当前扫描')
    item=next((r for r in load_window(job,index)['records'] if r['id']==record_id),None)
    if item is None:raise HTTPException(404,'记录不存在')
    event=next((r for r in scan.get('records',[])+scan.get('clue_records',[]) if r['id']==record_id),None)
    if event is None:
        event=next((r for r in scan.get('motion_events',[]) if record_id in r.get('event_window_ids',[])),None)
    if event:
        item.update({k:event[k] for k in ['event_start_utc','event_end_utc','event_window_ids','priority_window_ids','event_types','event_type_labels','related_events','related_event_count'] if k in event})
        item['episode_review_level']=event.get('review_level')
    if diagnostic:
        cached=job.setdefault('_diagnostics',{})
        if record_id not in cached:
            req={'action':'detail','params':job['params'],'dataset':scan['dataset'],'record_id':record_id,'start_utc':item['window_start_utc'],'end_utc':item['window_end_utc']}
            try:
                proc=subprocess.run([python_worker(),'-m','anomaly_workbench.live_worker'],input=json.dumps(req),capture_output=True,text=True,encoding='utf-8',env=environment(),timeout=120)
                messages=[]
                for line in proc.stdout.splitlines():
                    try:messages.append(json.loads(line))
                    except json.JSONDecodeError:pass
                message=next((v for v in messages if v.get('event')=='detail'),None)
                expected=evidence_signature(item.get('motion_observations'),item.get('neighbor_frames'))
                if message and expected and message.get('evidence_signature')==expected:
                    cached[record_id]=message['diagnostics']
                elif message:
                    item['diagnostics']=unavailable_diagnostics(item,'当前入库观测或邻船背景与本次保存扫描不一致，保留原扫描轨迹；模型曲线需重新扫描后生成。')
                else:
                    item['diagnostics']=unavailable_diagnostics(item,'模型诊断暂不可用，保留本次扫描已保存的原始轨迹、观测可信程度和核查证据。')
            except (OSError,ValueError,subprocess.TimeoutExpired):
                item['diagnostics']=unavailable_diagnostics(item,'模型诊断暂不可用，保留本次扫描已保存的原始轨迹、观测可信程度和核查证据。')
        if record_id in cached:item['diagnostics']=cached[record_id]
    elif record_id in job.get('_diagnostics',{}):
        item['diagnostics']=job['_diagnostics'][record_id]
    item['intent_assessment']=assess_maneuver_intent(item)
    item['review_history']=STORE.history(job_id,record_id)
    return scan,item

@router.get('')
@router.get('/')
def page():return FileResponse(ROOT/'static/live.html',media_type='text/html',headers={'Cache-Control':'no-store'})

@router.get('/static/{filename}')
def static(filename:str):
    if filename not in ['live.css','live.js','catalog-entry.js','neighbor-geometry.js','chart-logic.js']:raise HTTPException(404)
    return FileResponse(ROOT/'static'/filename,headers={'Cache-Control':'no-store'})

@router.get('/api/catalog')
def catalog(refresh:bool=False):
    try:return sqlite_stream.catalog(refresh=refresh)
    except Exception as exc:raise HTTPException(503,f'SQLite目录读取失败：{exc}') from exc

@router.get('/api/datasets/{dataset_id}')
def metadata(dataset_id:str):
    try:return sqlite_stream.dataset_metadata(dataset_id)
    except (ValueError,sqlite_stream.sqlite3.Error) as exc:raise HTTPException(422,str(exc)) from exc

@router.post('/api/jobs')
def scan(body:ScanRequest,request:Request):
    same_origin(request)
    try:
        ds=sqlite_stream.resolve(body.dataset_id);params=body.model_dump();params['dataset_id']=ds['id'];params['neighbor_revision']=3;params['policy_version']=5
        if not body.full_period:
            start=sqlite_stream.epoch(body.start_utc or ds['default_start_utc']);end=sqlite_stream.epoch(body.end_utc or ds['default_end_utc'])
            if end<=start:raise ValueError('结束时间必须晚于开始时间')
        python_worker()
    except (ValueError,FileNotFoundError) as exc:raise HTTPException(422,str(exc)) from exc
    with LOCK:
        if sum(j['status'] in ['queued','running'] for j in JOBS.values())>=3:raise HTTPException(429,'已有三个扫描任务，请先等待或取消')
        identifier=uuid4().hex
        job={'id':identifier,'scan_id':identifier,'status':'queued','message':'等待执行','completed':0,'total':0,'params':params,'dataset':ds,'created_at_utc':datetime.now(timezone.utc).isoformat(),'_folder':RUNTIME/'jobs'/identifier}
        JOBS[identifier]=job
        while len(JOBS)>12:
            expired=next((k for k,v in JOBS.items() if v['status'] not in ['queued','running']),None)
            if expired is None:break
            JOBS.pop(expired)
    threading.Thread(target=run_job,args=(job,),daemon=True).start()
    return public_job(job)

@router.get('/api/jobs/{job_id}')
def status(job_id:str):return public_job(job_by_id(job_id))

@router.post('/api/jobs/{job_id}/cancel')
def cancel(job_id:str,request:Request):
    same_origin(request);job=job_by_id(job_id);job.update(status='cancelled',message='已取消本次扫描');p=job.get('_process')
    if p and p.poll() is None:p.terminate()
    return public_job(job)

@router.get('/api/jobs/{job_id}/result')
def scan_result(job_id:str):
    _,scan=result(job_id);out={k:v for k,v in scan.items() if k!='record_index'}
    if scan.get('policy_version',0)<5:
        out={**out,'needs_rescan':True,'latest_policy_version':5,'records':[],'quality_records':[],'context_records':[],'clue_records':[],
             'scatter':[],'timeline':[],'type_counts':{},'notices':['该结果使用旧业务规则，已停止推荐旧候选；请重新扫描当前入库数据。']}
    return out

@router.get('/api/scans/{scan_id}/windows/{index}')
def window_result(scan_id:str,index:int):
    job,scan=result(scan_id)
    if not 0<=index<len(scan['timeline']):raise HTTPException(404)
    window=load_window(job,index);window['scan_id']=scan_id;return window

@router.get('/api/scans/{scan_id}/queue')
def queue_view(scan_id:str,kind:Literal['candidate','clue','quality','all']='candidate',
               type:str=Query(default='',max_length=80),limit:int=Query(default=20,ge=1,le=200),
               window_index:int|None=Query(default=None,ge=0),hide_stationary:bool=False,
               focus_record:str|None=Query(default=None,max_length=150)):
    job,scan=result(scan_id)
    if scan.get('policy_version',0)<5:raise HTTPException(409,'该扫描尚未检查可能正常操纵的解释，请重新执行')
    if window_index is not None and window_index>=len(scan.get('timeline',[])):raise HTTPException(404,'该扫描没有此时间窗')
    window=load_window(job,window_index) if window_index is not None else None
    value=build_queue(scan,kind=kind,category=type,limit=limit,window=window,
                      hide_stationary=hide_stationary,focus_record=focus_record)
    value['window_index']=window_index
    return value

@router.get('/api/scans/{scan_id}/records/{record_id}')
def record_detail(scan_id:str,record_id:str,diagnostic:bool=True):return record(scan_id,record_id,diagnostic)[1]

@router.post('/api/scans/{scan_id}/records/{record_id}/brief')
def brief(scan_id:str,record_id:str,body:BriefRequest,request:Request):
    same_origin(request);scan,item=record(scan_id,record_id);return generate_brief(scan,item,body.use_llm)

@router.post('/api/scans/{scan_id}/records/{record_id}/chart-explanations')
def chart_explanations(scan_id:str,record_id:str,request:Request):
    same_origin(request)
    scan,item=record(scan_id,record_id,False)
    if scan.get('policy_version',0)<5:
        raise HTTPException(409,'请按当前海事规则重新扫描后再解读图表')
    return generate_chart_explanations(scan,item)

@router.post('/api/reviews')
def review(body:ReviewRequest,request:Request):
    same_origin(request);scan,item=record(body.scan_id,body.record_id)
    value=STORE.append(scan_id=body.scan_id,record_id=body.record_id,dataset_id=scan['dataset']['id'],status=body.status,elapsed_seconds=body.elapsed_seconds,note=body.note,evidence={'record':item,'calibration':scan['calibration'],'params':scan['params']})
    return {'review':value,'history':STORE.history(body.scan_id,body.record_id)}

@router.get('/api/scans/{scan_id}/records/{record_id}/export')
def export(scan_id:str,record_id:str,format:Literal['json','md']='json'):
    scan,item=record(scan_id,record_id,True)
    payload={'dataset':scan['dataset'],'params':scan['params'],'calibration':scan['calibration'],'record':item,'source':'system3 SQLite live query','exported_at_utc':datetime.now(timezone.utc).isoformat()}
    if item.get('event_window_ids'):
        job=job_by_id(scan_id);evidence=[]
        for wid in item['event_window_ids']:
            index=scan['record_index'].get(wid)
            if index is not None:
                evidence.extend(r for r in load_window(job,index)['records'] if r['id']==wid)
        payload['event_window_evidence']=evidence
    canonical=json.dumps(payload,ensure_ascii=False,sort_keys=True);payload['sha256']=hashlib.sha256(canonical.encode()).hexdigest();body=json.dumps(payload,ensure_ascii=False,indent=2)
    if format=='md':body='# AIS 动态核查证据包\n\n'+generate_brief(scan,item)['text']+'\n\n```json\n'+body+'\n```\n'
    return Response(body,media_type='application/json' if format=='json' else 'text/markdown; charset=utf-8',headers={'Content-Disposition':f'attachment; filename="AIS-{record_id}.{format}"','Cache-Control':'no-store'})

def install_catalog_entry(app):
    @app.middleware('http')
    async def catalog_entry(request,call_next):
        response=await call_next(request)
        if request.url.path!='/data-management' or response.status_code!=200:return response
        body=b''.join([chunk async for chunk in response.body_iterator]);hook=b'<script src="/anomaly-workbench/static/catalog-entry.js" defer></script>'
        body=body.replace(b'</body>',hook+b'</body>');headers=dict(response.headers);headers.pop('content-length',None)
        return Response(body,status_code=response.status_code,headers=headers)
