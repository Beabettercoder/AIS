(() => {
  'use strict';
  const $=id=>document.getElementById(id), API='/anomaly-workbench/api';
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const good=v=>v!==null&&v!==undefined&&Number.isFinite(Number(v));
  const fmt=(v,n=2)=>good(v)?Number(v).toFixed(n):'—';
  const integer=v=>good(v)?Number(v).toLocaleString():'—';
  const types=[['turn_back','明显折返'],['deviation','方向变化线索'],['stall','快速减速后持续低速'],['sinuous','持续大幅交替转向'],['speed_drop','持续快速减速'],['formation_departure','编队脱离'],['group_inconsistency','群体运动不一致'],['congestion','拥堵 / 航线转移迹象'],['observation_glitch','位置跳变 / 运动学不一致']];
  const state={datasets:[],dataset:null,job:null,result:null,queue:'candidate',cards:[],queueInfo:null,windowIndex:null,selected:null,pendingRecordId:null,selectedAt:0,selectSerial:0,queueSerial:0,catalogSerial:0,epoch:0,metadataSerial:0,pollTimer:null,step:19,dirty:false,runPending:false,appliedCriteria:null,manualRange:null};
  const neighborGeometry=window.AISNeighborGeometry,chartLogic=window.AISChartLogic;
  let map,base,layers,positionLayer,toastTimer,neighborMarkers=new Map(),mapMode='context',requestedRecord=null,requestedStep=null;
  async function api(path,options={}) {
    const response=await fetch(API+path,{headers:{'Content-Type':'application/json'},...options});
    if(!response.ok){let data;try{data=await response.json();}catch(_){data={};}throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||`HTTP ${response.status}`));}
    return response.json();
  }
  function notice(text,error=false){$('message').textContent=text;$('message').hidden=!text;$('message').classList.toggle('error',error);}
  function toast(text){clearTimeout(toastTimer);$('toast').textContent=text;$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,4500);}
  const plainText=v=>String(v??'').replace(/SOG/g,'航速').replace(/COG/g,'航向').replace(/m\/s/g,'米/秒').replace(/\bkn\b/g,'节').replace(/\bnm\b/g,'海里').replace(/wₜ/g,'观测可信程度').replace(/个体流场残差|个体残差/g,'相对船群运动差异').replace(/运动学残差/g,'定位与航速不一致程度').replace(/(\d+(?:\.\d+)?) m\b/g,'$1 米').replace(/PG-DRGN/g,'轨迹检测模型');
  function dateInput(v){return String(v||'').replace('Z','').slice(0,19);}
  function maritimePolicy(record){return record?.maritime_policy||{};}
  function isStationary(record){const p=maritimePolicy(record);return p.stationary===true||record?.stationary===true||record?.motion_state==='berthed_anchored';}
  function isBehaviorCandidate(record){return Boolean(record?.is_candidate)&&!isStationary(record)&&good(record?.score)&&Number(record.score)>=.5&&Number(record.score)<=.99&&!state.result?.needs_rescan;}
  function priorityLabel(record){if(record?.review_level==='clue')return '一般线索';return good(record?.score)&&Number(record.score)>=0&&Number(record.score)<=.99?fmt(record.score,3):'待重算';}
  function sparseFallback(record){const p=maritimePolicy(record);return p.neighbor_supported===false||record?.neighbor_supported===false;}
  function scoreExplanation(record){
    const p=maritimePolicy(record);
    if(record?.review_reason)return record.review_reason;
    if(isStationary(record))return '静止/低速背景，已从行为候选剔除；模型原始输出不能改变该业务过滤。人工标注不改变算法排除与候选排序。';
    if(sparseFallback(record))return '按窗口终点及连续局部支持检查，评分采用自身运动学；未使用模型原始分或邻船速率差。';
    return p.score_explanation||p.explanation||record?.score_explanation||'业务优先级由实际运动变化证据与有效局部参考共同决定，原始模型输出单独保留。';
  }
  function cardWhy(record){
    const p=maritimePolicy(record);
    if(!Object.keys(p).length)return String(record.why_flagged||'点击查看完整核查依据。').slice(0,110);
    if(isStationary(record))return `原始最高航速 ${fmt(p.source_sog_max_kn)} 节，中位 ${fmt(p.source_sog_median_kn)} 节；可信累计位移 ${fmt(p.path_length_m,0)} m，已退出行为候选。`;
    if(['formation_departure','group_inconsistency'].includes(record.type)){
      const comparison=(p.reasons||[]).find(text=>/同一批|与相同/.test(text));
      if(comparison)return `可信在航轨迹 ${fmt(p.path_length_m,0)} m。${comparison}`;
    }
    if(p.evidence?.dynamic_speed_drop)return `前段 ${fmt(p.initial_sog_kn)} → 后段 ${fmt(p.late_sog_kn)} 节，下降 ${fmt(p.drop_fraction*100,1)}%；可信累计位移 ${fmt(p.path_length_m,0)} m。`;
    if(record.type==='deviation')return `可信连续轨迹 ${fmt(p.continuous_path_length_m,0)} m，相对自身初始航向的侧向偏移 ${fmt(p.cross_track_from_initial_heading_m,0)} m；未认定偏离官方航路。`;
    return `原始最高航速 ${fmt(p.source_sog_max_kn)} 节；可信连续轨迹 ${fmt(p.continuous_path_length_m,0)} m，真实位置转向 ${fmt(p.geometric_heading_change_deg,1)}°。`;
  }
  function requireRescan(){
    if(!state.result?.needs_rescan)return false;
    state.result=null;state.cards=[];state.queueInfo=null;requestedRecord=null;requestedStep=null;clearSelection('历史结果需要重新扫描。');
    $('candidates').innerHTML='<div class="empty">历史结果尚未区分普通运动线索与重点核查。请重新扫描，检查持续动作、历史反复模式及同船连续运动段。</div>';
    $('list-count').textContent='0';$('evidence').hidden=true;setReviewAvailability(false);
    notice('旧结果使用旧规则，已停止展示其候选排名。当前数据集、时间段和核查量已保留；点击“执行异常初筛”重新扫描。',true);
    return true;
  }
  const chartKeys=['motion','quality','timeline','types','residual'];
  const chartExplanationCache=new Map();
  let chartExplanationController=null,diagnosticController=null;
  function resetChartExplanations(message='选中船舶后，结合当前数据生成一句话解读。'){
    if(chartExplanationController)chartExplanationController.abort();
    chartExplanationController=null;
    for(const key of chartKeys){$('chart-'+key+'-explanation').textContent=message;$('chart-'+key+'-source').textContent='读图说明';}
  }
  function showChartExplanations(data){
    for(const key of chartKeys){const value=data.explanations?.[key];
      if(value?.purpose)$('chart-'+key+'-purpose').textContent=plainText(value.purpose);
      $('chart-'+key+'-explanation').textContent=plainText(value?.reading||'当前数据不足，先按读图说明核查。');
      $('chart-'+key+'-source').textContent=data.source==='local_llm'?'本地大模型解读':'规则说明';
    }
  }
  async function loadChartExplanations(serial){
    const scanId=state.result?.scan_id,recordId=state.selected?.id;if(!scanId||!recordId)return;
    const key=scanId+':'+recordId;
    if(chartExplanationCache.has(key)){showChartExplanations(chartExplanationCache.get(key));return;}
    const controller=new AbortController();chartExplanationController=controller;
    for(const name of chartKeys){$('chart-'+name+'-explanation').textContent='正在结合当前数据生成一句话解读…';$('chart-'+name+'-source').textContent='解读中';}
    try{
      const data=await api(`/scans/${scanId}/records/${encodeURIComponent(recordId)}/chart-explanations`,{method:'POST',signal:controller.signal});
      if(serial!==state.selectSerial||state.result?.scan_id!==scanId||state.selected?.id!==recordId)return;
      if(data.source==='local_llm'){chartExplanationCache.set(key,data);while(chartExplanationCache.size>20)chartExplanationCache.delete(chartExplanationCache.keys().next().value);}
      showChartExplanations(data);
    }catch(error){
      if(controller.signal.aborted||serial!==state.selectSerial||state.result?.scan_id!==scanId||state.selected?.id!==recordId)return;
      for(const name of chartKeys){$('chart-'+name+'-explanation').textContent='本地模型暂未完成解读，请先按上面的读图说明核查当前数据。';$('chart-'+name+'-source').textContent='规则说明';}
    }finally{if(chartExplanationController===controller)chartExplanationController=null;}
  }
  function setReviewAvailability(available){document.querySelectorAll('[data-review]').forEach(button=>{button.disabled=!available;});}
  function datasetBounds(){const b=state.dataset?.bbox;if(b?.length===4)map.fitBounds([[b[1],b[0]],[b[3],b[2]]],{padding:[28,28],maxZoom:13});}
  function initMap(){
    if(!window.L){$('map').textContent='地图组件未加载，请检查原系统的 Leaflet 资源。';return;}
    map=L.map('map',{zoomControl:true,preferCanvas:true}).setView([30,-88],8);
    base=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
    base.on('tileerror',()=>$('map-status').textContent='底图暂未加载；真实坐标轨迹与物理向量仍可核查');
    layers=L.layerGroup().addTo(map);positionLayer=L.layerGroup().addTo(map);
  }
  function context(){return {epoch:state.epoch,datasetId:state.dataset?.id,scanId:state.result?.scan_id};}
  function current(c,withScan=true){return c.epoch===state.epoch&&c.datasetId===state.dataset?.id&&(!withScan||c.scanId===state.result?.scan_id);}
  function changeAddress(values){const u=new URL(location.href);for(const [key,value] of Object.entries(values)){if(value===null||value===undefined||value==='')u.searchParams.delete(key);else u.searchParams.set(key,String(value));}history.replaceState(null,'',u);}
  function scannedRange(scan=state.result){const rows=scan?.timeline||[];return [rows[0]?.start_utc||scan?.params?.start_utc||'',rows.at(-1)?.end_utc||scan?.params?.end_utc||''];}
  function criteriaKey(){return JSON.stringify([state.dataset?.id,$('full').checked,$('full').checked?'':$('start').value,$('full').checked?'':$('end').value]);}
  function criteriaStatus(){
    const range=scannedRange();$('applied-range').textContent=state.result?`当前结果的扫描区间：${range[0]} 至 ${range[1]}（世界时，结束时刻不含）；下方四项统计始终覆盖这次完整扫描。`:'尚无扫描结果。';
    $('criteria-status').hidden=!state.dirty;$('criteria-status').textContent='时间条件已修改，尚未应用；当前清单和统计仍属于上面的已扫描区间，请重新执行初筛后复核。';
  }
  function clearSelection(message='当前清单没有选中船舶。',{redrawList=true}={}){
    state.selectSerial++;state.selected=null;state.pendingRecordId=null;state.selectedAt=0;resetChartExplanations(message);if(diagnosticController)diagnosticController.abort();diagnosticController=null;neighborMarkers=new Map();
    if(layers)layers.clearLayers();if(positionLayer)positionLayer.clearLayers();
    $('evidence').hidden=true;setReviewAvailability(false);$('brief-button').disabled=true;$('brief').hidden=true;$('review-note').value='';
    for(const id of ['export-json','export-md']){$(id).removeAttribute('href');$(id).setAttribute('aria-disabled','true');$(id).tabIndex=-1;}
    $('map-title').textContent='AIS 轨迹与船群运动参考';$('neighbor-count').textContent='等待选择当前清单中的目标';$('neighbor-note').textContent='按当前实际时间步显示同向邻船。';$('neighbor-list').innerHTML='';$('fit-busiest').disabled=true;
    $('step').disabled=true;$('step').value=0;$('step-time').textContent='选择船舶后显示实际世界时';$('diagnostic-target').textContent=message;
    for(const id of ['motion-chart','gate-chart']){$(id).textContent=message;$(id).classList.add('empty');}
    $('diagnostic-note').textContent='图中只画实际采样点；原始值缺失时断线，模型提示不等于异常概率或核查结论。';$('diagnostic-progress').textContent='';changeAddress({record:null,step:null});if(state.result)safeRender('scatter',renderScatter);if(redrawList)safeRender('list',renderList);
  }
  function clearScanDisplay(message='等待当前数据集的实际计算结果。'){
    state.cards=[];state.queueInfo=null;state.windowIndex=null;$('candidates').innerHTML='<div class="empty">'+esc(message)+'</div>';$('list-count').textContent='0';
    for(const id of ['timeline','type-chart','scatter']){$(id).textContent=message;$(id).classList.add('empty');}
    for(const id of ['kpi-ships','kpi-flow','kpi-quality','kpi-candidates'])$(id).textContent='—';
    $('kpi-points').textContent='等待扫描';$('kpi-speed').textContent='报文均速（含静止） · 船群参考有效率';$('kpi-window').textContent='等待扫描';$('timeline-selection').textContent='点击时间柱切换窗口';
    $('timeline-readout').textContent='折线分别看各自趋势，不能按同一高度比较船数与航速。';$('scatter-note').textContent='颜色表示当前筛选状态，不替代人工业务定性。';$('queue-scope').textContent='尚无扫描清单';$('return-scan').hidden=true;criteriaStatus();
  }
  function markCriteriaChanged(){
    const wasDirty=state.dirty;state.dirty=Boolean(state.result)&&criteriaKey()!==state.appliedCriteria;criteriaStatus();
    if(state.dirty){clearSelection('时间条件尚未应用，请重新扫描后选择船舶。');renderList();}
    else if(wasDirty&&state.result)loadQueue();
  }
  function exclusiveDatasetEnd(dataset=state.dataset){const end=Date.parse(dataset?.end_utc);return Number.isFinite(end)?new Date(end+1000).toISOString():dataset?.end_utc||'';}
  function syncTimeInputs(){const full=$('full').checked;for(const id of ['start','end'])$(id).disabled=full;if(full&&state.dataset){const range=state.result?.params?.full_period?scannedRange():[state.dataset.start_utc,exclusiveDatasetEnd()];$('start').value=dateInput(range[0]);$('end').value=dateInput(range[1]);}}
  function loadMetadata(d,c){
    const serial=++state.metadataSerial,valid=()=>state.dataset?.id===d.id&&serial===state.metadataSerial;
    $('dataset-meta').textContent=`${integer(d.row_count)} 条实际 AIS · 唯一船舶数正在核对 · 世界时 UTC ${d.start_utc} — ${d.end_utc}`;
    api('/datasets/'+encodeURIComponent(d.id)).then(meta=>{if(!valid())return;Object.assign(state.dataset,meta);$('dataset-meta').textContent=`${integer(meta.row_count)} 条实际 AIS · ${integer(meta.unique_mmsi)} 艘唯一船舶 · 数据完整范围 ${meta.start_utc} — ${meta.end_utc}（世界时 UTC） · 经纬度 [${meta.bbox.map(v=>Number(v).toFixed(4)).join(', ')}]`;if($('full').checked&&!state.result)syncTimeInputs();if(!state.selected)datasetBounds();}).catch(error=>{if(valid())$('dataset-meta').textContent+=' · 元数据核对未完成：'+error.message;});
  }
  async function catalog(refresh=false){
    const serial=++state.catalogSerial,c=context(),address=new URLSearchParams(location.search),wanted=refresh?state.dataset?.id:address.get('dataset'),resume=refresh?null:address.get('scan');let activeContext=c;
    try{
      const data=await api('/catalog'+(refresh?'?refresh=true':''));if(serial!==state.catalogSerial||!current(c,false))return;
      const datasets=data.datasets||[];if(!datasets.length)throw new Error('当前没有可分析的 AIS 入库轨迹。请先接入数据。');
      const match=d=>[d.id,d.name,...(d.sources||[]).map(v=>v.id)].some(x=>String(x).toLowerCase()===String(wanted||state.dataset?.id||'').toLowerCase());
      if(!refresh&&wanted&&!datasets.some(match))throw new Error('指定数据集在当前 AIS 目录中不可用，未自动运行其他数据集。');
      const chosen=datasets.find(match)||datasets[0];state.datasets=datasets;$('dataset').innerHTML=datasets.map(d=>`<option value="${esc(d.id)}">${esc(d.label)}</option>`).join('');$('dataset').value=chosen.id;$('dataset').disabled=false;$('connection').textContent='已连接入库轨迹';
      if(refresh&&state.dataset?.id===chosen.id){state.dataset={...state.dataset,...chosen};loadMetadata(state.dataset,context());notice('目录已刷新；当前扫描、时间条件和核查目标已保留，未重新扫描。');return;}
      changeDataset(true);const restoredContext=context();activeContext=restoredContext;
      if(resume){
        const job=await api('/jobs/'+encodeURIComponent(resume));if(!current(restoredContext,false))return;
        if(job.dataset.id!==chosen.id)throw new Error('该扫描属于其他数据集，未加载到当前数据源。');
        const scan=await api(`/jobs/${resume}/result`);if(!current(restoredContext,false))return;state.job=job;state.result=scan;
        $('far').value=String(job.params.far_budget);$('full').checked=Boolean(job.params.full_period);const range=scannedRange(scan);$('start').value=dateInput(range[0]);$('end').value=dateInput(range[1]);for(const id of ['start','end'])$(id).disabled=$('full').checked;
        state.appliedCriteria=criteriaKey();state.dirty=false;requestedRecord=address.get('record');requestedStep=address.has('step')?Number(address.get('step')):null;
        const limit=Number(address.get('limit'));if(Number.isInteger(limit)&&limit>=1&&limit<=200)$('topk').value=limit;$('topk-value').textContent=$('topk').value;
        state.queue=['candidate','clue','quality','all'].includes(address.get('queue'))?address.get('queue'):'candidate';$('type-filter').value=address.get('type')||'';
        const index=address.has('window')?Number(address.get('window')):null;state.windowIndex=Number.isInteger(index)&&index>=0&&index<scan.timeline.length?index:null;
        if(requireRescan())return;renderResult(false);notice('已恢复这次扫描；当前结果区间已在控制栏下方注明。');return;
      }
      if(!refresh&&wanted)await run();
    }catch(error){if(serial===state.catalogSerial&&current(activeContext,false)){notice(error.message,true);$('connection').textContent='数据目录需检查';}}
  }
  function changeDataset(fromCatalog=false){
    const d=state.datasets.find(v=>v.id===$('dataset').value);if(!d)return;if(!fromCatalog)state.catalogSerial++;
    const old=state.job;if(old&&['running','queued'].includes(old.status))api(`/jobs/${old.id}/cancel`,{method:'POST'}).catch(()=>{});
    state.epoch++;state.queueSerial++;clearTimeout(state.pollTimer);state.dataset=d;state.job=null;state.result=null;state.appliedCriteria=null;state.dirty=false;state.runPending=false;state.manualRange=null;requestedRecord=null;requestedStep=null;
    clearSelection('选择已接入数据集并执行初筛。');clearScanDisplay();changeAddress({dataset:d.id,scan:null,window:null,queue:null,type:null});
    $('type-filter').value='';state.queue='candidate';$('start').value=dateInput(d.default_start_utc);$('end').value=dateInput(d.default_end_utc);for(const id of ['start','end']){$(id).min=dateInput(d.start_utc);$(id).max=dateInput(exclusiveDatasetEnd(d));}syncTimeInputs();
    $('progress').hidden=true;$('cancel').hidden=true;$('run').disabled=false;loadMetadata(d,context());datasetBounds();notice('数据源已切换，等待你执行初筛。');
  }
  async function run(){
    if(!state.dataset||state.runPending||state.job&&['running','queued'].includes(state.job.status))return;
    state.epoch++;state.queueSerial++;const c=context();state.runPending=true;$('run').disabled=true;clearTimeout(state.pollTimer);clearSelection('扫描进行中，完成后再选择当前船舶。');state.result=null;state.cards=[];state.job=null;state.appliedCriteria=null;state.dirty=false;clearScanDisplay('正在扫描当前已入库轨迹…');changeAddress({scan:null,window:null});
    const body={dataset_id:state.dataset.id,start_utc:$('start').value?$('start').value+'Z':'',end_utc:$('end').value?$('end').value+'Z':'',full_period:$('full').checked,far_budget:Number($('far').value)||.01,top_k:20,exclude_stationary:true,window_minutes:20};
    const submittedCriteria=criteriaKey();
    try{
      const job=await api('/jobs',{method:'POST',body:JSON.stringify(body)});if(!current(c,false)){api(`/jobs/${job.id}/cancel`,{method:'POST'}).catch(()=>{});return;}
      state.job=job;state.appliedCriteria=submittedCriteria;$('cancel').hidden=false;$('progress').hidden=false;notice('正在读取已入库轨迹，计算真实运动变化和可信邻船参考。');poll();
    }catch(error){if(current(c,false)){notice(error.message,true);$('run').disabled=false;}}
    finally{if(current(c,false))state.runPending=false;}
  }
  async function poll(){
    if(!state.job)return;const id=state.job.id,c=context();
    try{
      const job=await api('/jobs/'+id);if(!current(c,false)||state.job?.id!==id)return;state.job=job;
      $('progress-bar').max=Math.max(1,job.total||1);$('progress-bar').value=job.completed||0;$('progress-text').textContent=`${job.message||'计算中'} · ${job.completed||0}/${job.total||'…'} 时窗${job.window_start?' · '+job.window_start:''}`;
      if(job.status==='complete'){
        const scan=await api(`/jobs/${id}/result`);if(!current(c,false)||state.job?.id!==id)return;state.result=scan;state.windowIndex=null;state.queue='candidate';
        const range=scannedRange(scan);if(criteriaKey()===state.appliedCriteria){$('start').value=dateInput(range[0]);$('end').value=dateInput(range[1]);state.appliedCriteria=criteriaKey();state.dirty=false;}else state.dirty=true;
        renderResult();$('run').disabled=false;$('cancel').hidden=true;$('progress').hidden=true;notice(state.dirty?'扫描已完成，但当前时间条件已变更；请按扫描区间查看结果，重新扫描后复核。':'扫描已完成，已显示当前范围的核查清单；点击船舶即可先查看保存轨迹。');return;
      }
      if(['error','cancelled'].includes(job.status)){notice(job.message,job.status==='error');$('run').disabled=false;$('cancel').hidden=true;return;}
      state.pollTimer=setTimeout(poll,1400);
    }catch(error){if(current(c,false)&&state.job?.id===id){notice(error.message,true);$('run').disabled=false;$('cancel').hidden=true;}}
  }
  function renderResult(resetQueue=true){
    if(requireRescan())return;const s=state.result.summary,rows=state.result.timeline||[],jam=rows.filter(r=>r.macro_label==='拥堵积压迹象').length;
    $('kpi-ships').textContent=integer(s.unique_mmsi);$('kpi-points').textContent=integer(s.point_count)+' 条 AIS / '+integer(s.window_count)+' 个实际时窗';$('kpi-flow').textContent=jam?`${jam} 个积压迹象时窗`:s.field_valid_ratio<.5?'船群参考覆盖不足':'未见整体积压判据';$('kpi-speed').textContent=`报文均速（含静止） ${fmt(s.mean_sog_kn)} 节 · 参考有效率 ${fmt(100*s.field_valid_ratio,1)}%`;
    $('kpi-quality').textContent=integer(s.quality_count);$('kpi-candidates').textContent=integer(s.candidate_vessel_count);$('kpi-window').textContent=`${integer(s.candidate_count)} 个优先运动段 · ${integer(s.general_clue_event_count)} 个一般线索段；均覆盖完整扫描`;
    if(resetQueue)state.queue='candidate';criteriaStatus();safeRender('timeline',renderTimeline);safeRender('distribution',renderDistribution);changeAddress({scan:state.result.scan_id});
    const preferredId=requestedRecord;requestedRecord=null;loadQueue({preferredId});
  }
  async function loadQueue({preferredId=null}={}){
    if(!state.result||state.result.needs_rescan)return;const c=context(),serial=++state.queueSerial,kind=state.queue,type=$('type-filter').value,windowIndex=state.windowIndex,limit=Math.max(1,Math.min(200,Number($('topk').value)||20));
    const query=new URLSearchParams({kind,type,limit:String(limit),hide_stationary:String(kind==='all'&&$('exclude-stationary').checked)});if(windowIndex!==null)query.set('window_index',String(windowIndex));if(preferredId)query.set('focus_record',preferredId);
    document.querySelectorAll('[data-queue]').forEach(b=>b.classList.toggle('active',b.dataset.queue===kind));$('candidates').innerHTML='<div class="empty">正在读取当前范围的船舶清单…</div>';$('list-count').textContent='…';
    changeAddress({queue:kind,type,window:windowIndex,limit});$('return-scan').hidden=windowIndex===null;
    const windowRow=state.result.timeline?.[windowIndex];$('timeline-selection').textContent=windowIndex===null?'完整扫描清单':`${windowRow?.start_utc||''} — ${windowRow?.end_utc||''}`;
    if(windowRow)$('timeline-readout').textContent=`当前时间窗：窗内在航船数 ${windowRow.moving_vessels||0} 艘 · 报文均速（含静止） ${fmt(windowRow.mean_sog_kn)} 节 · 优先核查 ${windowRow.candidate_count||0} 窗；四项统计仍覆盖完整扫描。`;
    try{
      const data=await api(`/scans/${c.scanId}/queue?${query}`);if(!current(c)||serial!==state.queueSerial)return;state.cards=data.records||[];state.queueInfo=data;safeRender('list',renderList);
      if(state.dirty){clearSelection('时间条件尚未应用，请重新扫描后复核。');return;}
      const desired=state.cards.find(r=>r.id===preferredId)||state.cards.find(r=>r.id===state.selected?.id)||state.cards[0];
      if(!desired){clearSelection(kind==='candidate'?'当前范围没有证据充分的优先核查船舶。':'当前筛选没有匹配的船舶。');return;}
      if(state.selected?.id!==desired.id&&state.pendingRecordId!==desired.id)selectRecord(desired.id);
    }catch(error){if(current(c)&&serial===state.queueSerial){state.cards=[];state.queueInfo=null;clearSelection('清单读取未完成。');$('list-count').textContent='0';$('candidates').innerHTML='<div class="empty">'+esc(error.message)+'</div>';}}
  }
  function renderList(){
    const records=state.cards||[],info=state.queueInfo;$('list-count').textContent=integer(records.length);
    const scope=state.windowIndex===null?'全扫描清单（每船一项）':`当前时间窗 ${state.result?.timeline?.[state.windowIndex]?.start_utc||''}`;
    $('queue-scope').textContent=`${scope} · 显示 ${records.length} 艘 / 匹配 ${integer(info?.total_vessels)} 艘；共 ${integer(info?.total_matching)} 个${info?.matching_unit||'记录'}。上方统计仍为完整扫描。`;
    if($('type-filter').value==='congestion'){$('candidates').innerHTML='<div class="empty">区域交通聚散提示请在下方时间轴回看，不给单艘船定性。</div>';return;}
    $('candidates').innerHTML=records.map(r=>{const stationary=isStationary(r),candidate=isBehaviorCandidate(r),kind=stationary?'静止背景':state.queue==='quality'?'观测质量疑点':candidate||r.review_level==='clue'?r.type_label:'背景参照';
      return `<button type="button" ${state.dirty?'disabled':''} class="candidate ${state.queue==='quality'?'quality':''} ${!candidate?'is-context':''} ${stationary?'is-stationary':''} ${(state.pendingRecordId||state.selected?.id)===r.id?'active':''} ${state.pendingRecordId===r.id?'is-loading':''}" aria-busy="${state.pendingRecordId===r.id}" data-record="${esc(r.id)}"><div class="candidate-head"><strong>${esc(r.vessel_id)}</strong><span class="score">${priorityLabel(r)}<small>${candidate?' 业务优先级':' 不是异常定性'}</small></span></div>${state.pendingRecordId===r.id?'<p class="loading-copy">正在加载这艘船的已保存轨迹…</p>':''}<time>${esc(r.event_start_utc||r.start_utc)}<br>至 ${esc(r.event_end_utc||r.end_utc)}</time><span class="tag ${!candidate?'neutral':''}">${esc(kind)}</span><p class="metrics">相对船群运动差异（窗口可信时段均值） ${good(r.field_residual_mps)?fmt(r.field_residual_mps)+' 米/秒':'未形成有效参考'} · 观测可信程度 ${fmt(r.mean_gate_weight)}</p><p>${esc(plainText(r.intent_assessment?.observed_action||cardWhy(r)))}</p>${r.intent_assessment?.normal_operation_not_excluded?'<p class="subtle">正常操纵尚未排除，可能意图待核实。</p>':''}<p class="policy-source">${esc(plainText(scoreExplanation(r)))}</p>${r.event_window_ids?.length>1?`<p class="subtle">合并 ${r.event_window_ids.length} 个连续动作窗口${r.event_type_labels?.length>1?' · '+esc(r.event_type_labels.join('、')):''}</p>`:''}${r.related_event_count>1?`<p class="subtle">该船匹配 ${r.related_event_count} 个${esc(r.related_matching_unit||'运动段')}，时间轴保留对应证据。</p>`:''}</button>`;
    }).join('')||'<div class="empty">当前范围和类型没有匹配对象；不会用普通线索填补优先核查清单。</div>';
  }
  function safeRender(panel,render,{fit=false}={}){
    try{render();return true;}catch(error){
      console.error('AIS '+panel+' render failed',error);const text='该面板暂未完成显示：'+String(error.message||error);
      if(panel==='map'){$('map-status').textContent=text;return drawSavedTrackFallback(fit);}
      if(panel==='diagnostics')$('diagnostic-note').textContent=text+'；已保存轨迹和当前目标继续保留。';
      if(panel==='scatter')$('scatter-note').textContent=text;if(panel==='timeline')$('timeline-readout').textContent=text;if(panel==='distribution')$('type-chart').textContent=text;
      if(panel==='list'){$('candidates').innerHTML='<div class="empty">'+esc(text)+'</div>';$('list-count').textContent='显示失败';}
      return false;
    }
  }
  function drawSavedTrackFallback(fit=false){
    const r=state.selected,points=(r?.latlon||[]).filter(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite));
    if(!map||!r||!points.length)return false;
    try{layers.clearLayers();positionLayer.clearLayers();L.polyline(points,{color:'#0f6c5c',weight:3}).bindTooltip('已保存的原始轨迹；正常操纵是否排除仍需业务核查。').addTo(layers);const k=Math.min(state.step,points.length-1);L.circleMarker(points[k],{radius:7,color:'#fff',weight:2,fillColor:'#0f6c5c',fillOpacity:1}).addTo(positionLayer);$('map-title').textContent=`MMSI ${r.vessel_id} · 已保存的原始轨迹`;$('neighbor-count').textContent='已显示原始轨迹，邻船叠加暂未完成';if(fit)map.fitBounds(L.latLngBounds(points),{padding:[40,40],maxZoom:15});return true;}
    catch(error){console.error('AIS saved trajectory fallback failed',error);$('map-status').textContent='海图暂未完成显示，已保存的轨迹证据与当前目标仍保留：'+String(error.message||error);return false;}
  }
  function renderIntent(record){
    const value=record.intent_assessment;
    $('intent-action').textContent='已观察动作：'+(value?.observed_action||'未提供');
    const intents=Array.isArray(value?.possible_intents)?value.possible_intents:[];$('intent-options').innerHTML=intents.length?intents.map(text=>'<li>'+esc(plainText(text))+'</li>').join(''):'<li>未提供可核实的意图线索。</li>';
    $('intent-explanation').textContent=value?.explanation||'未提供动作与可能意图的说明。';
    $('intent-normal').textContent=value?.normal_operation_not_excluded===true?'正常操纵尚未排除，不能据此确认异常意图。':value?.normal_operation_not_excluded===false?'可能意图仍需结合业务资料核实。':'是否排除正常操纵：未提供证据。';
    $('intent-context').textContent='待补充资料：'+(Array.isArray(value?.missing_context)&&value.missing_context.length?value.missing_context.join('、'):'未提供');
  }
  async function enrichDiagnostics(c,id,serial){
    const controller=new AbortController();diagnosticController=controller;
    const selected=()=>current(c)&&serial===state.selectSerial&&state.selected?.id===id&&!state.dirty;
    $('diagnostic-progress').textContent='已显示保存轨迹和观测可信程度，模型曲线正在后台计算；你可以继续切船和核查。';
    try{
      const detail=await api(`/scans/${c.scanId}/records/${encodeURIComponent(id)}?diagnostic=true`,{signal:controller.signal});if(!selected())return;
      state.selected.diagnostics=detail.diagnostics||{};safeRender('diagnostics',renderDiagnostics);safeRender('map',()=>renderMap(false));
      const diagnostics=state.selected.diagnostics,available=Array.isArray(diagnostics.pg_probability)&&diagnostics.pg_probability.length>0;
      $('diagnostic-progress').textContent=available?'后台模型诊断已更新；船舶、时间步与海图视野保持当前选择。':'模型诊断未提供，已保存轨迹和观测可信程度继续保留。';
    }catch(error){if(controller.signal.aborted||!selected())return;$('diagnostic-progress').textContent='模型曲线暂未完成：'+String(error.message||error)+'；已保存轨迹继续显示，可继续核查。';}
    finally{if(diagnosticController===controller)diagnosticController=null;if(selected())loadChartExplanations(serial);}
  }
  async function selectRecord(id){
    if(!state.result||state.dirty||!state.cards.some(r=>r.id===id))return;const c=context(),card=state.cards.find(v=>v.id===id);clearSelection('正在读取当前船舶…',{redrawList:false});const serial=state.selectSerial;state.selectedAt=performance.now();state.pendingRecordId=id;
    safeRender('list',renderList);$('map-title').textContent='正在加载 MMSI '+card.vessel_id;$('neighbor-count').textContent='正在读取这艘船已保存的轨迹…';$('diagnostic-target').textContent='正在加载 MMSI '+card.vessel_id;
    try{
      const r=await api(`/scans/${c.scanId}/records/${encodeURIComponent(id)}?diagnostic=false`);if(!current(c)||serial!==state.selectSerial||!state.cards.some(v=>v.id===id))return;
      if(state.result.needs_rescan||!r.maritime_policy){clearSelection('记录缺少当前规则证据，请重新扫描。');notice('记录缺少当前规则证据，请重新扫描。',true);return;}
      for(const key of ['event_start_utc','event_end_utc','event_window_ids','priority_window_ids','event_types','event_type_labels','related_event_count','related_matching_unit'])if(card?.[key]!==undefined)r[key]=card[key];
      state.selected=r;state.pendingRecordId=null;
      try{state.step=neighborGeometry.preferredStep(r,requestedStep);}catch(error){state.step=Math.max(0,(r.latlon||[]).length-1);console.error('AIS preferred step unavailable',error);}requestedStep=null;
      $('step').max=Math.max(0,(r.latlon||[]).length-1);$('step').value=state.step;$('step').disabled=!(r.latlon||[]).length;mapMode='context';changeAddress({record:r.id,step:state.step});$('map-title').textContent=`MMSI ${r.vessel_id} · 已保存轨迹与船群运动参考`;
      $('evidence').hidden=false;$('evidence-title').textContent=`MMSI ${r.vessel_id} · ${isStationary(r)?'静止背景':isBehaviorCandidate(r)?r.type_label:r.review_level==='clue'?'一般线索 · '+r.type_label:r.has_quality_issue?'观测质量核查':'背景参照'}`;$('why').textContent=plainText(r.why_flagged);
      const motion=maritimePolicy(r),raw=good(r.model_raw_score)?` · 模型提示 ${fmt(r.model_raw_score,4)}（仅作诊断）`:'';
      $('evidence-numbers').textContent=`${r.start_utc} — ${r.end_utc} · 业务优先级 ${priorityLabel(r)}${raw} · 相对船群运动差异（窗口可信时段均值） ${fmt(r.field_residual_mps,3)} 米/秒 · 定位与航速不一致峰值 ${fmt(r.e_kin_mps,3)} 米/秒 · 轨迹长度 ${fmt(motion.path_length_m,0)} 米。${plainText(scoreExplanation(r))}`;
      renderIntent(r);setReviewAvailability(true);$('brief-button').disabled=false;$('review-status').textContent=r.review_history?.length?'已归档 '+r.review_history.length+' 次复核':'待人工复核';
      const prefix=`/anomaly-workbench/api/scans/${c.scanId}/records/${encodeURIComponent(r.id)}/export?format=`;for(const [element,format] of [['export-json','json'],['export-md','md']]){$(element).href=prefix+format;$(element).removeAttribute('aria-disabled');$(element).tabIndex=0;}
      safeRender('list',renderList);safeRender('map',()=>renderMap(true),{fit:true});safeRender('diagnostics',renderDiagnostics);safeRender('scatter',renderScatter);enrichDiagnostics(c,id,serial);
    }catch(error){if(current(c)&&serial===state.selectSerial){if(state.selected?.id===id){state.pendingRecordId=null;notice('已保存轨迹已加载，部分说明尚未完成：'+String(error.message||error),true);safeRender('map',()=>renderMap(false));safeRender('list',renderList);}else{clearSelection('这艘船的保存轨迹读取未完成。');notice(error.message,true);}}}
  }
  function arrow(origin,vector,color,label){
    if(!origin||!vector||!vector.every(good))return;
    const lat=origin[0],lon=origin[1],dx=vector[0]*60,dy=vector[1]*60;
    const end=[lat+dy/111195,lon+dx/(111195*Math.max(.001,Math.cos(lat*Math.PI/180)))];
    L.polyline([origin,end],{color,weight:3,opacity:.9}).addTo(layers);
    const angle=Math.atan2(dx,dy),length=Math.min(80,Math.max(15,Math.hypot(dx,dy)*.2));
    const wings=[-.5,.5].map(a=>[end[0]-length*Math.cos(angle+a)/111195,end[1]-length*Math.sin(angle+a)/(111195*Math.max(.001,Math.cos(lat*Math.PI/180)))]);
    L.polyline([wings[0],end,wings[1]],{color,weight:3}).addTo(layers);
    L.marker(end,{interactive:false,icon:L.divIcon({className:'vector-label'+(color==='#ef4444'?' residual':''),html:esc(label),iconSize:null})}).addTo(layers);
  }
  function renderMap(fit=false){
    const r=state.selected;if(!map||!r)return;layers.clearLayers();positionLayer.clearLayers();
    neighborMarkers=new Map();const frame=neighborGeometry.frameAt(r,state.step),neighbors=neighborGeometry.visibleNeighbors(r,state.step),show=$('show-neighbors').checked,fieldConfirmed=neighbors.length>=3&&frame.field_valid===true;
    if(show)for(const [neighborIndex,n] of neighbors.entries()){
      for(const segment of n.visibleSegments||[n.visibleTrack])if(segment.length>1)L.polyline(segment,{color:'#167f93',weight:2.5,opacity:.85}).addTo(layers);
      const heading=good(n.cog_deg)?n.cog_deg:Math.atan2(n.velocity?.[0]||0,n.velocity?.[1]||0)*180/Math.PI;
      const icon=L.divIcon({className:'neighbor-ship-marker',iconSize:[23,23],iconAnchor:[11.5,11.5],html:`<svg viewBox="0 0 26 26" style="transform:rotate(${heading}deg)"><path d="M13 2L22 23L13 18L4 23Z" fill="#087f98" stroke="white" stroke-width="2"/></svg>`});
      const marker=L.marker(n.position,{icon,title:'同向邻船 MMSI '+n.mmsi}).addTo(layers);
      marker.bindTooltip(`MMSI ${esc(n.mmsi)}`,{permanent:$('show-neighbor-labels').checked,direction:neighborIndex%2?'right':'left',offset:[neighborIndex%2?10:-10,0],className:'neighbor-mmsi-label'});
      marker.bindPopup(`<strong>同向邻船 ${esc(n.mmsi)}</strong><br>报告世界时：${esc(n.report_utc||'末步快照')}<br>航速 ${fmt(n.sog_kn)} 节 · 航向 ${fmt(n.cog_deg,1)}°<br>距目标 ${fmt(n.distance_nm)} 海里 · 可信度 ${fmt(n.weight,3)}<br>报文距当前步 ${fmt(n.age_seconds,0)} 秒`);
      neighborMarkers.set(String(n.mmsi),marker);
    }
    $('neighbor-count').textContent=show?`当前时间步：${neighbors.length} 艘同向邻船`:`同向邻船图层已关闭（本步 ${neighbors.length} 艘）`;
    const historicalIds=new Set();for(let i=0;i<(r.latlon||[]).length;i++)for(const n of neighborGeometry.visibleNeighbors(r,i))historicalIds.add(String(n.mmsi));const historical=historicalIds.size;
    const basis=frame.direction_basis||'当前报告航向';
    $('neighbor-note').textContent=`3.5 海里内 · 在航航速不低于2节 · 航向差 <45° · ${plainText(basis)}${frame.direction_reference_utc?'（'+frame.direction_reference_utc+'）':''}。${neighbors.length<3?'当前不足3艘，本时间步不使用相对邻船速率差。':!fieldConfirmed?'本时间步流场未通过有效性检查，不使用此步邻船速率差。':''}${historical&&!neighbors.length?'本窗口其他时间步有 '+historical+' 艘满足3.5海里条件，可切换到“邻船最多时刻”。':''}`;
    $('neighbor-list').innerHTML=show?neighbors.map(n=>`<button type="button" data-neighbor-mmsi="${esc(n.mmsi)}" title="定位这艘真实邻船">${esc(n.mmsi)} <span>${fmt(n.distance_nm)} 海里</span></button>`).join(''):'';
    $('fit-busiest').disabled=!historical;
    if($('show-raw').checked){L.polyline(r.latlon,{color:'#ef4444',weight:1.8,dashArray:'5 6',opacity:.7}).bindTooltip('原始轨迹；正常操纵可能尚未排除，可能意图待核实。').addTo(layers);for(const i of r.quality_steps||[])if(r.latlon[i])L.marker(r.latlon[i],{icon:L.divIcon({className:'',html:'<span style="color:#d08307;font-size:23px;font-weight:bold">×</span>',iconSize:[16,16],iconAnchor:[8,8]})}).bindTooltip(`步 ${i} · 观测可信程度=${fmt(r.gate_weights[i],3)}`).addTo(layers);}
    const points=r.repaired_latlon||r.latlon,p=r.diagnostics?.pg_probability||[];
    if($('show-repair').checked)for(let i=1;i<points.length;i++){const score=p[i],color=isStationary(r)?'#8b9b94':isBehaviorCandidate(r)&&good(score)?`rgb(${Math.round(15+220*score)},${Math.round(130-60*score)},${Math.round(115-45*score)})`:'#0f6c5c';L.polyline([points[i-1],points[i]],{color,weight:3.2}).bindTooltip(`UTC ${r.start_utc} + ${fmt(r.time_seconds[i],0)}秒 · 原始模型诊断 ${fmt(score,3)} · 业务优先级 ${priorityLabel(r)} · 正常操纵可能尚未排除，意图待核实。`).addTo(layers);}
    const k=Math.min(state.step,points.length-1);L.circleMarker(points[k],{radius:7,color:'#fff',weight:2,fillColor:'#0f6c5c',fillOpacity:1}).bindTooltip('目标 '+r.vessel_id,{permanent:true,direction:'left',offset:[-9,0],className:'target-mmsi-label'}).addTo(positionLayer);
    if(fieldConfirmed){arrow(points[k],r.field_vectors?.[k],'#10b981','船群参考运动');arrow(points[k],r.residual_vectors?.[k],'#ef4444','当前步运动差异');}
    $('step-time').textContent=new Date(Date.parse(r.start_utc)+(r.time_seconds[k]||0)*1000).toISOString().replace('.000Z','Z');
    $('map-title').textContent=`MMSI ${r.vessel_id} · 真实轨迹与船群运动参考`;
    if(fit){const bounds=neighborGeometry.boundsPoints(r,state.step,show&&mapMode==='context');if(bounds.length){map.invalidateSize({pan:false});map.fitBounds(L.latLngBounds(bounds),{padding:[48,48],maxZoom:15,animate:false});}}
  }
  function chart(series,length,{rightMax=1,leftLabel='',rightLabel='',times=[]}={}){
    const W=680,H=235,L=47,R=44,T=22,B=32,inner=W-L-R,positions=chartLogic.timePositions(times,length);
    if(!positions.some(good))return '<div class="empty">采样时间缺失，未按等间隔虚构曲线。</div>';
    const leftVals=series.filter(v=>!v.right).flatMap(v=>v.values.filter(good)),lm=Math.max(...leftVals,1)*1.05,x=i=>good(positions[i])?L+positions[i]*inner:null,y=(v,right)=>H-B-Number(v)/(right?rightMax:lm)*(H-T-B);
    let out=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(leftLabel+' 与 '+rightLabel)}">`;
    if(state.selected?.gate_weights)state.selected.gate_weights.forEach((v,i)=>{if(good(v)&&v<.3&&good(x(i))){const left=i>0&&good(x(i-1))?(x(i-1)+x(i))/2:x(i),right=i+1<length&&good(x(i+1))?(x(i+1)+x(i))/2:x(i);out+=`<rect x="${left}" y="${T}" width="${Math.max(2,right-left)}" height="${H-T-B}" fill="#edf0f0"/>`;}});
    for(let j=0;j<5;j++){const v=j/4;out+=`<path d="M${L} ${y(lm*v,false)}H${W-R}" stroke="#e5ede9"/><text x="${L-7}" y="${y(lm*v,false)+4}" text-anchor="end" fill="#7b8e87" font-size="10">${fmt(lm*v,lm>10?0:2)}</text><text x="${W-R+6}" y="${y(rightMax*v,true)+4}" fill="#7b8e87" font-size="10">${fmt(rightMax*v,rightMax>10?0:2)}</text>`;}
    if(leftLabel.startsWith('本船 / 邻船航速')&&state.selected?.neighbor_sog_low){const lo=state.selected.neighbor_sog_low,hi=state.selected.neighbor_sog_high,values=lo.map((v,i)=>good(v)&&good(hi[i])&&good(x(i))?v:null);for(const segment of chartLogic.seriesSegments(values)){if(segment.length>1)out+=`<polygon points="${segment.map(i=>`${x(i)},${y(hi[i],false)}`).join(' ')} ${segment.slice().reverse().map(i=>`${x(i)},${y(lo[i],false)}`).join(' ')}" fill="#91aaa3" opacity=".18"/>`;}}
    for(const seriesItem of series){const values=seriesItem.values.map((v,i)=>good(v)&&good(x(i))?v:null);for(const segment of chartLogic.seriesSegments(values,{circular:Boolean(seriesItem.circular)})){if(segment.length===1){const i=segment[0];out+=`<circle cx="${x(i)}" cy="${y(values[i],seriesItem.right)}" r="2" fill="${seriesItem.color}"/>`;}else out+=`<polyline points="${segment.map(i=>`${x(i)},${y(values[i],seriesItem.right)}`).join(' ')}" fill="none" stroke="${seriesItem.color}" stroke-width="2" stroke-dasharray="${seriesItem.dash||''}"/>`;}}
    for(const i of [...new Set([0,5,10,15,length-1])].filter(i=>i<length&&i>=0&&good(x(i))))out+=`<text x="${x(i)}" y="${H-11}" text-anchor="middle" fill="#7b8e87" font-size="10">${fmt(times[i]/60,1)}分</text>`;
    return out+`<text x="${L}" y="12" fill="#64748b" font-size="10">${esc(leftLabel)}</text><text x="${W-R}" y="12" fill="#64748b" text-anchor="end" font-size="10">${esc(rightLabel)}</text></svg>`;
  }
  function renderDiagnostics(){
    const r=state.selected;if(!r)return;const d=r.diagnostics||{},size=r.time_seconds?.length||r.latlon?.length||0,original=chartLogic.originalSeries(r);
    $('diagnostic-target').textContent=`MMSI ${r.vessel_id} · 图中 ${size} 个采样点`;
    $('motion-chart').classList.remove('empty');$('motion-chart').innerHTML=chart([{values:original.sog_kn,color:'#0f6c5c'},{values:r.neighbor_mean_sog||[],color:'#92aaa3',dash:'4 3'},{values:original.cog_deg,color:'#0284c7',right:true,circular:true}],size,{rightMax:360,leftLabel:'本船 / 邻船航速（节）',rightLabel:'航向（度）',times:r.time_seconds||[]});
    $('gate-chart').classList.remove('empty');$('gate-chart').innerHTML=chart([{values:d.pg_probability||[],color:'#0284c7'},{values:d.raw_probability||[],color:'#ef4444',dash:'5 4'},{values:r.gate_weights||[],color:'#334155',right:true,dash:'4 2 1 2'}],size,{rightMax:1,leftLabel:'模型变化提示（不是异常概率）',rightLabel:'观测可信程度',times:r.time_seconds||[]});
    $('diagnostic-note').textContent=`图中采样点按实际时间绘制；原始航速缺失 ${original.missing_sog_count} 点、原始航向缺失 ${original.missing_cog_count} 点时断线，航向跨越0度时也断线，避免画出虚假的大转向。灰底表示观测可信程度低于0.3，可能来自定位不一致或原始字段缺失。${plainText(d.notice||(Array.isArray(d.notices)?d.notices.join(' '):'')||d.method||'模型诊断未提供；未提供的模型曲线不绘制。')} 模型提示不等于异常概率或核查结论。`;
  }
  function renderTimeline(){
    const rows=state.result?.timeline||[];if(!rows.length)return;const W=1300,H=235,L=40,R=18,T=18,B=37,n=rows.length,x=i=>L+(i+.5)*(W-L-R)/n;
    const series=[['moving_vessels','#0f6c5c'],['mean_sog_kn','#0284c7'],['divergence_mean','#10b981']];let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="当前数据集实际世界时时窗统计与待核查事件">`;
    if(state.windowIndex!==null)svg+=`<rect x="${L+state.windowIndex*(W-L-R)/n}" y="${T}" width="${Math.max(2,(W-L-R)/n)}" height="${H-B-T}" fill="#0f6c5c" opacity=".12"/>`;
    for(let j=0;j<4;j++)svg+=`<path d="M${L} ${T+j*42}H${W-R}" stroke="#e5ede9"/>`;
    const maxCount=Math.max(1,...rows.map(r=>r.candidate_count||0));
    rows.forEach((r,i)=>{const h=(r.candidate_count||0)/maxCount*48;svg+=`<rect x="${x(i)-(W-L-R)/n*.32}" y="${H-B-h}" width="${Math.max(.5,(W-L-R)/n*.64)}" height="${h}" fill="#ef4444" opacity=".65"/>`;});
    for(const [key,color] of series){const values=rows.map(r=>r[key]).filter(good);if(!values.length)continue;const low=Math.min(0,...values),high=Math.max(...values,low+.0000001);let chunks=[],s=[];rows.forEach((r,i)=>{if(good(r[key]))s.push(`${x(i)},${T+8+(1-(r[key]-low)/(high-low))*112}`);else if(s.length){chunks.push(s);s=[];}});if(s.length)chunks.push(s);for(const p of chunks)svg+=`<polyline points="${p.join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;}
    rows.forEach((r,i)=>{svg+=`<rect class="timeline-hit" data-bin="${i}" x="${L+i*(W-L-R)/n}" y="${T}" width="${Math.max(1,(W-L-R)/n)}" height="${H-B-T}" tabindex="0"><title>${esc(r.start_utc)} · 窗内在航船数 ${r.moving_vessels||0} 艘 · 报文均速（含静止） ${fmt(r.mean_sog_kn)} 节 · 船群运动聚散程度 ${good(r.divergence_mean)?Number(r.divergence_mean).toExponential(3):'不可用'} 每秒 · 候选 ${r.candidate_count||0}</title></rect>`;});
    for(const i of [...new Set([0,Math.floor(n/3),Math.floor(n*2/3),n-1])])svg+=`<text x="${x(i)}" y="${H-12}" text-anchor="${i===0?'start':i===n-1?'end':'middle'}" fill="#64748b" font-size="11">${esc(rows[i].start_utc.replace('T',' ').slice(5,16))} UTC</text>`;
    $('timeline').classList.remove('empty');$('timeline').innerHTML=svg+'</svg>';
  }
  function selectWindow(index){
    if(!state.result||!Number.isInteger(index)||index<0||index>=state.result.timeline.length)return;state.windowIndex=index;clearSelection('正在读取所选时间窗的清单…');const r=state.result.timeline[index];
    $('timeline-selection').textContent=r.start_utc+' — '+r.end_utc;$('timeline-readout').textContent=`当前时间窗：窗内在航船数 ${r.moving_vessels||0} 艘 · 窗口船数 ${r.vessels||0} · 报文均速（含静止） ${fmt(r.mean_sog_kn)} 节 · 船群运动聚散程度 ${good(r.divergence_mean)?Number(r.divergence_mean).toExponential(4)+' 每秒':'未形成有效参考'} · 优先核查 ${r.candidate_count||0} 窗 · 观测疑点 ${r.quality_count||0}；上方统计仍覆盖完整扫描。`;
    renderTimeline();loadQueue();
  }
  function returnToScan(){if(!state.result)return;state.windowIndex=null;clearSelection('正在返回完整扫描清单…');$('timeline-selection').textContent='完整扫描清单';$('timeline-readout').textContent='三条折线分别看各自变化趋势，不能按同一高度比较船数与航速；点击时间柱回看该时段。';renderTimeline();loadQueue();}
  function selectScatterRecord(id){
    if(!state.result||state.dirty)return;const r=(state.result.scatter||[]).find(v=>v.id===id);if(!r)return;
    const t=Date.parse(r.window_start_utc||r.start_utc||''),index=state.result.timeline.findIndex(v=>t>=Date.parse(v.start_utc)&&t<Date.parse(v.end_utc));state.windowIndex=index>=0?index:null;state.queue='all';$('type-filter').value='';if(isStationary(r))$('exclude-stationary').checked=false;
    clearSelection('正在定位这个散点对应的船舶…');renderTimeline();loadQueue({preferredId:id});
  }
  function renderDistribution(){
    const summary=state.result.summary||{},items=chartLogic.typeDistribution(state.result.type_counts||{},{priorityWindows:summary.candidate_windows||0,qualityWindows:(state.result.scatter||[]).length,regionalWindows:(state.result.timeline||[]).length});
    $('type-chart').classList.remove('empty');$('type-chart').innerHTML=items.map(item=>`<div class="type-row"><div class="barrow"><span>${esc(item.label)}</span><span class="barbg"><i style="width:${good(item.percent)?Math.max(0,Math.min(100,item.percent)):0}%;background:${item.key==='observation_glitch'?'#f59e0b':item.key==='congestion'?'#0284c7':'#0f6c5c'}"></i></span><b>${integer(item.count)} ${esc(item.unit)} · ${good(item.percent)?fmt(item.percent,1)+'%':'无统计'}</b></div><small>${esc(item.basis)} ${integer(item.denominator)} ${esc(item.unit)}</small></div>`).join('');renderScatter();
  }
  function renderScatter(){
    if(!state.result)return;const all=state.result.scatter,rows=all.filter(r=>good(r.field_residual_mps)&&good(r.e_kin_mps)),W=650,H=290,L=48,R=15,T=15,B=41;
    $('scatter').classList.remove('empty');if(!rows.length){$('scatter').innerHTML='<div class="empty">当前批次没有同时具备有效船群运动参考和定位一致性指标的窗口，可信邻船不足时不绘制散点。</div>';return;}
    const mx=Math.max(1,...rows.map(r=>r.field_residual_mps)),my=Math.max(1,...rows.map(r=>r.e_kin_mps)),x=v=>L+Math.log1p(v)/Math.log1p(mx)*(W-L-R),y=v=>H-B-Math.log1p(v)/Math.log1p(my)*(H-T-B);
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="真实窗口相对船群运动差异与定位与航速不一致程度散点图">`;
    for(const value of [0,.5,1,2,5,10,20,50,100,500,1000]){if(value<=mx)svg+=`<path d="M${x(value)} ${T}V${H-B}" stroke="#edf1ef"/><text x="${x(value)}" y="${H-B+15}" fill="#64748b" font-size="9" text-anchor="middle">${value}</text>`;if(value<=my)svg+=`<path d="M${L} ${y(value)}H${W-R}" stroke="#edf1ef"/><text x="${L-7}" y="${y(value)+3}" fill="#64748b" font-size="9" text-anchor="end">${value}</text>`;}
    for(const r of rows){const xx=x(r.field_residual_mps),yy=y(r.e_kin_mps),candidate=isBehaviorCandidate(r),stationary=isStationary(r),color=candidate?'#ef4444':r.has_quality_issue?'#f59e0b':stationary?'#8b9b94':'#10b981',selected=state.selected?.id===r.id;svg+=`<g class="scatter-point" data-record="${esc(r.id)}" tabindex="0"><title>MMSI ${esc(r.vessel_id)} · ${stationary?'静止背景，已排除 · ':''}相对船群运动差异窗口均值 ${fmt(r.field_residual_mps,3)} 米/秒 · 定位与航速不一致峰值 ${fmt(r.e_kin_mps,3)} 米/秒 · 业务优先级 ${priorityLabel(r)}</title>`;if(candidate)svg+=`<path d="M${xx},${yy-6}l1.8,4 4.2,.2-3.3,2.8 1.1,4.2-3.8-2.4-3.8,2.4 1.1-4.2-3.3-2.8 4.2-.2Z" fill="${color}" stroke="${selected?'#0f172a':color}"/>`;else svg+=`<circle cx="${xx}" cy="${yy}" r="${selected?5:3.5}" fill="${color}" opacity=".65" stroke="${selected?'#0f172a':'white'}"/>`;svg+='</g>';}
    svg+=`<text x="${W/2}" y="${H-6}" fill="#64748b" font-size="10" text-anchor="middle">相对船群运动差异窗口均值（米/秒，压缩刻度）</text><text x="9" y="${H/2}" fill="#64748b" font-size="10" transform="rotate(-90 9 ${H/2})" text-anchor="middle">定位与航速不一致峰值（米/秒，压缩刻度）</text></svg>`;
    $('scatter').innerHTML=svg;$('scatter-note').textContent=`${integer(rows.length)} 个可比较运动差异的窗口；${integer(all.length-rows.length)} 个窗口因可信邻船不足等原因未绘制。分布按窗口统计，质量提示与行为可重叠。星标为待人工核查候选，不是业务定性。`;
  }
  async function review(status){
    if(!state.result||!state.selected||state.dirty||state.result.needs_rescan)return;const c=context(),id=state.selected.id,serial=state.selectSerial,note=$('review-note').value,elapsed=Math.min(86400,(performance.now()-state.selectedAt)/1000);setReviewAvailability(false);
    const selected=()=>current(c)&&serial===state.selectSerial&&state.selected?.id===id&&!state.dirty;
    try{const data=await api('/reviews',{method:'POST',body:JSON.stringify({scan_id:c.scanId,record_id:id,status,elapsed_seconds:elapsed,note})});if(!selected())return;$('review-status').textContent='已归档 '+data.history.length+' 次复核';toast('当前船舶复核结果已写入独立日志。');}
    catch(error){if(selected())notice(error.message,true);}finally{if(selected())setReviewAvailability(true);}
  }
  async function generateBrief(){
    if(!state.result||!state.selected||state.dirty)return;const c=context(),id=state.selected.id,serial=state.selectSerial;$('brief-button').disabled=true;
    const selected=()=>current(c)&&serial===state.selectSerial&&state.selected?.id===id&&!state.dirty;
    try{const data=await api(`/scans/${c.scanId}/records/${encodeURIComponent(id)}/brief`,{method:'POST',body:JSON.stringify({use_llm:false})});if(selected()){$('brief').textContent=plainText(data.text);$('brief').hidden=false;}}
    catch(error){if(selected())notice(error.message,true);}finally{if(selected())$('brief-button').disabled=false;}
  }
  async function cancelScan(){
    const job=state.job;if(!job)return;state.epoch++;state.queueSerial++;clearTimeout(state.pollTimer);state.job=null;state.runPending=false;const c=context();clearSelection('已停止查看这次扫描。');$('run').disabled=false;$('cancel').hidden=true;$('progress').hidden=true;
    try{await api(`/jobs/${job.id}/cancel`,{method:'POST'});if(current(c,false))notice('已取消这次扫描，等待重新执行。');}catch(error){if(current(c,false))notice('取消请求未完成：'+error.message,true);}
  }
  $('refresh').addEventListener('click',()=>catalog(true));$('dataset').addEventListener('change',()=>changeDataset());$('run').addEventListener('click',run);$('cancel').addEventListener('click',cancelScan);
  $('full').addEventListener('change',()=>{if($('full').checked){state.manualRange=[$('start').value,$('end').value];syncTimeInputs();}else{for(const id of ['start','end'])$(id).disabled=false;const range=state.manualRange||[dateInput(state.dataset?.default_start_utc),dateInput(state.dataset?.default_end_utc)];$('start').value=range[0];$('end').value=range[1];}markCriteriaChanged();});
  for(const id of ['start','end'])$(id).addEventListener('input',markCriteriaChanged);
  $('topk').addEventListener('input',()=>{$('topk-value').textContent=$('topk').value;changeAddress({limit:$('topk').value});if(state.result)loadQueue();});
  $('exclude-stationary').addEventListener('change',()=>{if(state.queue==='all'&&state.result){clearSelection('正在更新静止背景的显示…');loadQueue();}});
  $('return-scan').addEventListener('click',returnToScan);$('fit-region').addEventListener('click',()=>{mapMode='region';datasetBounds();});
  $('type-filter').addEventListener('change',()=>{clearSelection('正在读取这个类型的船舶…');loadQueue();});
  $('fit-target').addEventListener('click',()=>{mapMode='target';safeRender('map',()=>renderMap(true),{fit:true});});$('fit-neighbors').addEventListener('click',()=>{$('show-neighbors').checked=true;mapMode='context';safeRender('map',()=>renderMap(true),{fit:true});});
  $('fit-busiest').addEventListener('click',()=>{if(!state.selected)return;state.step=neighborGeometry.busiestStep(state.selected);$('step').value=state.step;mapMode='context';$('show-neighbors').checked=true;safeRender('map',()=>renderMap(true),{fit:true});changeAddress({step:state.step});});
  document.querySelectorAll('[data-queue]').forEach(b=>b.addEventListener('click',()=>{state.queue=b.dataset.queue;clearSelection('正在读取当前队列…');loadQueue();}));
  function activateRecord(element){if(!element)return;if(element.closest('#scatter'))selectScatterRecord(element.dataset.record);else selectRecord(element.dataset.record);}
  document.addEventListener('click',event=>{activateRecord(event.target.closest('[data-record]'));const bin=event.target.closest('[data-bin]');if(bin)selectWindow(Number(bin.dataset.bin));const button=event.target.closest('[data-review]');if(button)review(button.dataset.review);const neighbor=event.target.closest('[data-neighbor-mmsi]');if(neighbor){const marker=neighborMarkers.get(neighbor.dataset.neighborMmsi);if(marker&&state.selected){map.panTo(marker.getLatLng());marker.openPopup();}}});
  document.addEventListener('keydown',event=>{if(event.key!=='Enter')return;const r=event.target.closest('[data-record]');if(r&&r.tagName.toLowerCase()!=='button')activateRecord(r);const bin=event.target.closest('[data-bin]');if(bin)selectWindow(Number(bin.dataset.bin));});
  for(const id of ['show-raw','show-repair','show-neighbor-labels'])$(id).addEventListener('change',()=>safeRender('map',()=>renderMap()));$('show-neighbors').addEventListener('change',()=>{if($('show-neighbors').checked)mapMode='context';safeRender('map',()=>renderMap(true),{fit:true});});
  $('step').addEventListener('input',()=>{if(!state.selected)return;state.step=Number($('step').value);safeRender('map',()=>renderMap(mapMode==='context'),{fit:mapMode==='context'});changeAddress({step:state.step});});$('brief-button').addEventListener('click',generateBrief);
  window.addEventListener('storage',event=>{if(event.key==='maritimeDataSourcesChanged')catalog(true);});
  setReviewAvailability(false);$('brief-button').disabled=true;initMap();catalog();
})();
