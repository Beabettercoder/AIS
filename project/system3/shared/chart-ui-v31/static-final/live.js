(() => {
  'use strict';
  const $=id=>document.getElementById(id), API='/anomaly-workbench/api';
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const good=v=>v!==null&&v!==undefined&&Number.isFinite(Number(v));
  const fmt=(v,n=2)=>good(v)?Number(v).toFixed(n):'—';
  const integer=v=>good(v)?Number(v).toLocaleString():'—';
  const types=[['turn_back','早期折返'],['deviation','航线偏离'],['stall','异常停滞或徘徊'],['sinuous','蛇行 / 不规则转向'],['speed_drop','突然减速'],['formation_departure','编队脱离'],['group_inconsistency','群体运动不一致'],['congestion','拥堵 / 航线转移迹象'],['observation_glitch','位置跳变 / 运动学不一致']];
  const state={datasets:[],dataset:null,job:null,result:null,queue:'candidate',windowRecords:null,selected:null,selectedAt:0,selectSerial:0,pollTimer:null,step:19};
  const neighborGeometry=window.AISNeighborGeometry;
  let map,base,layers,positionLayer,toastTimer,neighborMarkers=new Map(),mapMode='context',requestedRecord=null,requestedStep=null;
  async function api(path,options={}) {
    const response=await fetch(API+path,{headers:{'Content-Type':'application/json'},...options});
    if(!response.ok){let data;try{data=await response.json();}catch(_){data={};}throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||`HTTP ${response.status}`));}
    return response.json();
  }
  function notice(text,error=false){$('message').textContent=text;$('message').hidden=!text;$('message').classList.toggle('error',error);}
  function toast(text){clearTimeout(toastTimer);$('toast').textContent=text;$('toast').hidden=false;toastTimer=setTimeout(()=>$('toast').hidden=true,4500);}
  const plainText=v=>String(v??'').replace(/SOG/g,'航速').replace(/COG/g,'航向').replace(/m\/s/g,'米/秒').replace(/\bkn\b/g,'节').replace(/\bnm\b/g,'海里').replace(/wₜ/g,'定位可信程度').replace(/个体流场残差|个体残差/g,'相对船群运动差异').replace(/运动学残差/g,'定位与航速不一致程度').replace(/(\d+(?:\.\d+)?) m\b/g,'$1 米').replace(/PG-DRGN/g,'轨迹检测模型');
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
    state.result=null;state.selected=null;state.windowRecords=null;requestedRecord=null;requestedStep=null;resetChartExplanations();
    $('candidates').innerHTML='<div class="empty">历史结果尚未区分普通运动线索与重点核查。请重新扫描，检查持续动作、历史反复模式及同船连续运动段。</div>';
    $('list-count').textContent='0';$('evidence').hidden=true;setReviewAvailability(false);
    notice('旧结果使用旧规则，已停止展示其候选排名。当前数据集、时间段和核查量已保留；点击“执行异常初筛”重新扫描。',true);
    return true;
  }
  const chartKeys=['motion','quality','timeline','types','residual'];
  const chartExplanationCache=new Map();
  let chartExplanationController=null;
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
  async function catalog(refresh=false){
    try{
      const data=await api('/catalog'+(refresh?'?refresh=true':''));state.datasets=data.datasets||[];
      const address=new URLSearchParams(location.search),resume=!refresh?address.get('scan'):null;
      const wanted=refresh?state.dataset?.id:address.get('dataset');
      const prior=state.dataset?.id;
      $('dataset').innerHTML=state.datasets.map(d=>`<option value="${esc(d.id)}">${esc(d.label)}</option>`).join('');
      if(!state.datasets.length)throw new Error('当前 SQLite 尚无可分析的 AIS 入库轨迹。请先接入数据。');
      const chosen=state.datasets.find(d=>[d.id,d.name,...(d.sources||[]).map(s=>s.id)].some(x=>String(x).toLowerCase()===String(wanted||prior).toLowerCase()))||state.datasets[0];
      $('dataset').value=chosen.id;$('dataset').disabled=false;$('run').disabled=false;$('connection').textContent='已连接 system3 SQLite';
      changeDataset();
      if(resume){
        const job=await api('/jobs/'+encodeURIComponent(resume));
        if(job.dataset.id!==chosen.id)throw new Error('该扫描属于其他数据集，未加载到当前数据源。');
        state.job=job;state.result=await api(`/jobs/${resume}/result`);
        $('far').value=String(job.params.far_budget);$('topk').value=job.params.top_k;$('topk-value').textContent=job.params.top_k;
        $('exclude-stationary').checked=job.params.exclude_stationary!==false;
        $('full').checked=Boolean(job.params.full_period);for(const id of ['start','end'])$(id).disabled=$('full').checked;
        if(job.params.start_utc)$('start').value=dateInput(job.params.start_utc);if(job.params.end_utc)$('end').value=dateInput(job.params.end_utc);
        requestedRecord=address.get('record');requestedStep=address.has('step')?Number(address.get('step')):null;
        if(requireRescan())return;
        renderResult();notice('已恢复这次真实扫描；重新执行会读取当前 SQLite 数据。');return;
      }
      if(wanted){if(!state.datasets.some(d=>[d.id,d.name,...(d.sources||[]).map(s=>s.id)].some(x=>String(x).toLowerCase()===wanted.toLowerCase())))throw new Error('指定数据集在当前 AIS 目录中不可用，未自动运行其他数据集。');await run();}
    }catch(error){$('connection').textContent='数据目录需检查';notice(error.message,true);}
  }
  function changeDataset(){
    state.dataset=state.datasets.find(d=>d.id===$('dataset').value);const d=state.dataset;if(!d)return;
    if(state.job&&['running','queued'].includes(state.job.status))api(`/jobs/${state.job.id}/cancel`,{method:'POST'}).catch(()=>{});
    clearTimeout(state.pollTimer);state.job=null;state.result=null;state.selected=null;state.windowRecords=null;state.selectSerial++;resetChartExplanations();
    const address=new URL(location.href);address.searchParams.set('dataset',d.id);for(const key of ['scan','record','step'])address.searchParams.delete(key);history.replaceState(null,'',address);
    $('start').value=dateInput(d.default_start_utc);$('end').value=dateInput(d.default_end_utc);
    for(const id of ['start','end']){$(id).min=dateInput(d.start_utc);$(id).max=dateInput(d.end_utc);}
    $('dataset-meta').textContent=`${integer(d.row_count)} 条实际 AIS · 唯一船舶数正在核对 · 世界时 UTC ${d.start_utc} — ${d.end_utc}`;
    api('/datasets/'+encodeURIComponent(d.id)).then(meta=>{if(state.dataset?.id!==d.id)return;Object.assign(state.dataset,meta);$('dataset-meta').textContent=`${integer(meta.row_count)} 条实际 AIS · ${integer(meta.unique_mmsi)} 艘唯一船舶 · 世界时 UTC ${meta.start_utc} — ${meta.end_utc} · 范围 [${meta.bbox.map(v=>Number(v).toFixed(4)).join(', ')}]`;if(!state.selected)datasetBounds();}).catch(error=>{if(state.dataset?.id===d.id)$('dataset-meta').textContent+=' · 元数据核对未完成：'+error.message;});
    $('progress').hidden=true;$('cancel').hidden=true;$('run').disabled=false;$('evidence').hidden=true;setReviewAvailability(false);
    $('candidates').innerHTML='<div class="empty">当前选择 '+esc(d.name)+'。执行初筛后查看实际船舶候选。</div>';
    if(layers)layers.clearLayers();if(positionLayer)positionLayer.clearLayers();datasetBounds();
    for(const id of ['motion-chart','gate-chart','timeline','type-chart','scatter']){$(id).innerHTML='等待当前数据集的实际计算结果。';$(id).classList.add('empty');}
    for(const id of ['kpi-ships','kpi-flow','kpi-quality','kpi-candidates'])$(id).textContent='—';
    notice('数据源已切换。扫描直接读取当前 SQLite，不使用缓存演示窗口。');
  }
  async function run(){
    if(!state.dataset)return;
    state.selectSerial++;resetChartExplanations('扫描完成后，结合当前数据生成一句话解读。');
    try{
      const body={dataset_id:state.dataset.id,start_utc:$('start').value?$('start').value+'Z':'',end_utc:$('end').value?$('end').value+'Z':'',full_period:$('full').checked,far_budget:Number($('far').value),top_k:Number($('topk').value),exclude_stationary:$('exclude-stationary').checked,window_minutes:20};
      state.job=await api('/jobs',{method:'POST',body:JSON.stringify(body)});$('run').disabled=true;$('cancel').hidden=false;$('progress').hidden=false;state.result=null;state.windowRecords=null;
      notice('正在读取当前已入库轨迹，计算定位一致性、可信邻船和轨迹变化提示。');poll();
    }catch(error){notice(error.message,true);$('run').disabled=false;}
  }
  async function poll(){
    if(!state.job)return;const id=state.job.id;
    try{
      const job=await api('/jobs/'+id);if(state.job?.id!==id)return;state.job=job;
      $('progress-bar').max=Math.max(1,job.total||1);$('progress-bar').value=job.completed||0;
      $('progress-text').textContent=`${job.message||'计算中'} · ${job.completed||0}/${job.total||'…'} 时窗${job.window_start?' · '+job.window_start:''}`;
      if(job.status==='complete'){
        state.result=await api(`/jobs/${id}/result`);state.windowRecords=null;renderResult();$('run').disabled=false;$('cancel').hidden=true;return;
      }
      if(['error','cancelled'].includes(job.status)){notice(job.message,job.status==='error');$('run').disabled=false;$('cancel').hidden=true;return;}
      state.pollTimer=setTimeout(poll,1400);
    }catch(error){notice(error.message,true);$('run').disabled=false;$('cancel').hidden=true;}
  }
  function renderResult(){
    if(requireRescan())return;
    const s=state.result.summary,rows=state.result.timeline||[],jam=rows.filter(r=>r.macro_label==='拥堵积压迹象').length;
    $('kpi-ships').textContent=integer(s.unique_mmsi);$('kpi-points').textContent=integer(s.point_count)+' 条 AIS / '+integer(s.window_count)+' 个实际时窗';
    $('kpi-flow').textContent=jam?`${jam} 个积压迹象时窗`:s.field_valid_ratio<.5?'流场覆盖不足':'未见整体积压判据';
    $('kpi-speed').textContent=`均速 ${fmt(s.mean_sog_kn)} 节 · 场有效率 ${fmt(100*s.field_valid_ratio,1)}%`;
    $('kpi-quality').textContent=integer(s.quality_count);$('kpi-candidates').textContent=integer(s.candidate_vessel_count);
    const excluded=s.stationary_excluded_count??s.excluded_stationary_count??s.stationary_count;
    $('kpi-window').textContent=`${integer(s.candidate_count)} 个优先运动段 · ${integer(s.general_clue_event_count)} 个一般线索段；原始运动变化 ${integer(s.motion_clue_windows)} 窗，历史重复 ${integer(s.repeated_pattern_windows)} 窗`;
    notice('扫描完成。优先核查需持续真实动作或独立邻船佐证；普通变向、证据不足及历史反复模式保留在一般线索。数量由证据决定，不强制填满核查清单。');
    state.queue='candidate';
    document.querySelectorAll('[data-queue]').forEach(b=>b.classList.toggle('active',b.dataset.queue===state.queue));
    renderList();renderTimeline();renderDistribution();
    const initial=state.result.records[0]||state.result.quality_records[0]||state.result.clue_records?.[0]||state.result.context_records[0];
    const address=new URL(location.href);address.searchParams.set('scan',state.result.scan_id);history.replaceState(null,'',address);
    if(requestedRecord){const wanted=requestedRecord;requestedRecord=null;selectRecord(wanted);}else if(initial)selectRecord(initial.id);
  }
  function queueRecords(){
    if(!state.result||state.result.needs_rescan)return [];
    const rows=state.windowRecords?.records;
    const result=rows?rows.filter(r=>state.queue==='candidate'?isBehaviorCandidate(r):state.queue==='clue'?r.review_level==='clue':state.queue==='quality'?r.has_quality_issue:true):state.queue==='candidate'?state.result.records.filter(isBehaviorCandidate):state.queue==='clue'?(state.result.clue_records||[]):state.queue==='quality'?state.result.quality_records:state.result.scatter;
    return state.queue==='all'&&$('exclude-stationary').checked?result.filter(r=>!isStationary(r)):result;
  }
  function renderList(){
    const filter=$('type-filter').value;let records=queueRecords().filter(r=>!filter||(filter==='observation_glitch'?r.has_quality_issue:r.type===filter));
    $('list-count').textContent=integer(records.length);
    if(filter==='congestion'){$('candidates').innerHTML='<div class="empty">拥堵与航线转移属于区域层提示，请在下方实际世界时时间轴查看流场散度、在航船数和均速，不据此给单船定性。</div>';return;}
    const cap=state.queue==='all'?200:100;const visible=records.slice(0,cap);
    $('candidates').innerHTML=visible.map(r=>{
      const stationary=isStationary(r),candidate=isBehaviorCandidate(r),kind=stationary?'静止/低速背景，已排除':state.queue==='quality'?'观测质量疑点':candidate||r.review_level==='clue'?r.type_label:'未列行为候选';
      return `<button type="button" class="candidate ${state.queue==='quality'?'quality':''} ${!candidate?'is-context':''} ${stationary?'is-stationary':''} ${state.selected?.id===r.id?'active':''}" data-record="${esc(r.id)}"><div class="candidate-head"><strong>${esc(r.vessel_id)}</strong><span class="score">${priorityLabel(r)}<small>${candidate?" 业务优先级":" 不是异常定性"}</small></span></div><time>${esc(r.event_start_utc||r.start_utc)}<br>至 ${esc(r.event_end_utc||r.end_utc)}</time><span class="tag ${stationary?'neutral':state.queue==='quality'?'quality':!candidate?'neutral':''}">${esc(kind)}</span><span class="tag ${!candidate?'neutral':''}">${stationary?'静止背景':state.queue==='quality'?'观测质量':esc(({"Tier-I":'单船运动',"Tier-II":'船群关系',"Tier-III":'船群关系'})[r.tier]||r.tier||'轨迹背景')}</span><p class="metrics">相对船群运动差异 ${good(r.field_residual_mps)?fmt(r.field_residual_mps)+' 米/秒':'可信邻船不足，未计算'} · 定位可信程度 ${fmt(r.mean_gate_weight)}</p><p>${esc(plainText(cardWhy(r)))}</p><p class="policy-source">${esc(plainText(scoreExplanation(r)))}</p>${r.event_window_ids?.length>1?`<p class="subtle">合并 ${r.event_window_ids.length} 个连续动作窗口${r.event_type_labels?.length>1?" · "+esc(r.event_type_labels.join("、")):""}</p>`:''}${r.related_event_count>1?`<p class="subtle">该船有 ${r.related_event_count} 个独立运动段，时间轴保留各段证据。</p>`:''}</button>`;
    }).join('')||'<div class="empty">当前筛选没有证据充分的优先核查对象，不用普通运动变化补足数量。“一般线索”保留可回看的动作与重复模式。</div>';
    if(records.length>cap)$('candidates').insertAdjacentHTML('beforeend',`<p class="subtle">列表先显示 ${cap} 项；下方散点图和时间轴保留全部 ${integer(records.length)} 个窗口。</p>`);
  }
  async function selectRecord(id){
    if(!state.result)return;const serial=++state.selectSerial;state.selectedAt=performance.now();resetChartExplanations('正在读取当前船舶，随后生成一句话解读。');
    $('diagnostic-target').textContent='正在读取当前船舶…';
    try{
      const r=await api(`/scans/${state.result.scan_id}/records/${encodeURIComponent(id)}`);if(serial!==state.selectSerial)return;
      if(state.result.needs_rescan||!r.maritime_policy){setReviewAvailability(false);notice('该记录缺少当前海事过滤与评分证据，已停止展示旧版推荐。请重新执行异常初筛。',true);return;}
      state.selected=r;state.step=neighborGeometry.preferredStep(r,requestedStep);requestedStep=null;
      $('step').max=Math.max(0,r.latlon.length-1);$('step').value=state.step;$('step').disabled=false;mapMode='context';
      const address=new URL(location.href);address.searchParams.set('record',r.id);address.searchParams.set('step',String(state.step));history.replaceState(null,'',address);
      $('evidence').hidden=false;$('evidence-title').textContent=`MMSI ${r.vessel_id} · ${isStationary(r)?'静止/低速背景，已排除':isBehaviorCandidate(r)?r.type_label:r.review_level==='clue'?'一般线索 · '+r.type_label:r.has_quality_issue?'观测质量核查':'背景参照'}`;
      $('why').textContent=plainText(r.why_flagged);
      const motion=maritimePolicy(r),raw=good(r.model_raw_score)?` · 原始模型输出 ${fmt(r.model_raw_score,4)}（仅作诊断）`:'';
      $('evidence-numbers').textContent=`${r.start_utc} — ${r.end_utc} · 业务优先级 ${priorityLabel(r)}${raw} · 相对船群运动差异 ${fmt(r.field_residual_mps,3)} 米/秒 · 定位与航速不一致峰值 ${fmt(r.e_kin_mps,3)} 米/秒 · 轨迹长度 ${fmt(motion.path_length_m,0)} 米 · 原始最大航速 ${fmt(motion.source_sog_max_kn)} 节。${plainText(scoreExplanation(r))}`;
      setReviewAvailability(true);
      $('review-status').textContent=r.review_history?.length?'已归档 '+r.review_history.length+' 次复核':'待人工复核';$('review-note').value='';$('brief').hidden=true;
      const prefix=`/anomaly-workbench/api/scans/${state.result.scan_id}/records/${encodeURIComponent(r.id)}/export?format=`;
      $('export-json').href=prefix+'json';$('export-md').href=prefix+'md';
      renderList();renderMap(true);renderDiagnostics();renderScatter();loadChartExplanations(serial);
    }catch(error){notice(error.message,true);$('diagnostic-target').textContent='读取未完成';}
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
    if($('show-raw').checked){L.polyline(r.latlon,{color:'#ef4444',weight:1.8,dashArray:'5 6',opacity:.7}).addTo(layers);for(const i of r.quality_steps||[])if(r.latlon[i])L.marker(r.latlon[i],{icon:L.divIcon({className:'',html:'<span style="color:#d08307;font-size:23px;font-weight:bold">×</span>',iconSize:[16,16],iconAnchor:[8,8]})}).bindTooltip(`步 ${i} · 定位可信程度=${fmt(r.gate_weights[i],3)}`).addTo(layers);}
    const points=r.repaired_latlon||r.latlon,p=r.diagnostics?.pg_probability||[];
    if($('show-repair').checked)for(let i=1;i<points.length;i++){const score=p[i],color=isStationary(r)?'#8b9b94':isBehaviorCandidate(r)&&good(score)?`rgb(${Math.round(15+220*score)},${Math.round(130-60*score)},${Math.round(115-45*score)})`:'#0f6c5c';L.polyline([points[i-1],points[i]],{color,weight:3.2}).bindTooltip(`UTC ${r.start_utc} + ${fmt(r.time_seconds[i],0)}秒 · 原始模型诊断 ${fmt(score,3)} · 业务优先级 ${priorityLabel(r)}`).addTo(layers);}
    const k=Math.min(state.step,points.length-1);L.circleMarker(points[k],{radius:7,color:'#fff',weight:2,fillColor:'#0f6c5c',fillOpacity:1}).bindTooltip('目标 '+r.vessel_id,{permanent:true,direction:'left',offset:[-9,0],className:'target-mmsi-label'}).addTo(positionLayer);
    if(fieldConfirmed){arrow(points[k],r.field_vectors?.[k],'#10b981','船群参考运动');arrow(points[k],r.residual_vectors?.[k],'#ef4444','相对船群运动差异');}
    $('step-time').textContent=new Date(Date.parse(r.start_utc)+(r.time_seconds[k]||0)*1000).toISOString().replace('.000Z','Z');
    $('map-title').textContent=`MMSI ${r.vessel_id} · 真实轨迹与船群运动参考`;
    if(fit){const bounds=neighborGeometry.boundsPoints(r,state.step,show&&mapMode==='context');if(bounds.length){map.invalidateSize({pan:false});map.fitBounds(L.latLngBounds(bounds),{padding:[48,48],maxZoom:15,animate:false});}}
  }
  function chart(series,length,{rightMax=1,leftLabel='',rightLabel='',times=[]}={}){
    const W=680,H=235,L=47,R=44,T=22,B=32,inner=W-L-R;const leftVals=series.filter(s=>!s.right).flatMap(s=>s.values.filter(good));const lm=Math.max(...leftVals,1)*1.05;
    const x=i=>L+i*inner/Math.max(1,length-1),y=(v,right)=>H-B-Number(v)/(right?rightMax:lm)*(H-T-B);
    let out=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(leftLabel+' 与 '+rightLabel)}">`;
    if(state.selected?.gate_weights)state.selected.gate_weights.forEach((v,i)=>{if(v<.3)out+=`<rect x="${x(Math.max(0,i-.5))}" y="${T}" width="${inner/Math.max(1,length-1)}" height="${H-T-B}" fill="#edf0f0"/>`;});
    for(let j=0;j<5;j++){let v=j/4;out+=`<path d="M${L} ${y(lm*v,false)}H${W-R}" stroke="#e5ede9"/><text x="${L-7}" y="${y(lm*v,false)+4}" text-anchor="end" fill="#7b8e87" font-size="10">${fmt(lm*v,lm>10?0:2)}</text><text x="${W-R+6}" y="${y(rightMax*v,true)+4}" fill="#7b8e87" font-size="10">${fmt(rightMax*v,rightMax>10?0:2)}</text>`;}
    if(leftLabel.startsWith('本船 / 邻船航速')&&state.selected?.neighbor_sog_low){const lo=state.selected.neighbor_sog_low,hi=state.selected.neighbor_sog_high;let run=[];const draw=()=>{if(run.length>1)out+=`<polygon points="${run.map(i=>`${x(i)},${y(hi[i],false)}`).join(' ')} ${run.slice().reverse().map(i=>`${x(i)},${y(lo[i],false)}`).join(' ')}" fill="#91aaa3" opacity=".18"/>`;run=[];};lo.forEach((v,i)=>{if(good(v)&&good(hi[i]))run.push(i);else draw();});draw();}
    for(const s of series){let chunks=[],segment=[];s.values.forEach((v,i)=>{if(good(v))segment.push(`${x(i)},${y(v,s.right)}`);else if(segment.length){chunks.push(segment);segment=[];}});if(segment.length)chunks.push(segment);for(const c of chunks)out+=`<polyline points="${c.join(' ')}" fill="none" stroke="${s.color}" stroke-width="2" stroke-dasharray="${s.dash||''}"/>`;}
    for(const i of [...new Set([0,5,10,15,length-1])].filter(i=>i<length&&i>=0))out+=`<text x="${x(i)}" y="${H-11}" text-anchor="middle" fill="#7b8e87" font-size="10">${good(times[i])?fmt(times[i]/60,1)+'分':i}</text>`;
    return out+`<text x="${L}" y="12" fill="#64748b" font-size="10">${esc(leftLabel)}</text><text x="${W-R}" y="12" fill="#64748b" text-anchor="end" font-size="10">${esc(rightLabel)}</text></svg>`;
  }
  function renderDiagnostics(){
    const r=state.selected;if(!r)return;const d=r.diagnostics||{},size=r.sog_kn.length;
    $('diagnostic-target').textContent=`MMSI ${r.vessel_id} · ${size} 条实际观测`;
    $('motion-chart').classList.remove('empty');$('motion-chart').innerHTML=chart([{values:r.sog_kn,color:'#0f6c5c'},{values:r.neighbor_mean_sog,color:'#92aaa3',dash:'4 3'},{values:r.cog_deg,color:'#0284c7',right:true}],size,{rightMax:360,leftLabel:'本船 / 邻船航速（节）',rightLabel:'航向（度）',times:r.time_seconds});
    $('gate-chart').classList.remove('empty');$('gate-chart').innerHTML=chart([{values:d.pg_probability||[],color:'#0284c7'},{values:d.raw_probability||[],color:'#ef4444',dash:'5 4'},{values:r.gate_weights,color:'#334155',right:true,dash:'4 2 1 2'}],size,{rightMax:1,leftLabel:'模型变化提示（不是异常概率）',rightLabel:'定位可信程度',times:r.time_seconds});
    $('diagnostic-note').textContent=plainText(d.method||'模型诊断未完成，未保存的曲线不绘制。')+' 模型输出只作诊断，行为推荐还须通过静止船过滤、真实运动变化与局部邻船有效性规则。';
  }
  function renderTimeline(){
    const rows=state.result?.timeline||[];if(!rows.length)return;const W=1300,H=235,L=40,R=18,T=18,B=37,n=rows.length,x=i=>L+(i+.5)*(W-L-R)/n;
    const series=[['moving_vessels','#0f6c5c'],['mean_sog_kn','#0284c7'],['divergence_mean','#10b981']];let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="当前数据集实际世界时时窗统计与待核查事件">`;
    for(let j=0;j<4;j++)svg+=`<path d="M${L} ${T+j*42}H${W-R}" stroke="#e5ede9"/>`;
    const maxCount=Math.max(1,...rows.map(r=>r.candidate_count||0));
    rows.forEach((r,i)=>{const h=(r.candidate_count||0)/maxCount*48;svg+=`<rect x="${x(i)-(W-L-R)/n*.32}" y="${H-B-h}" width="${Math.max(.5,(W-L-R)/n*.64)}" height="${h}" fill="#ef4444" opacity=".65"/>`;});
    for(const [key,color] of series){const values=rows.map(r=>r[key]).filter(good);if(!values.length)continue;const low=Math.min(0,...values),high=Math.max(...values,low+.0000001);let chunks=[],s=[];rows.forEach((r,i)=>{if(good(r[key]))s.push(`${x(i)},${T+8+(1-(r[key]-low)/(high-low))*112}`);else if(s.length){chunks.push(s);s=[];}});if(s.length)chunks.push(s);for(const p of chunks)svg+=`<polyline points="${p.join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;}
    rows.forEach((r,i)=>{svg+=`<rect class="timeline-hit" data-bin="${i}" x="${L+i*(W-L-R)/n}" y="${T}" width="${Math.max(1,(W-L-R)/n)}" height="${H-B-T}" tabindex="0"><title>${esc(r.start_utc)} · 在航 ${r.moving_vessels||0} 艘 · 均速 ${fmt(r.mean_sog_kn)} 节 · 船群运动聚散程度 ${good(r.divergence_mean)?Number(r.divergence_mean).toExponential(3):'不可用'} 每秒 · 候选 ${r.candidate_count||0}</title></rect>`;});
    for(const i of [...new Set([0,Math.floor(n/3),Math.floor(n*2/3),n-1])])svg+=`<text x="${x(i)}" y="${H-12}" text-anchor="${i===0?'start':i===n-1?'end':'middle'}" fill="#64748b" font-size="11">${esc(rows[i].start_utc.replace('T',' ').slice(5,16))} UTC</text>`;
    $('timeline').classList.remove('empty');$('timeline').innerHTML=svg+'</svg>';
  }
  async function selectWindow(index){
    try{const value=await api(`/scans/${state.result.scan_id}/windows/${index}`);state.windowRecords=value;renderList();const r=state.result.timeline[index];$('timeline-selection').textContent=r.start_utc+' — '+r.end_utc;$('timeline-readout').textContent=`在航 ${r.moving_vessels||0} 艘 · 窗口船数 ${r.vessels||0} · 均速 ${fmt(r.mean_sog_kn)} 节 · 船群运动聚散程度 ${good(r.divergence_mean)?Number(r.divergence_mean).toExponential(4):'邻船不足，未拟合'} 每秒 · 候选 ${r.candidate_count||0} · 观测疑点 ${r.quality_count||0}`;const first=value.records.find(v=>v.is_candidate)||value.records.find(v=>v.motion_candidate)||value.records[0];if(first)selectRecord(first.id);else toast('此实际时片没有达到最小报文条件的船舶窗口。');}catch(error){notice(error.message,true);}
  }
  function renderDistribution(){
    const counts=state.result.type_counts||{},den=Math.max(1,state.result.scatter.length),max=Math.max(1,...Object.values(counts));
    $('type-chart').classList.remove('empty');$('type-chart').innerHTML=types.map(([key,label])=>`<div class="barrow"><span>${esc(label)}</span><span class="barbg"><i style="width:${100*(counts[key]||0)/max}%;background:${key==='observation_glitch'?'#f59e0b':key==='congestion'?'#0284c7':'#0f6c5c'}"></i></span><b>${integer(counts[key]||0)} · ${fmt(100*(counts[key]||0)/den,1)}%</b></div>`).join('');renderScatter();
  }
  function renderScatter(){
    if(!state.result)return;const all=state.result.scatter,rows=all.filter(r=>good(r.field_residual_mps)&&good(r.e_kin_mps)),W=650,H=290,L=48,R=15,T=15,B=41;
    $('scatter').classList.remove('empty');if(!rows.length){$('scatter').innerHTML='<div class="empty">当前批次没有同时具备有效船群运动参考和定位一致性指标的窗口，可信邻船不足时不绘制散点。</div>';return;}
    const mx=Math.max(1,...rows.map(r=>r.field_residual_mps)),my=Math.max(1,...rows.map(r=>r.e_kin_mps)),x=v=>L+Math.log1p(v)/Math.log1p(mx)*(W-L-R),y=v=>H-B-Math.log1p(v)/Math.log1p(my)*(H-T-B);
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="真实窗口相对船群运动差异与定位与航速不一致程度散点图">`;
    for(const value of [0,.5,1,2,5,10,20,50,100,500,1000]){if(value<=mx)svg+=`<path d="M${x(value)} ${T}V${H-B}" stroke="#edf1ef"/><text x="${x(value)}" y="${H-B+15}" fill="#64748b" font-size="9" text-anchor="middle">${value}</text>`;if(value<=my)svg+=`<path d="M${L} ${y(value)}H${W-R}" stroke="#edf1ef"/><text x="${L-7}" y="${y(value)+3}" fill="#64748b" font-size="9" text-anchor="end">${value}</text>`;}
    for(const r of rows){const xx=x(r.field_residual_mps),yy=y(r.e_kin_mps),candidate=isBehaviorCandidate(r),stationary=isStationary(r),color=candidate?'#ef4444':r.has_quality_issue?'#f59e0b':stationary?'#8b9b94':'#10b981',selected=state.selected?.id===r.id;svg+=`<g class="scatter-point" data-record="${esc(r.id)}" tabindex="0"><title>MMSI ${esc(r.vessel_id)} · ${stationary?'静止背景，已排除 · ':''}相对船群运动差异 ${fmt(r.field_residual_mps,3)} 米/秒 · 定位与航速不一致程度 ${fmt(r.e_kin_mps,3)} 米/秒 · 业务优先级 ${priorityLabel(r)}</title>`;if(candidate)svg+=`<path d="M${xx},${yy-6}l1.8,4 4.2,.2-3.3,2.8 1.1,4.2-3.8-2.4-3.8,2.4 1.1-4.2-3.3-2.8 4.2-.2Z" fill="${color}" stroke="${selected?'#0f172a':color}"/>`;else svg+=`<circle cx="${xx}" cy="${yy}" r="${selected?5:3.5}" fill="${color}" opacity=".65" stroke="${selected?'#0f172a':'white'}"/>`;svg+='</g>';}
    svg+=`<text x="${W/2}" y="${H-6}" fill="#64748b" font-size="10" text-anchor="middle">相对船群运动差异（米/秒，压缩刻度）</text><text x="9" y="${H/2}" fill="#64748b" font-size="10" transform="rotate(-90 9 ${H/2})" text-anchor="middle">定位与航速不一致程度（米/秒，压缩刻度）</text></svg>`;
    $('scatter').innerHTML=svg;$('scatter-note').textContent=`${integer(rows.length)} 个可比较运动差异的窗口；${integer(all.length-rows.length)} 个窗口因可信邻船不足等原因未绘制。分布按窗口统计，质量提示与行为可重叠。星标为待人工核查候选，不是业务定性。`;
  }
  async function review(status){
    if(!state.selected)return;
    if(state.result?.needs_rescan||!state.selected.maritime_policy){notice('旧规则记录需重算后才能提交当前业务复核。',true);return;}
    try{const data=await api('/reviews',{method:'POST',body:JSON.stringify({scan_id:state.result.scan_id,record_id:state.selected.id,status,elapsed_seconds:Math.min(86400,(performance.now()-state.selectedAt)/1000),note:$('review-note').value})});$('review-status').textContent='已归档 '+data.history.length+' 次复核';toast('复核结果已写入独立日志，源数据保持只读。');}catch(error){notice(error.message,true);}
  }
  $('refresh').addEventListener('click',()=>catalog(true));$('dataset').addEventListener('change',changeDataset);$('run').addEventListener('click',run);
  $('cancel').addEventListener('click',async()=>{if(state.job)state.job=await api(`/jobs/${state.job.id}/cancel`,{method:'POST'});clearTimeout(state.pollTimer);$('run').disabled=false;$('cancel').hidden=true;notice('已取消本次扫描。');});
  $('full').addEventListener('change',()=>{for(const id of ['start','end'])$(id).disabled=$('full').checked;});
  $('topk').addEventListener('input',()=>$('topk-value').textContent=$('topk').value);
  $('exclude-stationary').addEventListener('change',()=>{renderList();notice('静止背景显示已更新；算法行为候选始终剔除静止船，取消勾选仅允许在“全部船舶”中查看背景。');});
  $('fit-region').addEventListener('click',()=>{mapMode='region';datasetBounds();});$('type-filter').addEventListener('change',renderList);
  $('fit-target').addEventListener('click',()=>{mapMode='target';renderMap(true);});
  $('fit-neighbors').addEventListener('click',()=>{$('show-neighbors').checked=true;mapMode='context';renderMap(true);});
  $('fit-busiest').addEventListener('click',()=>{if(!state.selected)return;state.step=neighborGeometry.busiestStep(state.selected);$('step').value=state.step;mapMode='context';$('show-neighbors').checked=true;renderMap(true);const address=new URL(location.href);address.searchParams.set('step',String(state.step));history.replaceState(null,'',address);});
  document.querySelectorAll('[data-queue]').forEach(b=>b.addEventListener('click',()=>{state.queue=b.dataset.queue;document.querySelectorAll('[data-queue]').forEach(v=>v.classList.toggle('active',v===b));renderList();}));
  document.addEventListener('click',event=>{const r=event.target.closest('[data-record]');if(r)selectRecord(r.dataset.record);const bin=event.target.closest('[data-bin]');if(bin)selectWindow(Number(bin.dataset.bin));const rv=event.target.closest('[data-review]');if(rv)review(rv.dataset.review);});
  document.addEventListener('click',event=>{const target=event.target.closest('[data-neighbor-mmsi]');if(!target)return;const marker=neighborMarkers.get(target.dataset.neighborMmsi);if(marker){map.panTo(marker.getLatLng());marker.openPopup();}});
  document.addEventListener('keydown',event=>{if(event.key!=='Enter')return;const r=event.target.closest('[data-record]');if(r)selectRecord(r.dataset.record);const bin=event.target.closest('[data-bin]');if(bin)selectWindow(Number(bin.dataset.bin));});
  for(const id of ['show-raw','show-repair','show-neighbor-labels'])$(id).addEventListener('change',()=>renderMap());
  $('show-neighbors').addEventListener('change',()=>{if($('show-neighbors').checked)mapMode='context';renderMap(true);});
  $('step').addEventListener('input',()=>{state.step=Number($('step').value);renderMap(mapMode==='context');const address=new URL(location.href);address.searchParams.set('step',String(state.step));history.replaceState(null,'',address);});
  $('brief-button').addEventListener('click',async()=>{if(!state.selected)return;try{$('brief-button').disabled=true;const b=await api(`/scans/${state.result.scan_id}/records/${encodeURIComponent(state.selected.id)}/brief`,{method:'POST',body:JSON.stringify({use_llm:false})});$('brief').textContent=plainText(b.text);$('brief').hidden=false;}catch(error){notice(error.message,true);}finally{$('brief-button').disabled=false;}});
  window.addEventListener('storage',event=>{if(event.key==='maritimeDataSourcesChanged')catalog(true);});
  initMap();catalog();
})();
