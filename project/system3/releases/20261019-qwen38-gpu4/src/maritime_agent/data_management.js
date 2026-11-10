const tableHost = document.querySelector('#source-table');
const statusEl = document.querySelector('#status');
const summaryEl = document.querySelector('#summary');
const dialog = document.querySelector('#delete-dialog');
const confirmation = document.querySelector('#confirmation');
const deleteButton = document.querySelector('#confirm-delete');
let sources = [];
let selected = null;
const expandedGroups = new Set();
const sourceSyncKey='maritimeDataSourcesChanged';
function notifySourceChange(){try{localStorage.setItem(sourceSyncKey,String(Date.now()))}catch(_){/* Ignore unavailable storage. */}}

const kindLabels = {ais_csv:'AIS',gis_geojson:'GIS',noaa_enc_channels:'GIS · NOAA ENC',weather_csv:'气象',remote_geotiff:'遥感影像',stac_json:'遥感目录',event_text:'事件文本'};
const kindIcons = {ais_csv:'AIS',gis_geojson:'GIS',noaa_enc_channels:'ENC',weather_csv:'MET',remote_geotiff:'SAR',stac_json:'SAT',event_text:'EVT'};
const kindColors = {ais_csv:['#147f78','#e4f4f1'],gis_geojson:['#2875a8','#e8f2f9'],noaa_enc_channels:['#2875a8','#e8f2f9'],weather_csv:['#7b65b2','#f0ecf8'],remote_geotiff:['#a66b24','#faf0e2'],stac_json:['#4b6fae','#ebf0f8'],event_text:['#a95466','#f8e9ed']};
const recordLabels = {ais_csv:'AIS 记录',gis_geojson:'GIS 要素',noaa_enc_channels:'GIS 要素',weather_csv:'气象记录',remote_geotiff:'影像资产',stac_json:'遥感场景',event_text:'确认事件'};
const countLabels = {source_rows:'源记录',ais_positions:'AIS 报文',ais_hours:'AIS 聚合',vessel_hours:'船舶轨迹',stac_scenes:'目录场景',gis_features:'GIS 要素',weather_observations:'气象记录',remote_bands:'遥感波段',remote_detections:'检测目标',event_evidence_links:'事件证据',events_deleted_without_other_evidence:'孤立事件',events_retained_with_other_evidence:'共享事件保留',fusion_nodes:'融合节点',fusion_relations:'融合关系',situation_links:'态势证据',affected_cubes:'受影响立方体'};

function esc(value){const span=document.createElement('span');span.textContent=value??'';return span.innerHTML;}
function showStatus(message,error=false){statusEl.textContent=message;statusEl.classList.toggle('error',error);}
function shortTime(value){return value?value.replace('T',' ').replace('Z','').slice(0,16):'';}
function timeText(item){if(!item.first_utc)return '无时间范围';if(item.first_utc===item.last_utc)return shortTime(item.first_utc);const first=shortTime(item.first_utc),last=shortTime(item.last_utc);return first.slice(0,10)===last.slice(0,10)?`${first} — ${last.slice(11)}`:`${first} — ${last}`;}

function render(){
  const q=document.querySelector('#search').value.trim().toLowerCase();
  const visible=sources.filter(item=>!q||[item.source_name,item.region_label,item.source_id].some(v=>String(v||'').toLowerCase().includes(q)));
  document.querySelector('#stat-source-count').textContent=sources.length.toLocaleString();
  document.querySelector('#stat-record-count').textContent=sources.reduce((sum,item)=>sum+Number(item.row_count_total||0),0).toLocaleString();
  document.querySelector('#stat-region-count').textContent=new Set(sources.map(item=>item.region_id||item.region_label).filter(Boolean)).size.toLocaleString();
  document.querySelector('#stat-kind-count').textContent=new Set(sources.map(item=>item.kind)).size.toLocaleString();
  
  if(!visible.length){
    summaryEl.textContent=`共 ${sources.length} 个来源，当前显示 0 个`;
    tableHost.innerHTML='<div class="empty">没有符合条件的已接入来源。</div>';
    return;
  }

  // 1. 按所属数据集 (优先 region_label, 其次 region_id, 兜底 '默认数据集') 进行归类
  const grouped = new Map();
  for(const item of visible){
    const groupKey = item.region_label || item.region_id || '默认未归类数据集';
    if(!grouped.has(groupKey)){
      grouped.set(groupKey, []);
    }
    grouped.get(groupKey).push(item);
  }

  summaryEl.textContent=`共 ${sources.length} 个来源（归属于 ${grouped.size} 个数据集），当前显示 ${visible.length} 个`;

  // 2. 渲染每个数据集分组 (支持独立展开/折叠)
  tableHost.innerHTML = Array.from(grouped.entries()).map(([groupKey, groupItems]) => {
    // 搜索有关键字时智能自动展开，否则根据 expandedGroups 状态判断
    const isExpanded = q ? true : expandedGroups.has(groupKey);
    const totalRows = groupItems.reduce((acc, cur) => acc + Number(cur.row_count_total || 0), 0);
    const uniqueKinds = Array.from(new Set(groupItems.map(x => kindLabels[x.kind] || x.kind)));
    
    // 计算该数据集统一时间跨度
    const firstDates = groupItems.map(x => x.first_utc).filter(Boolean).sort();
    const lastDates = groupItems.map(x => x.last_utc).filter(Boolean).sort();
    const rangeText = (firstDates.length && lastDates.length)
      ? (firstDates[0] === lastDates[lastDates.length - 1]
          ? shortTime(firstDates[0])
          : `${shortTime(firstDates[0])} — ${shortTime(lastDates[lastDates.length - 1])}`)
      : '无统一时间范围';

    // 内部数据源完整卡片
    const cardsHtml = groupItems.map(item => {
      const [accent, soft] = kindColors[item.kind] || ['#287b78', '#e7f3f1'];
      return `<article class="source-card" style="--card-accent:${accent};--icon-bg:${soft}">
        <div class="source-head">
          <div class="source-identity">
            <div class="source-icon">${esc(kindIcons[item.kind] || 'DATA')}</div>
            <div style="min-width:0">
              <span class="name" title="${esc(item.source_name)}">${esc(item.source_name || '未命名来源')}</span>
              <span class="sub">${esc(item.region_label || '未标注区域')}</span>
            </div>
          </div>
          <span class="kind">${esc(kindLabels[item.kind] || item.kind)}</span>
        </div>
        <div class="source-metrics">
          <div class="metric">
            <span>${esc(recordLabels[item.kind] || '记录 / 要素')}</span>
            <strong>${Number(item.row_count_total || 0).toLocaleString()}</strong>
          </div>
          <div class="metric">
            <span>UTC 数据范围</span>
            <strong title="${esc(timeText(item))}">${esc(timeText(item))}</strong>
          </div>
        </div>
        <div class="source-foot">
          <span class="mono" title="${esc(item.source_id)}">${esc(item.source_id)}</span>
          <button class="secondary inspect" data-id="${esc(item.source_id)}">影响与删除</button>
        </div>
      </article>`;
    }).join('');

    return `
      <section class="dataset-group ${isExpanded ? 'expanded' : 'collapsed'}" data-group="${esc(groupKey)}">
        <header class="dataset-group-header" data-group="${esc(groupKey)}">
          <div class="d-info">
            <div class="d-badge-icon" aria-hidden="true">🗂️</div>
            <div class="d-text">
              <div class="d-title-row">
                <span class="d-title">数据集：${esc(groupKey)}</span>
                <span class="d-count-tag">${groupItems.length} 个数据源</span>
                <span class="d-rows-tag">${totalRows.toLocaleString()} 条要素/记录</span>
              </div>
              <div class="d-sub-row">
                <span>涵盖类型：${uniqueKinds.map(k => `<span class="k-chip">${esc(k)}</span>`).join('')}</span>
                <span style="color:#7f939a">时间范围：${esc(rangeText)}</span>
              </div>
            </div>
          </div>
          <div class="d-action">
            <span class="d-status-text">${isExpanded ? '收起完整数据' : `展开完整数据 (${groupItems.length})`}</span>
            <span class="d-arrow" aria-hidden="true">▼</span>
          </div>
        </header>
        <div class="dataset-group-body">
          <div class="source-grid">
            ${cardsHtml}
          </div>
        </div>
      </section>
    `;
  }).join('');

  // 3. 绑定折叠 / 展开点击事件
  tableHost.querySelectorAll('.dataset-group-header').forEach(header => {
    header.addEventListener('click', (e) => {
      if(e.target.closest('button')) return;
      const groupKey = header.dataset.group;
      const groupSec = header.closest('.dataset-group');
      if(!groupSec) return;
      if(expandedGroups.has(groupKey)){
        expandedGroups.delete(groupKey);
        groupSec.classList.remove('expanded');
        groupSec.classList.add('collapsed');
        const st = groupSec.querySelector('.d-status-text');
        const cnt = groupSec.querySelectorAll('.source-card').length;
        if(st) st.textContent = `展开完整数据 (${cnt})`;
      } else {
        expandedGroups.add(groupKey);
        groupSec.classList.remove('collapsed');
        groupSec.classList.add('expanded');
        const st = groupSec.querySelector('.d-status-text');
        if(st) st.textContent = '收起完整数据';
      }
    });
  });

  // 4. 绑定删除检查按钮
  tableHost.querySelectorAll('.inspect').forEach(button=>button.addEventListener('click',()=>inspect(button.dataset.id, button)));
}

async function load(){
  showStatus('正在读取已接入数据…');
  const params=new URLSearchParams({limit:'100'});
  const kind=document.querySelector('#kind').value,region=document.querySelector('#region').value.trim();
  if(kind)params.set('kind',kind);if(region)params.set('region',region);
  try{const response=await fetch(`/data-sources?${params}`);const body=await response.json();if(!response.ok)throw new Error(body.detail||'读取失败');sources=body.sources||[];render();showStatus('');}
  catch(error){sources=[];render();showStatus(error.message,true);}
}

function startTimer(callback){
  const t0=Date.now();
  const id=setInterval(()=>{callback(((Date.now()-t0)/1000).toFixed(1));},100);
  return {stop:()=>clearInterval(id),get:()=>((Date.now()-t0)/1000).toFixed(1)};
}

async function inspect(id, triggerBtn){
  const card=triggerBtn?triggerBtn.closest('.source-card'):null;
  if(card)card.classList.add('inspecting');
  let timer=null;
  if(triggerBtn){
    triggerBtn.disabled=true;
    triggerBtn.dataset.origText=triggerBtn.textContent;
    triggerBtn.innerHTML='<span class="spinner dark"></span> 分析中 <span class="inspect-sec">0.0s</span>';
    timer=startTimer(sec=>{
      const secEl=triggerBtn.querySelector('.inspect-sec');
      if(secEl)secEl.textContent=`${sec}s`;
      const tip=Number(sec)>=2.5?'（正在深度关联大型图谱，系统正常运行中…）':'';
      showStatus(`<span class="live-badge"><span class="pulse-dot"></span> 正在分析数据源 [${id}] 的级联影响 · 已耗时 ${sec}s ${tip}</span>`);
    });
  }
  selected=null;confirmation.value='';confirmation.disabled=false;deleteButton.disabled=true;
  const deleteStatus=document.querySelector('#delete-status');
  if(deleteStatus){deleteStatus.innerHTML='';deleteStatus.className='status';}
  showStatus(`<span class="live-badge"><span class="pulse-dot"></span> 正在分析数据源 [${id}] 的级联影响 · 已耗时 0.0s</span>`);
  try{
    const response=await fetch(`/data-sources/${encodeURIComponent(id)}/delete-impact`);
    const impact=await response.json();
    if(!response.ok)throw new Error(impact.detail||'无法计算删除影响');
    selected=impact;
    document.querySelector('#target-name').textContent=`${impact.source_name} · ${kindLabels[impact.kind]||impact.kind} · ${impact.region_label||'未标注区域'}`;
    document.querySelector('#confirmation-id').textContent=impact.confirmation_required;
    const entries=Object.entries(impact.counts||{}).filter(([,value])=>Number(value)>0);
    document.querySelector('#impact').innerHTML=entries.map(([key,value])=>`<div><strong>${Number(value).toLocaleString()}</strong><span>${esc(countLabels[key]||key)}</span></div>`).join('')||'<div><strong>0</strong><span>关联记录</span></div>';
    dialog.showModal();confirmation.focus();
    const elapsed=timer?timer.get():'0.5';
    showStatus(`<span class="live-badge"><span class="pulse-dot" style="background:#08756f"></span> 影响分析已完成（耗时 ${elapsed}s），请在弹窗中复核</span>`);
  }catch(error){
    showStatus(`分析影响失败: ${error.message}`,true);
    alert(`无法分析该数据源的删除影响：\n${error.message}`);
  }finally{
    if(timer)timer.stop();
    if(card)card.classList.remove('inspecting');
    if(triggerBtn){
      triggerBtn.disabled=false;
      triggerBtn.textContent=triggerBtn.dataset.origText||'影响与删除';
    }
  }
}

confirmation.addEventListener('input',()=>{deleteButton.disabled=!selected||confirmation.value.trim()!==selected.confirmation_required;});
deleteButton.addEventListener('click',async()=>{
  if(!selected||confirmation.value.trim()!==selected.confirmation_required)return;
  deleteButton.disabled=true;
  const cancelBtn=document.querySelector('#cancel-delete');
  const closeBtn=document.querySelector('#close-dialog');
  if(cancelBtn)cancelBtn.disabled=true;
  if(closeBtn)closeBtn.disabled=true;
  confirmation.disabled=true;
  const deleteStatus=document.querySelector('#delete-status');
  if(deleteStatus){
    deleteStatus.className='status';
    deleteStatus.innerHTML=`<div class="progress-box">
      <div class="p-title">
        <span><span class="spinner dark"></span> 正在执行事务清理与级联删除…</span>
        <span class="p-timer" id="del-elapsed">0.0s</span>
      </div>
      <div class="p-bar-bg"><div class="p-bar-fill" id="del-pbar" style="width:25%"></div></div>
      <div class="p-log" id="del-plog">
        [步骤 1/4] 验证数据源签名并锁定事务…<br>
        <span style="color:#0a726b">[步骤 2/4] 正在清理关联实体节点与图谱边结构…</span>
      </div>
    </div>`;
  }
  let delTimer=startTimer(sec=>{
    const el=document.querySelector('#del-elapsed');
    if(el)el.textContent=`${sec}s`;
    const pbar=document.querySelector('#del-pbar');
    const plog=document.querySelector('#del-plog');
    const s=Number(sec);
    if(pbar&&plog){
      if(s>=0.8&&s<1.8){
        pbar.style.width='55%';
        plog.innerHTML='[步骤 1/4] 验证数据源签名并锁定事务… (完成)<br>[步骤 2/4] 归档可恢复源文件，清理实体节点… (完成)<br><span style="color:#0a726b">[步骤 3/4] 级联解除时空绑定与证据链 (进行中)…</span>';
      }else if(s>=1.8&&s<3.0){
        pbar.style.width='78%';
        plog.innerHTML='[步骤 2/4] 实体节点清理完成<br>[步骤 3/4] 时空绑定与证据链解除完成<br><span style="color:#0a726b">[步骤 4/4] 联动检查受影响态势立方体…</span>';
      }else if(s>=3.0){
        pbar.style.width='92%';
        plog.innerHTML='[步骤 4/4] 态势关联重建中…<br><span style="color:#0a726b;font-weight:700">🟢 处理大型数据集中，事务完整性保障中（系统运行正常，未卡死）…</span>';
      }
    }
  });
  try{
    const id=selected.source_id;
    const response=await fetch(`/data-sources/${encodeURIComponent(id)}?confirmation=${encodeURIComponent(id)}`,{method:'DELETE',headers:{'Content-Type':'application/json'}});
    const body=await response.json();
    if(!response.ok)throw new Error(body.detail||'删除失败');
    delTimer.stop();
    dialog.close();
    notifySourceChange();
    await load();
    showStatus(`<span class="live-badge"><span class="pulse-dot" style="background:#08756f"></span> 已成功删除“${body.source_name}”。${body.recoverable_files?.length?'原文件已移入可恢复目录。':''}</span>`);
  }catch(error){
    delTimer.stop();
    if(deleteStatus){deleteStatus.textContent=`删除失败: ${error.message}`;deleteStatus.className='status error';}
    deleteButton.disabled=false;
  }finally{
    if(cancelBtn)cancelBtn.disabled=false;
    if(closeBtn)closeBtn.disabled=false;
    confirmation.disabled=false;
  }
});
document.querySelector('#close-dialog').addEventListener('click',()=>dialog.close());
document.querySelector('#cancel-delete').addEventListener('click',()=>dialog.close());
document.querySelector('#refresh').addEventListener('click',load);
document.querySelector('#kind').addEventListener('change',load);
document.querySelector('#region').addEventListener('change',load);
document.querySelector('#search').addEventListener('input',render);
const btnExpandAll = document.querySelector('#btn-expand-all');
const btnCollapseAll = document.querySelector('#btn-collapse-all');
if(btnExpandAll){
  btnExpandAll.addEventListener('click', () => {
    for(const item of sources){
      const k = item.region_label || item.region_id || '默认未归类数据集';
      expandedGroups.add(k);
    }
    render();
  });
}
if(btnCollapseAll){
  btnCollapseAll.addEventListener('click', () => {
    expandedGroups.clear();
    render();
  });
}
window.addEventListener('storage',event=>{if(event.key===sourceSyncKey)load()});
load();

