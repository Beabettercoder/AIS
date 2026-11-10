(() => {
  "use strict";
  const byId = id => document.getElementById(id);
  let job = null;
  let previewReady = false;
  let validatedActions = [];
  let previewMap = null;
  let batchItems = [];
  let activeBatchIndex = -1;
  const regions = new Map();
  const sourceSyncKey="maritimeDataSourcesChanged";
  function notifySourceChange(){try{localStorage.setItem(sourceSyncKey,String(Date.now()))}catch(_){/* Ignore unavailable storage. */}}

  function startTimer(callback){
    const t0=Date.now();
    const id=setInterval(()=>{callback(((Date.now()-t0)/1000).toFixed(1));},100);
    return {stop:()=>clearInterval(id),get:()=>((Date.now()-t0)/1000).toFixed(1)};
  }
  function setStatus(id, message, error=false) {
    const node = byId(id);
    node.textContent = message;
    node.classList.toggle("error", error);
  }
  async function jsonResponse(response) {
    const raw = await response.text();
    let data;
    try { data = raw ? JSON.parse(raw) : {}; }
    catch (_) {
      throw new Error(`服务器返回 HTTP ${response.status}${raw ? "，请查看服务日志或联系维护人员。" : "。"}`);
    }
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    return data;
  }
  async function loadSources() {
    try {
      const data = await jsonResponse(await fetch("/data-sources"));
      const target = byId("source-list");
      target.replaceChildren();
      const sourceSelector = byId("discovery-source");
      sourceSelector.replaceChildren();
      const base = document.createElement("option"); base.value = ""; base.textContent = "选择已登记来源"; sourceSelector.append(base);
      const catalog = await jsonResponse(await fetch("/catalog"));
      if (catalog.dataset_id) {
        const option=document.createElement("option"); option.value=catalog.dataset_id;
        option.textContent=`${catalog.port_id || "Houston"} · 基础数据集`; sourceSelector.append(option);
      }
      if (!data.sources.length) { target.textContent = "尚未接入新来源。"; return; }
      const stack=document.createElement("div");stack.className="source-stack";
      for (const item of data.sources) {
        const profile = item.profile || {};
        const metrics = profile.available_hourly_metrics || (item.kind === "ais_csv" ? ["ais_message_count", "unique_mmsi"] : []);
        const typeLabels={ais_csv:"AIS点位",gis_geojson:"GIS GeoJSON",noaa_enc_channels:"NOAA GIS",
                          weather_csv:"气象数据",stac_json:"遥感STAC目录",remote_geotiff:"遥感GeoTIFF",
                          event_text:"事件文本"};
        const detail = metrics.length ? `小时指标：${metrics.join(", ")}` :
          (item.kind === "gis_geojson" || item.kind === "noaa_enc_channels") ? "GIS 空间要素" :
          item.kind === "weather_csv" ? `气象${item.profile?.data_mode === "forecast" ? "预报" : "观测"}` :
          item.kind === "remote_geotiff" ? "已登记遥感影像" :
          item.kind === "event_text" ? "已抽取并核验事件证据" : "遥感场景目录";
        const article=document.createElement("article");article.className="source-item";
        const head=document.createElement("div");head.className="source-head";
        const title=document.createElement("strong");title.textContent=item.source_name || item.region_label;
        const kindBadge=document.createElement("span");kindBadge.className="source-kind";kindBadge.textContent=typeLabels[item.kind] || item.kind;
        head.append(title,kindBadge);
        const meta=document.createElement("p");meta.className="source-meta";
        meta.textContent=`${item.region_label || "未标注区域"}｜${Number(item.row_count_total).toLocaleString()} 条｜${item.first_utc ? item.first_utc.slice(0,10) : "无时间"}${item.last_utc && item.last_utc.slice(0,10)!==item.first_utc?.slice(0,10) ? ` 至 ${item.last_utc.slice(0,10)}` : ""}`;
        const detailNode=document.createElement("p");detailNode.className="source-detail";detailNode.textContent=detail;
        const id=document.createElement("small");id.className="source-id";id.title=`区域 ${item.region_id || "未绑定"}｜来源 ${item.source_id}`;
        id.textContent=`${item.region_id ? `区域 ${item.region_id} · ` : ""}来源 ${item.source_id}`;
        article.append(head,meta,detailNode,id);stack.append(article);
        const option=document.createElement("option"); option.value=item.source_id;
        option.textContent=`${typeLabels[item.kind] || item.kind} · ${item.source_name || item.region_label} · ${item.region_label}`; sourceSelector.append(option);
      }
      target.append(stack);
    } catch (error) { byId("source-list").textContent = `读取失败：${error.message}`; }
  }
  async function loadRegions(selectedId="") {
    const selector=byId("region-select");
    try {
      const data=await jsonResponse(await fetch("/regions"));
      regions.clear(); selector.replaceChildren();
      for(const item of data.regions || []){
        regions.set(item.region_id,item);
        const option=document.createElement("option"); option.value=item.region_id;
        option.textContent=`${item.canonical_name} · ${item.source_count} 个来源${item.status === "provisional" ? " · 临时边界" : ""}`;
        selector.append(option);
      }
      const create=document.createElement("option");create.value="__new__";create.textContent="＋ 新建研究区域";selector.append(create);
      selector.value=(selectedId && regions.has(selectedId)) ? selectedId : (data.regions?.[0]?.region_id || "__new__");
      updateRegionFields();
    } catch(error) {
      selector.replaceChildren(); const create=document.createElement("option");create.value="__new__";create.textContent="＋ 新建研究区域";selector.append(create);updateRegionFields();
      setStatus("upload-status",`研究区域读取失败：${error.message}`,true);
    }
  }
  function updateRegionFields(){
    const isNew=byId("region-select").value==="__new__";
    byId("region-new-wrap").classList.toggle("hidden",!isNew);
    byId("region").required=isNew;
    const node=byId("region-summary");
    if(isNew){node.textContent="新区域将在确认入库时创建。建议先上传边界 GeoJSON。";return}
    const item=regions.get(byId("region-select").value);
    if(!item){node.textContent="请选择研究区域。";return}
    const status=item.status === "active" ? "已登记GIS边界" : "临时边界，建议后续用GIS边界完善";
    const bbox=item.bbox_wgs84?.length ? item.bbox_wgs84.map(value=>Number(value).toFixed(4)).join(", ") : "暂无空间范围";
    node.textContent=`${status}｜${item.source_count} 个来源｜范围 ${bbox}`;
  }
  function renderMap(data) {
    const node=byId("map-preview");
    if(previewMap){previewMap.remove();previewMap=null}
    if(!data.map_preview?.features?.length || typeof L === "undefined"){node.classList.add("hidden");return}
    node.classList.remove("hidden");
    previewMap=L.map(node,{zoomControl:true});
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:19,attribution:"© OpenStreetMap contributors"}).addTo(previewMap);
    const layer=L.geoJSON(data.map_preview,{style:{color:"#0d6d76",weight:3,fillOpacity:.18},pointToLayer:(feature,latlng)=>L.circleMarker(latlng,{radius:6,color:"#0d6d76",fillOpacity:.7})}).addTo(previewMap);
    const bounds=layer.getBounds();if(bounds.isValid())previewMap.fitBounds(bounds.pad(.12));
    setTimeout(()=>previewMap?.invalidateSize(),0);
  }
  function currentRepairActions() {
    return [...document.querySelectorAll("#repair-plan input[data-action]:checked:not(:disabled)")]
      .map(node => node.dataset.action);
  }
  function updateBatchCommitState(){
    const pending=batchItems.filter(item=>item.data&&!item.excluded&&!item.committed);
    const ready=pending.filter(item=>item.ready);
    const blocked=pending.filter(item=>!item.ready);
    byId("commit").textContent=batchItems.length>1 ? `确认入库（${ready.length} 个文件）` : "确认入库";
    byId("commit").disabled=!ready.length||blocked.length>0;
  }
  function renderBatchList(){
    const box=byId("batch-review"),list=byId("batch-list");
    box.classList.toggle("hidden",batchItems.length<=1);
    if(batchItems.length<=1)return;
    const ready=batchItems.filter(item=>item.ready&&!item.committed).length;
    const failed=batchItems.filter(item=>item.error).length;
    byId("batch-title").textContent=`批量预览 · ${batchItems.length} 个文件 · ${ready} 个可入库${failed?` · ${failed} 个解析失败`:""}`;
    list.replaceChildren();
    batchItems.forEach((item,index)=>{
      const row=document.createElement("div");row.className="batch-item";
      const name=document.createElement("strong");name.textContent=item.name;name.title=item.name;
      const state=document.createElement("span");state.className=`batch-state ${item.ready||item.committed?"ok":item.error?"bad":""}`;
      state.textContent=item.committed?"已入库":item.excluded?"已移出":item.error?`失败：${item.error}`:item.ready?"校验通过":"需要处理";
      const actions=document.createElement("span");actions.className="row";
      if(item.data&&!item.committed&&!item.excluded){
        const view=document.createElement("button");view.type="button";view.className="secondary";view.textContent=index===activeBatchIndex?"正在查看":"查看";
        view.addEventListener("click",()=>renderReport(item.data,index));actions.append(view);
        if(!item.ready){
          const skip=document.createElement("button");skip.type="button";skip.className="secondary";skip.textContent="移出批次";
          skip.addEventListener("click",()=>{item.excluded=true;if(index===activeBatchIndex){const next=batchItems.findIndex(entry=>entry.data&&!entry.excluded&&!entry.committed);if(next>=0)renderReport(batchItems[next].data,next)}renderBatchList();updateBatchCommitState()});actions.append(skip);
        }
      }
      row.append(name,state,actions);list.append(row);
    });
  }
  function renderQuality(data) {
    const target=byId("quality-report"); target.replaceChildren();
    const repairs=byId("repair-plan"); repairs.replaceChildren();
    const quality=data.quality_report;
    if (!quality) return;
    const heading=document.createElement("h3"); heading.textContent="质量报告"; target.append(heading);
    const grid=document.createElement("div"); grid.className="quality-grid";
    for(const [label,value] of [["原始记录",quality.summary.raw_rows],["可接收",quality.summary.accepted_rows],["拒绝",quality.summary.rejected_rows],["阻断问题",quality.summary.blockers]]){
      const node=document.createElement("div");node.className="quality-stat";
      const name=document.createElement("span");name.textContent=label;
      const number=document.createElement("strong");number.textContent=String(value);
      node.append(name,number);grid.append(node);
    }
    target.append(grid);
    if(!quality.issues.length){const ok=document.createElement("p");ok.className="help";ok.textContent="没有发现质量问题。";target.append(ok)}
    for(const item of quality.issues){
      const node=document.createElement("div");node.className=`issue ${item.severity}`;
      const title=document.createElement("strong");title.textContent=`${item.severity.toUpperCase()} · ${item.message}`;
      const detail=document.createElement("div");detail.textContent=`影响 ${item.affected_count} 条（${(item.affected_ratio*100).toFixed(2)}%）${item.blocking ? " · 当前阻止入库" : ""}`;
      node.append(title,detail);
      if(item.samples?.length){const sample=document.createElement("small");sample.textContent=`示例：${JSON.stringify(item.samples)}`;node.append(sample)}
      target.append(node);
    }
    if(data.repair_plan?.length){const repairHeading=document.createElement("h3");repairHeading.textContent="修复方案";repairs.append(repairHeading)}
    for(const item of data.repair_plan || []){
      const label=document.createElement("label");label.className="repair";
      const input=document.createElement("input");input.type="checkbox";input.dataset.action=item.action_id;input.checked=Boolean(item.selected);input.disabled=item.category==="safe_auto";
      input.addEventListener("change",()=>{previewReady=false;validatedActions=[];if(activeBatchIndex>=0){batchItems[activeBatchIndex].ready=false;batchItems[activeBatchIndex].actions=[];renderBatchList()}updateBatchCommitState();setStatus("review-status","修复方案已改变，请重新校验。")});
      const body=document.createElement("span");const title=document.createElement("strong");title.textContent=`${item.label} · ${item.affected_count} 条`;
      const detail=document.createElement("small");detail.textContent=`${item.category === "safe_auto" ? "安全自动处理" : "需要确认"}：${item.description}`;
      body.append(title,detail);label.append(input,body);repairs.append(label);
    }
    if(data.repair_preview){const compare=document.createElement("p");compare.className="help";compare.textContent=`处理前 ${data.repair_preview.before.rows} 条；预计入库 ${data.repair_preview.after.rows} 条；拒绝 ${data.repair_preview.after.rejected_rows} 条；去重 ${data.repair_preview.after.duplicates_removed} 条。`;repairs.append(compare)}
  }
  function renderReport(data, batchIndex=activeBatchIndex) {
    activeBatchIndex=batchIndex;
    job = data.job_id;
    byId("review-card").classList.remove("hidden");
    byId("review-summary").replaceChildren();
    const regionStatus=data.region_validation?.validation_status;
    const overlap=data.region_validation?.overlap_ratio;
    const timeBadge = data.first_utc && data.last_utc
      ? (data.first_utc === data.last_utc ? `拍摄时间：${data.first_utc}` : `${data.first_utc} 至 ${data.last_utc}`)
      : "";
    for (const text of [data.kind, data.region_label, regionStatus ? `区域校验：${regionStatus}${overlap == null ? "" : ` ${(overlap*100).toFixed(1)}%`}` : "",
                        data.row_count == null ? "待补字段映射" : `${data.row_count} 条`,
                        timeBadge,
                        data.profile?.observed_bbox_wgs84 ? `观测范围：${data.profile.observed_bbox_wgs84.join(", ")}` : ""]) {
      if (!text) continue;
      const span=document.createElement("span"); span.className="badge"; span.textContent=text; byId("review-summary").append(span);
    }
    if (data.kind === "remote_geotiff" && data.first_utc && !byId("remote-time").value) {
      try {
        byId("remote-time").value = data.first_utc.replace("Z", "").slice(0, 19);
      } catch (_) {}
    }
    byId("quality-note").textContent=(data.quality_note || "") +
      (data.profile?.available_hourly_metrics?.length ? " 入库后生成报文数、唯一/移动/停泊 MMSI、船舶小时位置和航迹；没有观测的小时不会补零。" : "");
    byId("sample").textContent=JSON.stringify(data.sample || [],null,2);
    renderQuality(data);
    renderMap(data);
    previewReady = data.status === "pending_review";
    validatedActions = data.status === "pending_review" ? currentRepairActions() : [];
    if(activeBatchIndex>=0 && batchItems[activeBatchIndex]){
      batchItems[activeBatchIndex].data=data;
      batchItems[activeBatchIndex].ready=previewReady;
      batchItems[activeBatchIndex].actions=[...validatedActions];
    }
    renderBatchList();
    updateBatchCommitState();
    setStatus("review-status", data.status === "needs_mapping" ? "系统无法自动完成必要字段映射，当前文件暂不能入库。" : data.status === "needs_repair" ? "质量报告存在阻断问题，请选择修复方案并重新校验。" : "系统已完成字段映射和预览校验，请核对后确认入库。");
  }
  byId("intake-form").addEventListener("submit", async event => {
    event.preventDefault();
    const kind=byId("kind").value;
    const pastedText = byId("event-paste").value.trim();
    const payloads=[...byId("file").files].map(file=>({payload:file,sourceName:file.name}));
    if(kind==="event_text"&&pastedText)payloads.unshift({payload:new Blob([pastedText],{type:"text/plain;charset=utf-8"}),sourceName:"pasted-event.txt"});
    if(!payloads.length){setStatus("upload-status",kind==="event_text" ? "请粘贴文本或选择TXT/PDF文件。" : "请选择一个或多个文件。",true);return}
    const maxBytes=kind==="event_text" ? 25*1024*1024 : 200*1024*1024;
    const oversized=payloads.find(item=>item.payload.size>maxBytes);
    if(oversized){setStatus("upload-status",`${oversized.sourceName} 超过 ${kind==="event_text" ? 25 : 200} MB。`,true);return}
    const extRules = {
      ais_csv: [".csv", ".gz", ".xls", ".xlsx", ".zip"],
      gis_geojson: [".geojson", ".json"],
      weather_csv: [".csv", ".gz", ".zip"],
      stac_json: [".json", ".geojson"],
      remote_geotiff: [".tif", ".tiff"],
      event_text: [".txt", ".pdf"]
    };
    const allowedExts = extRules[kind];
    if (allowedExts) {
      const mismatched = payloads.find(p => !allowedExts.some(ext => p.sourceName.toLowerCase().endsWith(ext)));
      if (mismatched) {
        setStatus("upload-status", `文件“${mismatched.sourceName}”与当前选择的数据类型不匹配，该类型支持的格式：${allowedExts.join("、")}。`, true);
        return;
      }
    }
    const selectedRegionId=byId("region-select").value;
    const selectedRegion=regions.get(selectedRegionId);
    const regionLabel=selectedRegion ? selectedRegion.canonical_name : byId("region").value.trim();
    if(!regionLabel){setStatus("upload-status","请选择研究区域，或填写新研究区域名称。",true);return}
    batchItems=[];activeBatchIndex=-1;job=null;previewReady=false;validatedActions=[];
    byId("review-card").classList.add("hidden");
    for(let index=0;index<payloads.length;index++){
      const {payload,sourceName}=payloads[index];
      const timer=startTimer(sec=>{
        const tip=Number(sec)>=2.5?'（正在解析大文件与校验时空坐标，系统正常运行中，请稍候…）':'';
        setStatus("upload-status",`正在解析并校验 ${index+1}/${payloads.length}：${sourceName} · 已耗时 ${sec}s ${tip}`);
      });
      const params=new URLSearchParams({kind,region_label:regionLabel,
                                        region_id:selectedRegion ? selectedRegion.region_id : "",source_name:sourceName,
                                        use_as_region_boundary:String(kind==="gis_geojson" && byId("use-as-region-boundary").checked),
                                        assume_utc:String(byId("assume-utc").checked),data_mode:byId("weather-mode").value,
                                        weather_source_type:byId("weather-source-type").value,location_id:byId("weather-location-id").value,
                                        location_name:byId("weather-location-name").value});
      if(kind==="event_text"){
        if(byId("event-source-url").value)params.set("source_url",byId("event-source-url").value);
        if(byId("event-published-at").value)params.set("published_at",byId("event-published-at").value);
      }
      if(byId("weather-lon").value)params.set("location_lon",byId("weather-lon").value);
      if(byId("weather-lat").value)params.set("location_lat",byId("weather-lat").value);
      if(kind==="remote_geotiff"){
        if(byId("remote-time").value){
          const tVal = byId("remote-time").value.trim();
          params.set("observed_at_utc", tVal.endsWith("Z") ? tVal : `${tVal}Z`);
        }
        for(const [key,id] of [["scene_id","remote-scene-id"],["collection","remote-collection"],
                                ["platform","remote-platform"],["sensor","remote-sensor"],
                                ["polarization","remote-polarization"]]){
          if(byId(id).value && !(key==="scene_id"&&payloads.length>1))params.set(key,byId(id).value);
        }
      }
      try{
        const data=await jsonResponse(await fetch(`/data-intake/preview?${params}`,{method:"POST",headers:{"Content-Type":"application/octet-stream"},body:payload}));
        batchItems.push({name:sourceName,data,ready:data.status==="pending_review",actions:[],error:"",excluded:false,committed:false});
      }catch(error){batchItems.push({name:sourceName,data:null,ready:false,actions:[],error:error.message,excluded:true,committed:false})}
      finally{timer.stop();}
    }
    const first=batchItems.findIndex(item=>item.data&&!item.excluded);
    renderBatchList();
    if(first>=0){renderReport(batchItems[first].data,first);byId("review-card").scrollIntoView({behavior:"smooth",block:"start"})}
    const good=batchItems.filter(item=>item.data).length,failed=batchItems.length-good;
    setStatus("upload-status",`已完成 ${batchItems.length} 个文件的预览：${good} 个已解析${failed?`，${failed} 个失败并已跳过`:""}。尚未写入数据库。`,failed>0&&good===0);
  });
  function updateKindFields(){
    const kind=byId("kind").value;
    byId("weather-mode-wrap").classList.toggle("hidden",kind!=="weather_csv");
    byId("remote-raster-wrap").classList.toggle("hidden",kind!=="remote_geotiff");
    byId("gis-boundary-wrap").classList.toggle("hidden",kind!=="gis_geojson");
    byId("event-text-wrap").classList.toggle("hidden",kind!=="event_text");
    const acceptMap = {
      ais_csv: ".csv,.gz,.xls,.xlsx,.zip",
      gis_geojson: ".geojson,.json",
      weather_csv: ".csv,.gz,.zip",
      stac_json: ".json,.geojson",
      remote_geotiff: ".tif,.tiff",
      event_text: ".txt,.pdf"
    };
    byId("file").accept = acceptMap[kind] || ".csv,.gz,.xls,.xlsx,.zip,.json,.geojson,.tif,.tiff";
    if(kind==="event_text")byId("assume-utc").checked=false;
  }
  byId("kind").addEventListener("change",updateKindFields);
  updateKindFields();
  byId("validate").addEventListener("click", async () => {
    if (!job) return;
    setStatus("review-status","正在重新校验…");
    try {
      const data=await jsonResponse(await fetch(`/data-intake/${encodeURIComponent(job)}/validate`,
        {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({repair_actions:currentRepairActions()})}));
      renderReport(data,activeBatchIndex);
    } catch (error) { previewReady=false;if(activeBatchIndex>=0)batchItems[activeBatchIndex].ready=false;renderBatchList();updateBatchCommitState();setStatus("review-status",error.message,true); }
  });
  byId("commit").addEventListener("click", async () => {
    const pending=batchItems.filter(item=>item.data&&!item.excluded&&!item.committed);
    if(!pending.length||pending.some(item=>!item.ready))return;
    byId("commit").disabled=true;
    let succeeded=0,totalRows=0,lastRegion="";const failures=[];
    for(let index=0;index<batchItems.length;index++){
      const item=batchItems[index];if(!item.data||item.excluded||item.committed||!item.ready)continue;
      const isRemote = item.data?.kind === "remote_geotiff";
      const timer=startTimer(sec=>{
        const tip=Number(sec)>=2.0?(isRemote?'（正在执行遥感多极化检测、多普勒补偿与时空轨迹融合，系统正常运行中，请稍候…）':'（正在批量构建空间切片与融合索引，系统正常运行中，请稍候…）'):'';
        setStatus("review-status",`正在入库 ${succeeded+failures.length+1}/${pending.length}：${item.name} · 已耗时 ${sec}s ${tip}`);
      });
      try{
        const data=await jsonResponse(await fetch(`/data-intake/${encodeURIComponent(item.data.job_id)}/commit`,
          {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({repair_actions:item.actions||[],auto_process:byId("remote-auto-process").checked})}));
        if(data.kind==="ais_csv")await jsonResponse(await fetch(`/data-sources/${encodeURIComponent(data.source_id)}/verify`));
        item.committed=true;item.ready=false;item.error="";succeeded++;totalRows+=Number(data.row_count||0);lastRegion=data.region_id||lastRegion;
      }catch(error){item.error=error.message;failures.push(`${item.name}：${error.message}`)}
      finally{timer.stop();}
      renderBatchList();
    }
    updateBatchCommitState();
    if(succeeded){notifySourceChange();await Promise.all([loadSources(),loadRegions(lastRegion)])}
    setStatus("review-status",`批量入库完成：${succeeded} 个来源、${totalRows.toLocaleString()} 条记录${failures.length?`；${failures.length} 个失败（已显化错误，可点击重试）。`:"。"}`,failures.length>0);
    byId("commit").disabled = batchItems.filter(item=>item.data&&!item.excluded&&!item.committed&&item.ready).length === 0;
  });
  byId("discover").addEventListener("click", async () => {
    const sourceId=byId("discovery-source").value;
    const kinds=[...byId("discovery-kinds").selectedOptions].map(option => option.value);
    if (!sourceId || !kinds.length) { setStatus("discovery-status","请选择来源及至少一种希望补充的数据。",true); return; }
    setStatus("discovery-status","正在检查来源与覆盖范围…");
    try {
      const params=new URLSearchParams({source_id:sourceId,desired_kinds:kinds.join(","),date_utc:byId("discovery-date").value});
      const data=await jsonResponse(await fetch(`/data/discover?${params}`));
      const target=byId("discovery-results"); target.replaceChildren();
      for (const item of data.candidates) {
        const section=document.createElement("section"); section.className="card"; section.style.marginTop="12px";
        const title=document.createElement("strong"); title.textContent=`${item.kind} · ${item.status}`; section.append(title);
        const note=document.createElement("p"); note.className="help"; note.textContent=item.note || item.provider || "已具备"; section.append(note);
        if (item.url) { const link=document.createElement("a"); link.href=item.url; link.target="_blank"; link.rel="noopener noreferrer"; link.textContent="查看官方来源"; section.append(link); }
        if (item.kind === "gis" && item.status === "connector_available" && data.search_bbox_wgs84) {
          const button=document.createElement("button"); button.type="button"; button.className="secondary"; button.style.marginLeft="10px";
          button.textContent="获取当前 NOAA 航道要素";
          button.addEventListener("click", async () => {
            button.disabled=true; setStatus("discovery-status","正在获取并校验 NOAA GIS 要素…");
            try {
              const result=await jsonResponse(await fetch("/gis/noaa/channels/cache",{method:"POST",headers:{"Content-Type":"application/json"},
                body:JSON.stringify({bbox_wgs84:data.search_bbox_wgs84,region_label:data.input.region_label,
                                     region_id:data.input.region_id || "",limit:50})}));
              setStatus("discovery-status",`已缓存 ${result.feature_count} 个 GIS 要素；来源 ID ${result.source_id}。当前图层不能证明历史边界。`);
              notifySourceChange();
              await loadSources();
            } catch(error) { setStatus("discovery-status",error.message,true); button.disabled=false; }
          });
          section.append(button);
        }
        target.append(section);
      }
      setStatus("discovery-status","候选来源已列出。请核对覆盖、日期及授权条件。 ");
    } catch(error) { setStatus("discovery-status",error.message,true); }
  });
  updateKindFields();
  byId("region-select").addEventListener("change",updateRegionFields);
  window.addEventListener("storage",event=>{if(event.key===sourceSyncKey)Promise.all([loadSources(),loadRegions(byId("region-select").value)])});
  loadRegions();
  loadSources();
})();
