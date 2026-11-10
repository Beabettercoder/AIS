(() => {
  "use strict";

  if (!window.L || !window.WindParticleEngine) {
    console.error("Leaflet 或 WindParticleEngine 依赖缺失");
    return;
  }

  // 1. DOM 元素缓存
  const kpiTime = document.getElementById("kpi-time");
  const kpiVessels = document.getElementById("kpi-vessels");
  const kpiMaxWind = document.getElementById("kpi-max-wind");
  const kpiPressure = document.getElementById("kpi-pressure");
  const advisoryText = document.getElementById("advisory-text");
  const railWeatherProbe = document.getElementById("rail-weather-probe");
  const mapToolStatus = document.getElementById("map-tool-status");

  const sidebarVessels = document.getElementById("sidebar-vessels");
  const sidebarToggle = document.getElementById("sidebar-toggle");
  const sidebarClose = document.getElementById("sidebar-close");
  const sidebarVesselTotal = document.getElementById("sidebar-vessel-total");
  const vesselSearchInput = document.getElementById("vessel-search-input");
  const vesselSearchClear = document.getElementById("vessel-search-clear");
  const filterUnderwayOnly = document.getElementById("filter-underway-only");
  const vesselListContainer = document.getElementById("vessel-list");
  const filterBtns = document.querySelectorAll(".v-filter-btn");

  const countAll = document.getElementById("count-all");
  const countTanker = document.getElementById("count-tanker");
  const countCargo = document.getElementById("count-cargo");
  const countTug = document.getElementById("count-tug");
  const countOther = document.getElementById("count-other");

  const layerWindStream = document.getElementById("layer-wind-stream");
  const particleCountSlider = document.getElementById("particle-count-slider");
  const particleDensityRow = document.getElementById("particle-density-row");
  const layerTyphoonTrack = document.getElementById("layer-typhoon-track");
  const layerIsobars = document.getElementById("layer-isobars");
  const layerWaves = document.getElementById("layer-waves");
  const layerStations = document.getElementById("layer-stations");
  const layerVessels = document.getElementById("layer-vessels");
  const layerVesselVectors = document.getElementById("layer-vessel-vectors");
  const layerFacilities = document.getElementById("layer-facilities");
  const layerPortBoundary = document.getElementById("layer-port-boundary");
  const layerFairways = document.getElementById("layer-fairways");
  const layerNavLines = document.getElementById("layer-nav-lines");
  const layerNavAids = document.getElementById("layer-nav-aids");
  const layerAnchorages = document.getElementById("layer-anchorages");
  const layerAnchorMarkers = document.getElementById("layer-anchor-markers");
  const layerDocks = document.getElementById("layer-docks");
  const layerBerths = document.getElementById("layer-berths");
  const gisOpacitySlider = document.getElementById("gis-opacity-slider");
  const workspaceRegionSelect = document.getElementById("workspace-region-select");
  const layerRemoteScenes = document.getElementById("layer-remote-scenes");
  const layerRemoteDetections = document.getElementById("layer-remote-detections");
  const countRemoteDetections = document.getElementById("count-remote-detections");

  const dockCollapse = document.getElementById("dock-collapse");
  const dockBody = document.getElementById("dock-body");

  const targetLockBar = document.getElementById("target-lock-bar");
  const lockTargetName = document.getElementById("lock-target-name");
  const lockTargetMeta = document.getElementById("lock-target-meta");
  const btnUnlockTarget = document.getElementById("btn-unlock-target");

  const weatherPickerCard = document.getElementById("weather-picker-card");
  const weatherPickerClose = document.getElementById("weather-picker-close");
  const pickerCoords = document.getElementById("picker-coords");
  const pickerWindSpd = document.getElementById("picker-wind-spd");
  const pickerWindKnots = document.getElementById("picker-wind-knots");
  const pickerWindDir = document.getElementById("picker-wind-dir");
  const pickerCompass = document.getElementById("picker-compass");
  const pickerPressure = document.getElementById("picker-pressure");
  const pickerPressureTrend = document.getElementById("picker-pressure-trend");
  const pickerWave = document.getElementById("picker-wave");
  const pickerConfidence = document.getElementById("picker-confidence");
  const pickerStationDist = document.getElementById("picker-station-dist");

  const vesselCard = document.getElementById("vessel-card");
  const vesselCardClose = document.getElementById("vessel-card-close");
  const vesselCardClass = document.getElementById("vessel-card-class");
  const vesselCardName = document.getElementById("vessel-card-name");
  const vesselCardMmsi = document.getElementById("vessel-card-mmsi");
  const vesselCardSog = document.getElementById("vessel-card-sog");
  const vesselCardCog = document.getElementById("vessel-card-cog");
  const vesselCardSize = document.getElementById("vessel-card-size");
  const vesselCardDraft = document.getElementById("vessel-card-draft");
  const vesselCardStormDist = document.getElementById("vessel-card-storm-dist");
  const vesselCardRegionBox = document.getElementById("vessel-card-region-box");
  const vesselCardRegion = document.getElementById("vessel-card-region");
  const btnShowTrack = document.getElementById("btn-show-track");
  const btnCruiseTrack = document.getElementById("btn-cruise-track");

  // 海事空间要素抽屉与图例
  const gisDrawer = document.getElementById("gis-drawer");
  const gisDrawerClose = document.getElementById("gis-drawer-close");
  const gisDrawerBadge = document.getElementById("gis-drawer-badge");
  const gisDrawerTitle = document.getElementById("gis-drawer-title");
  const gisDrawerEnc = document.getElementById("gis-drawer-enc");
  const gisDrawerDepth = document.getElementById("gis-drawer-depth");
  const gisDrawerGrade = document.getElementById("gis-drawer-grade");
  const gisDrawerId = document.getElementById("gis-drawer-id");
  const gisDrawerRules = document.getElementById("gis-drawer-rules");
  const gisDrawerVesselsCount = document.getElementById("gis-drawer-vessels-count");
  const gisDrawerStoppedCount = document.getElementById("gis-drawer-stopped-count");
  const gisDrawerMovingCount = document.getElementById("gis-drawer-moving-count");
  const gisDrawerVesselsList = document.getElementById("gis-drawer-vessels-list");
  const btnGisFocus = document.getElementById("btn-gis-focus");

  const maritimeLegendCard = document.getElementById("maritime-legend-card");
  const maritimeLegendHeader = document.getElementById("maritime-legend-header");
  const maritimeLegendBody = document.getElementById("maritime-legend-body");
  const btnLegendToggle = document.getElementById("btn-legend-toggle");

  const btnStepPrev = document.getElementById("btn-step-prev");
  const btnStepNext = document.getElementById("btn-step-next");
  const btnTimelinePlay = document.getElementById("btn-timeline-play");
  const gfwSpeedBtns = document.querySelectorAll(".gfw-speed-btn");
  const gfwGranBtns = document.querySelectorAll(".gfw-gran-btn");
  const gfwCurrentTimeText = document.getElementById("gfw-current-time-text");
  const gfwStageTag = document.getElementById("gfw-stage-tag");
  const gfwDatetimePicker = document.getElementById("gfw-datetime-picker");
  const btnApplyCustomTime = document.getElementById("btn-apply-custom-time");
  const gfwPresetBtns = document.querySelectorAll(".gfw-preset-btn");
  const gfwTrackContainer = document.getElementById("gfw-track-container");
  const gfwHoverTooltip = document.getElementById("gfw-hover-tooltip");
  const tooltipTime = document.getElementById("tooltip-time");
  const tooltipWind = document.getElementById("tooltip-wind");
  const gfwHistogramTrack = document.getElementById("gfw-histogram-track");
  const gfwProgressFill = document.getElementById("gfw-progress-fill");
  const gfwScrubberNeedle = document.getElementById("gfw-scrubber-needle");
  const gfwScaleTicks = document.getElementById("gfw-scale-ticks");

  // 2. 状态管理
  let availableRegions = [];
  let currentRegionId = "";
  let currentRegionData = null;
  let currentRemoteScenes = [];
  let currentRemoteDetections = [];

  let currentGridData = null;
  let currentStations = [];
  let currentVessels = [];
  let filteredVessels = [];
  let selectedVesselClass = "all";
  let activeSelectedVessel = null;
  let activeMapTool = "browse";
  let weatherProbeMarker = null;

  // GFW 时间轴全量数据与当前视窗
  let allHistogramItems = [];       // 当前海区全周期小时数据
  let activeHistogramItems = [];    // 当前预设/视窗下显示的直方图项
  let currentPreset = "all";        // "all" | "YYYY-MM-DD"
  let currentHasTyphoon = false;    // 当前海区是否存在台风气旋
  let activeTrackVesselData = null; // 当前绘制航迹的船舶全部点位数据
  let activeTrackSplineData = null; // 稠密平滑样条曲线数据（带经纬度与切线航向角）
  let cruisePulseMarker = null;     // 航迹上的巡航实时发光脉冲光标
  let isCruisePlaying = false;      // 是否正在进行 60 FPS 平滑巡航
  let cruiseFrame = null;           // 巡航 requestAnimationFrame 句柄
  let cruiseProgress = 0.0;         // 巡航进度 [0.0, 1.0]
  let cruiseLastTimestamp = null;
  let cruiseTrailLastFrameAt = 0;
  let cruiseTrailLastIndex = -1;
  let trackTrailLine = null;        // 随巡航实时生长的已航行黄金尾迹线
  let trackRemainingLine = null;    // 未航行的引导流光虚线
  let currentGranularity = "continuous"; // "continuous" | "1h" | "3h"
  let currentItemIndex = 0;
  let currentItem = null;
  let currentExactIsoTime = "";     // 精确到分钟的连续时间戳
  let isTimelinePlaying = false;
  let playbackFrame = null;
  let playbackRate = 600; // 模拟时间倍率：1 秒推进 10 分钟
  let playbackTimeMs = null;
  let lastPlaybackFrameAt = null;
  let lastPlaybackFetchAt = 0;
  let playbackLoadInFlight = false;
  let lastPlaybackEnvironmentRenderAt = 0;
  let lastPlaybackUiSecond = "";
  let lastPlaybackHighlightIndex = -1;
  let isScrubberDragging = false;
  let fetchThrottleTimer = null;
  // 时间轴连续拖动时只保留最后一次请求。否则旧切片会在网络返回后继续
  // 重绘 500+ 个 AIS DOM 标记，既造成卡顿也可能让画面短暂回跳。
  let dataLoadController = null;
  let dataLoadSerial = 0;

  // 维护活跃船舶的 Marker 与矢量对象，用于平滑过渡动画
  const activeVesselMarkers = new Map(); // mmsi -> { marker, vectorLine, data }

  // 3. 初始化 Leaflet 地图
  const map = L.map("map", {
    preferCanvas: true,
    zoomControl: false,
    // AIS 标记数量较多；关闭缩放期间的 DOM 淡入/位移动画，优先保证拖拽和点击响应。
    zoomAnimation: false,
    markerZoomAnimation: false,
    fadeAnimation: false,
    minZoom: 7,
    maxZoom: 16,
  });
  L.control.zoom({ position: "topright" }).addTo(map);

  // 初始视野居中休斯顿航道与加尔维斯敦海湾
  map.setView([29.45, -94.95], 10);

  // 4. 多底图配置 (Base Maps)
  const basemaps = {
    dark: L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      subdomains: "abcd", maxZoom: 19, attribution: '&copy; OSM &copy; CARTO'
    }),
    nautical: L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18, attribution: '&copy; OpenStreetMap contributors'
    }),
    satellite: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 18, attribution: '&copy; Esri World Imagery'
    }),
  };
  let currentBasemap = basemaps.dark.addTo(map);

  document.querySelectorAll(".bmap-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".bmap-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const bmapType = btn.getAttribute("data-bmap");
      if (basemaps[bmapType] && currentBasemap !== basemaps[bmapType]) {
        map.removeLayer(currentBasemap);
        currentBasemap = basemaps[bmapType].addTo(map);
        currentBasemap.bringToBack();
      }
    });
  });

  // 5. 态势图层组
  const vesselLayerGroup = L.layerGroup().addTo(map);
  const vesselVectorLayerGroup = L.layerGroup().addTo(map);
  const singleTrackLayerGroup = L.layerGroup().addTo(map);
  const typhoonLayerGroup = L.layerGroup().addTo(map);
  const isobarLayerGroup = L.layerGroup().addTo(map);
  const waveLayerGroup = L.layerGroup().addTo(map);
  const stationLayerGroup = L.layerGroup().addTo(map);
  const weatherProbeLayerGroup = L.layerGroup().addTo(map);
  const facilityLayerGroup = L.layerGroup().addTo(map);

  // 卫星遥感切片外廓与 CFAR 目标检测图层组
  const remoteScenesLayerGroup = L.layerGroup().addTo(map);
  const remoteDetectionsLayerGroup = L.layerGroup().addTo(map);

  // 目标选中雷达高亮光环与 GIS 空间面域图层组
  const highlightReticleLayerGroup = L.layerGroup().addTo(map);
  const gisHighlightLayerGroup = L.layerGroup().addTo(map);
  // 专业 GIS 图层不直接上图；是否显示完全由控制坞的开关决定。
  // 这样默认态势画面始终以 AIS 船舶为主，不会被航道/锚地标注淹没。
  const portBoundaryLayerGroup = L.layerGroup();
  const fairwaysLayerGroup = L.layerGroup();
  const navLinesLayerGroup = L.layerGroup();
  const navAidsLayerGroup = L.layerGroup();
  const anchoragesLayerGroup = L.layerGroup();
  const anchorMarkersLayerGroup = L.layerGroup();
  const docksLayerGroup = L.layerGroup();
  const berthsLayerGroup = L.layerGroup();

  // GIS 全要素缓存与拓扑状态
  let allGisFeatures = [];
  let fairwayFeatures = [];
  let anchorageFeatures = [];
  let dockFeatures = [];
  let activeGisFeature = null;
  let currentGisFillOpacity = 0.18;

  // 6. 粒子流线引擎
  const canvas = document.getElementById("wind-canvas");
  const engine = new WindParticleEngine(canvas, {
    particleCount: parseInt(particleCountSlider.value, 10) || 1000,
    speedFactor: 0.35,
    clearMode: "transparent",
    lineWidth: 1.8,
  });

  function syncProjections() {
    engine.setProjections(
      (lon, lat) => {
        const pt = map.latLngToContainerPoint([lat, lon]);
        return [pt.x, pt.y];
      },
      (px, py) => {
        const ll = map.containerPointToLatLng([px, py]);
        return [ll.lng, ll.lat];
      }
    );
  }
  syncProjections();

  map.on("resize move zoom", () => {
    engine.clear();
  });

  // 7. 工具与辅助函数
  function getCompass(deg) {
    if (deg === null || deg === undefined) return "--";
    const d = ["北(N)", "北东北(NNE)", "东北(NE)", "东东北(ENE)", "东(E)", "东东南(ESE)", "东南(SE)", "南东南(SSE)",
               "南(S)", "南西南(SSW)", "西南(SW)", "西西南(WSW)", "西(W)", "西西北(WNW)", "西北(NW)", "北西北(NNW)"];
    return d[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];
  }

  function getBeaufortColor(speed) {
    if (speed < 6.0) return "#38bdf8";
    if (speed < 12.0) return "#4ade80";
    if (speed < 18.0) return "#facc15";
    if (speed < 24.0) return "#fb923c";
    return "#ef4444";
  }

  function getVesselColor(vclass) {
    const c = (vclass || "").toLowerCase();
    if (c.includes("tanker")) return "#ef4444";
    if (c.includes("cargo")) return "#22c55e";
    if (c.includes("tug")) return "#f59e0b";
    return "#38bdf8";
  }

  // 计算未来 30 分钟位置 (根据航向与节数)
  function computeCourseVectorEndPoint(lat, lon, cogDeg, sogKn) {
    if (!sogKn || sogKn < 0.5) return null;
    const distanceNm = (sogKn * 0.5); // 30 分钟航程
    const distanceKm = distanceNm * 1.852;
    const rad = (cogDeg * Math.PI) / 180;
    const dLat = (distanceKm / 111.0) * Math.cos(rad);
    const dLon = (distanceKm / (111.0 * Math.cos((lat * Math.PI) / 180))) * Math.sin(rad);
    return [lat + dLat, lon + dLon];
  }

  // 8. 船舶船模生成与动画平滑过渡 (Dynamic Moving Vessels)
  function createVesselSvgHtml(vessel) {
    const color = getVesselColor(vessel.vessel_class);
    const heading = vessel.heading_deg || vessel.cog_deg || 0;
    const isMoving = vessel.motion_state === "moving" || (vessel.sog_kn && vessel.sog_kn > 0.5);
    const size = isMoving ? 20 : 16;

    return `
      <div style="width:${size}px; height:${size}px; transform: rotate(${heading}deg); transform-origin: 50% 50%;">
        <svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24">
          <polygon points="12,2 20,20 12,16 4,20" fill="${color}" stroke="#ffffff" stroke-width="1.6" opacity="0.95" />
          ${isMoving ? '<circle cx="12" cy="16" r="2.8" fill="#ffffff" />' : ''}
        </svg>
      </div>
    `;
  }

  function renderVesselsDynamic(vessels) {
    if (!layerVessels.checked) {
      vesselLayerGroup.clearLayers();
      vesselVectorLayerGroup.clearLayers();
      activeVesselMarkers.clear();
      return;
    }

    const showVectors = layerVesselVectors.checked;
    const underwayOnly = filterUnderwayOnly.checked;
    const query = (vesselSearchInput.value || "").trim().toLowerCase();
    const newMmsiSet = new Set();

    for (const v of vessels) {
      if (v.lat === null || v.lon === null) continue;
      const isMoving = v.motion_state === "moving" || (v.sog_kn && v.sog_kn > 0.5);
      if (underwayOnly && !isMoving) continue;

      newMmsiSet.add(v.mmsi);
      const matchClass = selectedVesselClass === "all" || (v.vessel_class || "").toLowerCase().includes(selectedVesselClass);
      const matchQuery = !query || (v.name && v.name.toLowerCase().includes(query)) || (v.mmsi && v.mmsi.includes(query));
      const isMatch = matchClass && matchQuery;
      const hasSelected = !!activeSelectedVessel;
      const isSelected = hasSelected && String(activeSelectedVessel.mmsi) === String(v.mmsi);
      const targetOpacity = isSelected ? "1.0" : (isMatch ? (hasSelected ? "0.35" : "1.0") : "0.15");

      const existing = activeVesselMarkers.get(v.mmsi);

      if (existing) {
        const previous = existing.data || {};
        const curLatLng = existing.marker.getLatLng();
        const positionChanged = Math.abs(curLatLng.lat - v.lat) > 0.0000001 || Math.abs(curLatLng.lng - v.lon) > 0.0000001;
        const courseChanged = Math.abs(Number(previous.cog_deg || 0) - Number(v.cog_deg || 0)) > 0.2;
        const speedChanged = Math.abs(Number(previous.sog_kn || 0) - Number(v.sog_kn || 0)) > 0.05;
        const headingChanged = Math.abs(Number(previous.heading_deg || previous.cog_deg || 0) - Number(v.heading_deg || v.cog_deg || 0)) > 0.2;

        // 只移动真实发生变化的目标；避免每个时间片强制刷新数百个静止 DOM 标记。
        if (positionChanged) {
          existing.marker.setLatLng([v.lat, v.lon]);
        }
        // 更新船模旋转
        const iconDiv = existing.marker.getElement()?.querySelector("div");
        if (iconDiv && headingChanged) {
          const heading = v.heading_deg || v.cog_deg || 0;
          iconDiv.style.transform = `rotate(${heading}deg)`;
        }
        const el = existing.marker.getElement();
        if (el) {
          el.style.willChange = "transform, opacity";
          const frameDriven = isSelected && (isTimelinePlaying || isCruisePlaying);
          el.style.transition = frameDriven
            ? "opacity .15s ease"
            : (isMoving && isTimelinePlaying
            ? "transform 0.72s linear, opacity .15s ease"
            : (isMoving ? "transform 0.28s cubic-bezier(0.25, 0.1, 0.25, 1.0), opacity .15s ease" : "opacity .15s ease"));
          el.style.opacity = targetOpacity;
          el.style.pointerEvents = (matchClass && matchQuery) || isSelected ? "auto" : "none";
          if (isSelected) {
            el.classList.add("is-selected");
            el.setAttribute("data-label", v.name || `MMSI ${v.mmsi}`);
            existing.marker.setZIndexOffset(10000);
          } else {
            el.classList.remove("is-selected");
            el.removeAttribute("data-label");
            existing.marker.setZIndexOffset(0);
          }
        }
        existing.data = v;

        // 航向线仅在位置、航向或航速变化时重建，避免每帧删建 Leaflet 图层。
        if (existing.vectorLine && (!showVectors || !isMoving)) {
          vesselVectorLayerGroup.removeLayer(existing.vectorLine);
          existing.vectorLine = null;
        }
        if (showVectors && isMoving && (!existing.vectorLine || positionChanged || courseChanged || speedChanged)) {
          if (existing.vectorLine) vesselVectorLayerGroup.removeLayer(existing.vectorLine);
          const endPt = computeCourseVectorEndPoint(v.lat, v.lon, v.cog_deg || 0, v.sog_kn || 0);
          if (endPt) {
            existing.vectorLine = L.polyline([[v.lat, v.lon], endPt], {
              color: "#38bdf8", weight: 2, dashArray: "4, 4", opacity: 0.85
            });
            vesselVectorLayerGroup.addLayer(existing.vectorLine);
          }
        }
      } else {
        // 新建 Marker
        const icon = L.divIcon({
          html: createVesselSvgHtml(v),
          className: "leaflet-vessel-icon",
          iconSize: [20, 20],
          iconAnchor: [10, 10],
        });

        const marker = L.marker([v.lat, v.lon], {
          icon: icon,
          title: `${v.name} (MMSI ${v.mmsi})`,
        });

        marker.on("click", (e) => {
          L.DomEvent.stopPropagation(e);
          // 读取缓存中的最新切片数据，避免时间轴拖动后仍锁定旧坐标。
          const latest = activeVesselMarkers.get(v.mmsi)?.data || v;
          focusAndLockTarget(latest, { flyTo: false, scrollList: true });
        });

        vesselLayerGroup.addLayer(marker);
        const el = marker.getElement();
        if (el) {
          el.style.willChange = "transform, opacity";
          const frameDriven = isSelected && (isTimelinePlaying || isCruisePlaying);
          el.style.transition = frameDriven
            ? "opacity .15s ease"
            : (isMoving && isTimelinePlaying
            ? "transform 0.72s linear, opacity .15s ease"
            : (isMoving ? "transform 0.28s cubic-bezier(0.25, 0.1, 0.25, 1.0), opacity .15s ease" : "opacity .15s ease"));
          el.style.opacity = targetOpacity;
          el.style.pointerEvents = (matchClass && matchQuery) || isSelected ? "auto" : "none";
          if (isSelected) {
            el.classList.add("is-selected");
            el.setAttribute("data-label", v.name || `MMSI ${v.mmsi}`);
            marker.setZIndexOffset(10000);
          }
        }

        let vectorLine = null;
        if (showVectors && isMoving) {
          const endPt = computeCourseVectorEndPoint(v.lat, v.lon, v.cog_deg || 0, v.sog_kn || 0);
          if (endPt) {
            vectorLine = L.polyline([[v.lat, v.lon], endPt], {
              color: "#38bdf8", weight: 2, dashArray: "4, 4", opacity: 0.85
            });
            vesselVectorLayerGroup.addLayer(vectorLine);
          }
        }

        activeVesselMarkers.set(v.mmsi, { marker, vectorLine, data: v });
      }
    }

    // 移除本时窗不存在的 Marker
    for (const [mmsi, item] of activeVesselMarkers.entries()) {
      if (!newMmsiSet.has(mmsi)) {
        vesselLayerGroup.removeLayer(item.marker);
        if (item.vectorLine) vesselVectorLayerGroup.removeLayer(item.vectorLine);
        activeVesselMarkers.delete(mmsi);
      }
    }

    // 若当前有锁定船舶且在当前切片中，刷新其高亮状态与卡片数据
    if (activeSelectedVessel) {
      const live = activeVesselMarkers.get(activeSelectedVessel.mmsi);
      if (live) {
        activeSelectedVessel = live.data;
        highlightReticleLayerGroup.clearLayers();
        const el = live.marker.getElement();
        if (el) {
          el.classList.add("is-selected");
          el.setAttribute("data-label", live.data.name || `MMSI ${live.data.mmsi}`);
        }
        live.marker.setZIndexOffset(10000);
        showVesselCard(live.data);
        if (targetLockBar && !targetLockBar.hidden) {
          lockTargetMeta.textContent = `${(live.data.vessel_class || "VESSEL").toUpperCase()} · ${live.data.sog_kn ? live.data.sog_kn.toFixed(1) + "节" : "在泊"}`;
        }
      }
    }
  }

  // 目标选中高亮锁定与双向联动 (Target Highlight & Clean Focus)
  function focusAndLockTarget(vessel, options = {}) {
    if (!vessel || vessel.lat === null || vessel.lon === null) return;
    deactivateWeatherProbe();
    activeSelectedVessel = vessel;

    // 从地图选择的目标也必须在左侧可见：展开目标抽屉，并将该船置入列表。
    sidebarVessels.classList.remove("collapsed");
    layerDock?.classList.remove("workspace-open");
    railLayers?.classList.remove("is-active");
    if (typeof setRailActive === "function") setRailActive(railVessels);

    // 1. 清空雷达圈圈图层（极简纯净高亮，彻底去除雷达脉冲大圈）
    highlightReticleLayerGroup.clearLayers();

    // 2. 更新地图上所有船模 Marker 的状态与对比度
    const q = (vesselSearchInput.value || "").trim().toLowerCase();
    for (const [mmsi, item] of activeVesselMarkers.entries()) {
      const el = item.marker.getElement();
      if (el) {
        if (String(mmsi) === String(vessel.mmsi)) {
          el.classList.add("is-selected");
          el.setAttribute("data-label", vessel.name || `MMSI ${vessel.mmsi}`);
          el.style.opacity = "1.0";
          item.marker.setZIndexOffset(10000);
        } else {
          el.classList.remove("is-selected");
          el.removeAttribute("data-label");
          item.marker.setZIndexOffset(0);
          const v = item.data;
          const matchClass = selectedVesselClass === "all" || (v.vessel_class || "").toLowerCase().includes(selectedVesselClass);
          const matchQuery = !q || (v.name && v.name.toLowerCase().includes(q)) || (v.mmsi && v.mmsi.includes(q));
          el.style.opacity = (matchClass && matchQuery) ? "0.35" : "0.15";
        }
      }
    }

    // 3. 展现船舶详情卡片
    showVesselCard(vessel);

    // 4. 左侧列表高亮与自动平滑居中滚动 (Bi-directional List Scroll & Highlight)
    // 当前筛选结果只渲染前 100 条。先重绘，使地图点中的船即使原本不在
    // 前 100 条或被筛选条件排除，也会作为“已选目标”固定显示并可定位。
    updateSidebarVessels();
    const listItems = vesselListContainer.querySelectorAll(".vessel-item");
    let matchedItem = null;
    listItems.forEach(item => {
      const itemMmsi = item.getAttribute("data-mmsi");
      if (itemMmsi === String(vessel.mmsi)) {
        item.classList.add("active-selected");
        matchedItem = item;
      } else {
        item.classList.remove("active-selected");
      }
    });

    if (matchedItem && options.scrollList !== false) {
      matchedItem.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    // 5. 顶栏状态条展示
    if (targetLockBar) {
      lockTargetName.textContent = vessel.name || `MMSI ${vessel.mmsi}`;
      lockTargetMeta.textContent = `${(vessel.vessel_class || "VESSEL").toUpperCase()} · ${vessel.sog_kn ? vessel.sog_kn.toFixed(1) + "节" : "在泊"}`;
      targetLockBar.hidden = false;
    }

    // 6. 地图视野平滑推进 (若有要求)
    if (options.flyTo) {
      map.flyTo([vessel.lat, vessel.lon], Math.max(map.getZoom(), 12), { duration: 1.0 });
    }
  }

  function unlockTarget() {
    activeSelectedVessel = null;
    highlightReticleLayerGroup.clearLayers();
    const q = (vesselSearchInput.value || "").trim().toLowerCase();
    for (const item of activeVesselMarkers.values()) {
      const el = item.marker.getElement();
      if (el) {
        el.classList.remove("is-selected");
        el.removeAttribute("data-label");
        item.marker.setZIndexOffset(0);
        const v = item.data;
        const matchClass = selectedVesselClass === "all" || (v.vessel_class || "").toLowerCase().includes(selectedVesselClass);
        const matchQuery = !q || (v.name && v.name.toLowerCase().includes(q)) || (v.mmsi && v.mmsi.includes(q));
        el.style.opacity = (matchClass && matchQuery) ? "1.0" : "0.15";
      }
    }
    const listItems = vesselListContainer.querySelectorAll(".vessel-item");
    listItems.forEach(item => item.classList.remove("active-selected"));
    vesselCard.hidden = true;
    if (targetLockBar) targetLockBar.hidden = true;
  }

  function showVesselCard(v) {
    if (aiCopilotPanel && !aiCopilotPanel.hidden) closeCopilot();
    layerDock?.classList.remove("workspace-open");
    railLayers?.classList.remove("is-active");
    vesselCardClass.textContent = (v.vessel_class || "VESSEL").toUpperCase();
    vesselCardClass.style.color = getVesselColor(v.vessel_class);
    vesselCardClass.style.borderColor = getVesselColor(v.vessel_class);

    vesselCardName.textContent = v.name || `MMSI ${v.mmsi}`;
    vesselCardMmsi.textContent = `MMSI ${v.mmsi} · 坐标: ${v.lon.toFixed(3)}°, ${v.lat.toFixed(3)}°`;
    vesselCardSog.textContent = `${v.sog_kn !== null ? v.sog_kn.toFixed(1) : "0.0"} 节`;
    vesselCardCog.textContent = `${Math.round(v.cog_deg || 0)}° (${getCompass(v.cog_deg)})`;
    vesselCardSize.textContent = v.length_m ? `${v.length_m}m 长` : "未标注";
    vesselCardDraft.textContent = `${v.draft_m ? v.draft_m + "m 吃水 · " : ""}${v.motion_state === "moving" ? "在航作业" : "在泊/锚泊"}`;
    const stormBox = document.getElementById("vessel-card-storm-box");
    if (stormBox) {
      if (currentHasTyphoon && v.storm_distance_km) {
        vesselCardStormDist.textContent = `${v.storm_distance_km} km`;
        stormBox.hidden = false;
      } else {
        stormBox.hidden = true;
      }
    }

    if (vesselCardRegion) {
      vesselCardRegion.textContent = getVesselSpatialRegion(v);
    }

    weatherPickerCard.hidden = true;
    if (gisDrawer) gisDrawer.hidden = true;
    gisHighlightLayerGroup.clearLayers();
    vesselCard.hidden = false;

    if (btnShowTrack) btnShowTrack.textContent = "🗺️ 绘制全周期连续航迹";
    if (btnCruiseTrack) {
      btnCruiseTrack.textContent = "▶ 航迹连续平滑巡航";
      btnCruiseTrack.classList.remove("active");
    }
    stopCruiseAnimation();
  }

  vesselCardClose.addEventListener("click", unlockTarget);
  if (btnUnlockTarget) btnUnlockTarget.addEventListener("click", unlockTarget);

  // Catmull-Rom 样条平滑曲线插值算法：消除拐弯锐角折线，生成水上流体力学级连续曲线
  function generateSmoothSplineTrack(rawPoints, segmentsPerSpan = 8) {
    if (!rawPoints || rawPoints.length < 2) return { linePoints: rawPoints || [], splineItems: [] };
    const n = rawPoints.length;
    const linePoints = [];
    const splineItems = [];

    for (let i = 0; i < n - 1; i++) {
      const p0 = i === 0 ? rawPoints[0] : rawPoints[i - 1];
      const p1 = rawPoints[i];
      const p2 = rawPoints[i + 1];
      const p3 = i + 2 < n ? rawPoints[i + 2] : p2;

      for (let s = 0; s < segmentsPerSpan; s++) {
        const t = s / segmentsPerSpan;
        const t2 = t * t;
        const t3 = t2 * t;

        // Centripetal Catmull-Rom 矩阵
        const lat = 0.5 * (
          (2 * p1.lat) +
          (-p0.lat + p2.lat) * t +
          (2 * p0.lat - 5 * p1.lat + 4 * p2.lat - p3.lat) * t2 +
          (-p0.lat + 3 * p1.lat - 3 * p2.lat + p3.lat) * t3
        );
        const lon = 0.5 * (
          (2 * p1.lon) +
          (-p0.lon + p2.lon) * t +
          (2 * p0.lon - 5 * p1.lon + 4 * p2.lon - p3.lon) * t2 +
          (-p0.lon + 3 * p1.lon - 3 * p2.lon + p3.lon) * t3
        );

        // 样条切线方向矢量求导（获得瞬时航向角）
        const dlat = 0.5 * (
          (-p0.lat + p2.lat) +
          2 * (2 * p0.lat - 5 * p1.lat + 4 * p2.lat - p3.lat) * t +
          3 * (-p0.lat + 3 * p1.lat - 3 * p2.lat + p3.lat) * t2
        );
        const dlon = 0.5 * (
          (-p0.lon + p2.lon) +
          2 * (2 * p0.lon - 5 * p1.lon + 4 * p2.lon - p3.lon) * t +
          3 * (-p0.lon + 3 * p1.lon - 3 * p2.lon + p3.lon) * t2
        );
        const cosLat = Math.cos((lat * Math.PI) / 180.0);
        let heading = p1.cog_deg || 0;
        if (Math.abs(dlat) > 1e-7 || Math.abs(dlon) > 1e-7) {
          heading = (Math.atan2(dlon * cosLat, dlat) * (180.0 / Math.PI) + 360.0) % 360.0;
        }

        linePoints.push([lat, lon]);
        const p1Ms = new Date(p1.timestamp_utc || p1.timestamp_hour_utc || 0).getTime();
        const p2Ms = new Date(p2.timestamp_utc || p2.timestamp_hour_utc || 0).getTime();
        const timestampMs = Number.isFinite(p1Ms) && Number.isFinite(p2Ms)
          ? p1Ms + (p2Ms - p1Ms) * t
          : null;
        splineItems.push({ lat, lon, heading: Math.round(heading * 10) / 10, timestampMs });
      }
    }
    const last = rawPoints[n - 1];
    linePoints.push([last.lat, last.lon]);
    const lastTimestampMs = new Date(last.timestamp_utc || last.timestamp_hour_utc || 0).getTime();
    splineItems.push({
      lat: last.lat,
      lon: last.lon,
      heading: last.cog_deg || splineItems[splineItems.length - 1]?.heading || 0,
      timestampMs: Number.isFinite(lastTimestampMs) ? lastTimestampMs : null,
    });

    // 按真实地理距离累计，而不是按数组下标推进。这样相邻点疏密不均时，
    // 船标仍保持恒定视觉速度，不会一段很慢、一段突然冲刺。
    let cumulativeKm = 0;
    splineItems.forEach((item, index) => {
      if (index > 0) {
        const previous = splineItems[index - 1];
        const meanLat = ((previous.lat + item.lat) * 0.5 * Math.PI) / 180;
        const dy = (item.lat - previous.lat) * 111.32;
        const dx = (item.lon - previous.lon) * 111.32 * Math.cos(meanLat);
        cumulativeKm += Math.hypot(dx, dy);
      }
      item.distanceKm = cumulativeKm;
    });
    return { linePoints, splineItems };
  }

  let cruiseVesselMarker = null;

  function createCruisePulseIcon() {
    return L.divIcon({
      className: "vessel-cruise-pulse-marker",
      html: `<div class="vessel-cruise-pulse-inner"><span class="cruise-pulse-ring"></span><span class="cruise-pulse-dot"></span></div>`,
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
  }

  function createCruiseVesselIcon(vesselClass = "other", heading = 0, name = "") {
    return L.divIcon({
      className: "cruise-vessel-marker-root",
      html: `
        <div class="cruise-vessel-anchor">
          <div class="cruise-vessel-wrapper" style="transform: rotate(${heading}deg);">
            <div class="cruise-vessel-body ${vesselClass}"></div>
            <div class="cruise-vessel-wake"></div>
          </div>
          <div class="cruise-vessel-pill">${name || "巡航船舶"}</div>
        </div>
      `,
      iconSize: [40, 40],
      iconAnchor: [20, 20],
    });
  }

  // 60FPS 毫秒级单船航迹巡航追踪与平滑吸附 (时间轴联动)
  function updateActiveTrackCruisePosition(targetMs) {
    if (!activeTrackVesselData || !activeTrackVesselData.points || activeTrackVesselData.points.length < 2) return;
    const pts = activeTrackVesselData.points;
    const firstMs = new Date(pts[0].timestamp_utc || pts[0].timestamp_hour_utc).getTime();
    const lastMs = new Date(pts[pts.length - 1].timestamp_utc || pts[pts.length - 1].timestamp_hour_utc).getTime();
    if (!Number.isFinite(firstMs) || !Number.isFinite(lastMs) || lastMs <= firstMs) return;

    // 限制在航迹有效时间范围内
    const boundedTimeMs = Math.max(firstMs, Math.min(lastMs, targetMs));

    // 寻找目标所在的前后两个真实报文时刻；只有缺少逐报文时间时才回退到整点。
    let idx = 0;
    while (idx < pts.length - 1 && new Date(pts[idx + 1].timestamp_utc || pts[idx + 1].timestamp_hour_utc).getTime() <= boundedTimeMs) {
      idx++;
    }
    const p1 = pts[idx];
    const p2 = pts[Math.min(idx + 1, pts.length - 1)];
    const t1 = new Date(p1.timestamp_utc || p1.timestamp_hour_utc).getTime();
    const t2 = new Date(p2.timestamp_utc || p2.timestamp_hour_utc).getTime();
    const ratio = t2 > t1 ? Math.max(0, Math.min(1, (boundedTimeMs - t1) / (t2 - t1))) : 0;

    const p1Lat = Number(p1.representative_lat !== undefined ? p1.representative_lat : p1.lat);
    const p1Lon = Number(p1.representative_lon !== undefined ? p1.representative_lon : p1.lon);
    const p2Lat = Number(p2.representative_lat !== undefined ? p2.representative_lat : p2.lat);
    const p2Lon = Number(p2.representative_lon !== undefined ? p2.representative_lon : p2.lon);

    // 毫秒级线性坐标与角度插值
    const curLat = p1Lat + (p2Lat - p1Lat) * ratio;
    const curLon = p1Lon + (p2Lon - p1Lon) * ratio;

    const cog1 = Number(p1.cog_deg || 0);
    const cog2 = Number(p2.cog_deg || p1.cog_deg || 0);
    const deltaCog = ((cog2 - cog1 + 180) % 360) - 180;
    const curCog = (cog1 + deltaCog * ratio + 360) % 360;

    // 1. 移动巡航脉冲光标
    if (cruisePulseMarker) {
      cruisePulseMarker.setLatLng([curLat, curLon]);
    }

    // 2. 移动专属巡航船标
    if (cruiseVesselMarker) {
      cruiseVesselMarker.setLatLng([curLat, curLon]);
      const wrap = cruiseVesselMarker.getElement()?.querySelector(".cruise-vessel-wrapper");
      if (wrap) wrap.style.transform = `rotate(${curCog}deg)`;
    }

    // 3. 将当前选中的船舶 Marker 在 60 FPS 模式下同步贴合
    const mmsiKey = activeTrackVesselData?.mmsi;
    const targetMarkerObj = activeVesselMarkers.get(mmsiKey)
      || activeVesselMarkers.get(String(mmsiKey))
      || activeVesselMarkers.get(Number(mmsiKey));
    if (targetMarkerObj && targetMarkerObj.marker) {
      targetMarkerObj.marker.setLatLng([curLat, curLon]);
      const iconDiv = targetMarkerObj.marker.getElement()?.querySelector("div");
      if (iconDiv) {
        iconDiv.style.transform = `rotate(${curCog}deg)`;
      }
    }
  }

  // 沿样条曲线的独立 60 FPS 连续航行巡航帧驱动器
  function cruiseTick(now) {
    if (!isCruisePlaying || !activeTrackSplineData || !activeTrackSplineData.length) return;
    if (cruiseLastTimestamp === null) cruiseLastTimestamp = now;
    const dt = Math.min(100, Math.max(0, now - cruiseLastTimestamp));
    cruiseLastTimestamp = now;

    // 巡航全程平滑耗时约 12 秒，无论数据长短均以恒定流畅速度推进
    const durationMs = 12000;
    cruiseProgress += dt / durationMs;
    if (cruiseProgress > 1.0) {
      cruiseProgress = 0.0; // 自动循环连续巡航
    }

    renderSplineCruiseProgress(cruiseProgress);
    cruiseFrame = requestAnimationFrame(cruiseTick);
  }

  function renderSplineCruiseProgress(progress) {
    if (!activeTrackSplineData || !activeTrackSplineData.length) return;
    const items = activeTrackSplineData;
    const totalDistance = items[items.length - 1].distanceKm || 0;
    const targetDistance = Math.max(0, Math.min(totalDistance, progress * totalDistance));
    let low = 0;
    let high = items.length - 1;
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if ((items[mid].distanceKm || 0) < targetDistance) low = mid + 1;
      else high = mid;
    }
    const i1 = Math.max(1, low);
    const i0 = i1 - 1;
    const spanDistance = Math.max(1e-9, (items[i1].distanceKm || 0) - (items[i0].distanceKm || 0));
    const frac = Math.max(0, Math.min(1, (targetDistance - (items[i0].distanceKm || 0)) / spanDistance));

    const p0 = items[i0];
    const p1 = items[i1];

    const curLat = p0.lat + (p1.lat - p0.lat) * frac;
    const curLon = p0.lon + (p1.lon - p0.lon) * frac;
    const deltaH = ((p1.heading - p0.heading + 180) % 360) - 180;
    const curHeading = (p0.heading + deltaH * frac + 360) % 360;

    // 1. 移动巡航脉冲光标
    if (cruisePulseMarker) {
      cruisePulseMarker.setLatLng([curLat, curLon]);
    }

    // 2. 移动或自动创建专属巡航发光大船模 (确保任何新海区船舶 100% 绝对能动)
    if (!cruiseVesselMarker) {
      const vName = activeTrackVesselData?.vessel_name || `MMSI ${activeTrackVesselData?.mmsi}`;
      const vClass = activeTrackVesselData?.vessel_class || "other";
      cruiseVesselMarker = L.marker([curLat, curLon], {
        icon: createCruiseVesselIcon(vClass, curHeading, vName),
        zIndexOffset: 20000,
      }).addTo(singleTrackLayerGroup);
    } else {
      cruiseVesselMarker.setLatLng([curLat, curLon]);
      const wrap = cruiseVesselMarker.getElement()?.querySelector(".cruise-vessel-wrapper");
      if (wrap) wrap.style.transform = `rotate(${curHeading}deg)`;
    }

    // 3. 同步移动主图层中的选中船舶 Marker（若在当前时窗存在）
    const mmsiKey = activeTrackVesselData?.mmsi;
    const targetMarkerObj = activeVesselMarkers.get(mmsiKey)
      || activeVesselMarkers.get(String(mmsiKey))
      || activeVesselMarkers.get(Number(mmsiKey));
    if (targetMarkerObj && targetMarkerObj.marker) {
      targetMarkerObj.marker.setLatLng([curLat, curLon]);
      const iconDiv = targetMarkerObj.marker.getElement()?.querySelector("div");
      if (iconDiv) {
        iconDiv.style.transform = `rotate(${curHeading}deg)`;
      }
    }

    // 4. 动态更新随船生长的已航行金色尾流线 (Trail Line)
    const now = performance.now();
    // 尾迹包含上千个坐标，限制为约 20 FPS 重绘；船标本身仍由 rAF 逐帧移动。
    // 将两者解耦可避免每帧重建整条 SVG path 造成主线程卡顿。
    if (trackTrailLine && (cruiseTrailLastIndex < 0 || now - cruiseTrailLastFrameAt >= 50 || progress >= 0.999)) {
      const trail = items.slice(0, i0 + 1).map(p => [p.lat, p.lon]);
      trail.push([curLat, curLon]);
      trackTrailLine.setLatLngs(trail);
      cruiseTrailLastIndex = i0;
      cruiseTrailLastFrameAt = now;
    }
  }

  function startCruiseAnimation() {
    if (!activeTrackSplineData || !activeTrackSplineData.length) return;
    if (isTimelinePlaying) stopTimelinePlayback(false);
    isCruisePlaying = true;
    cruiseLastTimestamp = null;
    cruiseTrailLastFrameAt = 0;
    cruiseTrailLastIndex = -1;
    if (btnCruiseTrack) {
      btnCruiseTrack.textContent = "⏸ 暂停平滑巡航";
      btnCruiseTrack.classList.add("active");
    }
    cruiseFrame = requestAnimationFrame(cruiseTick);
  }

  function stopCruiseAnimation() {
    isCruisePlaying = false;
    if (cruiseFrame !== null) {
      cancelAnimationFrame(cruiseFrame);
      cruiseFrame = null;
    }
    cruiseLastTimestamp = null;
    if (btnCruiseTrack) {
      btnCruiseTrack.textContent = "▶ 航迹连续平滑巡航";
      btnCruiseTrack.classList.remove("active");
    }
  }

  // 统一绘制单船全周期连续平滑流光航迹 (去除一切单日切片，统一连续标准)
  async function drawVesselTrack(mmsi) {
    btnShowTrack.textContent = "正在生成高精平滑航迹…";
    try {
      const regParam = currentRegionId ? `?region_id=${encodeURIComponent(currentRegionId)}` : "";
      const res = await fetch(`/api/vessel-track-history/${encodeURIComponent(mmsi)}${regParam}`);
      if (!res.ok) throw new Error("航迹获取失败");
      const data = await res.json();
      activeTrackVesselData = data;

      const rawPoints = (data.points || []).map(p => ({
        lat: Number(p.representative_lat !== undefined ? p.representative_lat : p.lat),
        lon: Number(p.representative_lon !== undefined ? p.representative_lon : p.lon),
        cog_deg: Number(p.cog_deg || 0),
        sog_knots: Number(p.sog_knots || 0),
        timestamp_utc: p.timestamp_utc || p.timestamp_hour_utc,
        timestamp_hour_utc: p.timestamp_hour_utc || p.timestamp_utc,
      }));

      stopCruiseAnimation();
      singleTrackLayerGroup.clearLayers();
      cruisePulseMarker = null;
      cruiseVesselMarker = null;
      trackTrailLine = null;
      trackRemainingLine = null;
      activeTrackSplineData = null;

      if (rawPoints.length < 2) {
        alert("该船在连续观测时段内数据点较少，暂无长程连续航迹。");
        btnShowTrack.textContent = "🗺️ 绘制全周期连续航迹";
        return;
      }

      // 1. 生成 Catmull-Rom 样条高精平滑连续曲线 (自适应高密度采样)
      const subSegments = rawPoints.length > 200 ? 2 : 6;
      const { linePoints, splineItems } = generateSmoothSplineTrack(rawPoints, subSegments);
      activeTrackSplineData = splineItems;

      // 2. 底层漫反射琥珀光晕线 (Amber Glow Base)
      L.polyline(linePoints, {
        color: "#eab308",
        weight: 7,
        opacity: 0.25,
        lineCap: "round",
        lineJoin: "round",
        className: "track-glow-base",
      }).addTo(singleTrackLayerGroup);

      // 3. 引导流光虚线 (Remaining Flowing AntPath Line)
      trackRemainingLine = L.polyline(linePoints, {
        color: "#38bdf8",
        weight: 2,
        opacity: 0.75,
        lineCap: "round",
        className: "track-flowing-line",
      }).addTo(singleTrackLayerGroup);

      // 4. 动态生长的已航行黄金主航线 (Active Dynamic Trail)
      trackTrailLine = L.polyline([linePoints[0]], {
        color: "#facc15",
        weight: 3.5,
        opacity: 0.95,
        lineCap: "round",
        lineJoin: "round",
      }).addTo(singleTrackLayerGroup);

      // 5. 沿线均匀打标关键航向微粒 (Smooth Waypoint Badges，均匀分布 20~25 处防扎堆)
      const badgeInterval = Math.max(1, Math.floor(rawPoints.length / 24));
      rawPoints.forEach((pt, idx) => {
        if (idx !== 0 && idx !== rawPoints.length - 1 && idx % badgeInterval === 0) {
          const timeLabel = (pt.timestamp_utc || pt.timestamp_hour_utc || "").replace("T", " ").replace("Z", "").slice(5, 16);
          const sogLabel = pt.sog_knots ? `${pt.sog_knots.toFixed(1)} kn` : "停泊/微速";
          const badge = L.circleMarker([pt.lat, pt.lon], {
            radius: 2.5,
            fillColor: "#facc15",
            color: "#ffffff",
            weight: 1.2,
            fillOpacity: 0.85,
          }).addTo(singleTrackLayerGroup);
          badge.bindTooltip(`📍 ${timeLabel} UTC · 航速 ${sogLabel}`);
        }
      });

      // 6. 起始与终止点醒目标记
      const startLabel = (data.start_utc || "").replace("T", " ").replace("Z", "").slice(5, 16);
      const endLabel = (data.end_utc || "").replace("T", " ").replace("Z", "").slice(5, 16);
      L.circleMarker([rawPoints[0].lat, rawPoints[0].lon], { radius: 6, fillColor: "#22c55e", color: "#fff", weight: 2.2, fillOpacity: 1 })
        .bindTooltip(`🟢 航迹起点 (${startLabel} UTC)`).addTo(singleTrackLayerGroup);
      L.circleMarker([rawPoints[rawPoints.length - 1].lat, rawPoints[rawPoints.length - 1].lon], { radius: 6, fillColor: "#ef4444", color: "#fff", weight: 2.2, fillOpacity: 1 })
        .bindTooltip(`🔴 航迹终点 (${endLabel} UTC)`).addTo(singleTrackLayerGroup);

      // 7. 航迹实时巡航发光脉冲光标 (Cruise Pulse Pin)
      cruisePulseMarker = L.marker([rawPoints[0].lat, rawPoints[0].lon], {
        icon: createCruisePulseIcon(),
      }).addTo(singleTrackLayerGroup);

      cruiseProgress = 0.0;
      renderSplineCruiseProgress(0.0);

      map.fitBounds(L.polyline(linePoints).getBounds(), { padding: [80, 80] });
      btnShowTrack.textContent = `✓ 已绘制全周期连续航迹 (${rawPoints.length}点·高密连续)`;
    } catch (e) {
      console.warn("轨迹绘制异常:", e);
      btnShowTrack.textContent = "航迹暂不可用";
    }
  }

  btnShowTrack.addEventListener("click", () => {
    if (!activeSelectedVessel) return;
    drawVesselTrack(activeSelectedVessel.mmsi);
  });

  if (btnCruiseTrack) {
    btnCruiseTrack.addEventListener("click", async () => {
      if (!activeSelectedVessel) return;
      if (isCruisePlaying) {
        stopCruiseAnimation();
        return;
      }
      btnCruiseTrack.textContent = "⏳ 准备平滑巡航…";
      if (!activeTrackSplineData || !activeTrackSplineData.length) {
        await drawVesselTrack(activeSelectedVessel.mmsi);
      }
      if (activeTrackSplineData && activeTrackSplineData.length > 0) {
        startCruiseAnimation();
      } else {
        btnCruiseTrack.textContent = "▶ 航迹连续平滑巡航";
      }
    });
  }

  map.on("click", () => {
    if (activeMapTool !== "browse") return;
    stopCruiseAnimation();
    singleTrackLayerGroup.clearLayers();
    cruisePulseMarker = null;
    cruiseVesselMarker = null;
    activeTrackVesselData = null;
    activeTrackSplineData = null;
    trackTrailLine = null;
    trackRemainingLine = null;
    if (activeSelectedVessel) {
      btnShowTrack.textContent = "🗺️ 绘制全周期连续航迹";
      if (btnCruiseTrack) {
        btnCruiseTrack.textContent = "▶ 航迹连续平滑巡航";
        btnCruiseTrack.classList.remove("active");
      }
    }
  });

  // 9. 左侧船舶检索与过滤列表
  function updateSidebarVessels() {
    const q = (vesselSearchInput.value || "").trim().toLowerCase();
    const underwayOnly = filterUnderwayOnly.checked;

    filteredVessels = currentVessels.filter(v => {
      const matchClass = selectedVesselClass === "all" || (v.vessel_class || "").toLowerCase().includes(selectedVesselClass);
      const matchQuery = !q || (v.name && v.name.toLowerCase().includes(q)) || (v.mmsi && v.mmsi.includes(q));
      const isMoving = v.motion_state === "moving" || (v.sog_kn && v.sog_kn > 0.5);
      const matchUnderway = !underwayOnly || isMoving;
      return matchClass && matchQuery && matchUnderway;
    });

    sidebarVesselTotal.textContent = `${filteredVessels.length} 艘`;
    vesselListContainer.innerHTML = "";

    if (!filteredVessels.length) {
      vesselListContainer.innerHTML = '<div class="vessel-list-loading">未找到匹配的船舶</div>';
      return;
    }

    const displayList = filteredVessels.slice(0, 100);
    const selectedMmsi = activeSelectedVessel ? String(activeSelectedVessel.mmsi) : null;
    const selectionIsVisible = selectedMmsi && displayList.some(v => String(v.mmsi) === selectedMmsi);

    // 保证地图上刚选中的船不会因列表上限或当前筛选而“消失”。
    if (selectedMmsi && !selectionIsVisible) {
      const selectedVessel = currentVessels.find(v => String(v.mmsi) === selectedMmsi) || activeSelectedVessel;
      if (selectedVessel) displayList.unshift(selectedVessel);
    }
    const uniqueDisplayList = displayList.filter((v, index, all) =>
      all.findIndex(candidate => String(candidate.mmsi) === String(v.mmsi)) === index
    ).slice(0, 100);
    for (const v of uniqueDisplayList) {
      const isSelected = activeSelectedVessel && String(activeSelectedVessel.mmsi) === String(v.mmsi);
      const item = document.createElement("div");
      item.className = `vessel-item ${isSelected ? "active-selected" : ""}`;
      item.setAttribute("data-mmsi", String(v.mmsi));

      let badgeClass = "badge-other";
      if ((v.vessel_class || "").includes("tanker")) badgeClass = "badge-tanker";
      else if ((v.vessel_class || "").includes("cargo")) badgeClass = "badge-cargo";
      else if ((v.vessel_class || "").includes("tug")) badgeClass = "badge-tug";

      item.innerHTML = `
        <div class="vessel-item-top">
          <span class="vessel-item-name">${v.name}</span>
          <span class="vessel-badge ${badgeClass}">${v.vessel_class || "其他"}</span>
        </div>
        <div class="vessel-item-sub">
          <span>MMSI: ${v.mmsi}</span>
          <span>${v.sog_kn ? v.sog_kn.toFixed(1) + "节" : "在泊"} · ${v.motion_state === "moving" ? "在航作业" : "停泊"}</span>
        </div>
      `;

      item.addEventListener("click", () => {
        focusAndLockTarget(v, { flyTo: true, scrollList: false });
      });

      vesselListContainer.appendChild(item);
    }
  }

  vesselSearchInput.addEventListener("input", () => {
    vesselSearchClear.hidden = !vesselSearchInput.value;
    updateSidebarVessels();
    renderVesselsDynamic(currentVessels);
  });
  vesselSearchClear.addEventListener("click", () => {
    vesselSearchInput.value = "";
    vesselSearchClear.hidden = true;
    updateSidebarVessels();
    renderVesselsDynamic(currentVessels);
  });
  filterUnderwayOnly.addEventListener("change", () => {
    updateSidebarVessels();
    renderVesselsDynamic(currentVessels);
  });

  filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      selectedVesselClass = btn.getAttribute("data-class");
      updateSidebarVessels();
      renderVesselsDynamic(currentVessels);
    });
  });

  sidebarToggle.addEventListener("click", () => {
    sidebarVessels.classList.toggle("collapsed");
  });
  sidebarClose?.addEventListener("click", () => {
    sidebarVessels.classList.add("collapsed");
    railVessels?.classList.remove("is-active");
  });

  dockCollapse.addEventListener("click", () => {
    if (dockBody.style.display === "none") {
      dockBody.style.display = "flex";
      dockCollapse.textContent = "−";
    } else {
      dockBody.style.display = "none";
      dockCollapse.textContent = "+";
    }
  });

  // 图层快捷视图：把高频任务变成一键切换，详细图层仍保留在控制坞中。
  const mapPresetButtons = document.querySelectorAll("[data-map-preset]");
  const mapPresetLayers = {
    overview: {
      wind: false, typhoon: true, isobars: false, waves: false, stations: true,
      vessels: true, vectors: false, boundary: false, fairways: false,
      navLines: false, navAids: false, anchorages: false, anchorMarkers: false,
      docks: false, berths: false,
    },
    navigation: {
      wind: false, typhoon: true, isobars: false, waves: false, stations: false,
      vessels: true, vectors: true, boundary: false, fairways: false,
      navLines: true, navAids: false, anchorages: false, anchorMarkers: true,
      docks: false, berths: false,
    },
    weather: {
      wind: true, typhoon: true, isobars: true, waves: true, stations: true,
      vessels: true, vectors: false, boundary: false, fairways: false,
      navLines: false, navAids: false, anchorages: false, anchorMarkers: false,
      docks: false, berths: false,
    },
    clean: {
      wind: false, typhoon: false, isobars: false, waves: false, stations: false,
      vessels: true, vectors: false, boundary: false, fairways: false,
      navLines: false, navAids: false, anchorages: false, anchorMarkers: false,
      docks: false, berths: false,
    },
  };
  const mapPresetInputs = {
    wind: layerWindStream, typhoon: layerTyphoonTrack, isobars: layerIsobars,
    waves: layerWaves, stations: layerStations, vessels: layerVessels,
    vectors: layerVesselVectors, boundary: layerPortBoundary, fairways: layerFairways,
    navLines: layerNavLines, navAids: layerNavAids, anchorages: layerAnchorages,
    anchorMarkers: layerAnchorMarkers, docks: layerDocks, berths: layerBerths,
  };
  function applyMapPreset(name) {
    const preset = mapPresetLayers[name];
    if (!preset) return;
    Object.entries(preset).forEach(([key, checked]) => {
      const input = mapPresetInputs[key];
      if (input && input.checked !== checked) {
        input.checked = checked;
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
    mapPresetButtons.forEach(button => button.classList.toggle("active", button.dataset.mapPreset === name));
  }
  mapPresetButtons.forEach(button => button.addEventListener("click", () => applyMapPreset(button.dataset.mapPreset)));

  // 10. 台风贝丽尔中心动态路径与 7/10 级风圈 (Dynamic Typhoon Track & Wind Radii)
  function renderTyphoonTrackAndCircles(item) {
    typhoonLayerGroup.clearLayers();
    if (!currentHasTyphoon || !layerTyphoonTrack.checked || !allHistogramItems.length) return;

    const current = item || currentItem || allHistogramItems[0];
    if (!current || !current.typhoon_center) return;

    // 1. 绘制台风全周期行进历史虚线轨迹
    const trackPoints = allHistogramItems
      .filter(h => h.typhoon_center && h.typhoon_center.length === 2)
      .map(h => [h.typhoon_center[1], h.typhoon_center[0]]);

    if (trackPoints.length >= 2) {
      L.polyline(trackPoints, {
        color: "rgba(239, 68, 68, 0.55)",
        weight: 2.5,
        dashArray: "5, 7",
      }).addTo(typhoonLayerGroup);

      // 每隔若干小时绘制轨迹微小锚点
      allHistogramItems.forEach((h, idx) => {
        if (!h.typhoon_center || idx % 4 !== 0) return;
        const isCur = h.time_utc === current.time_utc;
        L.circleMarker([h.typhoon_center[1], h.typhoon_center[0]], {
          radius: isCur ? 6 : 2.5,
          fillColor: isCur ? "#ef4444" : "#94a3b8",
          color: "#ffffff",
          weight: 1.2,
          fillOpacity: 1,
        }).bindTooltip(`${h.time_utc.slice(5, 16)} 台风中心`).addTo(typhoonLayerGroup);
      });
    }

    // 2. 当前时刻的台风中心与风圈
    const center = [current.typhoon_center[1], current.typhoon_center[0]];
    const maxWind = current.max_wind_speed || 25.0;
    const minP = current.min_pressure || 988.0;

    // 根据极大风速自适应动态风圈半径 (km)
    const galeRadius = Math.max(50, Math.min(180, Math.round(maxWind * 4.2)));
    const stormRadius = Math.max(20, Math.min(85, Math.round(maxWind * 1.8)));

    // 7 级大风圈 (黄色半透明)
    L.circle(center, {
      radius: galeRadius * 1000,
      color: "rgba(250, 204, 21, 0.5)",
      weight: 1.5,
      dashArray: "8, 6",
      fillColor: "rgba(250, 204, 21, 0.08)",
      fillOpacity: 0.15,
    }).bindTooltip(`7级大风圈 (半径 ${galeRadius} km)`).addTo(typhoonLayerGroup);

    // 10 级暴风圈 (红色半透明)
    L.circle(center, {
      radius: stormRadius * 1000,
      color: "rgba(239, 68, 68, 0.65)",
      weight: 2,
      fillColor: "rgba(239, 68, 68, 0.16)",
      fillOpacity: 0.25,
    }).bindTooltip(`10级暴风圈 (半径 ${stormRadius} km)`).addTo(typhoonLayerGroup);

    // 动态发光台风眼图标 "🌀 L"
    const eyeIcon = L.divIcon({
      html: `
        <div style="display:flex; flex-direction:column; align-items:center; transform:translate(-50%, -50%); pointer-events:none;">
          <div style="font-size:26px; filter:drop-shadow(0 0 10px #ef4444); animation:pulse-glow 1.5s infinite;">🌀</div>
          <strong style="font-size:11px; background:rgba(239,68,68,0.9); color:#fff; padding:2px 6px; border-radius:4px; font-family:monospace; margin-top:-2px; white-space:nowrap;">
            贝丽尔 L ${minP.toFixed(1)} hPa
          </strong>
        </div>
      `,
      className: "typhoon-eye-icon",
      iconSize: [80, 40],
    });
    typhoonLayerGroup.addLayer(L.marker(center, { icon: eyeIcon }));
  }

  // 11. 海面等压线 (随当前时刻动态演变)
  function renderIsobars(item) {
    isobarLayerGroup.clearLayers();
    if (!currentHasTyphoon || !layerIsobars.checked) return;

    const current = item || currentItem || allHistogramItems[0];
    if (!current || !current.typhoon_center) return;

    const center = [current.typhoon_center[1], current.typhoon_center[0]];
    const minP = current.min_pressure || 988.0;

    const isobarLevels = [
      { p: minP + 4, radiusKm: 28, color: "rgba(239, 68, 68, 0.4)" },
      { p: minP + 8, radiusKm: 55, color: "rgba(251, 146, 60, 0.35)" },
      { p: minP + 12, radiusKm: 85, color: "rgba(250, 204, 21, 0.3)" },
      { p: minP + 16, radiusKm: 120, color: "rgba(56, 189, 248, 0.25)" },
    ];

    for (const iso of isobarLevels) {
      const circle = L.circle(center, {
        radius: iso.radiusKm * 1000,
        color: iso.color,
        weight: 1.5,
        dashArray: "6, 6",
        fill: false,
      });
      circle.bindTooltip(`${iso.p.toFixed(0)} hPa 等压线`, { permanent: false, opacity: 0.8 });
      isobarLayerGroup.addLayer(circle);
    }
  }

  // 12. 海浪浪高图层 (随风力动态演化)
  function renderWaves(item) {
    waveLayerGroup.clearLayers();
    if (!currentHasTyphoon || !layerWaves.checked) return;

    const current = item || currentItem || allHistogramItems[0];
    const maxWind = current ? (current.max_wind_speed || 20.0) : 20.0;
    const estMaxWave = Math.max(1.2, Math.min(6.5, maxWind / 5.5));

    const wavePoints = [
      { lat: 29.20, lon: -94.75, hs: estMaxWave, label: `外海波浪带 (${estMaxWave.toFixed(1)}m)` },
      { lat: 29.35, lon: -94.70, hs: Math.max(0.8, estMaxWave * 0.75), label: `外海开阔波浪区 (${(estMaxWave * 0.75).toFixed(1)}m)` },
      { lat: 29.50, lon: -94.85, hs: Math.max(0.5, estMaxWave * 0.45), label: `湾内浪区 (${(estMaxWave * 0.45).toFixed(1)}m)` },
      { lat: 29.70, lon: -95.05, hs: Math.max(0.3, estMaxWave * 0.25), label: `内港泊位水域 (${(estMaxWave * 0.25).toFixed(1)}m)` },
    ];

    for (const wp of wavePoints) {
      const circle = L.circle([wp.lat, wp.lon], {
        radius: 15000,
        fillColor: wp.hs > 3.0 ? "#a855f7" : wp.hs > 2.0 ? "#0284c7" : "#06b6d4",
        color: "transparent",
        fillOpacity: Math.min(0.35, 0.12 + (wp.hs / 5.0) * 0.2),
      });
      circle.bindTooltip(`有效浪高 Hs: ${wp.hs.toFixed(1)}m · ${wp.label}`, { sticky: true });
      waveLayerGroup.addLayer(circle);
    }
  }

  // 13. NOAA 浮标测站
  function renderStations(stations) {
    stationLayerGroup.clearLayers();
    if (!layerStations.checked || !stations || !stations.length) return;

    for (const st of stations) {
      if (st.lat === null || st.lon === null) continue;
      const speed = st.wind_speed_mps || 0;
      const marker = L.circleMarker([st.lat, st.lon], {
        radius: 8,
        fillColor: getBeaufortColor(speed),
        color: "#ffffff",
        weight: 2,
        fillOpacity: 0.9,
      });

      const popupHtml = `
        <div style="font-family:sans-serif; min-width:190px; font-size:12px; color:#0f172a;">
          <strong style="font-size:13px; display:block; margin-bottom:4px; color:#0284c7;">
            📡 NOAA ${st.station_name}
          </strong>
          <div style="color:#64748b; font-size:11px; margin-bottom:6px;">测站编号: ${st.station_id}</div>
          <table style="width:100%; font-size:11px; border-collapse:collapse;">
            <tr><td style="color:#64748b;">实测风速</td><td style="font-weight:700; text-align:right;">${speed.toFixed(1)} m/s (${(speed * 1.94384).toFixed(1)}节)</td></tr>
            <tr><td style="color:#64748b;">极大阵风</td><td style="font-weight:700; text-align:right; color:#ef4444;">${st.wind_gust_mps ? st.wind_gust_mps.toFixed(1) + " m/s" : "无记录"}</td></tr>
            <tr><td style="color:#64748b;">海面气压</td><td style="font-weight:700; text-align:right; color:#a855f7;">${st.air_pressure_hpa ? st.air_pressure_hpa.toFixed(1) + " hPa" : "—"}</td></tr>
            <tr><td style="color:#64748b;">气温</td><td style="font-weight:700; text-align:right;">${st.air_temperature_c ? st.air_temperature_c.toFixed(1) + " °C" : "—"}</td></tr>
          </table>
        </div>
      `;
      marker.bindPopup(popupHtml);
      stationLayerGroup.addLayer(marker);
    }
  }

  // ============================================================
  // 14. 空间拓扑与电子海图符号引擎 (Ray-Casting Point-in-Polygon & S-52 LOD)
  // ============================================================

  // 14.1 射线法点在面算法 (Ray-Casting Point-in-Polygon Engine)
  function isPointInRing(lat, lon, ring) {
    if (!ring || ring.length < 3) return false;
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i][0], yi = ring[i][1];
      const xj = ring[j][0], yj = ring[j][1];
      const intersect = ((yi > lat) !== (yj > lat)) &&
        (lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function isPointInGeometry(lat, lon, geom) {
    if (!geom || !geom.coordinates) return false;
    if (geom.type === "Polygon") {
      return isPointInRing(lat, lon, geom.coordinates[0]);
    } else if (geom.type === "MultiPolygon") {
      for (const poly of geom.coordinates) {
        if (isPointInRing(lat, lon, poly[0])) return true;
      }
    }
    return false;
  }

  // 14.2 质心近似计算 (Centroid for Anchor Beacon)
  function getPolygonCentroid(coordinates) {
    let ring = coordinates;
    if (Array.isArray(ring[0]) && Array.isArray(ring[0][0])) ring = ring[0];
    if (!ring || !ring.length) return [29.45, -94.95];
    let latSum = 0, lonSum = 0, count = ring.length;
    for (let i = 0; i < count; i++) {
      lonSum += ring[i][0];
      latSum += ring[i][1];
    }
    return [latSum / count, lonSum / count];
  }

  // 14.3 船舶空间归属识别 (Spatial Join for Vessels)
  function getVesselSpatialRegion(vessel) {
    if (!vessel || vessel.lat === null || vessel.lon === null) return "休斯顿海域";
    // 1. 优先匹配法定防风锚地
    for (const feat of anchorageFeatures) {
      if (isPointInGeometry(vessel.lat, vessel.lon, feat.geometry)) {
        return `⚓ ${feat.properties?.object_name || "加尔维斯敦防风锚地"}`;
      }
    }
    // 2. 匹配主航道深水通航水道
    for (const feat of fairwayFeatures) {
      if (isPointInGeometry(vessel.lat, vessel.lon, feat.geometry)) {
        return `🌊 ${feat.properties?.object_name || "Houston Ship Channel 航槽"}`;
      }
    }
    // 3. 匹配突堤码头装卸区
    for (const feat of dockFeatures) {
      if (isPointInGeometry(vessel.lat, vessel.lon, feat.geometry)) {
        return `🏢 ${feat.properties?.object_name || "突堤作业码头"}`;
      }
    }
    return "休斯顿近海通航水域";
  }

  // 14.4 打开海事空间要素属性抽屉 (GIS Feature Inspector Drawer)
  function openGisDrawer(feat, layer) {
    deactivateWeatherProbe();
    activeGisFeature = feat;
    const props = feat.properties || {};
    const cat = props.category || "fairway";
    const name = props.object_name || "海事要素";

    // 高亮该多边形轮廓
    gisHighlightLayerGroup.clearLayers();
    if (layer && typeof layer.getLatLngs === "function") {
      const highlightPoly = L.polygon(layer.getLatLngs(), {
        className: "gis-feature-active-highlight",
        fill: false,
        interactive: false,
      });
      gisHighlightLayerGroup.addLayer(highlightPoly);
    }

    // 标题与分类徽章
    if (cat === "anchorage") {
      gisDrawerBadge.textContent = "⚓ 法定防风锚地";
      gisDrawerBadge.style.color = "#facc15";
      gisDrawerBadge.style.borderColor = "rgba(250, 204, 21, 0.4)";
      gisDrawerGrade.textContent = "一级防风避难水域";
      gisDrawerRules.textContent = "限速 8 节 · 候泊抛锚水域";
    } else if (cat === "fairway") {
      gisDrawerBadge.textContent = "🌊 主通航水道";
      gisDrawerBadge.style.color = "#38bdf8";
      gisDrawerBadge.style.borderColor = "rgba(56, 189, 248, 0.4)";
      gisDrawerGrade.textContent = "深水通航级航槽";
      gisDrawerRules.textContent = "严禁抛锚 · 保持航向通报";
    } else if (cat === "dock") {
      gisDrawerBadge.textContent = "🏢 突堤作业码头";
      gisDrawerBadge.style.color = "#fb923c";
      gisDrawerBadge.style.borderColor = "rgba(251, 146, 60, 0.4)";
      gisDrawerGrade.textContent = "港口装卸作业水域";
      gisDrawerRules.textContent = "引水员登轮引航区";
    } else if (cat === "berth") {
      gisDrawerBadge.textContent = "📍 深水泊位";
      gisDrawerBadge.style.color = "#38bdf8";
      gisDrawerBadge.style.borderColor = "rgba(56, 189, 248, 0.4)";
      gisDrawerGrade.textContent = "180m 安全作业范围";
      gisDrawerRules.textContent = "系缆防台管制水域";
    } else {
      gisDrawerBadge.textContent = "🌐 港区总管控水域";
      gisDrawerBadge.style.color = "#38bdf8";
      gisDrawerBadge.style.borderColor = "rgba(56, 189, 248, 0.4)";
      gisDrawerGrade.textContent = "USCG 法定管辖区";
      gisDrawerRules.textContent = "VTS 休斯顿交管中心监管";
    }

    gisDrawerTitle.textContent = name;
    gisDrawerEnc.textContent = `ENC 图幅: ${props.enc_chart || "US5HOUEC.000"} · 管辖: ${props.authority || "USCG / NOAA"}`;
    gisDrawerDepth.textContent = props.depth_m ? `${props.depth_m}m` : "13.7m (维护水深)";
    gisDrawerId.textContent = props.object_id || "ENC_OBJ";

    // 计算当前时刻区内船舶态势
    updateGisDrawerVessels(feat);

    // 绑定聚焦按钮
    btnGisFocus.onclick = () => {
      if (layer && typeof layer.getBounds === "function") {
        map.fitBounds(layer.getBounds(), { padding: [60, 60], maxZoom: 14 });
      }
    };

    if (aiCopilotPanel && !aiCopilotPanel.hidden) closeCopilot();
    layerDock?.classList.remove("workspace-open");
    railLayers?.classList.remove("is-active");
    vesselCard.hidden = true;
    weatherPickerCard.hidden = true;
    gisDrawer.hidden = false;
    if (window.innerWidth < 1000) sidebarVessels.classList.add("collapsed");
  }

  // 14.5 计算并刷新海事抽屉内区内船舶列表
  function updateGisDrawerVessels(feat) {
    if (!feat) return;
    const inZoneVessels = [];
    let stoppedCount = 0;
    let movingCount = 0;

    for (const v of currentVessels) {
      if (v.lat === null || v.lon === null) continue;
      if (isPointInGeometry(v.lat, v.lon, feat.geometry)) {
        inZoneVessels.push(v);
        const isMoving = v.motion_state === "moving" || (v.sog_kn && v.sog_kn > 0.5);
        if (isMoving) movingCount++; else stoppedCount++;
      }
    }

    gisDrawerVesselsCount.textContent = `共 ${inZoneVessels.length} 艘`;
    gisDrawerStoppedCount.textContent = stoppedCount;
    gisDrawerMovingCount.textContent = movingCount;

    gisDrawerVesselsList.innerHTML = "";
    if (!inZoneVessels.length) {
      gisDrawerVesselsList.innerHTML = '<div style="color:var(--text-dim); font-size:11px; padding:10px 0; text-align:center;">当前时刻该水域无船舶活动</div>';
      return;
    }

    for (const v of inZoneVessels.slice(0, 30)) {
      const item = document.createElement("div");
      item.className = "gis-vessel-mini-item";
      const color = getVesselColor(v.vessel_class);
      item.innerHTML = `
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:${color};"></span>
          <span class="gis-vessel-mini-name">${v.name || 'MMSI ' + v.mmsi}</span>
        </div>
        <span class="gis-vessel-mini-meta">${v.sog_kn ? v.sog_kn.toFixed(1) + '节' : '在泊'}</span>
      `;
      item.addEventListener("click", () => {
        focusAndLockTarget(v, { flyTo: true, scrollList: true });
      });
      gisDrawerVesselsList.appendChild(item);
    }
  }

  // 14.6 三级智能 LOD 缩放控制引擎 (Zoom-Adaptive LOD Controller)
  function updateLODVisibility() {
    const zoom = map.getZoom();

    // 1. 泊位圈与泊位点：只有在 zoom >= 13 时才显示，避免远景密集挤压
    if (layerBerths && layerBerths.checked) {
      if (zoom >= 13) {
        if (!map.hasLayer(berthsLayerGroup)) map.addLayer(berthsLayerGroup);
      } else {
        if (map.hasLayer(berthsLayerGroup)) map.removeLayer(berthsLayerGroup);
      }
    }

    // 2. 码头作业面：在 zoom >= 10 时显示
    if (layerDocks && layerDocks.checked) {
      if (zoom >= 10) {
        if (!map.hasLayer(docksLayerGroup)) map.addLayer(docksLayerGroup);
      } else {
        if (map.hasLayer(docksLayerGroup)) map.removeLayer(docksLayerGroup);
      }
    }

    // 3. AtoN 助航灯标：在 zoom >= 11 时显示
    if (layerNavAids && layerNavAids.checked) {
      if (zoom >= 11) {
        if (!map.hasLayer(navAidsLayerGroup)) map.addLayer(navAidsLayerGroup);
      } else {
        if (map.hasLayer(navAidsLayerGroup)) map.removeLayer(navAidsLayerGroup);
      }
    }

    // 4. 锚位中心发光标 ⚓：在 zoom >= 9 时显示
    if (layerAnchorMarkers && layerAnchorMarkers.checked) {
      if (zoom >= 9) {
        if (!map.hasLayer(anchorMarkersLayerGroup)) map.addLayer(anchorMarkersLayerGroup);
      } else {
        if (map.hasLayer(anchorMarkersLayerGroup)) map.removeLayer(anchorMarkersLayerGroup);
      }
    }
  }

  map.on("zoomend", updateLODVisibility);

  // 14.7 加载全要素电子海图空间几何 (Load S-52 ENC Facilities)
  async function loadPortFacilities() {
    try {
      const regParam = currentRegionId ? `&region_id=${encodeURIComponent(currentRegionId)}` : "";
      const res = await fetch(`/api/port-facilities?limit=500${regParam}`);
      if (!res.ok) return;
      const data = await res.json();

      facilityLayerGroup.clearLayers();
      portBoundaryLayerGroup.clearLayers();
      fairwaysLayerGroup.clearLayers();
      navLinesLayerGroup.clearLayers();
      navAidsLayerGroup.clearLayers();
      anchoragesLayerGroup.clearLayers();
      anchorMarkersLayerGroup.clearLayers();
      docksLayerGroup.clearLayers();
      berthsLayerGroup.clearLayers();

      const features = data.features || [];
      allGisFeatures = features;
      fairwayFeatures = [];
      anchorageFeatures = [];
      dockFeatures = [];

      for (const feat of features) {
        const props = feat.properties || {};
        const cat = props.category || "fairway";
        const name = props.object_name || "海事要素";

        // 1. 港区总管控范围外边界 (Port Extent Boundary)
        if (cat === "port_boundary") {
          L.geoJSON(feat, {
            className: "gis-port-boundary-polygon",
            onEachFeature: (_, layer) => {
              layer.bindTooltip(`🌐 ${name}<br><small style="color:#94a3b8">${props.description || "港区法定总管控边界"}</small>`, { sticky: true });
            }
          }).addTo(portBoundaryLayerGroup);
        }

        // 2. 主航道与深水通航水系多边形 (Fairways)
        else if (cat === "fairway") {
          fairwayFeatures.push(feat);
          L.geoJSON(feat, {
            className: "gis-fairway-polygon",
            onEachFeature: (_, layer) => {
              const depth = props.depth_m ? `${props.depth_m}m` : "13.7m (标准维护水深)";
              layer.bindTooltip(`🌊 <strong>${name}</strong><br><span style="color:#38bdf8">航道水深: ${depth} · 点击展开详情</span>`, { sticky: true });
              layer.on("click", (e) => {
                L.DomEvent.stopPropagation(e);
                openGisDrawer(feat, layer);
              });
            }
          }).addTo(fairwaysLayerGroup);
        }

        // 3. 通航推荐中心导线 (Navigation Centerlines)
        else if (cat === "nav_line") {
          L.geoJSON(feat, {
            className: "gis-nav-line",
            onEachFeature: (_, layer) => {
              layer.bindTooltip(`〰️ <strong>${name}</strong><br><span style="color:#38bdf8">通航推荐中心导线</span>`, { sticky: true });
            }
          }).addTo(navLinesLayerGroup);
        }

        // 4. AtoN 助航灯桩与浮标设施 (Aids to Navigation)
        else if (cat === "nav_aid") {
          L.geoJSON(feat, {
            pointToLayer: (feature, latlng) => {
              const beaconIcon = L.divIcon({
                html: '<div class="gis-navaid-beacon" title="' + name + '"></div>',
                className: "navaid-beacon-marker",
                iconSize: [14, 14],
                iconAnchor: [7, 7],
              });
              const m = L.marker(latlng, { icon: beaconIcon });
              m.bindTooltip(`🗼 <strong>${name}</strong><br><span style="color:#4ade80">USCG 助航灯桩导标</span>`, { sticky: true });
              return m;
            }
          }).addTo(navAidsLayerGroup);
        }

        // 5. 法定防风与待泊锚地多边形 (Anchorages)
        else if (cat === "anchorage") {
          anchorageFeatures.push(feat);
          L.geoJSON(feat, {
            className: "gis-anchorage-polygon",
            onEachFeature: (_, layer) => {
              layer.bindTooltip(`⚓ <strong>${name}</strong><br><span style="color:#facc15">法定防风/待泊锚位水域 · 点击展开详情</span>`, { sticky: true });
              layer.on("click", (e) => {
                L.DomEvent.stopPropagation(e);
                openGisDrawer(feat, layer);
              });
            }
          }).addTo(anchoragesLayerGroup);

          // 居中打上发光 ⚓ 锚位中心标
          if (feat.geometry && feat.geometry.coordinates) {
            const centroid = getPolygonCentroid(feat.geometry.coordinates);
            const anchorIcon = L.divIcon({
              html: `
                <div class="anchorage-center-badge" title="${name}">
                  <span class="anchor-icon">⚓</span>
                </div>
              `,
              className: "anchorage-marker",
              iconSize: [24, 24],
              iconAnchor: [12, 12],
            });
            const anchorMarker = L.marker(centroid, { icon: anchorIcon });
            anchorMarker.bindTooltip(`⚓ <strong>${name}</strong><br><span style="color:#facc15">点击查看锚泊水域详情</span>`, { sticky: true });
            anchorMarker.on("click", (e) => {
              L.DomEvent.stopPropagation(e);
              openGisDrawer(feat, null);
            });
            anchorMarker.addTo(anchorMarkersLayerGroup);
          }
        }

        // 6. 突堤码头作业区 (Docks & Piers)
        else if (cat === "dock") {
          dockFeatures.push(feat);
          L.geoJSON(feat, {
            className: "gis-dock-polygon",
            onEachFeature: (_, layer) => {
              layer.bindTooltip(`🏢 <strong>${name}</strong><br><span style="color:#fb923c">港口装卸码头作业区 · 点击展开详情</span>`, { sticky: true });
              layer.on("click", (e) => {
                L.DomEvent.stopPropagation(e);
                openGisDrawer(feat, layer);
              });
            }
          }).addTo(docksLayerGroup);
        }

        // 7. 深水泊位点 (Berths - 180m 安全作业缓冲圈)
        else if (cat === "berth") {
          L.geoJSON(feat, {
            pointToLayer: (feature, latlng) => {
              const circle = L.circle(latlng, {
                radius: feature.properties?.buffer_radius_m || 180,
                className: "gis-berth-buffer",
              });
              circle.bindTooltip(`📍 <strong>${name}</strong><br><span style="color:#38bdf8">深水泊位 · 180m 安全作业缓冲圈</span>`, { sticky: true });
              circle.on("click", (e) => {
                L.DomEvent.stopPropagation(e);
                openGisDrawer(feat, circle);
              });
              return circle;
            }
          }).addTo(berthsLayerGroup);
        }
      }

      // 执行一次 LOD 自适应计算
      updateLODVisibility();
    } catch (e) {
      console.warn("加载港口航道 GIS 空间要素跳过:", e);
    }
  }

  // 14.8 加载卫星遥感切片外廓与 CFAR 雷达目标检测要素 (时相感知协同)
  async function loadRemoteSensingOverlays() {
    try {
      const regParam = currentRegionId ? `?region_id=${encodeURIComponent(currentRegionId)}` : "";
      const res = await fetch(`/api/remote-sensing-overlays${regParam}`);
      if (!res.ok) return;
      const data = await res.json();

      remoteScenesLayerGroup.clearLayers();
      remoteDetectionsLayerGroup.clearLayers();

      const scenes = data.scenes || [];
      const detections = data.detections || [];
      currentRemoteScenes = scenes;
      currentRemoteDetections = detections;

      if (countRemoteDetections) {
        countRemoteDetections.textContent = Math.min(detections.length, 50);
      }

      // 渲染图层抽屉中的卫星过顶时刻快捷直达条
      const helperContainer = document.getElementById("remote-scenes-timeline-helper");
      if (helperContainer) {
        if (scenes.length > 0) {
          helperContainer.hidden = false;
          let html = '<span class="remote-helper-title">卫星过顶时相 (点击直达时空对齐):</span>';
          scenes.forEach(sc => {
            const timeLabel = (sc.observed_at_utc || "").replace("T", " ").replace("Z", "").slice(5, 16);
            html += `<button type="button" class="remote-helper-pill" data-time="${sc.observed_at_utc}">🛰️ ${sc.name || sc.scene_id} (${timeLabel})</button>`;
          });
          helperContainer.innerHTML = html;

          helperContainer.querySelectorAll(".remote-helper-pill").forEach(btn => {
            btn.addEventListener("click", () => {
              const targetTime = btn.getAttribute("data-time");
              if (targetTime) {
                jumpToTimeUtc(targetTime);
                if (layerRemoteScenes) { layerRemoteScenes.checked = true; map.addLayer(remoteScenesLayerGroup); }
                if (layerRemoteDetections) { layerRemoteDetections.checked = true; map.addLayer(remoteDetectionsLayerGroup); }
                renderRemoteOverlaysForCurrentTime();
                const matchedSc = scenes.find(s => s.observed_at_utc === targetTime);
                if (matchedSc && matchedSc.bbox && matchedSc.bbox.length === 4) {
                  const [minLon, minLat, maxLon, maxLat] = matchedSc.bbox;
                  map.flyToBounds([[minLat, minLon], [maxLat, maxLon]], { padding: [60, 60], duration: 1.0 });
                }
              }
            });
          });
        } else {
          helperContainer.hidden = true;
          helperContainer.innerHTML = "";
        }
      }

      // 渲染切片覆盖面 (多边形虚线框)
      for (const sc of scenes) {
        if (!sc.bbox || sc.bbox.length !== 4) continue;
        const [minLon, minLat, maxLon, maxLat] = sc.bbox;
        const bounds = [[minLat, minLon], [maxLat, maxLon]];
        const rect = L.rectangle(bounds, {
          color: "#38bdf8",
          weight: 1.5,
          dashArray: "6, 4",
          fillColor: "#0284c7",
          fillOpacity: 0.08,
        });
        rect.bindTooltip(
          `<div><strong>🛰️ ${sc.name || sc.scene_id}</strong><br><small>时相: ${sc.observed_at_utc || "UTC"} · 集: ${sc.collection || "Sentinel-1"}</small></div>`,
          { sticky: true }
        );
        remoteScenesLayerGroup.addLayer(rect);
      }

      // 根据当前时间智能绘制 CFAR 目标
      renderRemoteOverlaysForCurrentTime();

    } catch (e) {
      console.warn("加载遥感图层要素失败:", e);
    }
  }

  // 根据当前播放/选定时相智能筛选雷达红框 (避免跨日期幽灵悬挂)
  function renderRemoteOverlaysForCurrentTime() {
    remoteDetectionsLayerGroup.clearLayers();
    if (!layerRemoteDetections || !layerRemoteDetections.checked || !currentRemoteDetections.length) return;

    const curTimeMs = currentExactIsoTime ? new Date(currentExactIsoTime).getTime() : 0;

    // 检查当前时刻是否接近任一遥感切片时相 (前后 4 小时时空窗口)
    let inWindow = false;
    for (const sc of currentRemoteScenes) {
      if (sc.observed_at_utc) {
        const scTimeMs = new Date(sc.observed_at_utc).getTime();
        if (Math.abs(curTimeMs - scTimeMs) <= 4 * 3600 * 1000) {
          inWindow = true;
          break;
        }
      }
    }

    // 若当前时间不在卫星过顶窗口，且当前海区有遥感切片，则自动隐去，保持海图干净
    if (!inWindow && currentRemoteScenes.length > 0) {
      return;
    }

    const cfarIcon = L.divIcon({
      className: "sar-cfar-reticle",
      html: '<div class="sar-cfar-reticle-box"></div>',
      iconSize: [14, 14],
      iconAnchor: [7, 7],
    });

    // 优选置信度最高的前 50 个高价值目标
    const topDetections = currentRemoteDetections
      .filter(d => (d.confidence || 0) >= 0.75)
      .slice(0, 50);

    for (const det of topDetections) {
      if (det.lat == null || det.lon == null) continue;
      const marker = L.marker([det.lat, det.lon], { icon: cfarIcon });
      const lenText = det.length_m ? `${Math.round(det.length_m)}m` : "--";
      const widText = det.width_m ? `${Math.round(det.width_m)}m` : "--";
      marker.bindPopup(`
        <div style="font-size:12px; line-height:1.5; color:#102a2f; min-width:180px;">
          <strong style="color:#dc2626;">🎯 CFAR 船体雷达检测目标</strong><br>
          <b>编号:</b> ${det.detection_id || "--"}<br>
          <b>置信度:</b> <span style="color:#059669; font-weight:700;">${(det.confidence * 100).toFixed(1)}%</span><br>
          <b>尺度:</b> ${lenText} × ${widText}<br>
          <b>经纬度:</b> ${det.lon.toFixed(5)}, ${det.lat.toFixed(5)}<br>
          <small style="color:#64748b;">时相: ${det.observed_at_utc || "--"}</small>
        </div>
      `);
      remoteDetectionsLayerGroup.addLayer(marker);
    }
  }

  if (gisDrawerClose) {
    gisDrawerClose.addEventListener("click", () => {
      gisDrawer.hidden = true;
      gisHighlightLayerGroup.clearLayers();
      activeGisFeature = null;
    });
  }

  // 15. 海面气象点位探针：仅在用户明确开启工具后取样，避免普通地图点击
  // 同时触发目标取消、GIS 抽屉和气象卡片等互相冲突的副作用。
  function bilinearGridMetric(lon, lat, field) {
    const grid = currentGridData;
    const bbox = grid?.bbox;
    const dims = grid?.grid_dimensions;
    const cells = grid?.cells;
    if (!bbox || !dims || !cells?.length) return null;
    const [west, south, east, north] = bbox;
    if (lon < west || lon > east || lat < south || lat > north) return null;
    const { cols, rows } = dims;
    const fx = ((lon - west) / (east - west)) * (cols - 1);
    const fy = ((lat - south) / (north - south)) * (rows - 1);
    const x0 = Math.max(0, Math.min(cols - 1, Math.floor(fx)));
    const y0 = Math.max(0, Math.min(rows - 1, Math.floor(fy)));
    const x1 = Math.min(cols - 1, x0 + 1);
    const y1 = Math.min(rows - 1, y0 + 1);
    const sx = fx - x0;
    const sy = fy - y0;
    const at = (x, y) => Number(cells[y * cols + x]?.[field]);
    const values = [at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1)];
    if (values.some(value => !Number.isFinite(value))) return null;
    const top = values[0] * (1 - sx) + values[1] * sx;
    const bottom = values[2] * (1 - sx) + values[3] * sx;
    return top * (1 - sy) + bottom * sy;
  }

  function getNearestStationDistanceKm(lon, lat) {
    if (!currentStations.length) return null;
    const radians = Math.PI / 180;
    return Math.min(...currentStations.map(station => {
      const dLat = (station.lat - lat) * radians;
      const dLon = (station.lon - lon) * radians;
      const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat * radians) * Math.cos(station.lat * radians) * Math.sin(dLon / 2) ** 2;
      return 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }));
  }

  function estimatePointWaveHeight(windSpeed, lon) {
    if (!Number.isFinite(windSpeed) || !currentGridData?.bbox) return null;
    const [west, , east] = currentGridData.bbox;
    // 连续暴露度：由内港向外海平滑增加，而不是“湾内/外海”两档硬切换。
    const offshoreExposure = Math.max(0.22, Math.min(1, 0.22 + 0.78 * ((lon - west) / (east - west))));
    return Math.max(0.15, Math.min(6.5, windSpeed * 0.16 * offshoreExposure));
  }

  function renderWeatherProbe(lat, lon) {
    if (weatherProbeMarker) {
      weatherProbeMarker.setLatLng([lat, lon]);
      return weatherProbeMarker;
    }
    const probeIcon = L.divIcon({
      html: '<svg class="weather-probe-pin" viewBox="0 0 32 42" aria-hidden="true"><path d="M16 1.5C8.3 1.5 2 7.8 2 15.5c0 10.2 14 24.8 14 24.8s14-14.6 14-24.8C30 7.8 23.7 1.5 16 1.5Z"/><circle cx="16" cy="15.5" r="5"/></svg>',
      className: 'weather-probe-marker', iconSize: [32, 42], iconAnchor: [16, 40],
    });
    weatherProbeMarker = L.marker([lat, lon], {
      icon: probeIcon, keyboard: false, zIndexOffset: 950, draggable: true,
      title: "拖动探针重新取样",
    }).addTo(weatherProbeLayerGroup);
    weatherProbeMarker.on("drag", event => {
      const point = event.target.getLatLng();
      updateWeatherProbeReadout(point.lat, point.lng, false);
    });
    weatherProbeMarker.on("dragend", event => {
      const point = event.target.getLatLng();
      updateWeatherProbeReadout(point.lat, point.lng, false);
    });
    return weatherProbeMarker;
  }

  function formatCoordinate(value, positive, negative) {
    return `${Math.abs(value).toFixed(3)}°${value >= 0 ? positive : negative}`;
  }

  function updateWeatherProbeReadout(lat, lng, moveMarker = true) {
    if (moveMarker) renderWeatherProbe(lat, lng);
    pickerCoords.textContent = `${formatCoordinate(lng, "E", "W")}, ${formatCoordinate(lat, "N", "S")}`;

    const wind = engine ? engine.getWindVector(lng, lat) : null;
    if (wind && wind.speed !== null) {
      pickerWindSpd.textContent = `${wind.speed.toFixed(1)} m/s`;
      pickerWindKnots.textContent = `${(wind.speed * 1.94384).toFixed(1)} 节`;
      pickerWindDir.textContent = `${Math.round(wind.direction)}°`;
      pickerCompass.textContent = getCompass(wind.direction);
      pickerConfidence.textContent = wind.confidence !== undefined ? `${(wind.confidence * 100).toFixed(0)}%` : "95%";
    } else {
      pickerWindSpd.textContent = "-- m/s";
      pickerWindKnots.textContent = "超出插值网格";
      pickerWindDir.textContent = "--°";
      pickerCompass.textContent = "无可用网格";
      pickerConfidence.textContent = "--";
    }

    const pressure = bilinearGridMetric(lng, lat, "air_pressure_hpa");
    const spatialConfidence = bilinearGridMetric(lng, lat, "spatial_confidence");
    const wave = estimatePointWaveHeight(wind?.speed, lng);
    const nearestStationKm = getNearestStationDistanceKm(lng, lat);
    pickerPressure.textContent = Number.isFinite(pressure) ? `${pressure.toFixed(1)} hPa` : "-- hPa";
    pickerPressureTrend.textContent = Number.isFinite(pressure) ? "Barnes 空间插值" : "无可用网格";
    pickerWave.textContent = Number.isFinite(wave) ? `${wave.toFixed(1)} m` : "-- m";
    pickerConfidence.textContent = Number.isFinite(spatialConfidence) ? `${Math.round(spatialConfidence * 100)}%` : pickerConfidence.textContent;
    pickerStationDist.textContent = Number.isFinite(nearestStationKm) ? `${nearestStationKm.toFixed(1)} km` : "-- km";

    weatherPickerCard.hidden = false;
  }

  map.on("click", (e) => {
    if (activeMapTool !== "weather-probe") {
      unlockTarget();
      if (gisDrawer) gisDrawer.hidden = true;
      gisHighlightLayerGroup.clearLayers();
      activeGisFeature = null;
      return;
    }
    updateWeatherProbeReadout(e.latlng.lat, e.latlng.lng, true);
  });

  weatherPickerClose.addEventListener("click", () => {
    weatherPickerCard.hidden = true;
    weatherProbeLayerGroup.clearLayers();
    weatherProbeMarker = null;
    setMapTool("browse");
  });

  // 全局 Escape 键一键退出所有弹窗与锁定
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      unlockTarget();
      weatherPickerCard.hidden = true;
      weatherProbeLayerGroup.clearLayers();
      weatherProbeMarker = null;
      setMapTool("browse");
      if (gisDrawer) gisDrawer.hidden = true;
      gisHighlightLayerGroup.clearLayers();
      activeGisFeature = null;
    }
  });

  // ============================================================
  // 16. 时序数据加载与全要素插值联动 (Barnes Grid & AIS Interpolation)
  // ============================================================
  async function loadDataForTime(validTimeUtc, item, options = {}) {
    const vesselsOnly = options.vesselsOnly === true;
    const renderOverlays = options.renderOverlays !== false;
    const requestSerial = ++dataLoadSerial;
    if (dataLoadController) dataLoadController.abort();
    const controller = new AbortController();
    dataLoadController = controller;

    try {
      const targetItem = item || allHistogramItems.find(h => h.time_utc === validTimeUtc) || currentItem;
      const cleanTime = validTimeUtc.replace("T", " ").replace("Z", "").slice(0, 16);

      // 同步顶部大屏与时钟标签
      kpiTime.textContent = cleanTime;
      if (!isTimelinePlaying) {
        gfwCurrentTimeText.textContent = `${cleanTime} UTC`;
        gfwDatetimePicker.value = validTimeUtc.slice(0, 16);
      }

      if (targetItem) {
        if (!isTimelinePlaying) {
          gfwStageTag.textContent = currentHasTyphoon
            ? (targetItem.stage_desc || "气旋演变观测")
            : "常态航运与监管";
        }
        kpiPressure.textContent = `${(targetItem.min_pressure || 1013.0).toFixed(1)} hPa`;
        advisoryText.textContent = currentHasTyphoon
          ? `时刻 ${cleanTime} UTC · 极大风速 ${targetItem.max_wind_speed?.toFixed(1) || "--"} m/s · ${targetItem.stage_desc || ""}`
          : `时刻 ${cleanTime} UTC · 海区常态通航观测 · 浮标海况平稳`;
      }

      // 并行刷新风场与船舶快照 (Barnes 客观分析网格与 AIS 动态船舶分布)
      const regParam = currentRegionId ? `&region_id=${encodeURIComponent(currentRegionId)}` : "";
      const [gridRes, vesselRes] = await Promise.all([
        vesselsOnly ? Promise.resolve(null) : fetch(`/api/wind-field-grid?valid_time_utc=${encodeURIComponent(validTimeUtc)}&cols=16&rows=16${regParam}`, { signal: controller.signal }),
        fetch(`/api/vessels-snapshot?valid_time_utc=${encodeURIComponent(validTimeUtc)}&limit=800${regParam}`, { signal: controller.signal })
      ]);

      // 仅允许最新时间点写回地图，旧请求即使晚到也不能触发全量重绘。
      if (requestSerial !== dataLoadSerial) return;

      if (gridRes?.ok) {
        const gridData = await gridRes.json();
        currentGridData = gridData.grid_field;
        currentStations = gridData.stations || [];

        if (gridData.grid_field?.summary) {
          const s = gridData.grid_field.summary;
          kpiMaxWind.innerHTML = `${s.max_wind_speed_mps.toFixed(1)} m/s <small>(${(s.max_wind_speed_mps * 1.94384).toFixed(1)}节)</small>`;
        }

        // 刷新风场粒子流
        engine.setData(currentGridData, { preserveParticles: isTimelinePlaying });
        if (layerWindStream.checked) engine.start();

        // 刷新测站实况
        renderStations(currentStations);
      }

      // 环境图层按小时刷新；小时内部仅更新 AIS 插值位置，避免重复重绘风场、
      // 台风圈、等压线和波浪等高开销图层。
      if (!vesselsOnly && renderOverlays) {
        renderTyphoonTrackAndCircles(targetItem);
        renderIsobars(targetItem);
        renderWaves(targetItem);
        renderRemoteOverlaysForCurrentTime();
      }

      if (vesselRes.ok) {
        const vesselData = await vesselRes.json();
        currentVessels = vesselData.vessels || [];

        if (vesselData.summary) {
          const sum = vesselData.summary;
          kpiVessels.innerHTML = `${sum.total} 艘 <small>(在航 ${sum.moving})</small>`;
          countAll.textContent = sum.total;
          countTanker.textContent = sum.tanker;
          countCargo.textContent = sum.cargo;
          countTug.textContent = sum.tug;
          countOther.textContent = sum.other;
        }

        // 动态平滑移动船舶船模
        renderVesselsDynamic(currentVessels);
        // 左侧抽屉默认收起，收起时跳过 100 条列表的 DOM 重建；打开时再渲染。
        if (!sidebarVessels.classList.contains("collapsed")) updateSidebarVessels();

        // 若当前海事要素抽屉处于打开状态，同步更新区内船舶态势
        if (gisDrawer && !gisDrawer.hidden && activeGisFeature) {
          updateGisDrawerVessels(activeGisFeature);
        }
      }

    } catch (err) {
      if (err?.name !== "AbortError") console.error("加载数据切片失败:", err);
    } finally {
      if (requestSerial === dataLoadSerial) dataLoadController = null;
    }
  }

  // ============================================================
  // 17. GFW 风格时序活动直方图与自由拖拽时间轴交互引擎
  // ============================================================

  // 17.1 统一全域连续时空视窗驱动引擎 (Unified Global Continuous Timeline)
  // 不再按天拆解，统一海区完整生命周期，未来任何新数据无缝自适应接入
  function initDynamicTimelinePresets() {
    const container = document.getElementById("gfw-presets-container");
    if (!container || !allHistogramItems.length) return;

    if (gfwDatetimePicker && allHistogramItems.length > 0) {
      gfwDatetimePicker.min = allHistogramItems[0].time_utc.slice(0, 16);
      gfwDatetimePicker.max = allHistogramItems[allHistogramItems.length - 1].time_utc.slice(0, 16);
    }

    const startItem = allHistogramItems[0];
    const endItem = allHistogramItems[allHistogramItems.length - 1];
    const startStr = startItem.time_utc.replace("T", " ").replace("Z", "").slice(5, 16);
    const endStr = endItem.time_utc.replace("T", " ").replace("Z", "").slice(5, 16);

    container.innerHTML = `
      <span class="window-dot"></span>
      <span class="window-text">🌐 全域连续观测时段: ${startStr} ~ ${endStr} UTC (${allHistogramItems.length}h 连续覆盖)</span>
    `;

    applyTimelinePreset("all");
  }

  function applyTimelinePreset(presetKey = "all") {
    currentPreset = "all";
    activeHistogramItems = [...allHistogramItems];

    // 自适应生成标准、纯净的等分时间戳刻度
    const n = activeHistogramItems.length;
    if (n > 0) {
      const tickIndices = [0, Math.floor(n * 0.25), Math.floor(n * 0.5), Math.floor(n * 0.75), n - 1]
        .filter((v, i, a) => a.indexOf(v) === i && v < n);
      gfwScaleTicks.innerHTML = tickIndices.map((idx, i) => {
        const item = activeHistogramItems[idx];
        const label = item.time_utc.slice(5, 16).replace("T", " ");
        const isMid = i === Math.floor(tickIndices.length / 2);
        return `<span class="scale-tick ${isMid ? 'highlight' : ''}" data-time="${item.time_utc}">${label}</span>`;
      }).join("");
    } else {
      gfwScaleTicks.innerHTML = `<span>暂无时序直方图</span>`;
    }

    // 重新绑定刻度尺点击事件
    gfwScaleTicks.querySelectorAll(".scale-tick").forEach(tick => {
      tick.addEventListener("click", () => {
        const time = tick.getAttribute("data-time");
        if (time) jumpToTimeUtc(time);
      });
    });

    renderGfwHistogramTrack();

    // 优先保留当前选中的时间点，若不在当前范围则居中选定
    const existingIdx = activeHistogramItems.findIndex(h => h.time_utc === currentItem?.time_utc);
    if (existingIdx !== -1) {
      selectItemByIndex(existingIdx, false);
    } else {
      selectItemByIndex(Math.floor(activeHistogramItems.length / 2), true);
    }
  }

  // 兼容别名
  function setPreset(presetName) {
    applyTimelinePreset("all");
  }

  // 17.2 渲染直方图细柱 (Render Histogram Bars)
  function renderGfwHistogramTrack() {
    gfwHistogramTrack.innerHTML = "";
    if (!activeHistogramItems.length) return;

    const maxWindVal = Math.max(...activeHistogramItems.map(h => h.max_wind_speed || 15.0));

    activeHistogramItems.forEach((item, idx) => {
      const bar = document.createElement("div");
      bar.className = "gfw-hist-bar";
      bar.dataset.index = idx;

      const speed = item.max_wind_speed || 8.0;
      const heightPx = Math.max(4, Math.round((speed / maxWindVal) * 44));
      bar.style.height = `${heightPx}px`;
      bar.style.backgroundColor = getBeaufortColor(speed);

      // 点击柱子直接跳转
      bar.addEventListener("click", (e) => {
        e.stopPropagation();
        selectItemByIndex(idx, true);
      });

      // 悬浮显示 Tooltip
      bar.addEventListener("mouseenter", (e) => showHistogramTooltip(item, bar));
      bar.addEventListener("mouseleave", hideHistogramTooltip);

      gfwHistogramTrack.appendChild(bar);
    });
  }

  // 17.3 悬停气泡 Tooltip 逻辑
  function showHistogramTooltip(item, targetEl) {
    if (isScrubberDragging) return;
    const rect = targetEl.getBoundingClientRect();
    const containerRect = gfwTrackContainer.getBoundingClientRect();
    const offsetX = rect.left - containerRect.left + rect.width / 2;

    tooltipTime.textContent = `${item.time_utc.replace("T", " ").replace("Z", "").slice(0, 16)} UTC`;
    tooltipWind.textContent = `极大风速: ${item.max_wind_speed?.toFixed(1) || "--"} m/s · 气压: ${item.min_pressure?.toFixed(1) || "--"} hPa`;

    gfwHoverTooltip.style.left = `${Math.max(60, Math.min(containerRect.width - 60, offsetX))}px`;
    gfwHoverTooltip.hidden = false;
  }

  function hideHistogramTooltip() {
    if (!isScrubberDragging) {
      gfwHoverTooltip.hidden = true;
    }
  }

  // 17.4 更新游标滑块与进度遮罩位置 (Update Scrubber Position)
  function updateScrubberPosition(targetIdx) {
    if (activeHistogramItems.length <= 1) return;
    const pct = Math.max(0, Math.min(100, (targetIdx / (activeHistogramItems.length - 1)) * 100));

    gfwScrubberNeedle.style.left = `${pct}%`;
    gfwProgressFill.style.width = `${pct}%`;

    // 高亮直方图柱子
    const bars = gfwHistogramTrack.querySelectorAll(".gfw-hist-bar");
    bars.forEach((b, i) => b.classList.toggle("active", i === targetIdx));
  }

  // 17.5 索引级选定与平滑驱动
  function selectItemByIndex(targetIdx, shouldFetch = true) {
    if (!activeHistogramItems.length) return;
    targetIdx = Math.max(0, Math.min(activeHistogramItems.length - 1, targetIdx));

    currentItemIndex = targetIdx;
    currentItem = activeHistogramItems[targetIdx];
    currentExactIsoTime = currentItem.time_utc;
    playbackTimeMs = new Date(currentExactIsoTime).getTime();

    updateScrubberPosition(targetIdx);

    // 实时更新头部文字
    const cleanTime = currentItem.time_utc.replace("T", " ").replace("Z", "").slice(0, 16);
    gfwCurrentTimeText.textContent = `${cleanTime} UTC`;
    gfwDatetimePicker.value = currentItem.time_utc.slice(0, 16);
    gfwStageTag.textContent = currentItem.stage_desc || (currentHasTyphoon ? "台风气象过程观测" : "常态航运与监管");

    if (shouldFetch) {
      loadDataForTime(currentItem.time_utc, currentItem);
    }
  }

  // 17.6 精确时间戳跳转 (Jump to Any Date/Time UTC)
  function jumpToTimeUtc(timeUtcStr) {
    if (!allHistogramItems.length) return;

    // 如果当前选定时段不在当前切片范围内，自动扩展到全时段
    const targetDate = timeUtcStr.slice(0, 10);
    const inActive = activeHistogramItems.some(h => h.time_utc && h.time_utc.startsWith(targetDate));
    if (!inActive) {
      applyTimelinePreset("all");
    }

    // 寻找时间差最小的最接近项
    const targetMs = new Date(timeUtcStr).getTime();
    let bestIdx = 0;
    let minDiff = Infinity;

    activeHistogramItems.forEach((h, idx) => {
      const diff = Math.abs(new Date(h.time_utc).getTime() - targetMs);
      if (diff < minDiff) {
        minDiff = diff;
        bestIdx = idx;
      }
    });

    selectItemByIndex(bestIdx, true);
  }

  // 17.7 自由拖拽与轨道点击交互引擎 (Continuous Minute-Level Scrubber Drag & Click)
  const needleHandle = gfwScrubberNeedle.querySelector(".gfw-needle-handle");

  function handleTrackPointerEvent(e, isEnd = false) {
    if (!activeHistogramItems.length) return;
    const rect = gfwTrackContainer.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));

    const startMs = new Date(activeHistogramItems[0].time_utc).getTime();
    const endMs = new Date(activeHistogramItems[activeHistogramItems.length - 1].time_utc).getTime();
    let exactMs = startMs + ratio * (endMs - startMs);

    // 依所选粒度吸附
    if (currentGranularity === "1h") {
      exactMs = Math.round(exactMs / 3600000) * 3600000;
    } else if (currentGranularity === "3h") {
      exactMs = Math.round(exactMs / (3600000 * 3)) * (3600000 * 3);
    } else {
      // continuous: 对齐到最近 1 分钟
      exactMs = Math.round(exactMs / 60000) * 60000;
    }

    exactMs = Math.max(startMs, Math.min(endMs, exactMs));
    const effectiveRatio = endMs > startMs ? (exactMs - startMs) / (endMs - startMs) : ratio;

    // 寻找最邻近的小时直方图条目
    let closestIdx = 0;
    let minDiff = Infinity;
    activeHistogramItems.forEach((item, idx) => {
      const diff = Math.abs(new Date(item.time_utc).getTime() - exactMs);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = idx;
      }
    });

    const item = activeHistogramItems[closestIdx];
    const exactDate = new Date(exactMs);
    const isoUtc = exactDate.toISOString().replace(".000", "");
    currentExactIsoTime = isoUtc;

    // 格式化当前时刻（精确到分钟）
    const yyyy = exactDate.getUTCFullYear();
    const mm = String(exactDate.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(exactDate.getUTCDate()).padStart(2, "0");
    const hh = String(exactDate.getUTCHours()).padStart(2, "0");
    const min = String(exactDate.getUTCMinutes()).padStart(2, "0");
    const cleanTime = `${yyyy}-${mm}-${dd} ${hh}:${min}`;

    // 60fps 瞬时更新游标针位置与大屏指示
    const pct = (effectiveRatio * 100);
    gfwScrubberNeedle.style.left = `${pct}%`;
    gfwProgressFill.style.width = `${pct}%`;

    // 高亮临近直方图柱
    const bars = gfwHistogramTrack.querySelectorAll(".gfw-hist-bar");
    bars.forEach((b, i) => b.classList.toggle("active", i === closestIdx));

    gfwCurrentTimeText.textContent = `${cleanTime} UTC`;
    gfwDatetimePicker.value = `${yyyy}-${mm}-${dd}T${hh}:${min}`;
    gfwStageTag.textContent = item.stage_desc || "连续时空插值观测";

    // Tooltip 伴随游标
    tooltipTime.textContent = `${cleanTime} UTC`;
    tooltipWind.textContent = `极大风速: ${item.max_wind_speed?.toFixed(1) || "--"} m/s · 气压: ${item.min_pressure?.toFixed(1) || "--"} hPa`;
    gfwHoverTooltip.style.left = `${pct}%`;
    gfwHoverTooltip.hidden = false;

    currentItemIndex = closestIdx;
    currentItem = item;
    updateActiveTrackCruisePosition(exactMs);

    if (isEnd) {
      // 释放指针时执行确切高精插值数据拉取
      loadDataForTime(isoUtc, item);
    } else {
      // 拖动过程中降低数据刷新频率；游标仍即时移动，松手后再精确加载。
      if (fetchThrottleTimer) clearTimeout(fetchThrottleTimer);
      fetchThrottleTimer = setTimeout(() => {
        loadDataForTime(isoUtc, item);
      }, 260);
    }
  }

  // 轨道整体点击直接定位
  gfwTrackContainer.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".gfw-needle-handle")) return; // 若点在 handle 上交由拖拽处理
    handleTrackPointerEvent(e, true);
  });

  // 游标手柄 Pointer 拖拽捕捉 (setPointerCapture)
  needleHandle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    if (isTimelinePlaying) stopTimelinePlayback(false);
    isScrubberDragging = true;
    gfwScrubberNeedle.classList.add("dragging");
    needleHandle.setPointerCapture(e.pointerId);
  });

  needleHandle.addEventListener("pointermove", (e) => {
    if (!isScrubberDragging) return;
    handleTrackPointerEvent(e, false);
  });

  const onPointerDragEnd = (e) => {
    if (!isScrubberDragging) return;
    isScrubberDragging = false;
    gfwScrubberNeedle.classList.remove("dragging");
    gfwHoverTooltip.hidden = true;
    try {
      needleHandle.releasePointerCapture(e.pointerId);
    } catch (_) {}
    handleTrackPointerEvent(e, true);
  };

  needleHandle.addEventListener("pointerup", onPointerDragEnd);
  needleHandle.addEventListener("pointercancel", onPointerDragEnd);

  // 轨道鼠标移动跟随 Tooltip
  gfwTrackContainer.addEventListener("pointermove", (e) => {
    if (isScrubberDragging) return;
    const rect = gfwTrackContainer.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));

    const startMs = new Date(activeHistogramItems[0]?.time_utc || 0).getTime();
    const endMs = new Date(activeHistogramItems[activeHistogramItems.length - 1]?.time_utc || 0).getTime();
    const hoverMs = startMs + ratio * (endMs - startMs);
    const hoverDate = new Date(hoverMs);

    let closestIdx = 0;
    let minDiff = Infinity;
    activeHistogramItems.forEach((item, idx) => {
      const diff = Math.abs(new Date(item.time_utc).getTime() - hoverMs);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = idx;
      }
    });

    const item = activeHistogramItems[closestIdx];
    if (item) {
      const yyyy = hoverDate.getUTCFullYear();
      const mm = String(hoverDate.getUTCMonth() + 1).padStart(2, "0");
      const dd = String(hoverDate.getUTCDate()).padStart(2, "0");
      const hh = String(hoverDate.getUTCHours()).padStart(2, "0");
      const min = String(hoverDate.getUTCMinutes()).padStart(2, "0");
      tooltipTime.textContent = `${yyyy}-${mm}-${dd} ${hh}:${min} UTC`;
      tooltipWind.textContent = `极大风速: ${item.max_wind_speed?.toFixed(1) || "--"} m/s · 气压: ${item.min_pressure?.toFixed(1) || "--"} hPa`;
      gfwHoverTooltip.style.left = `${Math.max(60, Math.min(rect.width - 60, e.clientX - rect.left))}px`;
      gfwHoverTooltip.hidden = false;
    }
  });

  gfwTrackContainer.addEventListener("pointerleave", () => {
    if (!isScrubberDragging) gfwHoverTooltip.hidden = true;
  });

  // 17.8 自定义日期时间选择器输入跳转
  btnApplyCustomTime.addEventListener("click", () => {
    const val = gfwDatetimePicker.value;
    if (val) {
      const isoUtc = val.length === 16 ? `${val}:00Z` : val;
      jumpToTimeUtc(isoUtc);
    }
  });

  gfwDatetimePicker.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const val = gfwDatetimePicker.value;
      if (val) {
        const isoUtc = val.length === 16 ? `${val}:00Z` : val;
        jumpToTimeUtc(isoUtc);
      }
    }
  });

  // 时间粒度模式切换
  gfwGranBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      gfwGranBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentGranularity = btn.getAttribute("data-gran") || "continuous";
    });
  });

  // 17.9 预设胶囊按钮监听
  gfwPresetBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const preset = btn.getAttribute("data-preset");
      if (preset && preset !== currentPreset) {
        setPreset(preset);
      }
    });
  });

  // 17.10 步退与步进按钮 (Step Forward / Backward)
  btnStepPrev.addEventListener("click", () => {
    selectItemByIndex(currentItemIndex - 1, true);
  });
  btnStepNext.addEventListener("click", () => {
    selectItemByIndex(currentItemIndex + 1, true);
  });

  // 17.11 播放倍速切换
  gfwSpeedBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      gfwSpeedBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const factor = Math.max(1, Number.parseFloat(btn.textContent) || 1);
      playbackRate = 600 * factor;
    });
  });

  function interpolateTimelineItem(timeMs) {
    if (!activeHistogramItems.length) return null;
    if (activeHistogramItems.length === 1) return { ...activeHistogramItems[0] };

    let rightIndex = activeHistogramItems.findIndex(item => new Date(item.time_utc).getTime() >= timeMs);
    if (rightIndex <= 0) return { ...activeHistogramItems[0], time_utc: new Date(timeMs).toISOString() };
    if (rightIndex < 0) {
      return { ...activeHistogramItems[activeHistogramItems.length - 1], time_utc: new Date(timeMs).toISOString() };
    }
    const left = activeHistogramItems[rightIndex - 1];
    const right = activeHistogramItems[rightIndex];
    const leftMs = new Date(left.time_utc).getTime();
    const rightMs = new Date(right.time_utc).getTime();
    const ratio = Math.max(0, Math.min(1, (timeMs - leftMs) / Math.max(1, rightMs - leftMs)));
    const mix = (a, b, fallback = 0) => {
      const leftValue = Number.isFinite(Number(a)) ? Number(a) : fallback;
      const rightValue = Number.isFinite(Number(b)) ? Number(b) : leftValue;
      return leftValue + (rightValue - leftValue) * ratio;
    };
    const leftCenter = Array.isArray(left.typhoon_center) ? left.typhoon_center : right.typhoon_center;
    const rightCenter = Array.isArray(right.typhoon_center) ? right.typhoon_center : leftCenter;
    return {
      ...left,
      time_utc: new Date(timeMs).toISOString(),
      timestamp_epoch: Math.floor(timeMs / 1000),
      max_wind_speed: mix(left.max_wind_speed, right.max_wind_speed, 0),
      min_pressure: mix(left.min_pressure, right.min_pressure, 1012),
      obs_count: Math.round(mix(left.obs_count, right.obs_count, 0)),
      typhoon_center: leftCenter && rightCenter
        ? [mix(leftCenter[0], rightCenter[0]), mix(leftCenter[1], rightCenter[1])]
        : null,
      stage_desc: ratio < 0.5 ? left.stage_desc : right.stage_desc,
      interpolation_ratio: ratio,
      interpolation_from_utc: left.time_utc,
      interpolation_to_utc: right.time_utc,
    };
  }

  function renderContinuousEnvironment(item, frameTime) {
    if (!item || frameTime - lastPlaybackEnvironmentRenderAt < 120) return;
    lastPlaybackEnvironmentRenderAt = frameTime;
    renderTyphoonTrackAndCircles(item);
    renderIsobars(item);
    renderWaves(item);
    kpiPressure.textContent = `${Number(item.min_pressure || 988).toFixed(1)} hPa`;
    advisoryText.textContent = `连续时刻 ${item.time_utc.slice(0, 19).replace("T", " ")} UTC · 极大风速 ${Number(item.max_wind_speed || 0).toFixed(1)} m/s · ${item.stage_desc || ""}`;
  }

  // 17.12 连续播放：游标逐帧移动，AIS、测站风场与环境要素均按精确时刻插值。
  function renderContinuousPlaybackTime(timeMs) {
    if (!activeHistogramItems.length) return null;
    const startMs = new Date(activeHistogramItems[0].time_utc).getTime();
    const endMs = new Date(activeHistogramItems[activeHistogramItems.length - 1].time_utc).getTime();
    const durationMs = Math.max(1, endMs - startMs);
    const boundedMs = Math.max(startMs, Math.min(endMs, timeMs));
    const ratio = (boundedMs - startMs) / durationMs;

    let closestIdx = 0;
    let minDiff = Infinity;
    activeHistogramItems.forEach((item, idx) => {
      const diff = Math.abs(new Date(item.time_utc).getTime() - boundedMs);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = idx;
      }
    });

    const exactDate = new Date(boundedMs);
    const isoUtc = exactDate.toISOString();
    const yyyy = exactDate.getUTCFullYear();
    const mm = String(exactDate.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(exactDate.getUTCDate()).padStart(2, "0");
    const hh = String(exactDate.getUTCHours()).padStart(2, "0");
    const min = String(exactDate.getUTCMinutes()).padStart(2, "0");
    const sec = String(exactDate.getUTCSeconds()).padStart(2, "0");
    const item = interpolateTimelineItem(boundedMs) || activeHistogramItems[closestIdx];

    gfwScrubberNeedle.style.left = `${ratio * 100}%`;
    gfwProgressFill.style.width = `${ratio * 100}%`;
    if (lastPlaybackHighlightIndex !== closestIdx) {
      const bars = gfwHistogramTrack.querySelectorAll(".gfw-hist-bar");
      if (lastPlaybackHighlightIndex >= 0) bars[lastPlaybackHighlightIndex]?.classList.remove("active");
      bars[closestIdx]?.classList.add("active");
      lastPlaybackHighlightIndex = closestIdx;
    }
    const uiSecond = `${yyyy}-${mm}-${dd} ${hh}:${min}:${sec}`;
    if (uiSecond !== lastPlaybackUiSecond) {
      gfwCurrentTimeText.textContent = `${uiSecond} UTC`;
      gfwDatetimePicker.value = `${yyyy}-${mm}-${dd}T${hh}:${min}`;
      gfwStageTag.textContent = `${item.stage_desc || "连续时空插值观测"} · 连续插值`;
      lastPlaybackUiSecond = uiSecond;
    }
    gfwHoverTooltip.hidden = true;

    currentItemIndex = closestIdx;
    currentItem = item;
    currentExactIsoTime = isoUtc;
    updateActiveTrackCruisePosition(boundedMs);
    return { isoUtc, item, startMs, endMs };
  }

  function playbackTick(frameTime) {
    if (!isTimelinePlaying || !activeHistogramItems.length) return;
    if (lastPlaybackFrameAt === null) lastPlaybackFrameAt = frameTime;
    const elapsed = Math.min(100, Math.max(0, frameTime - lastPlaybackFrameAt));
    lastPlaybackFrameAt = frameTime;

    const startMs = new Date(activeHistogramItems[0].time_utc).getTime();
    const endMs = new Date(activeHistogramItems[activeHistogramItems.length - 1].time_utc).getTime();
    const durationMs = Math.max(1, endMs - startMs);
    playbackTimeMs = Number.isFinite(playbackTimeMs) ? playbackTimeMs : startMs;
    playbackTimeMs += elapsed * playbackRate;
    if (playbackTimeMs > endMs) playbackTimeMs = startMs + ((playbackTimeMs - startMs) % durationMs);

    const rendered = renderContinuousPlaybackTime(playbackTimeMs);
    if (rendered) renderContinuousEnvironment(rendered.item, frameTime);
    const fetchIntervalMs = Math.max(620, 1000 / Math.sqrt(playbackRate / 600));
    if (rendered && !playbackLoadInFlight && frameTime - lastPlaybackFetchAt >= fetchIntervalMs) {
      lastPlaybackFetchAt = frameTime;
      playbackLoadInFlight = true;
      loadDataForTime(rendered.isoUtc, rendered.item, { renderOverlays: false })
        .finally(() => { playbackLoadInFlight = false; });
    }
    playbackFrame = requestAnimationFrame(playbackTick);
  }

  function stopTimelinePlayback(commitCurrentTime = true) {
    isTimelinePlaying = false;
    if (playbackFrame !== null) cancelAnimationFrame(playbackFrame);
    playbackFrame = null;
    lastPlaybackFrameAt = null;
    playbackLoadInFlight = false;
    btnTimelinePlay.classList.remove("active");
    btnTimelinePlay.querySelector(".play-icon").textContent = "▶";
    btnTimelinePlay.querySelector(".play-text").textContent = "播放";
    if (commitCurrentTime && currentExactIsoTime) loadDataForTime(currentExactIsoTime, currentItem);
  }

  function toggleTimelinePlay() {
    if (isTimelinePlaying) {
      stopTimelinePlayback(true);
    } else {
      stopCruiseAnimation();
      isTimelinePlaying = true;
      btnTimelinePlay.classList.add("active");
      btnTimelinePlay.querySelector(".play-icon").textContent = "⏸";
      btnTimelinePlay.querySelector(".play-text").textContent = "暂停";
      playbackTimeMs = new Date(currentExactIsoTime || currentItem?.time_utc || activeHistogramItems[0]?.time_utc).getTime();
      lastPlaybackFrameAt = null;
      lastPlaybackFetchAt = 0;
      lastPlaybackEnvironmentRenderAt = 0;
      lastPlaybackUiSecond = "";
      lastPlaybackHighlightIndex = -1;
      playbackFrame = requestAnimationFrame(playbackTick);
    }
  }
  btnTimelinePlay.addEventListener("click", toggleTimelinePlay);

  // ============================================================
  // 18. 图层开关监听
  // ============================================================
  layerWindStream.addEventListener("change", (e) => {
    canvas.style.display = e.target.checked ? "block" : "none";
    particleDensityRow.style.display = e.target.checked ? "flex" : "none";
    if (e.target.checked) engine.start(); else engine.stop();
  });

  particleCountSlider.addEventListener("input", (e) => {
    engine.setParticleCount(parseInt(e.target.value, 10));
  });

  layerTyphoonTrack.addEventListener("change", (e) => {
    if (e.target.checked) {
      map.addLayer(typhoonLayerGroup);
      renderTyphoonTrackAndCircles(currentItem);
    } else {
      map.removeLayer(typhoonLayerGroup);
    }
  });

  layerIsobars.addEventListener("change", (e) => {
    if (e.target.checked) {
      map.addLayer(isobarLayerGroup);
      renderIsobars(currentItem);
    } else {
      map.removeLayer(isobarLayerGroup);
    }
  });

  layerWaves.addEventListener("change", (e) => {
    if (e.target.checked) {
      map.addLayer(waveLayerGroup);
      renderWaves(currentItem);
    } else {
      map.removeLayer(waveLayerGroup);
    }
  });

  layerStations.addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(stationLayerGroup); else map.removeLayer(stationLayerGroup);
  });

  layerVessels.addEventListener("change", () => {
    renderVesselsDynamic(currentVessels);
  });

  layerVesselVectors.addEventListener("change", () => {
    renderVesselsDynamic(currentVessels);
  });

  if (layerRemoteScenes) {
    layerRemoteScenes.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(remoteScenesLayerGroup); else map.removeLayer(remoteScenesLayerGroup);
    });
  }

  if (layerRemoteDetections) {
    layerRemoteDetections.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(remoteDetectionsLayerGroup); else map.removeLayer(remoteDetectionsLayerGroup);
    });
  }

  layerFacilities.addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(facilityLayerGroup); else map.removeLayer(facilityLayerGroup);
  });

  if (layerPortBoundary) {
    layerPortBoundary.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(portBoundaryLayerGroup); else map.removeLayer(portBoundaryLayerGroup);
    });
  }

  if (layerFairways) {
    layerFairways.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(fairwaysLayerGroup); else map.removeLayer(fairwaysLayerGroup);
    });
  }

  if (layerAnchorages) {
    layerAnchorages.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(anchoragesLayerGroup); else map.removeLayer(anchoragesLayerGroup);
    });
  }

  if (layerDocks) {
    layerDocks.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(docksLayerGroup); else map.removeLayer(docksLayerGroup);
    });
  }

  if (layerBerths) {
    layerBerths.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(berthsLayerGroup); else map.removeLayer(berthsLayerGroup);
    });
  }

  if (layerNavLines) {
    layerNavLines.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(navLinesLayerGroup); else map.removeLayer(navLinesLayerGroup);
    });
  }

  if (layerNavAids) {
    layerNavAids.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(navAidsLayerGroup); else map.removeLayer(navAidsLayerGroup);
    });
  }

  if (layerAnchorMarkers) {
    layerAnchorMarkers.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(anchorMarkersLayerGroup); else map.removeLayer(anchorMarkersLayerGroup);
    });
  }

  // 海图填充透明度滑块
  if (gisOpacitySlider) {
    gisOpacitySlider.addEventListener("input", (e) => {
      const val = parseInt(e.target.value, 10) / 100;
      currentGisFillOpacity = val;
      fairwaysLayerGroup.eachLayer(layer => {
        if (layer.setStyle) layer.setStyle({ fillOpacity: val });
      });
      anchoragesLayerGroup.eachLayer(layer => {
        if (layer.setStyle) layer.setStyle({ fillOpacity: val });
      });
      docksLayerGroup.eachLayer(layer => {
        if (layer.setStyle) layer.setStyle({ fillOpacity: Math.min(0.6, val + 0.05) });
      });
      berthsLayerGroup.eachLayer(layer => {
        if (layer.setStyle) layer.setStyle({ fillOpacity: val });
      });
    });
  }

  // 海图标注图例折叠切换
  if (btnLegendToggle && maritimeLegendBody) {
    const toggleLegend = () => {
      const isCollapsed = maritimeLegendBody.classList.toggle("collapsed");
      btnLegendToggle.textContent = isCollapsed ? "+" : "−";
    };
    btnLegendToggle.addEventListener("click", toggleLegend);
    if (maritimeLegendHeader) {
      maritimeLegendHeader.addEventListener("click", (e) => {
        if (maritimeLegendCard?.dataset.justDragged === "true") {
          maritimeLegendCard.dataset.justDragged = "false";
          return;
        }
        if (e.target !== btnLegendToggle) toggleLegend();
      });
    }
  }

  // 可移动浮窗：信息卡、图例和图层坞通过标题栏拖动。拖动范围限制在地图
  // 可视区域且不会覆盖底部时间轴；表单控件和关闭按钮保持正常点击行为。
  function makePanelDraggable(panel, handle) {
    if (!panel || !handle || panel.dataset.dragReady === "true") return;
    panel.dataset.dragReady = "true";
    handle.classList.add("panel-drag-handle");
    handle.setAttribute("title", `${handle.getAttribute("title") || ""} 按住可拖动`.trim());
    L.DomEvent.disableClickPropagation(panel);
    L.DomEvent.disableScrollPropagation(panel);

    handle.addEventListener("pointerdown", event => {
      if (event.button !== 0 || event.target.closest("button,input,select,textarea,a")) return;
      const panelRect = panel.getBoundingClientRect();
      if (!panelRect.width || !panelRect.height) return;
      const parent = panel.offsetParent || document.documentElement;
      const parentRect = parent.getBoundingClientRect();
      const startX = event.clientX;
      const startY = event.clientY;
      const initialLeft = panelRect.left - parentRect.left;
      const initialTop = panelRect.top - parentRect.top;
      let moved = false;

      panel.style.left = `${initialLeft}px`;
      panel.style.top = `${initialTop}px`;
      panel.style.right = "auto";
      panel.style.bottom = "auto";
      panel.classList.add("is-panel-dragging");
      handle.setPointerCapture?.(event.pointerId);
      event.preventDefault();

      const onMove = moveEvent => {
        const dx = moveEvent.clientX - startX;
        const dy = moveEvent.clientY - startY;
        if (Math.abs(dx) + Math.abs(dy) > 4) moved = true;
        const timelineRect = document.getElementById("gfw-timeline-panel")?.getBoundingClientRect();
        const availableBottom = timelineRect && timelineRect.top > parentRect.top
          ? timelineRect.top - parentRect.top - 8
          : parentRect.height - 8;
        const maxLeft = Math.max(8, parentRect.width - panelRect.width - 8);
        const maxTop = Math.max(8, availableBottom - panelRect.height);
        panel.style.left = `${Math.max(8, Math.min(maxLeft, initialLeft + dx))}px`;
        panel.style.top = `${Math.max(8, Math.min(maxTop, initialTop + dy))}px`;
      };
      const onUp = () => {
        document.removeEventListener("pointermove", onMove);
        document.removeEventListener("pointerup", onUp);
        document.removeEventListener("pointercancel", onUp);
        panel.classList.remove("is-panel-dragging");
        if (moved) panel.dataset.justDragged = "true";
      };
      document.addEventListener("pointermove", onMove);
      document.addEventListener("pointerup", onUp, { once: true });
      document.addEventListener("pointercancel", onUp, { once: true });
    });
  }

  // ============================================================
  // 19. 全局海区管理与多源态势联动 (Region Management & Situation Sync)
  // ============================================================
  async function loadTimelineHistogram() {
    try {
      const regParam = currentRegionId ? `?region_id=${encodeURIComponent(currentRegionId)}` : "";
      const histRes = await fetch(`/api/timeline-histogram${regParam}`);
      if (histRes.ok) {
        const histData = await histRes.json();
        allHistogramItems = histData.items || [];
        currentHasTyphoon = Boolean(histData.has_typhoon);
      }
    } catch (e) {
      console.warn("读取全周期时间直方图失败:", e);
    }
    updateRegionEnvironmentUI();
    initDynamicTimelinePresets();
  }

  function updateRegionEnvironmentUI() {
    const rowTyphoon = document.getElementById("row-typhoon-track");
    const detailsModels = document.getElementById("details-typhoon-models");
    const stormBox = document.getElementById("vessel-card-storm-box");

    if (rowTyphoon) rowTyphoon.hidden = !currentHasTyphoon;
    if (detailsModels) detailsModels.hidden = !currentHasTyphoon;
    if (stormBox) stormBox.hidden = !currentHasTyphoon;

    if (!currentHasTyphoon) {
      typhoonLayerGroup.clearLayers();
      isobarLayerGroup.clearLayers();
      waveLayerGroup.clearLayers();
    }
  }

  async function switchRegion(regionId, doFly = true) {
    if (!regionId) return;
    currentRegionId = regionId;
    currentRegionData = availableRegions.find(r => r.region_id === regionId) || null;

    // 同步 URL 参数便于刷新与持久分享
    try {
      const url = new URL(window.location);
      url.searchParams.set("region_id", regionId);
      window.history.replaceState({}, "", url);
    } catch (_) {}

    // 视口平滑飞行至该海区包围盒
    if (currentRegionData) {
      let bbox = currentRegionData.bbox_json || currentRegionData.bbox_wgs84;
      if (typeof bbox === "string") {
        try { bbox = JSON.parse(bbox); } catch (_) { bbox = null; }
      }
      if (Array.isArray(bbox) && bbox.length === 4) {
        const [minLon, minLat, maxLon, maxLat] = bbox;
        map.fitBounds([[minLat, minLon], [maxLat, maxLon]], {
          padding: [50, 50],
          maxZoom: 13,
          animate: Boolean(doFly),
          duration: 1.2,
        });
      }
    }

    // 同步智能体 Copilot 上下文
    copilotRegionId = currentRegionId;
    if (copilotContextText && currentRegionData) {
      copilotContextText.textContent = `${currentRegionData.canonical_name || currentRegionData.display_name || "当前海区"} · ${document.getElementById("gfw-current-time-text")?.textContent || "当前时间"}`;
    }

    // 刷新该海区的各模态图层
    await Promise.all([
      loadPortFacilities(),
      loadRemoteSensingOverlays(),
      loadTimelineHistogram(),
    ]);
  }

  async function initRegionsAndWorkspace() {
    try {
      const res = await fetch("/regions");
      if (!res.ok) throw new Error("获取海区列表失败");
      const payload = await res.json();
      availableRegions = payload.regions || payload.items || payload || [];

      if (workspaceRegionSelect) {
        workspaceRegionSelect.innerHTML = "";
        for (const reg of availableRegions) {
          const opt = document.createElement("option");
          opt.value = reg.region_id;
          opt.textContent = `${reg.canonical_name || reg.display_name || reg.region_id}`;
          workspaceRegionSelect.appendChild(opt);
        }

        workspaceRegionSelect.addEventListener("change", (e) => {
          switchRegion(e.target.value, true);
        });
      }

      // 确定初始海区：优先从 URL 读取，若无则优先选休斯敦，若无则选第一个有效海区
      const urlParams = new URLSearchParams(window.location.search);
      const urlRegionId = urlParams.get("region_id");
      let selectedReg = availableRegions.find(r => r.region_id === urlRegionId);
      if (!selectedReg) {
        selectedReg = availableRegions.find(r => /houston-galveston/i.test(r.canonical_name || ""))
          || availableRegions.find(r => /houston/i.test(r.canonical_name || ""))
          || availableRegions[0];
      }

      if (selectedReg) {
        currentRegionId = selectedReg.region_id;
        currentRegionData = selectedReg;
        if (workspaceRegionSelect) workspaceRegionSelect.value = selectedReg.region_id;
        await switchRegion(selectedReg.region_id, false);
      }
    } catch (err) {
      console.warn("初始化海区列表失败，采用默认态势加载:", err);
      await loadPortFacilities();
      await loadRemoteSensingOverlays();
      await loadTimelineHistogram();
    }
  }

  async function init() {
    await initRegionsAndWorkspace();
  }

  // ============================================================
  // 20. 地图工作区与智能调查面板
  // ============================================================
  const railVessels = document.getElementById("rail-vessels");
  const railLayers = document.getElementById("rail-layers");
  const railAnalysis = document.getElementById("rail-analysis");
  const aiCopilotToggle = document.getElementById("ai-copilot-toggle");
  const aiCopilotPanel = document.getElementById("ai-copilot-panel");
  const aiCopilotClose = document.getElementById("ai-copilot-close");
  const copilotForm = document.getElementById("copilot-form");
  const copilotQuestion = document.getElementById("copilot-question");
  const copilotSubmit = document.getElementById("copilot-submit");
  const copilotThread = document.getElementById("copilot-thread");
  const copilotContextText = document.getElementById("copilot-context-text");
  const layerDock = document.getElementById("layer-dock");
  let copilotBusy = false;
  let copilotConversationId = null;
  let copilotRegionId = null;
  let copilotCloseTimer = null;

  function setMapTool(tool = "browse") {
    activeMapTool = tool;
    const probeActive = tool === "weather-probe";
    railWeatherProbe?.classList.toggle("is-active", probeActive);
    railWeatherProbe?.setAttribute("aria-pressed", String(probeActive));
    if (mapToolStatus) mapToolStatus.hidden = !probeActive;
    document.getElementById("map")?.classList.toggle("map-probe-active", probeActive);
  }

  function deactivateWeatherProbe() {
    setMapTool("browse");
    weatherPickerCard.hidden = true;
    weatherProbeLayerGroup.clearLayers();
    weatherProbeMarker = null;
  }

  function setRailActive(target) {
    [railVessels, railLayers, railAnalysis].forEach(item => item?.classList.toggle("is-active", item === target));
  }

  function openCopilot() {
    deactivateWeatherProbe();
    if (window.innerWidth < 1000) sidebarVessels.classList.add("collapsed");
    layerDock?.classList.remove("workspace-open");
    railLayers?.classList.remove("is-active");
    // 调查工作台独占右侧信息区，避免与目标、气象、GIS 抽屉叠在一起。
    vesselCard.hidden = true;
    weatherPickerCard.hidden = true;
    if (gisDrawer) gisDrawer.hidden = true;
    gisHighlightLayerGroup.clearLayers();
    if (copilotCloseTimer) {
      clearTimeout(copilotCloseTimer);
      copilotCloseTimer = null;
    }
    aiCopilotPanel.hidden = false;
    requestAnimationFrame(() => aiCopilotPanel.classList.add("is-open"));
    setRailActive(railAnalysis);
    copilotQuestion?.focus();
  }

  function closeCopilot() {
    aiCopilotPanel.classList.remove("is-open");
    if (copilotCloseTimer) clearTimeout(copilotCloseTimer);
    copilotCloseTimer = setTimeout(() => {
      aiCopilotPanel.hidden = true;
      copilotCloseTimer = null;
    }, 180);
  }

  railVessels?.addEventListener("click", () => {
    deactivateWeatherProbe();
    const opening = sidebarVessels.classList.contains("collapsed");
    if (!opening) {
      sidebarVessels.classList.add("collapsed");
      setRailActive(null);
      return;
    }
    sidebarVessels.classList.remove("collapsed");
    updateSidebarVessels();
    layerDock?.classList.remove("workspace-open");
    closeCopilot();
    setRailActive(railVessels);
  });
  railLayers?.addEventListener("click", () => {
    deactivateWeatherProbe();
    if (window.innerWidth < 1000) sidebarVessels.classList.add("collapsed");
    const opening = !layerDock.classList.contains("workspace-open");
    layerDock.classList.toggle("workspace-open", opening);
    if (opening) dockBody.style.display = "flex";
    closeCopilot();
    setRailActive(opening ? railLayers : null);
  });
  railWeatherProbe?.addEventListener("click", () => {
    if (activeMapTool === "weather-probe") {
      deactivateWeatherProbe();
      return;
    }
    closeCopilot();
    layerDock?.classList.remove("workspace-open");
    setRailActive(null);
    setMapTool("weather-probe");
  });
  [railAnalysis, aiCopilotToggle].forEach(button => button?.addEventListener("click", openCopilot));
  aiCopilotClose?.addEventListener("click", () => {
    closeCopilot();
    setRailActive(null);
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && aiCopilotPanel && !aiCopilotPanel.hidden) {
      closeCopilot();
      setRailActive(sidebarVessels.classList.contains("collapsed") ? null : railVessels);
    }
  });

  document.getElementById("copilot-starters")?.addEventListener("click", event => {
    const button = event.target.closest("[data-question]");
    if (!button) return;
    copilotQuestion.value = button.dataset.question;
    copilotQuestion.focus();
  });

  function appendCopilotMessage(kind, text, meta = "") {
    const item = document.createElement("article");
    item.className = `copilot-message ${kind}`;
    const label = document.createElement("span");
    label.className = "copilot-message-label";
    label.textContent = kind === "user" ? "你的问题" : "海事智能体";
    const body = document.createElement("div");
    body.className = "copilot-message-body";
    body.textContent = text;
    item.append(label, body);
    if (meta) {
      const foot = document.createElement("small");
      foot.textContent = meta;
      item.append(foot);
    }
    copilotThread.append(item);
    copilotThread.scrollTop = copilotThread.scrollHeight;
    return body;
  }

  async function resolveCopilotRegion() {
    try {
      const response = await fetch("/regions");
      if (!response.ok) return;
      const payload = await response.json();
      const regions = payload.regions || payload.items || payload || [];
      const preferred = regions.find(item => /houston-galveston/i.test(item.canonical_name || item.region_label || ""))
        || regions.find(item => /houston/i.test(item.canonical_name || item.region_label || ""));
      copilotRegionId = preferred?.region_id || null;
      if (preferred && copilotContextText) {
        copilotContextText.textContent = `${preferred.canonical_name || preferred.region_label} · ${document.getElementById("gfw-current-time-text")?.textContent || "当前时间"}`;
      }
    } catch (_) { /* 地图仍可独立使用。 */ }
  }

  copilotForm?.addEventListener("submit", async event => {
    event.preventDefault();
    const question = copilotQuestion.value.trim();
    if (!question || copilotBusy) return;
    copilotBusy = true;
    copilotSubmit.disabled = true;
    copilotQuestion.value = "";
    copilotThread.querySelector(".copilot-welcome")?.remove();
    appendCopilotMessage("user", question);
    const pending = appendCopilotMessage("assistant", "正在检索当前区域的多源证据…");
    try {
      const currentTime = document.getElementById("gfw-current-time-text")?.textContent || "";
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: `${question}\n\n地图上下文：当前显示时间为 ${currentTime}。`,
          ...(copilotConversationId ? { conversation_id: copilotConversationId } : {}),
          ...(copilotRegionId ? { region_id: copilotRegionId } : {}),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      copilotConversationId = data.conversation_id || copilotConversationId;
      pending.textContent = data.answer || data.response || "智能体未返回可展示的文本。";
      const evidenceCount = Array.isArray(data.evidence) ? data.evidence.length : 0;
      if (evidenceCount) pending.parentElement.querySelector("small")?.remove();
      if (evidenceCount) {
        const foot = document.createElement("small");
        foot.textContent = `已核验 ${evidenceCount} 条数据证据`;
        pending.parentElement.append(foot);
      }
    } catch (error) {
      pending.textContent = `调查失败：${error.message || error}`;
      pending.parentElement.classList.add("is-error");
    } finally {
      copilotBusy = false;
      copilotSubmit.disabled = false;
      copilotQuestion.focus();
    }
  });

  // 默认把专业图层控制收进工具入口，让地图成为第一视觉层级。
  layerDock.classList.remove("workspace-open");
  if (maritimeLegendBody && btnLegendToggle) {
    maritimeLegendBody.classList.add("collapsed");
    btnLegendToggle.textContent = "+";
  }
  makePanelDraggable(layerDock, layerDock?.querySelector(".dock-header"));
  makePanelDraggable(weatherPickerCard, weatherPickerCard?.querySelector(".card-header"));
  makePanelDraggable(vesselCard, vesselCard?.querySelector(".card-header"));
  makePanelDraggable(gisDrawer, gisDrawer?.querySelector(".card-header"));
  makePanelDraggable(maritimeLegendCard, maritimeLegendHeader);

  // ==========================================================================
  // 态势研判简报与数据审计指挥抽屉 (Situation Intelligence Briefing Drawer)
  // ==========================================================================
  const btnOpenBriefing = document.getElementById("btn-open-briefing");
  const briefingDrawer = document.getElementById("briefing-drawer");
  const btnCloseBriefing = document.getElementById("btn-close-briefing");
  const btnCopyBriefing = document.getElementById("btn-copy-briefing");
  const btnPrintBriefing = document.getElementById("btn-print-briefing");
  const briefingLoadingState = document.getElementById("briefing-loading-state");
  const briefingDisplay = document.getElementById("briefing-display");
  const briefingRegionSub = document.getElementById("briefing-region-sub");
  const auditGradeBox = document.getElementById("audit-grade-box");
  const briefingGradeLetter = document.getElementById("briefing-grade-letter");
  const briefingGradeTitle = document.getElementById("briefing-grade-title");
  const briefingConfidenceText = document.getElementById("briefing-confidence-text");
  const briefingReportId = document.getElementById("briefing-report-id");
  const briefingGeneratedAt = document.getElementById("briefing-generated-at");
  const briefingRegionName = document.getElementById("briefing-region-name");
  const briefingModalitiesGrid = document.getElementById("briefing-modalities-grid");
  const briefingLimitationsBox = document.getElementById("briefing-limitations-box");
  const briefingLimitationsList = document.getElementById("briefing-limitations-list");
  const briefingKpiGrid = document.getElementById("briefing-kpi-grid");
  const briefingMarkdownBody = document.getElementById("briefing-markdown-body");

  let currentBriefingMarkdown = "";
  let isBriefingLoading = false;

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatInlineMarkdown(str) {
    let text = escapeHtml(str);
    // **粗体**
    text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // `代码`
    text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
    return text;
  }

  function renderMarkdownToHtml(mdText) {
    if (!mdText) return "<p>暂无研判内容。</p>";
    const lines = mdText.split("\n");
    const htmlParts = [];
    let inList = false;
    let inBlockquote = false;

    for (let rawLine of lines) {
      const line = rawLine.trim();
      if (!line) {
        if (inList) { htmlParts.push("</ul>"); inList = false; }
        if (inBlockquote) { htmlParts.push("</blockquote>"); inBlockquote = false; }
        continue;
      }

      if (line.startsWith("#### ")) {
        if (inList) { htmlParts.push("</ul>"); inList = false; }
        if (inBlockquote) { htmlParts.push("</blockquote>"); inBlockquote = false; }
        htmlParts.push(`<h4>${escapeHtml(line.slice(5))}</h4>`);
      } else if (line.startsWith("### ")) {
        if (inList) { htmlParts.push("</ul>"); inList = false; }
        if (inBlockquote) { htmlParts.push("</blockquote>"); inBlockquote = false; }
        htmlParts.push(`<h3>${escapeHtml(line.slice(4))}</h3>`);
      } else if (line.startsWith("## ")) {
        if (inList) { htmlParts.push("</ul>"); inList = false; }
        if (inBlockquote) { htmlParts.push("</blockquote>"); inBlockquote = false; }
        htmlParts.push(`<h2>${escapeHtml(line.slice(3))}</h2>`);
      } else if (line.startsWith("# ")) {
        if (inList) { htmlParts.push("</ul>"); inList = false; }
        if (inBlockquote) { htmlParts.push("</blockquote>"); inBlockquote = false; }
        htmlParts.push(`<h1>${escapeHtml(line.slice(2))}</h1>`);
      } else if (line.startsWith("> ")) {
        if (inList) { htmlParts.push("</ul>"); inList = false; }
        if (!inBlockquote) { htmlParts.push("<blockquote>"); inBlockquote = true; }
        htmlParts.push(`<p>${formatInlineMarkdown(line.slice(2))}</p>`);
      } else if (line.startsWith("- ") || line.startsWith("* ")) {
        if (inBlockquote) { htmlParts.push("</blockquote>"); inBlockquote = false; }
        if (!inList) { htmlParts.push("<ul>"); inList = true; }
        htmlParts.push(`<li>${formatInlineMarkdown(line.slice(2))}</li>`);
      } else {
        if (inList) { htmlParts.push("</ul>"); inList = false; }
        if (inBlockquote) { htmlParts.push("</blockquote>"); inBlockquote = false; }
        htmlParts.push(`<p>${formatInlineMarkdown(line)}</p>`);
      }
    }

    if (inList) htmlParts.push("</ul>");
    if (inBlockquote) htmlParts.push("</blockquote>");
    return htmlParts.join("");
  }

  function renderModalityCards(audit) {
    if (!briefingModalitiesGrid || !audit || !audit.modalities) return;
    const mods = audit.modalities;
    if (Array.isArray(mods)) {
      briefingModalitiesGrid.innerHTML = mods.map(m => {
        const isAvail = m.status === "available" || m.status === "active";
        const statusClass = isAvail ? "status-active" : "status-missing";
        const statusText = isAvail ? "🟢 完备接入" : "⚪ 模态缺失";
        return `
          <div class="modality-card ${statusClass}">
            <div class="modality-header">
              <span>${escapeHtml(m.name)}</span>
              <span class="modality-status-tag">${statusText}</span>
            </div>
            <div class="modality-detail">${escapeHtml(m.details || m.badge || "")}</div>
          </div>
        `;
      }).join("");
    }
  }

  function renderBriefingKpis(kpi) {
    if (!briefingKpiGrid || !kpi) return;
    const total = kpi.total_vessels ?? kpi.total_mmsi ?? 0;
    const underway = kpi.underway_vessels ?? 0;
    const moored = kpi.moored_vessels ?? Math.max(0, total - underway);
    const underwayRatio = total > 0 ? ((underway / total) * 100).toFixed(1) + "%" : "0%";
    const maxSog = kpi.max_sog_knots ?? 0;
    const darks = kpi.dark_vessel_candidates ?? kpi.dark_target_candidates ?? 0;
    const maxWind = kpi.max_wind_speed ?? kpi.max_wind_speed_mps ?? 0;
    const minP = kpi.min_pressure ?? kpi.min_pressure_hpa ?? "--";

    const items = [
      { label: "全域通航总量", val: `${total} 艘`, sub: `在航率 ${underwayRatio}` },
      { label: "在泊/系泊避风", val: `${moored} 艘`, sub: "动静态比率审计" },
      { label: "极大航速", val: `${maxSog} 节`, sub: "单船峰值航速" },
      { label: "雷达暗标候选", val: `${darks} 艘`, sub: "未报 AIS 强反射目标" },
      { label: "全域极大风速", val: `${maxWind} m/s`, sub: `气压 ${minP} hPa` },
    ];
    briefingKpiGrid.innerHTML = items.map(it => `
      <div class="briefing-kpi-card">
        <span class="briefing-kpi-label">${it.label}</span>
        <strong class="briefing-kpi-val">${it.val}</strong>
        <span class="briefing-kpi-sub">${it.sub}</span>
      </div>
    `).join("");
  }

  function renderLimitations(limitations) {
    if (!briefingLimitationsList || !briefingLimitationsBox) return;
    const badge = briefingLimitationsBox.querySelector(".limitations-badge");
    if (!limitations || limitations.length === 0) {
      briefingLimitationsBox.style.borderLeftColor = "#34d399";
      if (badge) {
        badge.textContent = "✅ 多源交叉审计认证";
        badge.style.color = "#34d399";
      }
      briefingLimitationsList.innerHTML = "<li>各情报模态时空全维对齐，证据链置信度极高，研判结论具备直接指挥参考价值。</li>";
      return;
    }
    briefingLimitationsBox.style.borderLeftColor = "#fbbf24";
    if (badge) {
      badge.textContent = "⚠️ 态势研判边界与限度声明";
      badge.style.color = "#fbbf24";
    }
    briefingLimitationsList.innerHTML = limitations.map(lim => `<li>${escapeHtml(lim)}</li>`).join("");
  }

  async function openBriefingDrawer() {
    if (!briefingDrawer) return;
    briefingDrawer.removeAttribute("hidden");
    requestAnimationFrame(() => briefingDrawer.classList.add("open"));

    if (isBriefingLoading) return;
    isBriefingLoading = true;

    // 切换至加载动画
    briefingLoadingState?.removeAttribute("hidden");
    briefingDisplay?.setAttribute("hidden", "");

    const step1 = document.getElementById("pipeline-step-1");
    const step2 = document.getElementById("pipeline-step-2");
    const step3 = document.getElementById("pipeline-step-3");
    if (step1 && step2 && step3) {
      step1.classList.add("active");
      step2.classList.remove("active");
      step3.classList.remove("active");
    }

    const regionId = workspaceRegionSelect?.value || "";
    const selectedOptText = workspaceRegionSelect?.selectedOptions?.[0]?.text || "休斯敦海区";
    if (briefingRegionSub) {
      briefingRegionSub.textContent = `海区情报自适应审计 · 跨模态时空融合研判 · ${selectedOptText}`;
    }

    const t1 = setTimeout(() => { step2?.classList.add("active"); }, 400);
    const t2 = setTimeout(() => { step3?.classList.add("active"); }, 800);

    try {
      const url = `/api/situation-report?region_id=${encodeURIComponent(regionId)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const report = await res.json();

      currentBriefingMarkdown = report.markdown || "";

      // 报告编号与元数据
      if (briefingReportId) briefingReportId.textContent = report.report_id || "--";
      if (briefingGeneratedAt) briefingGeneratedAt.textContent = report.generated_at_utc || "--";
      if (briefingRegionName) briefingRegionName.textContent = report.region_name || selectedOptText;

      // 评级与置信度
      const grade = report.data_sufficiency?.grade || "B";
      if (auditGradeBox) {
        auditGradeBox.className = `audit-grade-box grade-${grade}`;
      }
      if (briefingGradeLetter) briefingGradeLetter.textContent = grade;
      if (briefingGradeTitle) briefingGradeTitle.textContent = report.data_sufficiency?.grade_title || `${grade} 级态势感知`;
      if (briefingConfidenceText) briefingConfidenceText.textContent = `置信度: ${report.data_sufficiency?.confidence_level || "良好"}`;

      // 四模态体检
      renderModalityCards(report.data_sufficiency);

      // 边界声明
      renderLimitations(report.data_sufficiency?.limitations);

      // KPI 量化卡片
      renderBriefingKpis(report.kpi_metrics);

      // Markdown 正文渲染
      if (briefingMarkdownBody) {
        briefingMarkdownBody.innerHTML = renderMarkdownToHtml(report.markdown);
      }
    } catch (err) {
      console.error("生成态势研判简报失败:", err);
      if (briefingMarkdownBody) {
        briefingMarkdownBody.innerHTML = `<p style="color:#ef4444;">简报生成失败：${escapeHtml(err.message || String(err))}</p>`;
      }
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      isBriefingLoading = false;
      briefingLoadingState?.setAttribute("hidden", "");
      briefingDisplay?.removeAttribute("hidden");
    }
  }

  function closeBriefingDrawer() {
    if (!briefingDrawer) return;
    briefingDrawer.classList.remove("open");
    setTimeout(() => {
      if (!briefingDrawer.classList.contains("open")) {
        briefingDrawer.setAttribute("hidden", "");
      }
    }, 380);
  }

  // 事件监听绑定
  btnOpenBriefing?.addEventListener("click", openBriefingDrawer);
  btnCloseBriefing?.addEventListener("click", closeBriefingDrawer);

  btnCopyBriefing?.addEventListener("click", async () => {
    if (!currentBriefingMarkdown) return;
    try {
      await navigator.clipboard.writeText(currentBriefingMarkdown);
      const originalText = btnCopyBriefing.innerHTML;
      btnCopyBriefing.innerHTML = "<span>✅</span> 已复制 Markdown";
      setTimeout(() => { btnCopyBriefing.innerHTML = originalText; }, 2000);
    } catch (e) {
      alert("复制失败，请手动选中文本复制。");
    }
  });

  btnPrintBriefing?.addEventListener("click", () => {
    window.print();
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && briefingDrawer && !briefingDrawer.hasAttribute("hidden")) {
      closeBriefingDrawer();
    }
  });

  init();
})();
