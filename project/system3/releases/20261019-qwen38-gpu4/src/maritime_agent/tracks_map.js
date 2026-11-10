(() => {
  "use strict";
  const day = document.getElementById("day");
  const vesselClass = document.getElementById("vessel-class");
  const mmsi = document.getElementById("mmsi");
  const granularity = document.getElementById("granularity");
  const limit = document.getElementById("limit");
  const status = document.getElementById("status");
  const detail = document.getElementById("track-detail");
  const basemapStatus = document.getElementById("basemap-status");
  const colors = { cargo: "#0d7f87", tanker: "#9761ac", tug: "#d28b27",
    fishing: "#3783ba", passenger: "#54a96b", high_speed_craft: "#d46c78",
    other_or_unknown: "#687d96" };
  let selectedFeature = null;
  let selectedLine = null;
  let selectionNumber = 0;
  let allBounds = null;
  let selectedBounds = null;

  if (!window.L) {
    status.textContent = "地图组件加载失败，请刷新页面。";
    return;
  }
  const map = L.map("map", { preferCanvas: true, zoomControl: true });
  map.setView([29.65, -95.05], 10);
  const background = L.layerGroup().addTo(map);
  const selection = L.layerGroup().addTo(map);
  const tiles = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    minZoom: 5, maxZoom: 18,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap contributors</a>'
  }).addTo(map);
  tiles.on("tileerror", () => {
    basemapStatus.hidden = false;
    basemapStatus.textContent = "底图暂时无法加载；航迹仍可查看。请检查网络连接。";
  });
  tiles.on("tileload", () => { basemapStatus.hidden = true; });

  function textLine(tag, value) {
    const element = document.createElement(tag);
    element.textContent = value;
    return element;
  }
  function showDetail(data, observed) {
    detail.replaceChildren();
    detail.append(textLine("h3", data.vessel_name || `MMSI ${data.mmsi}`));
    const meta = textLine("div", observed
      ? `MMSI ${data.mmsi} · ${data.vessel_class || "未知船型"} · ${data.date_utc} UTC · ${data.observation_count} 条观测记录，地图展示 ${data.display_point_count} 点、${data.gap_segment_count} 个连续片段${data.sampled_for_display ? "（显示已抽样，数据库保留全部记录）" : ""}。来源：${data.source}。`
      : `MMSI ${data.mmsi} · ${data.vessel_class || "未知船型"} · ${data.date_utc} UTC · ${data.hour_points} 个小时代表点。来源：${data.source}。`);
    meta.className = "detail-meta";
    detail.append(meta);
    const table = document.createElement("table");
    const head = document.createElement("thead");
    const heading = document.createElement("tr");
    for (const label of observed ? ["UTC 时间", "纬度", "经度", "航速(节)"] : ["UTC 小时", "纬度", "经度", "状态"]) {
      heading.append(textLine("th", label));
    }
    head.append(heading); table.append(head);
    const body = document.createElement("tbody");
    for (const point of data.points.slice(0, 100)) {
      const row = document.createElement("tr");
      const cells = observed
        ? [point.timestamp_utc, Number(point.lat).toFixed(5), Number(point.lon).toFixed(5), point.sog_kn ?? "—"]
        : [point.timestamp_hour_utc, Number(point.representative_lat).toFixed(5),
          Number(point.representative_lon).toFixed(5), point.motion_state || "—"];
      for (const value of cells) row.append(textLine("td", String(value)));
      body.append(row);
    }
    table.append(body); detail.append(table);
    if (data.points.length > 100) {
      const note = textLine("p", "表格仅列出前 100 个显示点。");
      note.className = "detail-meta";
      detail.append(note);
    }
  }
  function popupFor(point, observed) {
    const box = document.createElement("div");
    box.append(textLine("strong", observed ? point.timestamp_utc : point.timestamp_hour_utc));
    const lat = observed ? point.lat : point.representative_lat;
    const lon = observed ? point.lon : point.representative_lon;
    box.append(textLine("div", `${Math.abs(Number(lat)).toFixed(5)}°${lat >= 0 ? "N" : "S"}, ${Math.abs(Number(lon)).toFixed(5)}°${lon >= 0 ? "E" : "W"}`));
    box.append(textLine("div", observed ? `航速 ${point.sog_kn ?? "—"} 节 · 航向 ${point.cog_deg ?? "—"}°`
      : `状态 ${point.motion_state || "—"}`));
    return box;
  }
  async function selectTrack(feature, line) {
    if (selectedLine) selectedLine.setStyle({ color: selectedLine.options.baseColor, weight: 3, opacity: .7 });
    selectedFeature = feature;
    selectedLine = line;
    line.setStyle({ color: "#e36a38", weight: 5, opacity: 1 });
    selection.clearLayers();
    const current = ++selectionNumber;
    const props = feature.properties;
    const observed = granularity.value === "message";
    try {
      const url = `/tracks/${encodeURIComponent(props.date_utc)}/${encodeURIComponent(props.mmsi)}?granularity=${granularity.value}`;
      const response = await fetch(url);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      if (current !== selectionNumber) return;
      showDetail(data, observed);
      const locations = [];
      if (observed) {
        const segments = new Map();
        for (const point of data.points) {
          const location = [point.lat, point.lon];
          locations.push(location);
          if (!segments.has(point.segment)) segments.set(point.segment, []);
          segments.get(point.segment).push(location);
        }
        for (const segment of segments.values()) {
          if (segment.length > 1) L.polyline(segment, { color: "#057e87", weight: 3,
            opacity: .95, interactive: false }).addTo(selection);
        }
      } else {
        for (const point of data.points) locations.push([point.representative_lat, point.representative_lon]);
        if (locations.length > 1) L.polyline(locations, { color: "#057e87", weight: 3,
          opacity: .95, interactive: false }).addTo(selection);
      }
      const markerStep = Math.max(1, Math.ceil(data.points.length / 350));
      for (let index = 0; index < data.points.length; index += markerStep) {
        const point = data.points[index];
        const location = observed ? [point.lat, point.lon] : [point.representative_lat, point.representative_lon];
        L.circleMarker(location, { radius: observed ? 3 : 4, color: "#fff", weight: 1,
          fillColor: observed ? "#087f88" : "#e36a38", fillOpacity: 1 })
          .bindPopup(popupFor(point, observed)).addTo(selection);
      }
      if (locations.length) {
        selectedBounds = L.latLngBounds(locations);
        map.fitBounds(selectedBounds.pad(.1), { maxZoom: 15, animate: false });
      }
    } catch (error) {
      if (current === selectionNumber) detail.textContent = `航迹详情读取失败：${error.message}`;
    }
  }
  async function loadTracks() {
    status.textContent = "正在读取航迹…";
    selectionNumber++;
    selectedFeature = null; selectedLine = null; selectedBounds = null;
    background.clearLayers(); selection.clearLayers();
    const params = new URLSearchParams({ date_utc: day.value, limit: limit.value });
    if (vesselClass.value) params.set("vessel_class", vesselClass.value);
    if (mmsi.value.trim()) params.set("mmsi", mmsi.value.trim());
    try {
      const response = await fetch(`/tracks?${params}`);
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      allBounds = L.latLngBounds([]);
      let autoSelect = null;
      for (const feature of data.features) {
        const coordinateParts = feature.geometry.type === "MultiLineString"
          ? feature.geometry.coordinates : [feature.geometry.coordinates];
        const positions = coordinateParts.map(part => part.map(([lon, lat]) => [lat, lon]));
        const baseColor = colors[feature.properties.vessel_class] || colors.other_or_unknown;
        const line = L.polyline(positions, { color: baseColor, baseColor, weight: 3, opacity: .7 });
        const tooltip = textLine("span", `${feature.properties.vessel_name || "未知船名"} · MMSI ${feature.properties.mmsi}`);
        line.bindTooltip(tooltip);
        line.on("click", () => selectTrack(feature, line));
        line.addTo(background);
        allBounds.extend(line.getBounds());
        if (mmsi.value.trim() === feature.properties.mmsi) autoSelect = [feature, line];
      }
      status.textContent = `显示 ${data.returned} 条航迹${data.returned === Number(limit.value) ? "（已达显示上限，可筛选 MMSI 或船型）" : ""}`;
      if (allBounds.isValid()) map.fitBounds(allBounds.pad(.08), { maxZoom: 13, animate: false });
      if (autoSelect) await selectTrack(...autoSelect);
      else detail.textContent = data.returned ? "点击地图中的航迹线查看船舶详情。" : "没有符合筛选条件的日航迹。";
    } catch (error) {
      status.textContent = `读取失败：${error.message}`;
    }
  }
  document.getElementById("reset-zoom").addEventListener("click", () => {
    const bounds = selectedBounds?.isValid() ? selectedBounds : allBounds;
    if (bounds?.isValid()) map.fitBounds(bounds.pad(.1), { maxZoom: 15, animate: false });
    else map.setView([29.65, -95.05], 10);
  });
  document.getElementById("filters").addEventListener("submit", event => {
    event.preventDefault(); loadTracks();
  });
  granularity.addEventListener("change", () => {
    if (selectedFeature && selectedLine) selectTrack(selectedFeature, selectedLine);
  });
  async function init() {
    try {
      const response = await fetch("/catalog");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const catalog = await response.json();
      const first = catalog.time_range_utc[0].slice(0, 10);
      const last = catalog.time_range_utc[1].slice(0, 10);
      const incoming = new URLSearchParams(window.location.search);
      const selectedDay = incoming.get("date_utc") || "2024-07-08";
      for (let cursor = new Date(first + "T00:00:00Z"); cursor <= new Date(last + "T00:00:00Z"); cursor.setUTCDate(cursor.getUTCDate() + 1)) {
        const option = document.createElement("option");
        option.value = cursor.toISOString().slice(0, 10);
        option.textContent = option.value;
        if (option.value === selectedDay) option.selected = true;
        day.append(option);
      }
      if (incoming.get("mmsi")) { mmsi.value = incoming.get("mmsi"); limit.value = "50"; }
      if (incoming.get("granularity") === "message") granularity.value = "message";
      for (const value of catalog.vessel_classes || []) {
        const option = document.createElement("option"); option.value = value; option.textContent = value;
        vesselClass.append(option);
      }
      if (!catalog.track_count) throw new Error("当前数据集尚未导入逐船航迹");
      await loadTracks();
    } catch (error) {
      status.textContent = `地图不可用：${error.message}`;
    }
  }
  init();
})();
