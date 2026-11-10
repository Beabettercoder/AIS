/* Render evidence-backed track and port GIS cards inside chat answers. */
(function () {
  "use strict";
  const ns = "http://www.w3.org/2000/svg";

  function svgNode(tag, attrs = {}) {
    const element = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, String(value));
    return element;
  }

  function makeCard(title, note) {
    const card = document.createElement("section");
    card.className = "inline-viz";
    const heading = document.createElement("div");
    heading.className = "inline-viz-heading";
    const label = document.createElement("strong");
    label.textContent = title;
    const status = document.createElement("span");
    status.textContent = "正在读取空间数据…";
    heading.append(label, status);
    const frame = document.createElement("div");
    frame.className = "inline-viz-frame";
    const foot = document.createElement("p");
    foot.className = "inline-viz-note";
    foot.textContent = note;
    card.append(heading, frame, foot);
    return { card, status, frame, foot };
  }

  function drawMap(frame, parts, isArea) {
    const points = parts.flatMap(part => part.points).filter(point =>
      Array.isArray(point) && point.length >= 2 && Number.isFinite(Number(point[0])) && Number.isFinite(Number(point[1])));
    if (!points.length) throw new Error("没有可显示的坐标");
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [rawX, rawY] of points) {
      const x = Number(rawX), y = Number(rawY);
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;
    const spanX = Math.max(maxX - minX, isArea ? 0.005 : 0.02);
    const spanY = Math.max(maxY - minY, isArea ? 0.005 : 0.02);
    const scale = Math.min(690 / (spanX * Math.cos(midY * Math.PI / 180)), 270 / spanY);
    const project = ([x, y]) => [380 + (Number(x) - midX) * Math.cos(midY * Math.PI / 180) * scale,
      155 - (Number(y) - midY) * scale];
    const svg = svgNode("svg", { viewBox: "0 0 760 310", role: "img", "aria-label": isArea ? "GIS 空间对象范围" : "船舶小时代表位置航迹" });
    for (let index = 1; index < 5; index++) {
      const x = index * 152, y = index * 62;
      svg.append(svgNode("line", { x1: x, y1: 0, x2: x, y2: 310, class: "inline-viz-grid" }));
      svg.append(svgNode("line", { x1: 0, y1: y, x2: 760, y2: y, class: "inline-viz-grid" }));
    }
    for (const part of parts) {
      const projected = part.points.map(project);
      if (projected.length === 1) {
        svg.append(svgNode("circle", { cx: projected[0][0], cy: projected[0][1], r: 6, class: "inline-viz-point" }));
        continue;
      }
      const path = projected.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)}`).join(" ") +
        (part.closed ? " Z" : "");
      svg.append(svgNode("path", { d: path, class: part.closed ? "inline-viz-area" : "inline-viz-track" }));
    }
    if (!isArea) {
      const first = project(points[0]);
      const last = project(points[points.length - 1]);
      svg.append(svgNode("circle", { cx: first[0], cy: first[1], r: 5, class: "inline-viz-start" }));
      svg.append(svgNode("circle", { cx: last[0], cy: last[1], r: 5, class: "inline-viz-end" }));
    }
    const north = svgNode("text", { x: 728, y: 27, class: "inline-viz-north" });
    north.textContent = "N ↑";
    svg.append(north);
    frame.replaceChildren(svg);
  }

  function geometryParts(geometry) {
    const coordinates = geometry?.coordinates;
    if (!coordinates) return [];
    switch (geometry.type) {
      case "Point": return [{ points: [coordinates], closed: false }];
      case "LineString": return [{ points: coordinates, closed: false }];
      case "MultiLineString": return coordinates.map(points => ({ points, closed: false }));
      case "Polygon": return coordinates.map(points => ({ points, closed: true }));
      case "MultiPolygon": return coordinates.flatMap(polygon => polygon.map(points => ({ points, closed: true })));
      default: return [];
    }
  }

  async function renderTrack(evidence, target) {
    const title = evidence.vessel_name || `MMSI ${evidence.mmsi}`;
    const view = makeCard(`${title} · ${evidence.date_utc} UTC`,
      "线段仅连接已观测位置，不代表确切航路。经纬度网格不是航海底图。");
    target.append(view.card);
    const selector = document.createElement("select");
    selector.className = "inline-viz-selector";
    for (const [value, label] of [["message", "逐报文 AIS（默认）"], ["hourly", "小时代表点（派生）"]]) {
      const option = document.createElement("option"); option.value = value; option.textContent = label;
      selector.append(option);
    }
    selector.value = String(evidence.evidence_id || "").startsWith("message_track:") || evidence.granularity === "message" ? "message" : "hourly";
    selector.setAttribute("aria-label", "航迹精度");
    view.card.querySelector(".inline-viz-heading").append(selector);
    const link = document.createElement("a");
    link.textContent = "在航迹页查看";
    view.foot.append(" ", link);
    let requestNumber = 0;
    async function load() {
      const current = ++requestNumber;
      try {
        const imported = String(evidence.evidence_id || "").startsWith("ingested_track:") && evidence.source_id;
        const url = imported
          ? `/data-sources/${encodeURIComponent(evidence.source_id)}/tracks/${encodeURIComponent(evidence.date_utc)}/${encodeURIComponent(evidence.mmsi)}?granularity=${selector.value}`
          : `/tracks/${encodeURIComponent(evidence.date_utc)}/${encodeURIComponent(evidence.mmsi)}?granularity=${selector.value}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (current !== requestNumber) return;
        const messageMode = selector.value === "message";
        const parts = [];
        if (messageMode) {
          const groups = new Map();
          for (const point of data.points) {
            const segment = point.segment ?? 0;
            if (!groups.has(segment)) groups.set(segment, []);
            groups.get(segment).push([point.lon, point.lat]);
          }
          for (const points of groups.values()) parts.push({ points, closed: false });
          view.status.textContent = `${data.observation_count} 条观测 · 展示 ${data.display_point_count ?? data.points.length} 点`;
          view.foot.firstChild.textContent = data.sampled_for_display
            ? "逐报文观测点；显示点经过抽样，数据库保留全部记录；超过 15 分钟的空缺断线。"
            : "逐报文观测点；超过 15 分钟的空缺断线。";
        } else {
          parts.push({ points: data.points.map(point => [point.representative_lon ?? point.lon, point.representative_lat ?? point.lat]), closed: false });
          view.status.textContent = `${data.hour_points ?? data.observation_count} 个小时代表点 · MMSI ${data.mmsi}`;
          view.foot.firstChild.textContent = "每小时代表位置连线；线段不代表确切航路。";
        }
        drawMap(view.frame, parts, false);
        if (imported) { link.hidden = true; }
        else { link.hidden = false; link.href = `/tracks-map?date_utc=${encodeURIComponent(data.date_utc)}&mmsi=${encodeURIComponent(data.mmsi)}&granularity=${selector.value}`; }
      } catch (error) {
        if (current === requestNumber) view.status.textContent = `航迹图读取失败：${error.message}`;
      }
    }
    selector.addEventListener("change", load);
    load();
  }

  async function renderArea(evidence, target) {
    const view = makeCard(evidence.object_name || evidence.object_id,
      "GIS 来源对象的经纬度几何，仅供态势展示，不能用于航海导航；图层来源日期可能与 AIS 日期不同。");
    target.append(view.card);
    try {
      const url = evidence.external_source_id && evidence.external_feature_id
        ? `/gis/external/${encodeURIComponent(evidence.external_source_id)}/${encodeURIComponent(evidence.external_feature_id)}`
        : `/gis-objects/${encodeURIComponent(evidence.object_id)}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const feature = await response.json();
      drawMap(view.frame, geometryParts(feature.geometry), true);
      const stats = evidence.ais_hourly;
      const rawDate = String(feature.properties.source_date || "");
      const sourceDate = /^\d{8}$/.test(rawDate) ? `${rawDate.slice(0, 4)}-${rawDate.slice(4, 6)}-${rawDate.slice(6)}` : (rawDate || "未标注");
      view.status.textContent = `${feature.properties.object_type || "GIS 对象"} · 来源日期 ${sourceDate}` +
        (stats?.observed_hours ? ` · ${evidence.date_utc} UTC 有 ${stats.observed_hours} 个 AIS 匹配小时` : "");
      if (stats?.observed_hours && stats.mean_unique_mmsi != null) {
        view.foot.prepend(`${evidence.date_utc} 平均 ${stats.mean_unique_mmsi} 艘/小时（按主对象匹配）。`);
      }
    } catch (error) {
      view.status.textContent = `海域图读取失败：${error.message}`;
    }
  }

  function renderRemoteSensing(evidence, target) {
    const view = makeCard(`${evidence.collection} · ${evidence.datetime_utc} UTC`,
      "卫星场景仅供对照；云层和成像时刻限制解读，不能据此认定船舶位置。");
    target.append(view.card);
    const assets = Array.isArray(evidence.local_assets) ? evidence.local_assets : [];
    if (assets.some(item => String(item.asset_key).endsWith("_thumbnail.jpg"))) {
      const img = document.createElement("img");
      img.src = `/remote-sensing/${encodeURIComponent(evidence.scene_id)}/preview`;
      img.alt = `${evidence.collection} 场景缩略图，可能受云层遮挡`;
      img.style.cssText = "width:100%;height:100%;object-fit:contain;background:#101c29";
      view.frame.replaceChildren(img);
      view.status.textContent = `场景云量 ${evidence.scene_cloud_cover_pct ?? "未知"}% · ${evidence.local_visual_available ? "有本地真彩色局部裁剪" : "有目录及部分本地资产"}`;
    } else {
      view.status.textContent = "仅有场景目录元数据，尚无本地可视影像";
    }
    const link = document.createElement("a");
    link.href = evidence.catalog_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "查看来源场景";
    view.foot.append(" ", link);
  }

  function renderWindField(evidence, target) {
    const isNowcast = String(evidence.evidence_id || "").startsWith("wind_nowcast") || Boolean(evidence.nowcast_summary);
    const isPoint = String(evidence.evidence_id || "").startsWith("wind_field_point") || evidence.query_mode === "point_interpolation";
    const timeLabel = evidence.valid_time_utc || evidence.base_time_utc || "2024-07-08T12:00:00Z";
    const cleanTime = timeLabel.replace("T", " ").replace("Z", " UTC");

    let title = `连续海况风场 · ${cleanTime}`;
    let noteText = "基于 Barnes 客观分析高斯距离核与正交矢量分解合成，彻底消除 0°/360° 翻转失真与孤立测站空间断崖。";

    if (isNowcast) {
      title = `海况时效短临外推预测 · ${cleanTime} +${evidence.lead_hours || 1.0}h`;
      noteText = "基于时空微元变率与阻尼平流动力学方程外推，包含气压骤降与极端大风预警。";
    } else if (isPoint) {
      title = `海况定点气象插值 · ${cleanTime}`;
    }

    const view = makeCard(title, noteText);
    target.append(view.card);

    // Header metrics
    if (isNowcast && evidence.nowcast_summary) {
      const s = evidence.nowcast_summary;
      view.status.textContent = `外推风速 ${Number(s.extrapolated_speed_mps || 0).toFixed(1)} m/s · 预警 [${s.alert_level || "平稳"}] · 气压趋势 ${s.pressure_trend || "平稳"}`;
    } else if (isPoint && evidence.result) {
      const r = evidence.result;
      view.status.textContent = `风速 ${Number(r.wind_speed_mps || 0).toFixed(1)} m/s (${Number(r.wind_direction_deg || 0).toFixed(0)}°) · 气压 ${Number(r.air_pressure_hpa || 0).toFixed(1)} hPa · 置信度 ${(Number(r.spatial_confidence || 1) * 100).toFixed(0)}%`;
    } else if (evidence.grid_field && evidence.grid_field.summary) {
      const g = evidence.grid_field.summary;
      view.status.textContent = `全域极大 ${Number(g.max_wind_speed_mps || 0).toFixed(1)} m/s · 全域平均 ${Number(g.avg_wind_speed_mps || 0).toFixed(1)} m/s · ${evidence.active_station_count || 8} 个测站`;
    } else {
      view.status.textContent = "已提取气象正交风场态势";
    }

    // Mini Canvas Frame
    const container = document.createElement("div");
    container.style.cssText = "position:relative; width:100%; height:230px; background:#070d17; border-radius:8px; overflow:hidden;";

    const miniCanvas = document.createElement("canvas");
    miniCanvas.style.cssText = "width:100%; height:100%; display:block;";
    container.appendChild(miniCanvas);

    // Overlay CTA button
    const cta = document.createElement("a");
    cta.href = "/wind-map";
    cta.target = "_blank";
    cta.className = "inline-viz-wind-btn";
    cta.textContent = "⛶ 全屏展开海况风场大屏";
    cta.style.cssText = "position:absolute; bottom:12px; right:12px; padding:6px 14px; background:rgba(56,189,248,0.22); border:1px solid rgba(56,189,248,0.5); color:#38bdf8; border-radius:6px; font-size:12px; text-decoration:none; font-weight:600; backdrop-filter:blur(4px); transition:all 0.2s;";
    cta.onmouseenter = () => { cta.style.background = "rgba(56,189,248,0.4)"; };
    cta.onmouseleave = () => { cta.style.background = "rgba(56,189,248,0.22)"; };
    container.appendChild(cta);

    // Floating badges for quick facts
    const overlayBadges = document.createElement("div");
    overlayBadges.style.cssText = "position:absolute; top:10px; left:12px; display:flex; gap:8px; font-size:11px;";

    let badgeText = "台风贝丽尔实时态势";
    if (evidence.grid_field?.summary) {
      badgeText = `极大风速: ${Number(evidence.grid_field.summary.max_wind_speed_mps).toFixed(1)} m/s`;
    } else if (evidence.nowcast_summary) {
      badgeText = `外推风速: ${Number(evidence.nowcast_summary.extrapolated_speed_mps).toFixed(1)} m/s`;
    }

    overlayBadges.innerHTML = `
      <span style="background:rgba(15,23,42,0.85); color:#38bdf8; padding:3px 8px; border-radius:4px; border:1px solid rgba(56,189,248,0.3);">${badgeText}</span>
      <span style="background:rgba(15,23,42,0.85); color:#4ade80; padding:3px 8px; border-radius:4px; border:1px solid rgba(74,222,128,0.3);">Barnes 连续正交流场</span>
    `;
    container.appendChild(overlayBadges);

    view.frame.replaceChildren(container);

    // Link in footer
    const link = document.createElement("a");
    link.href = "/wind-map";
    link.target = "_blank";
    link.textContent = "在风场态势大屏中查看完整演变与雷达探针 →";
    view.foot.append(" ", link);

    // Initialize mini particle engine if available
    if (window.WindParticleEngine) {
      requestAnimationFrame(() => {
        try {
          const miniEngine = new window.WindParticleEngine(miniCanvas, {
            particleCount: 500,
            speedFactor: 0.35,
            clearMode: "solid",
            lineWidth: 1.5,
          });

          const onGridLoaded = (gridData) => {
            const bbox = gridData.bbox || [-95.8, 28.8, -94.3, 30.1];
            const [west, south, east, north] = bbox;
            miniEngine.setProjections(
              (lon, lat) => {
                const px = ((lon - west) / (east - west)) * miniEngine.width;
                const py = ((north - lat) / (north - south)) * miniEngine.height;
                return [px, py];
              },
              (px, py) => {
                const lon = west + (px / miniEngine.width) * (east - west);
                const lat = north - (py / miniEngine.height) * (north - south);
                return [lon, lat];
              }
            );
            miniEngine.setData(gridData);
            miniEngine.start();
          };

          if (evidence.grid_field && evidence.grid_field.cells) {
            onGridLoaded(evidence.grid_field);
          } else {
            const queryTime = evidence.valid_time_utc || evidence.base_time_utc || "2024-07-08T12:00:00Z";
            fetch(`/api/wind-field-grid?valid_time_utc=${encodeURIComponent(queryTime)}&cols=12&rows=12`)
              .then(res => res.json())
              .then(data => {
                if (data.grid_field && data.grid_field.cells) {
                  onGridLoaded(data.grid_field);
                }
              })
              .catch(e => console.warn("微缩风场加载跳过:", e));
          }
        } catch (err) {
          console.warn("微缩风场渲染跳过:", err);
        }
      });
    }
  }

  function renderAnswerVisualizations(answer, data) {
    const evidence = Array.isArray(data.evidence) ? data.evidence : [];
    const trackSeen = new Set(), areaSeen = new Set(), sceneSeen = new Set(), windSeen = new Set();
    const tracks = [], areas = [], scenes = [], winds = [];
    for (const item of evidence) {
      if (item?.mmsi && item?.date_utc && /^(track|message_track|ingested_track):/.test(String(item.evidence_id || ""))) {
        const key = `${item.date_utc}/${item.mmsi}`;
        if (!trackSeen.has(key)) { trackSeen.add(key); tracks.push(item); }
      } else if (item?.object_id && String(item.evidence_id || "").startsWith("gis:")) {
        if (!areaSeen.has(item.object_id)) { areaSeen.add(item.object_id); areas.push(item); }
      } else if (item?.scene_id && String(item.evidence_id || "").startsWith("remote_sensing:")) {
        if (!sceneSeen.has(item.scene_id)) { sceneSeen.add(item.scene_id); scenes.push(item); }
      } else if (
        String(item?.evidence_id || "").startsWith("wind_field") ||
        String(item?.evidence_id || "").startsWith("wind_nowcast") ||
        item?.grid_field || item?.nowcast_summary || item?.query_mode === "continuous_grid_field"
      ) {
        const key = item.evidence_id || `${item.valid_time_utc || item.base_time_utc}`;
        if (!windSeen.has(key)) { windSeen.add(key); winds.push(item); }
      }
    }
    if (!tracks.length && !areas.length && !scenes.length && !winds.length) return;
    const group = document.createElement("div");
    group.className = "inline-viz-group";
    answer.append(group);
    for (const item of tracks.slice(0, 3)) renderTrack(item, group);
    for (const item of areas.slice(0, 2)) renderArea(item, group);
    for (const item of scenes.slice(0, 2)) renderRemoteSensing(item, group);
    for (const item of winds.slice(0, 2)) renderWindField(item, group);
  }

  window.renderAnswerVisualizations = renderAnswerVisualizations;
})();
