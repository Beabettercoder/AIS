"""Narrow, auditable tool surface exposed to the language model."""

from __future__ import annotations

import contextvars
import json
import re
from typing import Any, Callable

from langchain_core.tools import tool

from .repository import DataRepository, evidence_id


ACTIVE_REGION_VAR: contextvars.ContextVar[str] = contextvars.ContextVar("active_region_var", default="")

KNOWN_REGION_ALIASES = {
    "hau": "region-d3529b060d6d4d65",
    "mobile": "region-0a589e4f05844f47",
    "danmai": "region-96113d130a3e4901",
    "丹麦": "region-96113d130a3e4901",
    "houston": "region-f5812c64b6934007",
    "休斯顿": "region-f5812c64b6934007",
    "houston-galveston": "region-5b33d883d4734d58",
}


def set_active_region(region_id: str) -> None:
    """Set the active study-region context for hard spatial sandboxing."""
    if not region_id:
        ACTIVE_REGION_VAR.set("")
        return
    norm = normalize_region_arg(region_id)
    ACTIVE_REGION_VAR.set(norm)


def get_active_region() -> str:
    """Get the active study-region context ID."""
    return ACTIVE_REGION_VAR.get()


def normalize_region_arg(val: Any) -> str:
    if not val:
        return ""
    text = str(val).strip()
    m = re.search(r"(region-[0-9a-f]{16,36})", text, re.I)
    if m:
        return m.group(1).lower()
    return KNOWN_REGION_ALIASES.get(text.lower(), text)


def normalize_date_arg(val: Any) -> str:
    if not isinstance(val, str):
        return val
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", val)
    if m:
        y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mon:02d}-{d:02d}"
    return val.strip()


def normalize_mmsi_arg(val: Any) -> str:
    if not val:
        return val
    m = re.search(r"\b(\d{9})\b", str(val))
    if m:
        return m.group(1)
    return str(val).strip()


def normalize_tool_arg(k: str, v: Any) -> Any:
    if v is None:
        return v
    if k in ("region_id", "region", "region_label"):
        return normalize_region_arg(v)
    if k in ("day_utc", "date_utc", "baseline_day_utc"):
        return normalize_date_arg(v)
    if k == "mmsi":
        return normalize_mmsi_arg(v)
    if k in ("lon", "lat") and isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def sanitize_tool_payload(obj: Any, max_list_items: int = 10, max_str_len: int = 1500) -> Any:
    """Safely compact large tool payloads to protect model context window and token budget.
    
    Preserves all evidence_ids for citation verification while summarizing or truncating
    huge collections, nested geometries, or verbose raw metadata.
    """
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            if k in ("provenance", "quality") and isinstance(v, dict) and len(str(v)) > 400:
                clean[k] = {"source": v.get("source_title", v.get("source", "recorded"))}
            elif k in ("geometry", "geojson", "coordinates") and isinstance(v, (dict, list)) and len(str(v)) > 500:
                clean[k] = {"type": "GeoJSON/Geometry", "note": "已由地图前端可视化渲染，模型上下文已精简"}
            elif isinstance(v, list) and len(v) > max_list_items:
                kept = []
                for i, item in enumerate(v):
                    if i < max_list_items:
                        kept.append(sanitize_tool_payload(item, max_list_items=5, max_str_len=max_str_len))
                    elif isinstance(item, dict) and "evidence_id" in item:
                        stub = {"evidence_id": item["evidence_id"]}
                        for f in ("modality", "object_type", "predicate", "mmsi", "date_utc", "timestamp_utc"):
                            if f in item:
                                stub[f] = item[f]
                        kept.append(stub)
                clean[k] = kept
                clean["_total_count"] = len(v)
                clean["_truncated_items"] = len(v) - max_list_items
            else:
                clean[k] = sanitize_tool_payload(v, max_list_items=max_list_items, max_str_len=max_str_len)
        return clean
    elif isinstance(obj, list):
        if len(obj) > max_list_items:
            res = [sanitize_tool_payload(x, max_list_items=5, max_str_len=max_str_len) for x in obj[:max_list_items]]
            for item in obj[max_list_items:]:
                if isinstance(item, dict) and "evidence_id" in item:
                    stub = {"evidence_id": item["evidence_id"]}
                    for f in ("modality", "object_type", "predicate", "mmsi", "date_utc"):
                        if f in item:
                            stub[f] = item[f]
                    res.append(stub)
            return res
        return [sanitize_tool_payload(x, max_list_items=max_list_items) for x in obj]
    elif isinstance(obj, str) and len(obj) > max_str_len:
        return obj[:max_str_len] + "... [截断以保护Token预算]"
    return obj


def _json_call(fn: Callable, **kwargs: object) -> str:
    try:
        active_region = get_active_region()
        clean_kwargs = {k: normalize_tool_arg(k, v) for k, v in kwargs.items()}
        if active_region and ("region_id" in clean_kwargs or getattr(fn, "__name__", "") in
                              ("query_situation_units", "situation_cube_overview", "search_fusion_relations",
                               "detect_dark_vessels", "query_wind_field", "query_wind_nowcast", "compare_situation_periods")):
            if not clean_kwargs.get("region_id") or clean_kwargs.get("region_id") != active_region:
                clean_kwargs["region_id"] = active_region
        result = fn(**clean_kwargs)
        sanitized = sanitize_tool_payload(result)
        return json.dumps(sanitized, ensure_ascii=False, default=str)
    except (ValueError, FileNotFoundError, OSError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def build_tools(repo: DataRepository) -> list:
    @tool
    def ais_daily_change(day_utc: str, baseline_day_utc: str = "") -> str:
        """Best first tool for broad questions such as '7月8日AIS数据变化'. Compare port AIS messages, hourly unique MMSI, moving MMSI and stopped MMSI for a UTC day against the previous UTC day by default. Dates are YYYY-MM-DD. Returns four metrics, units, quality flags and evidence IDs in one call."""
        return _json_call(repo.ais_daily_change, day_utc=day_utc, baseline_day_utc=baseline_day_utc)

    @tool
    def data_overview() -> str:
        """List coverage, available zones, vessel classes, metrics and data tables for the base Houston-Beryl dataset ONLY. DO NOT use when analyzing research regions (use situation_cube_overview with region_id instead)."""
        active_region = get_active_region()
        if active_region and hasattr(repo, "situation_cube_overview"):
            return _json_call(repo.situation_cube_overview, region_id=active_region)
        return _json_call(repo.overview)

    @tool
    def aggregate_hourly(table: str, metric: str, start_utc: str, end_utc: str,
                         selector: str = "", statistic: str = "mean") -> str:
        """Compute a bounded hourly statistic. table=port|zone|vessel_class; selector is a zone or vessel class, empty for port. UTC interval is [start_utc,end_utc). statistic=mean|min|max|sum; sum only for additive counts. Returns evidence_id, source, locator, units and quality flags."""
        return _json_call(repo.aggregate, table=table, metric=metric, start_utc=start_utc,
                          end_utc=end_utc, selector=selector or None, statistic=statistic)

    @tool
    def compare_periods(table: str, metric: str, baseline_start_utc: str, baseline_end_utc: str,
                        target_start_utc: str, target_end_utc: str, selector: str = "",
                        statistic: str = "mean") -> str:
        """Compare two UTC periods using the same allowlisted hourly metric and statistic. Returns each period's evidence, absolute and percent difference, and quality flags. Does not establish causation."""
        return _json_call(repo.compare, table=table, metric=metric,
                          baseline_start_utc=baseline_start_utc, baseline_end_utc=baseline_end_utc,
                          target_start_utc=target_start_utc, target_end_utc=target_end_utc,
                          selector=selector or None, statistic=statistic)

    @tool
    def search_confirmed_events(query: str = "", limit: int = 10) -> str:
        """Search confirmed port and waterway operation events by text. Returns a search-level evidence_id even when match_count is zero; cite it for any zero-result claim. Dates are local and may have only day precision; publication date is separate."""
        return _json_call(repo.search_events, query=query, limit=limit)

    @tool
    def get_event_evidence(event_id: str) -> str:
        """Retrieve one confirmed event and its provenance: source title/URL, quote excerpt, verification flag and locator. Use before citing documentary claims."""
        return _json_call(repo.event_evidence, event_id=event_id)

    @tool
    def search_object_phases(query: str = "", limit: int = 10) -> str:
        """Search facility/port-object phase comparison rows. Returns a search-level evidence_id even when match_count is zero, plus observed descriptive measures and ranking eligibility."""
        return _json_call(repo.object_comparisons, query=query, limit=limit)

    @tool
    def search_vessel_tracks(date_utc: str = "", mmsi: str = "", vessel_class: str = "",
                             limit: int = 10) -> str:
        """Find up to 20 distinct vessels from cleaned message-level AIS observations. date_utc is optional YYYY-MM-DD; mmsi and vessel_class are optional exact filters. Returns message counts, observation times, provenance and evidence IDs. Connecting observations is not a proven route."""
        return _json_call(repo.search_tracks, date_utc=date_utc, mmsi=mmsi,
                          vessel_class=vessel_class, limit=limit)

    @tool
    def get_vessel_track(date_utc: str, mmsi: str, granularity: str = "message") -> str:
        """Get one vessel's track on a UTC date. Default and preferred granularity='message' uses cleaned real AIS reports; use 'hourly' only when the user explicitly requests the derived hourly representative product. observation_gap_count is the count of gaps over 15 minutes; gap_segment_count is the count of continuous segments and must not be called a gap count. Connecting observations is not a proven traveled route."""
        if granularity == "message":
            return _json_call(repo.get_message_track, date_utc=date_utc, mmsi=mmsi, max_points=0)
        if granularity != "hourly":
            raise ValueError("granularity 必须是 hourly 或 message")
        return _json_call(repo.get_track, date_utc=date_utc, mmsi=mmsi)

    @tool
    def search_gis_objects(query: str, date_utc: str = "", limit: int = 5) -> str:
        """Find named water areas, fairways, anchorages and other port GIS objects by name. Use for a question naming a sea area, bay, channel, harbour or marine facility. Optional date_utc=YYYY-MM-DD adds AIS hourly means matched to the exact GIS object. Returns object IDs, source provenance, temporal caveats and evidence IDs; geometry is visualized by the webpage."""
        return _json_call(repo.search_gis_objects, query=query, date_utc=date_utc, limit=limit)

    @tool
    def search_external_gis(query: str = "", source_id: str = "", date_utc: str = "", limit: int = 5) -> str:
        """Search cached NOAA channel features or user-uploaded GeoJSON by name or source ID. Returns provenance and GIS geometry evidence IDs. Current NOAA geometry is not a proven historical boundary."""
        if not hasattr(repo, "search_external_gis"):
            return json.dumps({"error": "外部 GIS 查询需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.search_external_gis, query=query, source_id=source_id,
                          date_utc=date_utc, limit=limit)

    @tool
    def discover_complementary_data(source_id: str, desired_kinds: list[str], date_utc: str = "") -> str:
        """Given any registered AIS, GIS or remote-sensing source ID, discover which desired kinds have configured providers for the source footprint. Kinds: ais,gis,remote_sensing,weather,events. This is source discovery, not proof that missing data has been downloaded."""
        if not hasattr(repo, "discover_complementary_data"):
            return json.dumps({"error": "补充数据发现需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.discover_complementary_data, source_id=source_id,
                          desired_kinds=desired_kinds, date_utc=date_utc)

    @tool
    def acquire_noaa_channels(bbox_wgs84: list[float], region_label: str = "Houston", limit: int = 50) -> str:
        """Fetch and cache one bounded, allowlisted NOAA maintained-channel GIS layer for a US coastal WGS84 bbox. Use only when the question needs channel geometry. Returns a source ID; call search_external_gis for features. The current snapshot cannot prove a historical channel boundary."""
        if not hasattr(repo, "acquire_noaa_channels"):
            return json.dumps({"error": "NOAA GIS 连接需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.acquire_noaa_channels, bbox_wgs84=bbox_wgs84,
                          region_label=region_label, limit=limit)

    @tool
    def search_remote_sensing(date_utc: str = "", collection: str = "", limit: int = 10) -> str:
        """Search locally indexed public satellite scenes by UTC day and optional collection: sentinel-1-grd, sentinel-2-c1-l2a, landsat-c2-l2. Returns a search-level evidence_id even when match_count is zero, plus acquisition time, bbox, scene cloud cover, local asset availability and scene evidence IDs. Cite the search-level evidence_id for a zero-result claim. A catalog scene is not a verified vessel detection; missing dates are only absent from the searched catalog."""
        return _json_call(repo.search_remote_sensing, date_utc=date_utc,
                          collection=collection, limit=limit)

    @tool
    def search_ingested_data(date_utc: str = "", region: str = "", kind: str = "", limit: int = 10) -> str:
        """Search newly imported external data sources by UTC date, region label and kind=ais_csv|stac_json. Returns source provenance and dated AIS observation/unique MMSI counts or STAC scene metadata. These are separate datasets; do not mix raw point counts with Houston hourly aggregates."""
        return _json_call(repo.search_ingested_data, date_utc=date_utc, region=region,
                          kind=kind, limit=limit)

    @tool
    def query_data_hourly(source_id: str, metric: str, start_utc: str, end_utc: str) -> str:
        """Query a named source's bounded UTC hourly AIS series. source_id is a dataset ID from data_overview or an imported AIS source ID from search_ingested_data. metric=ais_message_count|unique_mmsi|moving_mmsi|stopped_mmsi. Imported AIS movement uses SOG>0.5 knots. Do not merge sources or treat missing hours as zero."""
        if not hasattr(repo, "query_data_hourly"):
            return json.dumps({"error": "统一来源查询需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.query_data_hourly, source_id=source_id, metric=metric,
                          start_utc=start_utc, end_utc=end_utc)

    @tool
    def ingested_ais_daily_change(source_id: str, day_utc: str, baseline_day_utc: str = "") -> str:
        """Compare an imported AIS source across two UTC days in one call. Returns daily AIS message totals plus hourly means for unique, moving and stopped MMSI. The default baseline is the previous UTC day. Movement uses SOG>0.5 knots."""
        if not hasattr(repo, "ingested_ais_daily_change"):
            return json.dumps({"error": "接入来源日变化查询需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.ingested_ais_daily_change, source_id=source_id, day_utc=day_utc,
                          baseline_day_utc=baseline_day_utc)

    @tool
    def search_ingested_vessel_tracks(source_id: str, date_utc: str = "", mmsi: str = "",
                                      limit: int = 10) -> str:
        """Find vessel-day tracks from real message-level observations inside one imported AIS source. Get source_id from search_ingested_data. Optional date_utc and MMSI narrow the result."""
        if not hasattr(repo, "search_ingested_tracks"):
            return json.dumps({"error": "接入来源航迹查询需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.search_ingested_tracks, source_id=source_id, date_utc=date_utc,
                          mmsi=mmsi, limit=limit)

    @tool
    def get_ingested_vessel_track(source_id: str, date_utc: str, mmsi: str,
                                  granularity: str = "message") -> str:
        """Get one vessel track from an imported AIS source. Default and preferred granularity=message returns real observed AIS reports; hourly is only for an explicitly requested derived hourly view. Connecting points is not a proven route."""
        if not hasattr(repo, "get_ingested_track"):
            return json.dumps({"error": "接入来源航迹查询需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.get_ingested_track, source_id=source_id, date_utc=date_utc,
                          mmsi=mmsi, granularity=granularity, max_points=0)

    @tool
    def assess_remote_ais_overlap(date_utc: str) -> str:
        """Retrieve a legacy sampled cloud-mask QA result based on 17:00 hourly representatives. Never use it for vessel identity, target matching, precise position, or simultaneous remote-AIS correspondence; prefer POSSIBLY_OBSERVES relations built from message-level AIS. Available for sampled days only."""
        return _json_call(repo.assess_remote_ais_overlap, date_utc=date_utc)

    @tool
    def fusion_overview() -> str:
        """Summarize indexed AIS, GIS, weather, remote-sensing and event nodes plus accepted cross-source relations for the base Houston-Beryl dataset ONLY. DO NOT use when analyzing research regions (use search_fusion_relations with region_id instead)."""
        if not hasattr(repo, "fusion_overview"):
            return json.dumps({"error": "统一关联查询需要 SQLite 模式"}, ensure_ascii=False)
        active_region = get_active_region()
        if active_region and hasattr(repo, "search_fusion_relations"):
            return _json_call(repo.search_fusion_relations, region_id=active_region, limit=10)
        return _json_call(repo.fusion_overview)

    @tool
    def search_fusion_relations(node_id: str = "", predicate: str = "",
                                modality: str = "", region_id: str = "",
                                mmsi: str = "", limit: int = 20) -> str:
        """Search evidence-backed cross-source relations, including spatial context, event regions, remote detections and open-set vessel associations for an optional study region (region_id) or vessel MMSI (mmsi). A relation is not causal proof; inspect confidence, method_version, has_more and evidence."""
        if not hasattr(repo, "search_fusion_relations"):
            return json.dumps({"error": "统一关联查询需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.search_fusion_relations, node_id=node_id, predicate=predicate,
                          modality=modality, region_id=region_id, mmsi=mmsi, limit=limit)

    @tool
    def situation_cube_overview(region_id: str = "") -> str:
        """List task-adaptive situation cubes for a study region. Returns native UTC windows, available and missing modalities, evidence counts and method versions. Missing modalities are unknown evidence, not zero activity."""
        if not hasattr(repo, "situation_cube_overview"):
            return json.dumps({"error": "态势立方体查询需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.situation_cube_overview, region_id=region_id)

    @tool
    def query_situation_units(region_id: str = "", start_time_utc: str = "",
                              end_time_utc: str = "", modality: str = "",
                              limit: int = 20) -> str:
        """Find overall maritime situation units by stable region ID and real UTC interval. Optional modality=ais|gis|weather|remote_sensing|event. This is the preferred tool for multi-source regional questions; it never rounds observations to hourly representative points."""
        if not hasattr(repo, "search_situation_units"):
            return json.dumps({"error": "态势立方体查询需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.search_situation_units, region_id=region_id,
                          start_time_utc=start_time_utc, end_time_utc=end_time_utc,
                          modality=modality, limit=limit)

    @tool
    def get_situation_unit(unit_id: str) -> str:
        """Inspect one situation unit and all AIS, GIS, remote-sensing, weather and event evidence. Returns native source times, spatial relations, separate score components, provenance and explicit missing modalities. Rule scores are not calibrated probabilities and co-occurrence is not causation."""
        if not hasattr(repo, "get_situation_unit"):
            return json.dumps({"error": "态势立方体查询需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.get_situation_unit, unit_id=unit_id)

    @tool
    def compare_situation_periods(region_id: str, baseline_start_utc: str,
                                  baseline_end_utc: str, target_start_utc: str,
                                  target_end_utc: str) -> str:
        """Compare modality evidence coverage between two real UTC intervals in one study region. Counts describe evidence availability, not causal impact or physical intensity."""
        if not hasattr(repo, "compare_situation_periods"):
            return json.dumps({"error": "态势立方体查询需要 SQLite 模式"}, ensure_ascii=False)
        return _json_call(repo.compare_situation_periods, region_id=region_id,
                          baseline_start_utc=baseline_start_utc,
                          baseline_end_utc=baseline_end_utc,
                          target_start_utc=target_start_utc,
                          target_end_utc=target_end_utc)

    @tool
    def data_intake_capabilities() -> str:
        """Describe the in-app data intake workflow, including GeoTIFF internal alignment and optional SAR automation. This tool is read-only and never writes the database."""
        info = {"evidence_id": evidence_id("intake", ["ais_csv", "stac_json", "gis_geojson", "weather_csv", "remote_geotiff", "event_text", "/data-intake"]),
                "url": "/data-intake", "supported_kinds": ["ais_csv", "stac_json", "gis_geojson", "weather_csv", "remote_geotiff", "event_text"],
                "max_file_mb": 200, "max_rows_per_batch": 2000000,
                "compressed_inputs": [".csv.gz", ".zip containing .csv or .csv.gz"],
                "ais_required_fields": ["timestamp_utc", "mmsi", "lon", "lat"],
                "derived_products": ["ais_message_count", "unique_mmsi", "moving_mmsi", "stopped_mmsi", "vessel_hour_positions", "vessel_tracks"],
                "workflow": ["上传单文件或批次 ZIP", "自动读取或映射元数据", "预览与质量校验", "确认后事务入库", "GeoTIFF 内部时空对齐", "受支持的 SAR 自动检测与关联"],
                "scope": "新增来源按区域、原文件 SHA-256 独立登记；GeoTIFF 读取 CRS、范围、网格和波段，STAC 仍只导入目录元数据。"}
        return json.dumps(info, ensure_ascii=False)

    @tool
    def detect_dark_vessels(region_id: str = "", date_utc: str = "",
                            min_confidence: float = 0.75,
                            max_reachability_hours: float = 3.0,
                            limit: int = 10) -> str:
        """排查疑似关闭 AIS 应答机的暗船/幽灵船目标。自动结合遥感未匹配目标、GIS人工设施排除（浮标/码头/礁石）与时空断点回溯（Kinematic Reachability Cone），研判暗船身份与可疑等级。"""
        if not hasattr(repo, "detect_dark_vessels"):
            return json.dumps({"error": "暗船检测需要 SQLite 数据库支持"}, ensure_ascii=False)
        return _json_call(repo.detect_dark_vessels, region_id=region_id, date_utc=date_utc,
                          min_confidence=min_confidence,
                          max_reachability_hours=max_reachability_hours, limit=limit)

    @tool
    def query_wind_field(region_id: str = "", valid_time_utc: str = "",
                         lon: float | None = None, lat: float | None = None,
                         grid_cols: int = 8, grid_rows: int = 8) -> str:
        """基于 Barnes 客观分析高斯加权插值法与正交矢量合成模型，查询连续网格风场或特定坐标点的气象场。彻底消除离散孤立测站空间断崖与角度平均翻转失真。指定 lon,lat 返回单点精确气象与指数衰减置信度；不指定返回全域规则网格风场。"""
        if not hasattr(repo, "query_wind_field"):
            return json.dumps({"error": "连续风场查询需要 SQLite 数据库支持"}, ensure_ascii=False)
        return _json_call(repo.query_wind_field, region_id=region_id, valid_time_utc=valid_time_utc,
                          lon=lon, lat=lat, grid_cols=grid_cols, grid_rows=grid_rows)

    @tool
    def query_wind_nowcast(region_id: str = "", base_time_utc: str = "",
                           lead_hours: float = 1.0,
                           lon: float | None = None, lat: float | None = None) -> str:
        """基于高频密集观测时序变率与阻尼动力学方程，进行 0~3 小时超短期短临外推预测（Nowcasting）。预测未来 1~3 小时指定坐标或海区的风速演变、风向顺逆时针偏转、气压急降倾向（∂P/∂t）、极大阵风风险及海上大风预警级别（蓝/黄/橙/红），并输出双重时空衰减置信度。"""
        if not hasattr(repo, "query_wind_nowcast"):
            return json.dumps({"error": "气象短临外推预测需要 SQLite 数据库支持"}, ensure_ascii=False)
        return _json_call(repo.query_wind_nowcast, region_id=region_id, base_time_utc=base_time_utc,
                          lead_hours=lead_hours, lon=lon, lat=lat)

    @tool
    def assess_collision_risk(time_utc: str = "", target_mmsi: str = "",
                              region_id: str = "", min_cri: float = 0.4,
                              max_distance_nm: float = 4.0) -> str:
        """评估指定时间海区内船舶的碰撞风险、最近会遇距离(DCPA/TCPA)及国际避碰规则(COLREGs)会遇局面。基于相对运动学与1972国际避碰规则判定对遇(Rule 14)、追越(Rule 13)、交叉相遇(Rule 15)及走锚失控责任(Rule 18)，输出碰撞危险度(CRI)及让路/直航义务。"""
        from .colregs import evaluate_traffic_conflict_matrix

        active_reg = get_active_region() or region_id
        vessels = []
        if hasattr(repo, "query_vessels_snapshot"):
            snap_res = repo.query_vessels_snapshot(valid_time_utc=time_utc or "2024-07-08T12:00:00Z",
                                                   limit=600, region_id=active_reg)
            vessels = snap_res.get("vessels", [])

        records = evaluate_traffic_conflict_matrix(
            vessels,
            target_mmsi=target_mmsi,
            min_cri=min_cri,
            max_distance_nm=max_distance_nm,
            top_limit=8,
        )

        ev_id = evidence_id("colregs", [time_utc, target_mmsi, active_reg, len(records)])
        payload = {
            "evidence_id": ev_id,
            "region_id": active_reg,
            "evaluated_time_utc": time_utc or "2024-07-08T12:00:00Z",
            "target_mmsi": target_mmsi or None,
            "total_conflicts_found": len(records),
            "conflicts": [r.to_dict() for r in records],
            "note": "依据 COLREGs 1972 国际海上避碰规则与相对运动学计算，DCPA(海里/米)及TCPA(分钟)为假设航向航速不变时的极限几何会遇参数。"
        }
        return json.dumps(sanitize_tool_payload(payload), ensure_ascii=False)

    @tool
    def assess_anchor_stability(
        target_mmsi: str = "",
        time_utc: str = "",
        region_id: str = "",
        anchor_drop_lat: float | None = None,
        anchor_drop_lon: float | None = None,
        seabed_type: str = "sandy_mud",
        water_depth_m: float = 15.0,
        chain_length_m: float | None = None,
    ) -> str:
        """基于流体力学、OCIMF 船舶受风流压阻力模型与底质极限持力机理，评估船舶锚泊稳定性与走锚态势。自动结合实测 AIS 轨迹与局部风场数据，计算拖力比(DFR)、安全回旋半径、偏离距离及风流合力一致性，判定是否正常锚泊、回旋偏摆、走锚临界预警或确诊失控走锚。若未指定 MMSI 则自动扫描当前海区高风险锚泊船。"""
        from .anchor_physics import assess_anchor_stability as eval_anchor

        active_reg = get_active_region() or region_id
        eval_time = time_utc or "2024-07-08T12:00:00Z"

        # 1. 查询快照船舶
        vessels = []
        if hasattr(repo, "query_vessels_snapshot"):
            snap_res = repo.query_vessels_snapshot(valid_time_utc=eval_time,
                                                   limit=600, region_id=active_reg)
            vessels = snap_res.get("vessels", [])

        # 2. 获取风场环境
        default_wind_speed = 18.0
        default_wind_dir = 140.0
        if hasattr(repo, "query_wind_field"):
            try:
                wf = repo.query_wind_field(region_id=active_reg, valid_time_utc=eval_time)
                stats = wf.get("statistics", {})
                if stats.get("mean_speed_knots") is not None:
                    default_wind_speed = float(stats["mean_speed_knots"])
                if stats.get("prevailing_direction_deg") is not None:
                    default_wind_dir = float(stats["prevailing_direction_deg"])
            except Exception:
                pass

        # 3. 若指定了 target_mmsi，进行单船高精度物理推演
        if target_mmsi:
            target_v = next((v for v in vessels if str(v.get("mmsi")) == str(target_mmsi)), None)
            if not target_v and hasattr(repo, "get_track"):
                try:
                    day_str = eval_time[:10]
                    track = repo.get_track(day_str, str(target_mmsi))
                    pts = track.get("points", [])
                    if pts:
                        p_last = pts[-1]
                        target_v = {
                            "mmsi": str(target_mmsi),
                            "name": track.get("vessel_name", f"MMSI {target_mmsi}"),
                            "lat": p_last.get("lat"),
                            "lon": p_last.get("lon"),
                            "sog_kn": p_last.get("sog_knots", 0.0),
                            "cog_deg": p_last.get("cog_degrees", 0.0),
                            "heading_deg": p_last.get("heading_degrees", p_last.get("cog_degrees", 0.0)),
                            "length_m": None,
                        }
                        if anchor_drop_lat is None:
                            anchor_drop_lat = pts[0].get("lat")
                            anchor_drop_lon = pts[0].get("lon")
                except Exception:
                    pass

            if not target_v:
                return json.dumps({"error": f"在时间 {eval_time} 未检索到 MMSI {target_mmsi} 的 AIS 船位"}, ensure_ascii=False)

            c_lat = float(target_v.get("lat") or 0.0)
            c_lon = float(target_v.get("lon") or 0.0)
            drop_lat = anchor_drop_lat if anchor_drop_lat is not None else c_lat
            drop_lon = anchor_drop_lon if anchor_drop_lon is not None else c_lon

            rec = eval_anchor(
                mmsi=str(target_mmsi),
                vessel_name=target_v.get("name") or f"MMSI {target_mmsi}",
                anchor_drop_lat=drop_lat,
                anchor_drop_lon=drop_lon,
                current_lat=c_lat,
                current_lon=c_lon,
                sog_knots=float(target_v.get("sog_kn") or 0.0),
                cog_deg=float(target_v.get("cog_deg") or 0.0),
                heading_deg=float(target_v.get("heading_deg") or target_v.get("cog_deg") or 0.0),
                wind_speed_knots=default_wind_speed,
                wind_direction_from_deg=default_wind_dir,
                vessel_length_m=float(target_v.get("length_m") or 140.0),
                seabed_type=seabed_type,
                water_depth_m=water_depth_m,
                chain_length_m=chain_length_m,
            )
            ev_id = evidence_id("anchor_mech", [target_mmsi, eval_time, rec.status.value])
            payload = {
                "evidence_id": ev_id,
                "region_id": active_reg,
                "evaluated_time_utc": eval_time,
                "target_mmsi": str(target_mmsi),
                "assessment": rec.to_dict(),
            }
            return json.dumps(sanitize_tool_payload(payload), ensure_ascii=False)

        # 4. 未指定 target_mmsi，批量扫描低速/锚泊船
        candidates = [v for v in vessels if (v.get("sog_kn") or 0.0) < 3.0]
        records = []
        for v in candidates[:50]:
            c_lat = float(v.get("lat") or 0.0)
            c_lon = float(v.get("lon") or 0.0)
            rec = eval_anchor(
                mmsi=str(v.get("mmsi")),
                vessel_name=v.get("name") or f"MMSI {v.get('mmsi')}",
                anchor_drop_lat=c_lat,
                anchor_drop_lon=c_lon,
                current_lat=c_lat,
                current_lon=c_lon,
                sog_knots=float(v.get("sog_kn") or 0.0),
                cog_deg=float(v.get("cog_deg") or 0.0),
                heading_deg=float(v.get("heading_deg") or v.get("cog_deg") or 0.0),
                wind_speed_knots=default_wind_speed,
                wind_direction_from_deg=default_wind_dir,
                seabed_type=seabed_type,
                water_depth_m=water_depth_m,
                chain_length_m=chain_length_m,
            )
            records.append(rec)

        # 优先保留高负荷/预警/走锚船
        records.sort(key=lambda r: r.drag_force_ratio, reverse=True)
        top_recs = records[:10]
        ev_id = evidence_id("anchor_scan", [eval_time, active_reg, len(top_recs)])
        payload = {
            "evidence_id": ev_id,
            "region_id": active_reg,
            "evaluated_time_utc": eval_time,
            "scanned_vessels_count": len(candidates),
            "urgent_count": sum(1 for r in top_recs if r.is_urgent),
            "top_records": [r.to_dict() for r in top_recs],
        }
        return json.dumps(sanitize_tool_payload(payload), ensure_ascii=False)

    @tool
    def get_target_vessel_dossier(date_utc: str, mmsi: str, region_id: str = "") -> str:
        """针对指定 MMSI 船舶和日期，一站式调取全维度的时空态势档案与多源情报交叉核验数据。
        包含：逐报文航迹完整性（报文数、15分钟缺口数、极大/均值SOG）、走锚与工况机理排查、COLREGs 会遇与极限逼近解算、卫星遥感切片精确关联与时空对齐、海洋气象环境及真实证据编号。
        """
        active_reg = get_active_region() or region_id
        if hasattr(repo, "get_target_vessel_dossier"):
            dossier = repo.get_target_vessel_dossier(date_utc=date_utc, mmsi=mmsi, region_id=active_reg)
            return json.dumps(sanitize_tool_payload(dossier), ensure_ascii=False)
        return json.dumps({"error": "底层存储未支持单船全景态势档案"}, ensure_ascii=False)

    return [ais_daily_change, data_overview, aggregate_hourly, compare_periods,
            search_confirmed_events, get_event_evidence, search_object_phases,
            search_vessel_tracks, get_vessel_track, search_gis_objects, search_external_gis,
            discover_complementary_data, acquire_noaa_channels, search_remote_sensing,
            search_ingested_data, query_data_hourly, ingested_ais_daily_change, search_ingested_vessel_tracks,
            get_ingested_vessel_track, assess_remote_ais_overlap, fusion_overview,
            search_fusion_relations, situation_cube_overview, query_situation_units,
            get_situation_unit, compare_situation_periods, detect_dark_vessels, query_wind_field,
            query_wind_nowcast, data_intake_capabilities, assess_collision_risk, assess_anchor_stability,
            get_target_vessel_dossier]
