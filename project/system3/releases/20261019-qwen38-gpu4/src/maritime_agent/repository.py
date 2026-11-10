"""Validated, read-only adapter over the Houston-Beryl DS outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


TABLES = {
    "port": "situation_analysis/01_port_hourly/port_situation_hourly.csv",
    "zone": "situation_analysis/02_zone_hourly/port_zone_situation_hourly.csv",
    "vessel_class": "situation_analysis/03_vessel_class_hourly/vessel_class_situation_hourly.csv",
    "events": "operations_events/03_confirmed_events/confirmed_events.csv",
    "event_links": "operations_events/04_evidence_links/event_evidence_links.csv",
    "sources": "operations_events/00_source_registry/source_registry.csv",
    "objects": "situation_analysis/09_object_event_chains/object_phase_comparison.csv",
}

# Metrics are deliberately allowlisted; the model never writes Python or SQL.
METRICS = {
    "port": {
        "ais_message_count": "条/小时", "unique_mmsi": "艘/小时",
        "moving_mmsi": "艘/小时", "stopped_mmsi": "艘/小时",
        "behavior_state_channel_transit": "艘/小时",
        "transition_port_zone_entry": "次/小时",
        "transition_port_zone_exit": "次/小时",
        "wind_speed_mean_ms": "米/秒", "wind_gust_max_ms": "米/秒",
        "air_pressure_mean_hpa": "百帕", "water_level_mean_m_mllw": "米 MLLW",
        "storm_distance_to_bbox_center_km": "千米",
    },
    "zone": {
        "unique_mmsi": "艘/小时", "motion_moving": "艘/小时",
        "motion_stopped": "艘/小时", "state_channel_transit": "艘/小时",
        "arrivals_port_zone_entry": "次/小时", "departures_port_zone_exit": "次/小时",
    },
    "vessel_class": {
        "unique_mmsi": "艘/小时", "moving_mmsi": "艘/小时",
        "stopped_mmsi": "艘/小时", "transition_port_zone_entry": "次/小时",
    },
}
ADDITIVE_METRICS = {
    "ais_message_count", "transition_port_zone_entry", "transition_port_zone_exit",
    "arrivals_port_zone_entry", "departures_port_zone_exit",
}


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"UTC 时间格式错误: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("时间必须明确标为 UTC，例如 2024-07-08T00:00:00Z")
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("小时表查询的起止时间必须位于整点")
    return parsed.astimezone(timezone.utc)


def _number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def evidence_id(prefix: str, payload: object) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}:{hashlib.sha256(data.encode('utf-8')).hexdigest()[:12]}"


class DataRepository:
    def __init__(self, root: Path):
        self.root = root.resolve()
        for name in ("port", "zone", "vessel_class", "events", "event_links", "sources"):
            path = self.root / TABLES[name]
            if not path.is_file():
                raise FileNotFoundError(f"缺少必需数据表: {path}")

    @lru_cache(maxsize=10)
    def rows(self, table: str) -> tuple[dict[str, str], ...]:
        if table not in TABLES:
            raise ValueError(f"未知数据表: {table}")
        path = self.root / TABLES[table]
        if not path.is_file():
            raise FileNotFoundError(f"缺少数据表: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return tuple(csv.DictReader(stream))

    def overview(self) -> dict[str, Any]:
        port = self.rows("port")
        if not port:
            raise ValueError("港口小时表为空")
        return {
            "evidence_id": evidence_id("overview", [port[0]["timestamp_hour_utc"], port[-1]["timestamp_hour_utc"]]),
            "dataset": "Houston-Beryl DS",
            "time_range_utc": [port[0]["timestamp_hour_utc"], port[-1]["timestamp_hour_utc"]],
            "port_hour_rows": len(port),
            "zones": sorted({r["macro_zone"] for r in self.rows("zone")}),
            "vessel_classes": sorted({r["vessel_class"] for r in self.rows("vessel_class")}),
            "available_metrics": METRICS,
            "confirmed_event_count": len(self.rows("events")),
            "track_count": 0,
            "gis_object_count": 0,
            "sources": {name: TABLES[name] for name in TABLES},
            "temporal_caveat": "AIS/气象为小时数据；部分运营事件只有本地日期精度，不能定位到具体小时。",
        }

    def search_tracks(self, date_utc: str = "", mmsi: str = "", vessel_class: str = "",
                      limit: int = 10) -> dict[str, Any]:
        raise ValueError("CSV 查询模式没有接入船舶小时位置和日航迹；请使用已导入航迹的 SQLite 数据集")

    def get_track(self, date_utc: str, mmsi: str) -> dict[str, Any]:
        raise ValueError("CSV 查询模式没有接入船舶小时位置和日航迹；请使用已导入航迹的 SQLite 数据集")

    def get_message_track(self, date_utc: str, mmsi: str, max_points: int = 3000) -> dict[str, Any]:
        raise ValueError("CSV 查询模式没有接入逐报文 AIS 观测点；请使用已导入 AIS 的 SQLite 数据集")

    def search_remote_sensing(self, date_utc: str = "", collection: str = "",
                              limit: int = 10) -> dict[str, Any]:
        raise ValueError("CSV 查询模式没有接入遥感场景目录；请使用已导入遥感数据的 SQLite 数据集")

    def get_remote_sensing_scene(self, scene_id: str) -> dict[str, Any]:
        raise ValueError("CSV 查询模式没有接入遥感场景目录")

    def search_ingested_data(self, date_utc: str = "", region: str = "",
                             kind: str = "", limit: int = 10) -> dict[str, Any]:
        raise ValueError("CSV 查询模式没有接入新增数据来源目录")

    def assess_remote_ais_overlap(self, date_utc: str) -> dict[str, Any]:
        raise ValueError("CSV 查询模式没有接入遥感与 AIS 空间质量核对")

    def _selection(self, table: str, metric: str, start_utc: str, end_utc: str,
                   selector: str | None = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
        if table not in METRICS or metric not in METRICS[table]:
            raise ValueError(f"不支持的指标。{table} 可用指标: {list(METRICS.get(table, {}))}")
        start, end = _utc(start_utc), _utc(end_utc)
        if end <= start:
            raise ValueError("end_utc 必须晚于 start_utc；区间为左闭右开")
        if (end - start).days > 366:
            raise ValueError("单次查询最多 366 天")
        selector_key = {"zone": "macro_zone", "vessel_class": "vessel_class"}.get(table)
        if selector_key:
            options = {r[selector_key] for r in self.rows(table)}
            if selector not in options:
                raise ValueError(f"{table} selector 必须选一个: {sorted(options)}")
        elif selector:
            raise ValueError("port 表不能使用 selector")
        rows = [r for r in self.rows(table)
                if start <= _utc(r["timestamp_hour_utc"]) < end
                and (not selector_key or r[selector_key] == selector)]
        return rows, {"table": table, "metric": metric, "start_utc": start_utc,
                      "end_utc_exclusive": end_utc, "selector": selector}

    def aggregate(self, table: str, metric: str, start_utc: str, end_utc: str,
                  selector: str | None = None, statistic: str = "mean") -> dict[str, Any]:
        rows, scope = self._selection(table, metric, start_utc, end_utc, selector)
        if statistic not in {"mean", "min", "max", "sum"}:
            raise ValueError("statistic 只能是 mean、min、max、sum")
        if statistic == "sum" and metric not in ADDITIVE_METRICS:
            raise ValueError("该指标不能跨小时直接求和；请选择 mean、min 或 max")
        values = [v for r in rows if (v := _number(r.get(metric))) is not None]
        if not values:
            result = None
        elif statistic == "mean":
            result = sum(values) / len(values)
        elif statistic == "sum":
            result = sum(values)
        elif statistic == "min":
            result = min(values)
        else:
            result = max(values)
        missing_hours = []
        if table == "port":
            missing_hours = [r["timestamp_hour_utc"] for r in rows
                             if r.get("ais_source_file_available") == "0" or r.get("ais_no_reports") == "1"]
        unit = METRICS[table][metric]
        if statistic == "sum":
            unit = "条" if metric == "ais_message_count" else "次"
        payload = {**scope, "statistic": statistic}
        expected_hours = int((_utc(end_utc) - _utc(start_utc)).total_seconds() // 3600)
        return {
            "evidence_id": evidence_id("aggregate", payload),
            "value": round(result, 4) if result is not None else None,
            "unit": unit,
            "statistic": statistic,
            "row_count": len(rows),
            "non_null_rows": len(values),
            "scope": scope,
            "source": TABLES[table],
            "record_locator": f"timestamp_hour_utc>={start_utc} AND <{end_utc}" +
                              (f" AND {('macro_zone' if table == 'zone' else 'vessel_class')}={selector}" if selector else ""),
            "quality_flags": {
                "requested_hour_count": expected_hours,
                "missing_hourly_row_count": max(0, expected_hours - len(rows)),
                "ais_missing_or_no_report_hours": missing_hours[:24],
                "ais_missing_or_no_report_count": len(missing_hours),
                "no_matching_rows": not rows,
                "partial_metric_coverage": bool(rows) and len(values) < len(rows),
            },
            "interpretation": "区间内每小时观测值的统计；同一 MMSI 在多小时中可能重复出现，不能当作全区间去重船舶数。"
                              if metric.endswith("mmsi") or metric in {"unique_mmsi", "moving_mmsi", "stopped_mmsi"} else "",
        }

    def compare(self, table: str, metric: str, baseline_start_utc: str, baseline_end_utc: str,
                target_start_utc: str, target_end_utc: str, selector: str | None = None,
                statistic: str = "mean") -> dict[str, Any]:
        baseline = self.aggregate(table, metric, baseline_start_utc, baseline_end_utc, selector, statistic)
        target = self.aggregate(table, metric, target_start_utc, target_end_utc, selector, statistic)
        a, b = baseline["value"], target["value"]
        delta = round(b - a, 4) if a is not None and b is not None else None
        pct = round((b - a) / a * 100, 2) if a not in (None, 0) and b is not None else None
        return {
            "evidence_id": evidence_id("comparison", [baseline["evidence_id"], target["evidence_id"]]),
            "baseline": baseline, "target": target, "difference": delta,
            "change_pct": pct, "unit": target["unit"],
            "warning": "基线为零或缺测，百分比不可计算。" if pct is None else "",
            "causality_caveat": "时间上的共同变化不足以证明风暴、运营事件与船舶行为之间的因果关系。",
        }

    def ais_daily_change(self, day_utc: str, baseline_day_utc: str = "") -> dict[str, Any]:
        """Compare core port AIS indicators for two UTC calendar days."""
        try:
            target_day = date.fromisoformat(day_utc)
            baseline_day = date.fromisoformat(baseline_day_utc) if baseline_day_utc else target_day - timedelta(days=1)
        except ValueError as exc:
            raise ValueError("日期必须为 YYYY-MM-DD，例如 2024-07-08") from exc
        def bounds(day: date) -> tuple[str, str]:
            return (day.isoformat() + "T00:00:00Z",
                    (day + timedelta(days=1)).isoformat() + "T00:00:00Z")
        baseline_start, baseline_end = bounds(baseline_day)
        target_start, target_end = bounds(target_day)
        metrics = ("ais_message_count", "unique_mmsi", "moving_mmsi", "stopped_mmsi")
        comparisons = {metric: self.compare("port", metric, baseline_start, baseline_end,
                                            target_start, target_end) for metric in metrics}
        return {
            "evidence_id": evidence_id("ais_day_change", [baseline_day.isoformat(), target_day.isoformat()]),
            "baseline_day_utc": baseline_day.isoformat(),
            "target_day_utc": target_day.isoformat(),
            "comparison_rule": "若未指定基准日期，使用前一个 UTC 自然日。",
            "metrics": comparisons,
            "source": TABLES["port"],
            "interpretation": "unique_mmsi 等船舶指标为每小时观测均值，不是整日去重船舶数。",
            "causality_caveat": "这些是描述性变化，不能单凭同期数据推断原因。",
        }

    def search_events(self, query: str = "", limit: int = 10) -> dict[str, Any]:
        limit = max(1, min(limit, 20))
        q = query.strip().casefold()
        fields = ("event_id", "event_name", "event_type", "event_subtype", "location_text", "operational_constraint")
        found = [r for r in self.rows("events") if not q or any(q in (r.get(k) or "").casefold() for k in fields)]
        events = [{**r, "evidence_id": f"event:{r['event_id']}", "source": TABLES["events"],
                   "record_locator": f"event_id={r['event_id']}"} for r in found[:limit]]
        return {"evidence_id": evidence_id("event_search", [q, len(found)]),
                "events": events, "match_count": len(found), "truncated": len(found) > limit,
                "precision_note": "valid_start_local/valid_end_local 的精度由 time_precision 指定；published_date 是发布日期，不是生效时间。"}

    def event_evidence(self, event_id: str) -> dict[str, Any]:
        matches = [r for r in self.rows("events") if r["event_id"] == event_id]
        if not matches:
            raise ValueError(f"事件不存在: {event_id}")
        links = [r for r in self.rows("event_links") if r["event_id"] == event_id]
        source_map = {r["source_id"]: r for r in self.rows("sources")}
        cited = []
        for r in links[:20]:
            source = source_map.get(r["source_id"], {})
            cited.append({
                "evidence_id": f"event_link:{r['evidence_link_id']}",
                "source_id": r["source_id"], "source_title": source.get("title", ""),
                "source_url": source.get("source_url", ""),
                "publication_date": source.get("publication_date", ""),
                "locator": r.get("evidence_locator", ""),
                "quote_verified": r.get("quote_verified", ""),
                "supports_or_conflicts": r.get("supports_or_conflicts", ""),
                "quote_excerpt": (r.get("evidence_quote") or "")[:800],
            })
        return {"event": {**matches[0], "evidence_id": f"event:{event_id}"},
                "evidence_links": cited, "link_count": len(links),
                "source": TABLES["event_links"]}

    def object_comparisons(self, query: str = "", limit: int = 10) -> dict[str, Any]:
        q = query.strip().casefold()
        rows = [r for r in self.rows("objects") if not q or q in (r.get("object_name") or "").casefold()
                or q in (r.get("object_id") or "").casefold()]
        selected = [{**r, "evidence_id": f"object:{r['object_id']}", "source": TABLES["objects"],
                     "record_locator": f"object_id={r['object_id']}"} for r in rows[:max(1, min(limit, 20))]]
        return {"evidence_id": evidence_id("object_search", [q, len(rows)]),
                "objects": selected, "match_count": len(rows), "truncated": len(rows) > len(selected),
                "caveat": "对象阶段对比是描述性统计；请检查 ranking_eligibility 和各阶段小时数。"}

    def map_tracks(self, date_utc: str, vessel_class: str = "", mmsi: str = "",
                   limit: int = 100) -> dict[str, Any]:
        raise ValueError("当前数据后端未导入逐船航迹；请使用 SQLite 数据库并运行航迹导入")

    def search_gis_objects(self, query: str, date_utc: str = "", limit: int = 5) -> dict[str, Any]:
        raise ValueError("当前数据后端没有港口 GIS 对象；请使用已导入 GIS 的 SQLite 数据集")

    def get_gis_object(self, object_id: str) -> dict[str, Any]:
        raise ValueError("当前数据后端没有港口 GIS 对象；请使用已导入 GIS 的 SQLite 数据集")

    def detect_dark_vessels(self, region_id: str = "", date_utc: str = "",
                            min_confidence: float = 0.75,
                            max_reachability_hours: float = 3.0,
                            limit: int = 10) -> dict[str, Any]:
        raise ValueError("暗船检测需要 SQLite 数据库后端支持")

    def query_wind_field(self, region_id: str = "", valid_time_utc: str = "",
                         lon: float | None = None, lat: float | None = None,
                         grid_cols: int = 8, grid_rows: int = 8) -> dict[str, Any]:
        raise ValueError("连续网格风场查询需要 SQLite 数据库后端支持")

    def query_wind_nowcast(self, region_id: str = "", base_time_utc: str = "",
                           lead_hours: float = 1.0,
                           lon: float | None = None, lat: float | None = None,
                           lead_steps: list[float] | None = None) -> dict[str, Any]:
        raise ValueError("气象短临外推预测需要 SQLite 数据库后端支持")

    def query_vessels_snapshot(self, valid_time_utc: str = "", limit: int = 800) -> dict[str, Any]:
        raise ValueError("船舶态势快照查询需要 SQLite 数据库后端支持")

    def query_port_facilities(self, limit: int = 200) -> dict[str, Any]:
        raise ValueError("港口泊位航道设施查询需要 SQLite 数据库后端支持")

    def query_timeline_histogram(self, start_utc: str = "2024-07-06T00:00:00Z",
                                 end_utc: str = "2024-07-10T23:00:00Z") -> dict[str, Any]:
        raise ValueError("时间轴直方图查询需要 SQLite 数据库后端支持")


