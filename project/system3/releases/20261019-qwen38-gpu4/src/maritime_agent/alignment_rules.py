"""Versioned canonical schemas and normalization rules for data intake."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


KIND_RULE_FILES = {
    "ais_csv": "ais.json",
    "gis_geojson": "gis.json",
    "weather_csv": "weather.json",
    "stac_json": "remote_sensing.json",
    "events": "events.json",
}


@lru_cache(maxsize=None)
def common_rules() -> dict[str, Any]:
    return _load("common.json")


@lru_cache(maxsize=None)
def rules_for(kind: str) -> dict[str, Any]:
    try:
        name = KIND_RULE_FILES[kind]
    except KeyError as exc:
        raise ValueError(f"没有注册的数据对齐类型: {kind}") from exc
    rules = _load(name)
    required = {"kind", "schema_version", "ruleset_version", "canonical_fields", "aliases"}
    missing = sorted(required - rules.keys())
    if missing:
        raise ValueError(f"对齐规则 {name} 缺少字段: {', '.join(missing)}")
    if rules["kind"] != kind:
        raise ValueError(f"对齐规则 {name} 的 kind 不匹配")
    return rules


def schema_identity(kind: str) -> dict[str, str]:
    rules = rules_for(kind)
    return {"schema_version": str(rules["schema_version"]),
            "ruleset_version": str(rules["ruleset_version"])}


def canonical_profile(kind: str) -> dict[str, Any]:
    common = common_rules()
    identity = schema_identity(kind)
    return {**identity, "canonical_crs": common["spatial"]["canonical_crs"],
            "canonical_time": common["temporal"]["canonical_time"],
            "canonical_fields": list(rules_for(kind)["canonical_fields"])}


def _load(name: str) -> dict[str, Any]:
    resource = files("maritime_agent").joinpath("alignment_rules", name)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"对齐规则 {name} 必须是 JSON 对象")
    return payload
