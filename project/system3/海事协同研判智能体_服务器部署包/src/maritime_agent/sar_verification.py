from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

AIS_SAR_ROOT = Path(r"E:\ais-sar")
S02_CSV_PATH = AIS_SAR_ROOT / "06_outputs" / "S02" / "01_association_baseline_comparison" / "association_method_comparison.csv"
S02_CANDS_PATH = AIS_SAR_ROOT / "06_outputs" / "S02" / "00_minimal_validation" / "candidate_pairs_within_500m.csv"
LABELS_DIR = AIS_SAR_ROOT / "05_labels"
LABELS_FILE = LABELS_DIR / "S02_ground_truth_labels.csv"
SQLITE_PATH = Path("storage/maritime.sqlite3")


def _get_vessel_metadata(mmsi_set: set[str]) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    if not SQLITE_PATH.exists() or not mmsi_set:
        return meta
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.cursor()
        for mmsi in mmsi_set:
            if not mmsi:
                continue
            row = cur.execute(
                "SELECT vessel_name, vessel_class, sog_kn, cog_deg "
                "FROM ais_messages WHERE mmsi = ? AND vessel_name IS NOT NULL "
                "ORDER BY timestamp_utc DESC LIMIT 1",
                (mmsi,),
            ).fetchone()
            if row:
                meta[mmsi] = {
                    "vessel_name": row[0] or "未知船名",
                    "vessel_class": row[1] or "未知类型",
                    "sog_kn": str(row[2]) if row[2] is not None else "",
                    "cog_deg": str(row[3]) if row[3] is not None else "",
                }
            else:
                meta[mmsi] = {
                    "vessel_name": "未收录船名",
                    "vessel_class": "未知船型",
                    "sog_kn": "",
                    "cog_deg": "",
                }
        conn.close()
    except Exception as exc:
        print(f"[Warning] Failed to fetch vessel metadata: {exc}")
    return meta


def load_verification_targets() -> dict[str, Any]:
    if not S02_CSV_PATH.exists():
        raise FileNotFoundError(f"未找到目标文件: {S02_CSV_PATH}")

    with open(S02_CSV_PATH, "r", encoding="utf-8-sig") as f:
        targets_raw = list(csv.DictReader(f))

    cands_by_target: dict[str, list[dict[str, Any]]] = {}
    mmsi_set: set[str] = set()

    if S02_CANDS_PATH.exists():
        with open(S02_CANDS_PATH, "r", encoding="utf-8-sig") as f:
            for c in csv.DictReader(f):
                tid = c.get("sar_target_id", "")
                if tid:
                    cands_by_target.setdefault(tid, []).append(c)
                    if c.get("mmsi"):
                        mmsi_set.add(str(c["mmsi"]))

    # Also extract MMSIs from target recommendation columns
    for t in targets_raw:
        for key in ("nn500_mmsi", "neighbour_mmsi", "world_v0_mmsi", "manual_truth_mmsi_or_NO_MATCH"):
            val = (t.get(key) or "").strip()
            if val and val.isdigit():
                mmsi_set.add(val)

    # 查取船舶静态属性
    vessel_meta = _get_vessel_metadata(mmsi_set)

    targets = []
    stats = {
        "total": len(targets_raw),
        "reviewed": 0,
        "unreviewed": 0,
        "difficult_count": 0,  # 候选 >= 2
        "single_count": 0,     # 候选 == 1
        "zero_count": 0,       # 候选 == 0 (暗船/杂波)
        "changed_count": 0,    # nn_vs_world_changed == True
        "confirmed_match": 0,
        "confirmed_no_match": 0,
        "ambiguous": 0,
        "false_alarm": 0,
    }

    for idx, t in enumerate(targets_raw):
        tid = t.get("sar_target_id", f"T_{idx}")
        cands = cands_by_target.get(tid, [])
        # enrich candidates with metadata
        enriched_cands = []
        for c in sorted(cands, key=lambda x: float(x.get("distance_m", 99999))):
            mmsi = str(c.get("mmsi", ""))
            meta = vessel_meta.get(mmsi, {})
            enriched_cands.append({
                "candidate_rank": c.get("candidate_rank_by_distance", "1"),
                "mmsi": mmsi,
                "distance_m": round(float(c.get("distance_m", 0)), 1),
                "ais_lon": float(c.get("ais_lon", 0)),
                "ais_lat": float(c.get("ais_lat", 0)),
                "gap_seconds": round(float(c.get("ais_support_gap_seconds", 0)), 1),
                "cfar_z": round(float(c.get("cfar_z", 0)), 2),
                "vessel_name": meta.get("vessel_name", "未知船名"),
                "vessel_class": meta.get("vessel_class", "未知船型"),
                "sog_kn": meta.get("sog_kn", ""),
                "cog_deg": meta.get("cog_deg", ""),
            })

        cand_len = len(enriched_cands)
        if cand_len >= 2:
            stats["difficult_count"] += 1
        elif cand_len == 1:
            stats["single_count"] += 1
        else:
            stats["zero_count"] += 1

        is_changed = str(t.get("nn_vs_world_changed", "")).lower() == "true"
        if is_changed:
            stats["changed_count"] += 1

        review_status = (t.get("manual_review_status") or "UNREVIEWED").strip()
        truth_val = (t.get("manual_truth_mmsi_or_NO_MATCH") or "").strip()

        if review_status and review_status != "UNREVIEWED":
            stats["reviewed"] += 1
            if review_status == "CONFIRMED" and truth_val and truth_val != "NO_MATCH":
                stats["confirmed_match"] += 1
            elif review_status == "CONFIRMED" and truth_val == "NO_MATCH":
                stats["confirmed_no_match"] += 1
            elif review_status == "AMBIGUOUS":
                stats["ambiguous"] += 1
            elif review_status == "FALSE_ALARM":
                stats["false_alarm"] += 1
        else:
            stats["unreviewed"] += 1

        targets.append({
            "sar_target_id": tid,
            "sar_lon": float(t.get("sar_lon", 0)),
            "sar_lat": float(t.get("sar_lat", 0)),
            "methods": {
                "nn500": {
                    "mmsi": t.get("nn500_mmsi", ""),
                    "distance_m": round(float(t.get("nn500_distance_m", 0)), 1) if t.get("nn500_distance_m") else None,
                    "vessel_name": vessel_meta.get(t.get("nn500_mmsi", ""), {}).get("vessel_name", ""),
                },
                "neighbour": {
                    "mmsi": t.get("neighbour_mmsi", ""),
                    "distance_m": round(float(t.get("neighbour_distance_m", 0)), 1) if t.get("neighbour_distance_m") else None,
                    "vessel_name": vessel_meta.get(t.get("neighbour_mmsi", ""), {}).get("vessel_name", ""),
                },
                "world_v0": {
                    "decision": t.get("world_v0_decision", ""),
                    "mmsi": t.get("world_v0_mmsi", ""),
                    "distance_m": round(float(t.get("world_v0_distance_m", 0)), 1) if t.get("world_v0_distance_m") else None,
                    "cost": round(float(t.get("world_v0_cost", 0)), 2) if t.get("world_v0_cost") else None,
                    "margin": round(float(t.get("world_v0_margin", 0)), 2) if t.get("world_v0_margin") else None,
                    "confidence": round(float(t.get("world_v0_pair_preference_uncalibrated", 0)), 3) if t.get("world_v0_pair_preference_uncalibrated") else None,
                    "vessel_name": vessel_meta.get(t.get("world_v0_mmsi", ""), {}).get("vessel_name", ""),
                },
            },
            "is_changed": is_changed,
            "candidates": enriched_cands,
            "candidate_count": cand_len,
            "manual_truth_mmsi_or_NO_MATCH": truth_val,
            "manual_review_status": review_status,
        })

    return {
        "scene_id": "S02",
        "scene_time_utc": "2024-07-06T12:23:36Z",
        "stats": stats,
        "targets": targets,
    }


def save_manual_label(
    sar_target_id: str,
    truth_mmsi_or_no_match: str,
    review_status: str,
    reviewer: str = "Reviewer-1",
    notes: str = "",
) -> dict[str, Any]:
    if not S02_CSV_PATH.exists():
        raise FileNotFoundError(f"未找到目标文件: {S02_CSV_PATH}")

    # 1. 更新原 CSV (E:\ais-sar\06_outputs\S02\01_association_baseline_comparison\association_method_comparison.csv)
    with open(S02_CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        fieldnames = reader[0].keys() if reader else []

    updated = False
    for row in reader:
        if row.get("sar_target_id") == sar_target_id:
            row["manual_truth_mmsi_or_NO_MATCH"] = truth_mmsi_or_no_match.strip()
            row["manual_review_status"] = review_status.strip()
            updated = True
            break

    if updated:
        with open(S02_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reader)

    # 2. 规范写入 05_labels 目录: scene_id,sar_target_id,mmsi,label_status,reviewer,notes,updated_at
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    label_records = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("sar_target_id"):
                    label_records[row["sar_target_id"]] = row

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    label_records[sar_target_id] = {
        "scene_id": "S02",
        "sar_target_id": sar_target_id,
        "mmsi": truth_mmsi_or_no_match.strip(),
        "label_status": review_status.strip(),
        "reviewer": reviewer.strip(),
        "notes": notes.strip(),
        "updated_at": now_str,
    }

    label_fieldnames = ["scene_id", "sar_target_id", "mmsi", "label_status", "reviewer", "notes", "updated_at"]
    with open(LABELS_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=label_fieldnames)
        writer.writeheader()
        writer.writerows(label_records.values())

    return {
        "success": True,
        "sar_target_id": sar_target_id,
        "saved_label": label_records[sar_target_id],
        "total_labels_in_05_labels": len(label_records),
    }
