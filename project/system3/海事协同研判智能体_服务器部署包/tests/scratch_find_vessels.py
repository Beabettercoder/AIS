import sqlite3
import json

conn = sqlite3.connect("storage/maritime.sqlite3")
conn.row_factory = sqlite3.Row

# 1. 查找在 2024-01-06 挪威 Haugesund 有遥感融合候选的船舶
rows = conn.execute("""
    SELECT candidate_id, decision, raw_score, details_json
    FROM fusion_candidates
    WHERE predicate = 'POSSIBLY_OBSERVES'
    ORDER BY CASE WHEN decision='accepted' THEN 0 ELSE 1 END, raw_score DESC
""").fetchall()

vessels = {}
for r in rows:
    if not r["details_json"]:
        continue
    try:
        d = json.loads(r["details_json"])
    except Exception:
        continue
    m = d.get("mmsi")
    if m and (m not in vessels or r["decision"] == "accepted"):
        vessels[m] = {
            "decision": r["decision"],
            "score": r["raw_score"],
            "time": d.get("ais_timestamp_utc"),
            "dist_m": round(d.get("distance_km", 0) * 1000, 1),
            "candidate_id": r["candidate_id"]
        }

print("=== 具有遥感雷达融合匹配的船舶 ===")
for m, info in list(vessels.items())[:10]:
    print(f"MMSI {m}: decision={info['decision']}, score={info['score']}, dist={info['dist_m']}m, time={info['time']}")

# 2. 查找 2024-01-06 典型的不同工况船舶：
# - 纯锚泊系泊船（max_sog < 0.5 节）
# - 持续在航高机动船（avg_sog > 10 节）
# - 产生紧迫会遇冲突的船
print("\n=== 典型锚泊系泊船舶 ===")
anchored = conn.execute("""
    SELECT mmsi, count(*) as cnt, max(sog_knots) as max_sog, avg(sog_knots) as avg_sog
    FROM ingested_ais_positions
    WHERE timestamp_utc LIKE '2024-01-06%'
    GROUP BY mmsi
    HAVING cnt > 2000 AND max_sog < 0.5
    LIMIT 5
""").fetchall()
for a in anchored:
    print(dict(a))

print("\n=== 典型高速在航巡航船舶 ===")
cruising = conn.execute("""
    SELECT mmsi, count(*) as cnt, max(sog_knots) as max_sog, avg(sog_knots) as avg_sog
    FROM ingested_ais_positions
    WHERE timestamp_utc LIKE '2024-01-06%'
    GROUP BY mmsi
    HAVING cnt > 3000 AND avg_sog > 8.0
    ORDER BY avg_sog DESC
    LIMIT 5
""").fetchall()
for c in cruising:
    print(dict(c))
