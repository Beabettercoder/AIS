import json
from pathlib import Path

import pandas as pd

base = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
CLASSES = ["turn_back", "deviation", "stall", "sinuous", "speed_drop"]
rows = []
for level in range(1, 6):
    for mode in ("raw", "mean", "ours"):
        p = base / f"eval_v2_l{level}" / f"res_{mode}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        row = {"level": level, "method": mode}
        for c in CLASSES:
            key = f"AP_{c}"
            vals = [r[key] for r in d["runs"] if key in r]
            row[c] = round(float(sum(vals) / len(vals)), 3) if vals else None
            row[c + "_std"] = round(float((sum((v - row[c]) ** 2 for v in vals) / len(vals)) ** 0.5), 3) if vals else None
        a = d["agg"]
        row["clean_AP"] = round(a["clean_AP"]["mean"], 3)
        row["full_AP"] = round(a["full_AP"]["mean"], 3)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(base / "table3_intensity_by_class.csv", index=False)

lines = ["# Table 3 — per-class AP by anomaly strength (3 seeds)",
         "",
         "| Level | Method | Clean AP | Full AP | turn_back | deviation | stall | sinuous | speed_drop |",
         "|---|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    lines.append("| L{level} | {method} | {clean_AP} | {full_AP} | {turn_back} | {deviation} | {stall} | {sinuous} | {speed_drop} |".format(**r))
txt = "\n".join(lines)
(base / "table3_intensity_by_class.md").write_text(txt, encoding="utf-8")
print(txt)
