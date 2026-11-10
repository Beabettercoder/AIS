import json
from pathlib import Path

import pandas as pd

base = Path("/mnt/sda/WangZijia/AIS/outputs/phase1/transfer2")
rows = []
for p in sorted(base.glob("*.json")):
    d = json.loads(p.read_text())
    a = d["agg"]
    name = p.stem
    parts = name.split("_")
    rows.append({
        "run": name,
        "setting": d["setting"],
        "method": d["method"],
        "shot": d.get("shot"),
        "clean_AP": round(a["clean_AP"]["mean"], 3),
        "full_AP": round(a["full_AP"]["mean"], 3),
        "R@1%": round(a["recall_1pct"]["mean"], 3),
        "R@5%": round(a["recall_5pct"]["mean"], 3),
    })
df = pd.DataFrame(rows)
df.to_csv(base / "table2_transfer.csv", index=False)
print(df.to_string(index=False))
