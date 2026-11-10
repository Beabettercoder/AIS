import json
from pathlib import Path

BASE = Path("/mnt/sda/WangZijia/AIS/outputs/phase1/transfer2")
FILES = {
    "Unseen Ports|NY": "recal_H2NY.json",
    "Unseen Ports|LA": "recal_H2LA.json",
    "Unseen Straits|Juan de Fuca": "recal_H2puget.json",
    "Unseen Straits|Florida": "recal_H2miami.json",
    "Unseen Open Seas|Outer GoM": "recal_H2gom_open.json",
    "Unseen Open Seas|NW Atlantic": "recal_H2atl_open.json",
}

rows = []
for name, fn in FILES.items():
    p = BASE / fn
    if not p.exists():
        print("missing", fn)
        continue
    d = json.loads(p.read_text())["summary"]
    scen, reg = name.split("|")
    rows.append({
        "scenario": scen, "region": reg,
        "S2_clean": d["S2_source_norm"]["clean_AP"], "S2_full": d["S2_source_norm"]["full_AP"],
        "S2p_clean": d["S2plus_target_norm"]["clean_AP"], "S2p_full": d["S2plus_target_norm"]["full_AP"],
    })
    print(rows[-1])

out = Path("/mnt/sda/WangZijia/AIS/outputs/phase1/table2_strict_zeroshot.json")
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

lines = ["# Strict zero-shot transfer: Ports -> Unseen Scenarios (S2 / S2+)", "",
         "| Scenario | Region | S2 Clean AP | S2 Full AP | S2+ Clean AP | S2+ Full AP | Δ (S2+−S2) |",
         "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['scenario']} | {r['region']} | {r['S2_clean']:.3f} | {r['S2_full']:.3f} | "
                 f"{r['S2p_clean']:.3f} | {r['S2p_full']:.3f} | {r['S2p_full']-r['S2_full']:+.3f} |")
txt = "\n".join(lines)
(Path("/mnt/sda/WangZijia/AIS/outputs/phase1") / "table2_strict_zeroshot.md").write_text(txt, encoding="utf-8")
print(txt)
