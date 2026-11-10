import json
from pathlib import Path

base = Path("/mnt/sda/WangZijia/AIS/outputs/phase1/transfer2")
SPEC = [
    ("NY (Houston→NY)", {
        "S1 目标域内": "s1_NY_ours.json",
        "S2 零样本": "recal_H2NY.json:S2_source_norm",
        "S2+ 重标定": "recal_H2NY.json:S2plus_target_norm",
        "S3 1-shot": "recal_H2NY.json:S3_proto_1shot",
        "S3 5-shot": "recal_H2NY.json:S3_proto_5shot",
    }),
    ("LA (Houston→LA)", {
        "S1 目标域内": "s1_LA_ours.json",
        "S2 零样本": "recal_H2LA.json:S2_source_norm",
        "S2+ 重标定": "recal_H2LA.json:S2plus_target_norm",
        "S3 1-shot": "recal_H2LA.json:S3_proto_1shot",
        "S3 5-shot": "recal_H2LA.json:S3_proto_5shot",
    }),
    ("Zone A (from Zone B)", {
        "S1 目标域内": "s1_A_ours.json",
        "S2 零样本": "s2_B2A_ours.json",
        "S2+ 重标定": "recal_B2A.json:S2plus_target_norm",
        "S3 1-shot": "recal_B2A.json:S3_proto_1shot",
        "S3 5-shot": "s3_B2A_ours_5shot.json",
    }),
    ("Zone B (from Zone A)", {
        "S1 目标域内": "s1_B_ours.json",
        "S2 零样本": "s2_A2B_ours.json",
        "S2+ 重标定": "recal_A2B.json:S2plus_target_norm",
        "S3 1-shot": "recal_A2B.json:S3_proto_1shot",
        "S3 5-shot": "recal_A2B.json:S3_proto_5shot",
    }),
]


def get(fname):
    if fname is None:
        return None
    key = None
    if ":" in fname:
        fname, key = fname.split(":")
    path = base / fname
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if key:
        return d["summary"][key]
    a = d.get("agg", {})
    return {k: v["mean"] for k, v in a.items() if k in ("clean_AP", "full_AP", "recall_1pct")}


lines = ["# Table 2 — cross-domain transfer (3 seeds)", "",
         "| Target | Setting | Clean AP | Full AP | Recall@1% |", "|---|---|---|---|---|"]
for name, settings in SPEC:
    for label, fname in settings.items():
        r = get(fname)
        if not r:
            lines.append(f"| {name} | {label} | - | - | - |")
            continue
        lines.append(f"| {name} | {label} | {r['clean_AP']:.3f} | {r['full_AP']:.3f} | {r['recall_1pct']:.3f} |")
txt = "\n".join(lines)
(base / "table2_final.md").write_text(txt, encoding="utf-8")
print(txt)
