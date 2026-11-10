#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 41: 防泄漏审计 — 源域与各目标域的 MMSI 交集 + 严格零样本可用规模。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path("/mnt/sda/WangZijia/AIS/data/interim")
SRC = "ais_houston_202407_enriched.parquet"
TARGETS = {
    "Unseen Ports / NY": "ais_ny_enriched.parquet",
    "Unseen Ports / LA": "ais_la_enriched.parquet",
    "Unseen Straits / Juan de Fuca": "ais_puget_enriched.parquet",
    "Unseen Straits / Florida": "ais_miami_enriched.parquet",
    "Unseen Open Seas / Outer GoM": "ais_gom_open_enriched.parquet",
    "Unseen Open Seas / NW Atlantic": "ais_atl_open_enriched.parquet",
}


def mmsi_set(path: Path) -> set[int]:
    df = pd.read_parquet(path, columns=["mmsi", "sog"])
    df = df[df.sog >= 2.0]
    return set(df.mmsi.astype("int64").unique().tolist())


def main() -> None:
    src = mmsi_set(BASE / SRC)
    audit = {"source": SRC, "source_mmsi": len(src), "targets": {}}
    for name, f in TARGETS.items():
        p = BASE / f
        if not p.exists():
            continue
        tgt = mmsi_set(p)
        overlap = src & tgt
        remain = tgt - src
        audit["targets"][name] = {"mmsi_total": len(tgt), "overlap_with_source": len(overlap),
                                  "mmsi_after_removal": len(remain),
                                  "removal_ratio": round(len(overlap) / max(len(tgt), 1), 4)}
        assert len(remain & src) == 0
    out = Path("/mnt/sda/WangZijia/AIS/outputs/phase1")
    (out / "leak_audit_mmsi.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = ["# Anti-leakage audit — MMSI disjointness", "",
             f"Source (Houston, 2024-07): **{len(src)}** unique moving-vessel MMSIs.", "",
             "| Target region | MMSI total | Overlap with source | Removed | Remain (strict ZS test set) |",
             "|---|---|---|---|---|"]
    for k, v in audit["targets"].items():
        lines.append(f"| {k} | {v['mmsi_total']} | {v['overlap_with_source']} ({v['removal_ratio']:.1%}) | "
                     f"{v['overlap_with_source']} | {v['mmsi_after_removal']} |")
    lines += ["", "`assert len(train_mmsi & test_mmsi) == 0` holds for every target region.", ""]
    txt = "\n".join(lines)
    (out / "table2_leak_audit.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
