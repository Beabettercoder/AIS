"""Independent release audit: reads actual samples, source rows and all features.

Does not import the sampler, feature builder, or context-feature implementation.
No training, labels, source mutation, or reliance on a generator validation result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
KNOT_MPS = 1852. / 3600.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def close(actual, expected, name):
    if not np.allclose(actual, expected, rtol=2e-6, atol=2e-7, equal_nan=False):
        difference = np.asarray(actual,dtype=float)-np.asarray(expected,dtype=float)
        raise ValueError(f"{name}: physical recomputation mismatch; max error={np.max(np.abs(difference))}")


def coverage_bbox(meta, start, end, lon0, lat0, meters):
    first = datetime.fromtimestamp(start-600, timezone.utc).date()
    last = datetime.fromtimestamp(end, timezone.utc).date()
    days = [(first+timedelta(days=j)).isoformat() for j in range((last-first).days+1)]
    lookup = meta["coverage_bbox_by_utc_day"]
    if any(day not in lookup for day in days):
        return None
    boxes = np.asarray([lookup[day] for day in days], float)
    box = [boxes[:,0].max(), boxes[:,1].max(), boxes[:,2].min(), boxes[:,3].min()]
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    sx = meters*np.cos(np.deg2rad(lat0))
    return np.array([(box[0]-lon0)*sx, (box[1]-lat0)*meters,
                     (box[2]-lon0)*sx, (box[3]-lat0)*meters])


def raw_physics(source, config, lon0, lat0, meters):
    """Independent seven-channel physics, trust and interval-mask calculation."""
    t = np.asarray(source["timestamp"],np.float64)
    x = (np.asarray(source["lon"],np.float64)-lon0)*meters*np.cos(np.deg2rad(lat0))
    y = (np.asarray(source["lat"],np.float64)-lat0)*meters
    positions = np.column_stack([x, y])
    dt = np.r_[0., np.diff(t)]
    dp = np.vstack([np.zeros(2), np.diff(positions, axis=0)])
    rad = np.deg2rad(np.asarray(source["cog"],np.float64))
    velocity = np.asarray(source["sog"],np.float64)[:,None]*KNOT_MPS*np.column_stack([np.sin(rad),np.cos(rad)])
    R = config["radius_nm"]*1852.
    scale = config["velocity_scale_mps"]
    raw = np.column_stack([dp/R, velocity/scale, dt/config["window_seconds"], np.sin(rad),np.cos(rad)])
    valid = (dt>0)&(dt<=config["max_kinematic_gap_seconds"])
    error = np.zeros(len(t))
    ix = np.flatnonzero(valid)
    error[ix] = np.linalg.norm(dp[ix]/dt[ix,None]-(velocity[ix]+velocity[ix-1])/2, axis=1)
    reliability = np.zeros(len(t))
    reliability[ix] = np.exp(-np.maximum(0,error[ix]-config["gate_deadzone_mps"])**2/config["gate_denominator_mps2"])
    gate = np.where(valid, reliability, 1.)
    gated = raw.copy()
    gated[:,[0,1,2,3,5,6]] *= gate[:,None]
    return dict(raw=raw, gated=gated, time_s=t, positions=positions, velocity=velocity,
                kinematic_valid=valid, time_interval_valid=dt>0,
                position_increment_valid=dt>0, e_kin_mps=error,
                self_reliability=reliability, self_gate_weight=gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--registry", type=Path)
    ap.add_argument("--expected-regions", type=int, default=8)
    ap.add_argument("--ships-per-region", type=int, default=300)
    ap.add_argument("--windows-per-region", type=int, default=2700)
    ap.add_argument("--split-ships", default="210,45,45")
    ap.add_argument("--split-windows", default="1890,405,405")
    ap.add_argument("--per-vessel-cap", type=int, default=20)
    ap.add_argument("--selection-only", action="store_true")
    ap.add_argument("--protocol", choices=("strict","scene"), default="scene")
    ap.add_argument("--feature-directory")
    args = ap.parse_args()
    started = time.time()
    release = args.release.resolve()
    registry_path = args.registry or (release/"source_registry.json" if (release/"source_registry.json").exists()
                                      else release.parent/"source_registry.json")
    feature_directory = args.feature_directory or ("features" if args.protocol=="strict" else "scene_features")
    output = release/("independent_selection_audit.json" if args.selection_only else
                      ("independent_release_audit.json" if args.protocol=="strict" else "independent_scene_release_audit.json"))
    result = dict(audit_schema="independent_release_audit_v1", status="running",
                  independent_implementation=True, imports_generator_code=False,
                  context_protocol=args.protocol, feature_directory=feature_directory,
                  label_status="Unlabeled observed source windows; no injection or training in this audit",
                  quotas=dict(regions=args.expected_regions, ships_per_region=args.ships_per_region,
                              windows_per_region=args.windows_per_region,
                              split_ships=args.split_ships, split_windows=args.split_windows,
                              per_vessel_cap=args.per_vessel_cap), regions={}, errors=[])
    try:
        split_ships = dict(zip(SPLITS, map(int,args.split_ships.split(","))))
        split_windows = dict(zip(SPLITS, map(int,args.split_windows.split(","))))
        require(len(split_ships)==3 and sum(split_ships.values())==args.ships_per_region,"Invalid ship quotas")
        require(len(split_windows)==3 and sum(split_windows.values())==args.windows_per_region,"Invalid window quotas")
        registry = json.loads(registry_path.read_text(encoding="utf-8"))["regions"]
        selection_path = release/"selected_windows.parquet"
        groups_path = release/"global_all_mmsi_split.parquet"
        sel = pd.read_parquet(selection_path)
        groups = pd.read_parquet(groups_path)
        declared = json.loads((release/"selection_report.json").read_text(encoding="utf-8"))
        input_hashes = dict(selected_windows_sha256=sha(selection_path),
                            global_all_mmsi_split_sha256=sha(groups_path),
                            source_registry_sha256=sha(registry_path),
                            selection_report_sha256=sha(release/"selection_report.json"),
                            verifier_script_sha256=sha(__file__))
        if not args.selection_only:
            builder_name = "build_features.py" if args.protocol=="strict" else "build_features_scene.py"
            input_hashes["build_features_script_sha256"] = sha(release.parent/builder_name)
            input_hashes["context_module_sha256"] = sha(release.parent/"context/context_features.py")
        result["input_hashes"] = input_hashes
        require(not groups.mmsi.duplicated().any(),"Duplicate MMSI in global assignment")
        require(set(groups.split)==set(SPLITS),"Unexpected global split names")
        group_map = dict(zip(groups.mmsi.astype(str),groups.split))
        require(set(sel.split)==set(SPLITS),"Unexpected selected split names")
        require(sel.region.nunique()==args.expected_regions,"Unexpected number of release regions")
        require(not sel.window_id.duplicated().any(),"Repeated source-window ID")
        require(len(sel)==args.expected_regions*args.windows_per_region,"Incorrect total window quota")
        declared_targets = {s:set(sel.loc[sel.split==s,"mmsi"].astype(str)) for s in SPLITS}
        all_source_ids = set(group_map)
        scene_allowed = dict(train=all_source_ids-declared_targets["val"]-declared_targets["test"],
                             val=all_source_ids-declared_targets["test"],test=all_source_ids)
        all_identity_sets = {s:set() for s in SPLITS}
        target_identity_sets = {s:set() for s in SPLITS}
        observation_sets = {s:set() for s in SPLITS}
        observation_regions = {s:{} for s in SPLITS}
        duplicate_observation_events = Counter()
        cross_region_examples = []
        payload_signatures = set()
        total_edges = 0
        total_raw_steps = 0
        for region, windows in sel.groupby("region",sort=True):
            region_started = time.time()
            require(region in registry,f"Missing registry region {region}")
            require(windows.mmsi.nunique()==args.ships_per_region,f"{region}: ship quota")
            require(len(windows)==args.windows_per_region,f"{region}: window quota")
            counts = windows.groupby("mmsi").size()
            require(int(counts.max())<=args.per_vessel_cap,f"{region}: per-vessel cap")
            meta = registry[region]
            source_path = Path(meta["canonical_file"])
            source_hash = sha(source_path)
            require(source_hash==declared["source_files"][region]["canonical_sha256"],f"{region}: source hash changed")
            require(source_hash==meta["sha256"],f"{region}: registry source hash mismatch")
            canonical = pd.read_parquet(source_path).reset_index(drop=True)
            require(canonical.mmsi.astype(str).isin(group_map).all(),f"{region}: source identities missing assignment")
            # Materialize numeric columns once. Indexing the complete Arrow-backed
            # DataFrame for every window needlessly copies all source-file strings.
            canonical_columns={key:canonical[key].to_numpy() for key in
                               ("mmsi","timestamp","lon","lat","sog","cog")}
            canonical_rows=len(canonical)
            identity_time=pd.DataFrame({key:canonical_columns[key] for key in ("mmsi","timestamp")})
            # Actual canonical times per identity allow independent latest-report
            # checks without importing the context index used by the generator.
            vessel_times = {str(m):np.sort(g.timestamp.to_numpy(np.float64),kind="stable")
                            for m,g in identity_time.groupby("mmsi",sort=False)}
            del canonical,identity_time
            roi = meta["core_bbox"]
            lon0, lat0 = (roi[0]+roi[2])/2, (roi[1]+roi[3])/2
            report_path = release/feature_directory/region/"feature_report.json"
            feature_report = None
            if not args.selection_only:
                require(report_path.is_file(),f"{region}: completed feature report not present")
                feature_report = json.loads(report_path.read_text(encoding="utf-8"))
                require(feature_report["canonical_sha256"]==source_hash,f"{region}: feature source hash")
                require(feature_report["module_sha256"]==input_hashes["context_module_sha256"],f"{region}: module changed")
                for key in ("selected_windows_sha256","global_all_mmsi_split_sha256",
                            "build_features_script_sha256","source_registry_sha256"):
                    require(feature_report["construction_provenance"][key]==input_hashes[key],f"{region}: provenance {key}")
                if args.protocol=="strict":
                    require("no artificial anomaly injection" in feature_report["label_status"],f"{region}: labels not declared")
                else:
                    require(feature_report["context_protocol"]=="target_disjoint_causal_scene","Scene protocol undeclared")
                    require("SAME unlabeled source windows" in feature_report["label_status"],"Scene duplicate-sample semantics undeclared")
            region_result = dict(source_sha256=source_hash, windows=len(windows),
                                 target_vessels=int(windows.mmsi.nunique()),
                                 maximum_windows_per_vessel=int(counts.max()),splits={})
            region_identity_sets = {}
            for split in SPLITS:
                sub = windows.loc[windows.split==split].reset_index(drop=True)
                require(len(sub)==split_windows[split],f"{region}/{split}: window quota")
                require(sub.mmsi.nunique()==split_ships[split],f"{region}/{split}: ship quota")
                arrays, audit_stream = None, None
                split_access, split_steps, split_edges = set(sub.mmsi.astype(str)), 0, 0
                if not args.selection_only:
                    dest = report_path.parent
                    npz_path, audit_path = dest/f"{split}.npz", dest/f"{split}_context_audit.jsonl.gz"
                    require(sha(npz_path)==feature_report["output_hashes"][split]["npz_sha256"],f"{region}/{split}: NPZ hash")
                    require(sha(audit_path)==feature_report["output_hashes"][split]["audit_sha256"],f"{region}/{split}: audit hash")
                    with np.load(npz_path,allow_pickle=False) as archive:
                        arrays = {k:archive[k] for k in archive.files}
                    require(not any(k.lower() in {"y","label","labels","anomaly_label"} for k in arrays),"Unexpected assigned labels")
                    for key, arr in arrays.items():
                        require(arr.shape[0]==len(sub),f"{region}/{split}: {key} length")
                        if arr.dtype.kind in "fc":
                            require(np.isfinite(arr).all(),f"{region}/{split}: nonfinite {key}")
                    require(arrays["raw"].shape==(len(sub),20,7),"Incorrect raw shape")
                    for key in ("context","plain_context","plain_affine_context"):
                        require(arrays[key].shape==(len(sub),20,16),f"Incorrect {key} shape")
                    require(arrays["time_s"].dtype==np.float64,"Absolute times lose float64 precision")
                    require(arrays["raw"].dtype==np.float32 and arrays["context"].dtype==np.float32,"Unexpected feature dtype")
                    require(arrays["mask"].dtype==np.bool_,"Invalid step-mask dtype")
                    require(np.array_equal(arrays["window_id"],sub.window_id.to_numpy()),"Window order mismatch")
                    require(np.array_equal(arrays["mmsi"],sub.mmsi.to_numpy(np.int64)),"MMSI order mismatch")
                    if args.protocol=="scene":
                        strict_dir = release/"features"/region
                        strict_report = json.loads((strict_dir/"feature_report.json").read_text(encoding="utf-8"))
                        strict_npz = strict_dir/f"{split}.npz"
                        require(sha(strict_npz)==strict_report["output_hashes"][split]["npz_sha256"],"Strict reference hash changed")
                        with np.load(strict_npz,allow_pickle=False) as strict_arrays:
                            for key in ("raw","gated","mask","time_s","window_id","mmsi","start_utc_s","end_utc_s",
                                        "kinematic_valid","time_interval_valid","position_increment_valid",
                                        "e_kin_mps","self_reliability","self_gate_weight","coverage_known","boundary_context_valid"):
                                require(np.array_equal(arrays[key],strict_arrays[key]),f"Scene/strict same-window target mismatch: {key}")
                    for key, arr in arrays.items():
                        if arr.ndim>=2 and arr.shape[1]==20:
                            require(np.all(arr[~arrays["mask"]]==0),f"{region}/{split}: padding pollution in {key}")
                    audit_stream = gzip.open(audit_path,"rt",encoding="utf-8")
                try:
                    for j,row in enumerate(sub.itertuples()):
                        identity = str(row.mmsi)
                        require(group_map[identity]==split,f"{region}/{split}: target in wrong group")
                        physical = np.asarray(json.loads(row.row_indices))
                        ii = physical.astype(np.int64)
                        require(np.array_equal(ii,physical) and np.all(ii>=0) and np.all(ii<canonical_rows),"Invalid physical row indices")
                        source={key:values[ii] for key,values in canonical_columns.items()}
                        times=np.asarray(source["timestamp"],np.float64)
                        require(np.all(source["mmsi"]==row.mmsi),"Source-row MMSI mismatch")
                        require(np.array_equal(times,json.loads(row.timestamps)),"Source-row timestamp mismatch")
                        n = len(ii)
                        require(n==row.n_obs and 8<=n<=20,"Incorrect observed-step count")
                        require(np.all(np.diff(times)>=60),"Unexpected noncausal/duplicate/dense target times")
                        require(times[0]==row.start_utc_s and times[-1]==row.end_utc_s,"Source envelope mismatch")
                        require(times[-1]-times[0]<=1380,"Source window exceeds 23 min")
                        signature = (identity,tuple(times))
                        require(signature not in payload_signatures,"Duplicate source-window payload")
                        payload_signatures.add(signature)
                        target_identity_sets[split].add(identity)
                        for t in times:
                            key = (identity,float(t))
                            if key in observation_regions[split]:
                                previous = observation_regions[split][key]
                                kind = "within_region" if previous==region else "cross_region_same_split"
                                duplicate_observation_events[kind] += 1
                                if previous!=region and len(cross_region_examples)<20:
                                    cross_region_examples.append(dict(mmsi=identity,time_s=float(t),first_region=previous,region=region,split=split))
                            else:
                                observation_regions[split][key] = region
                            observation_sets[split].add(key)
                        split_steps += n
                        if args.selection_only:
                            continue
                        line = next(audit_stream,None)
                        require(line is not None,"Missing per-window audit entry")
                        md = json.loads(line)
                        require(md["window_id"]==row.window_id and md["region"]==region,"Audit window order mismatch")
                        require(md["target_mmsi"]==identity and md["split"]==split,"Audit target/split mismatch")
                        require(md["decision_time_s"]==row.end_utc_s and md["valid_steps"]==n,"Audit decision/count mismatch")
                        require(md["schema_version"]=="reliable_context_v2","Wrong shared context version")
                        if args.protocol=="scene":
                            require(md["context_protocol"]=="target_disjoint_causal_scene","Per-window scene protocol undeclared")
                        require(np.array_equal(arrays["mask"][j],np.arange(20)<n),"Incorrect padding mask")
                        require(np.array_equal(arrays["time_s"][j,:n],times),"NPZ absolute-time mismatch")
                        require(np.all(arrays["time_s"][j,:n]<=row.end_utc_s),"Future target step")
                        expected = raw_physics(source,md["config"],lon0,lat0,111195.)
                        for key in ("raw","gated","kinematic_valid","time_interval_valid",
                                    "position_increment_valid","e_kin_mps","self_reliability","self_gate_weight"):
                            close(arrays[key][j,:n],expected[key],key)
                        close(arrays["plain_context"][j,:n,:7],expected["raw"],"B raw prefix")
                        close(arrays["plain_affine_context"][j,:n,:7],expected["raw"],"B affine raw prefix")
                        close(arrays["context"][j,:n,:7],expected["gated"],"C gated prefix")
                        close(arrays["context"][j,:n,7:9],arrays["residual_mps"][j,:n]/md["config"]["velocity_scale_mps"]*expected["self_gate_weight"][:,None],"C residual input")
                        bbox = coverage_bbox(meta,row.start_utc_s,row.end_utc_s,lon0,lat0,111195.)
                        known = bbox is not None
                        require(np.all(arrays["coverage_known"][j,:n]==known),"Coverage known-mask mismatch")
                        require(md["coverage_status"]==("known_source_roi" if known else "unknown"),"Coverage metadata mismatch")
                        boundary = np.ones(n,bool)
                        if known:
                            pos = expected["positions"];R=md["config"]["radius_nm"]*1852.
                            boundary = ((pos[:,0]-R>=bbox[0])&(pos[:,1]-R>=bbox[1])&
                                        (pos[:,0]+R<=bbox[2])&(pos[:,1]+R<=bbox[3]))
                        require(np.array_equal(arrays["boundary_context_valid"][j,:n],boundary),"Boundary mask mismatch")
                        require(len(md["steps"])==n,"Audit step count mismatch")
                        accepted_ids, plain_ids = set(), set()
                        for step_number,step in enumerate(md["steps"]):
                            step_time = times[step_number]
                            require(step["step"]==step_number and step["time_s"]==step_time,"Audit time/order mismatch")
                            seen = set(); accepted_count=0;plain_count=0
                            for nb in step["neighbors"]:
                                other = str(nb["mmsi"]);nbtime=float(nb["time_s"])
                                if args.protocol=="strict":
                                    require(other in group_map and group_map[other]==split,"Cross-group neighbor input")
                                else:
                                    require(other in scene_allowed[split],"Forbidden held-out target in scene context")
                                require(other!=identity and other not in seen,"Self/duplicate neighbor")
                                require(np.isfinite(nbtime) and nbtime<=step_time,"Future/nonfinite neighbor time")
                                require(0<=step_time-nbtime<=300,"Stale neighbor")
                                require(abs(nb["age_s"]-(step_time-nbtime))<1e-8,"Neighbor age mismatch")
                                timeline=vessel_times[other]
                                latest=int(np.searchsorted(timeline,step_time,side="right"))-1
                                require(latest>=0 and timeline[latest]==nbtime,"Neighbor is not latest causal source report")
                                require(0<=nb["distance_m"]<=md["config"]["radius_nm"]*1852.+1e-5,"Neighbor outside radius")
                                if nb["accepted"]:
                                    accepted_count+=1;accepted_ids.add(other)
                                    require(nb["kinematic_valid"] and nb["reliability"]>=md["config"]["min_neighbor_reliability"],"Unqualified reliable neighbor")
                                if nb["plain_accepted"]:
                                    plain_count+=1;plain_ids.add(other)
                                seen.add(other);split_access.add(other);split_edges+=1
                            require(accepted_count==arrays["valid_neighbor_count"][j,step_number],"Reliable-neighbor count mismatch")
                            require(plain_count==arrays["plain_neighbor_count"][j,step_number],"Plain-neighbor count mismatch")
                            if known and not boundary[step_number]:
                                for key in ("background_mean_mps","background_field_mps","residual_mps",
                                            "divergence_per_s","curl_per_s","plain_background_mean_mps",
                                            "context_valid","plain_context_valid","plain_affine_valid"):
                                    require(np.all(arrays[key][j,step_number]==0),f"Boundary leaked into {key}")
                        require(set(md["context_exposure_mmsi"])==accepted_ids,"Accepted context exposure mismatch")
                        require(set(md["plain_context_exposure_mmsi"])==plain_ids,"Plain context exposure mismatch")
                    if audit_stream is not None:
                        require(next(audit_stream,None) is None,"Extra audit windows")
                finally:
                    if audit_stream is not None:audit_stream.close()
                all_identity_sets[split].update(split_access)
                region_identity_sets[split]=split_access
                total_edges+=split_edges;total_raw_steps+=split_steps
                region_result["splits"][split] = dict(windows=len(sub),vessels=int(sub.mmsi.nunique()),
                    verified_target_observations=split_steps,verified_neighbor_edges=split_edges,
                    actual_accessed_identity_count=len(split_access))
                if feature_report is not None:
                    require(set(feature_report["all_accessed_identity_sets"][split])==split_access,"Report accessed IDs differ from actual audit")
            region_result["elapsed_seconds"]=time.time()-region_started
            if args.protocol=="scene" and feature_report is not None:
                actual_intersections={a+"__"+b:len(region_identity_sets[a]&region_identity_sets[b])
                                      for a,b in (("train","val"),("train","test"),("val","test"))}
                require(actual_intersections==feature_report["actual_background_identity_intersections"],"Scene identity intersections mismatch")
                region_result["actual_cross_split_input_identity_intersections"]=actual_intersections
            result["regions"][region]=region_result
            print("VERIFIED",region,json.dumps(region_result["splits"]),flush=True)
        comparisons={}
        for a,b in (("train","val"),("train","test"),("val","test")):
            comparisons[f"{a}_vs_{b}"]=dict(target_identity_overlap=len(target_identity_sets[a]&target_identity_sets[b]),
                all_accessed_identity_overlap=len(all_identity_sets[a]&all_identity_sets[b]),
                target_observation_overlap=len(observation_sets[a]&observation_sets[b]))
            require(comparisons[f"{a}_vs_{b}"]["target_identity_overlap"]==0 and
                    comparisons[f"{a}_vs_{b}"]["target_observation_overlap"]==0,f"Cross-split target overlap {a}/{b}")
            if args.protocol=="strict":
                require(comparisons[f"{a}_vs_{b}"]["all_accessed_identity_overlap"]==0,f"Cross-split input overlap {a}/{b}")
        result.update(status="passed",verified_windows=len(sel),verified_raw_steps=total_raw_steps,
                      verified_neighbor_edges=total_edges,cross_split_checks=comparisons,
                      duplicate_source_window_payloads=0,
                      repeated_target_observation_events=dict(duplicate_observation_events),
                      cross_region_same_split_observation_examples=cross_region_examples,
                      maximum_windows_per_global_mmsi=int(sel.groupby("mmsi").size().max()),
                      global_mmsi_exceeding_per_region_cap=int((sel.groupby("mmsi").size()>args.per_vessel_cap).sum()),
                      cap_scope="Per region per vessel; cross-region contributions reported separately",
                      target_observation_counts={s:len(v) for s,v in observation_sets.items()},
                      actual_accessed_identity_counts={s:len(v) for s,v in all_identity_sets.items()},
                      same_source_windows_as_strict_verified=args.protocol=="scene" and not args.selection_only,
                      number_of_independent_source_windows=len(sel),
                      context_view_is_additional_independent_data=False,
                      selection_only=args.selection_only)
    except Exception as error:
        result["status"]="failed"
        result["errors"].append(f"{type(error).__name__}: {error}")
    result["elapsed_seconds"]=time.time()-started
    result["completed_utc"]=datetime.now(timezone.utc).isoformat()
    if release.exists():output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"report":str(output),"errors":result["errors"]}),flush=True)
    return 0 if result["status"]=="passed" else 1


if __name__=="__main__":
    sys.exit(main())
