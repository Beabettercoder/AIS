"""Frozen, paired component diagnostics using training and validation only.

The imported vendor contains the previously exercised backbone definitions.
This driver never calls its data loader, training routine or entry point.
It only opens the explicitly enumerated train.npz and val.npz feature files.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
import pilot_train_frozen as base

FEATURES = (
    "raw16", "raw_meta", "gated_meta", "mean_meta", "affine_meta",
    "reliable_raw", "reliable_gate_target", "reliable_gate_residual",
    "legacy_full", "reliable_raw_fallback",
)
REGIONS = ("houston", "ny", "la", "puget", "great_belt", "haugesund", "kattegat", "shanghai")
SPLITS = ("train", "val")
METADATA = (
    "mmsi", "window_id", "base_window_id", "region", "behavior_type",
    "noise_type", "noise_scope", "target_noise_effective",
    "neighbor_noise_effective", "observation_noise_effective",
)
EXPECTED_TRAINING = {
    "seeds": [0, 1, 2], "epochs": 24, "hidden": 32, "dropout": 0.1,
    "optimizer": "AdamW", "learning_rate": 0.002, "weight_decay": 0.01,
    "batch_size": 128, "gradient_clip_norm": 5.0, "cpu_threads_per_worker": 4,
    "loss": "binary_cross_entropy_with_logits", "PSA": False,
    "consistency_loss": False, "CDG": False,
}
VENDOR_SHA256 = "4ee25368063f92a2be6aa5991fb228c2771ca4e981123deb13e763cc732f5281"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def state_digest(state):
    """Portable digest of names, dtypes, shapes and tensor payloads, not a zip."""
    h = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8") + b"\0")
        h.update(str(array.dtype).encode("ascii") + b"\0")
        h.update(canonical_json(list(array.shape)).encode("ascii") + b"\0")
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def id_digest(dataset):
    records = [[str(dataset[k][i]) for k in ("window_id", "base_window_id", "mmsi", "region")]
               + [int(dataset["y"][i])] for i in range(len(dataset["y"]))]
    return digest_json(records)


def epoch_orders(n, seed, epochs):
    rng = np.random.default_rng(seed + 100)
    return [rng.permutation(n).astype("<i8", copy=False) for _ in range(epochs)]


def order_digests(orders):
    total = hashlib.sha256()
    per_epoch = []
    for epoch, order in enumerate(orders, 1):
        payload = np.asarray([epoch], dtype="<i8").tobytes() + np.asarray(order, dtype="<i8").tobytes()
        per_epoch.append(hashlib.sha256(payload).hexdigest())
        total.update(payload)
    return total.hexdigest(), per_epoch


def selected_epoch(history):
    # Python max returns the first occurrence, implementing the frozen tie rule.
    return max(history, key=lambda row: row["validation_macro_AP"])["epoch"]


def read_protocol(path):
    path = Path(path)
    protocol = json.loads(path.read_text(encoding="utf-8-sig"))
    if protocol["feature_names"] != list(FEATURES):
        raise ValueError("Feature variants differ from frozen experiment")
    if protocol["regions"] != list(REGIONS) or protocol["splits_read"] != list(SPLITS):
        raise ValueError("Frozen region or train/validation-only scope changed")
    if protocol["training"] != EXPECTED_TRAINING:
        raise ValueError("Training budget/configuration differs from frozen experiment")
    if protocol["samples"] != {"train_per_region": 980, "val_per_region": 210,
                                "train_total": 7840, "val_total": 1680}:
        raise ValueError("Unexpected frozen sample allocation")
    if base.sha(HERE / "vendor" / "pilot_train_frozen.py") != VENDOR_SHA256:
        raise ValueError("Frozen backbone vendor changed")
    return protocol, base.sha(path)


def load_train_val(data_root, regions=REGIONS, expected_counts=None):
    """No directory globbing and no test path construction or access."""
    if expected_counts is None:
        expected_counts = {"train": 980, "val": 210}
    pieces = {split: [] for split in SPLITS}
    sources = []
    source_rows = []
    for region in regions:
        for split in SPLITS:
            path = Path(data_root) / region / (split + ".npz")
            source_sha = base.sha(path)
            with np.load(path, allow_pickle=False) as z:
                required = set(FEATURES) | set(METADATA) | {"y", "mask"}
                if required - set(z.files):
                    raise ValueError(f"Missing keys in {path}: {sorted(required - set(z.files))}")
                y = z["y"]
                if y.ndim != 1 or not np.isin(y, [0, 1]).all():
                    raise ValueError(f"Invalid labels: {path}")
                n = len(y)
                if expected_counts and n != expected_counts[split]:
                    raise ValueError(f"Unexpected count {path}: {n}")
                if len(np.unique(y)) != 2:
                    raise ValueError(f"Both classes required in {path}")
                mask = base.validate_mask(z["mask"])
                if mask.shape != (n, 20):
                    raise ValueError(f"Unexpected mask shape in {path}")
                part = {"y": y.astype(np.int64), "mask": mask.copy(), "features": {}}
                for key in METADATA:
                    if z[key].shape != (n,):
                        raise ValueError(f"Metadata shape {path}/{key}")
                    part[key] = z[key].astype(str)
                if not np.all(part["region"] == region):
                    raise ValueError(f"Region mismatch in {path}")
                for feature in FEATURES:
                    x = z[feature].astype(np.float32)
                    if x.shape != (n, 20, 16) or not np.isfinite(x).all():
                        raise ValueError(f"Invalid feature values/shape in {path}/{feature}")
                    if np.any(x[~mask] != 0):
                        raise ValueError(f"Nonzero feature padding in {path}/{feature}")
                    part["features"][feature] = x
                sources.append({"path": str(path.resolve()), "sha256": source_sha,
                                "region": region, "split": split, "samples": n,
                                "positive": int(y.sum()), "valid_steps": int(mask.sum())})
                for i in range(n):
                    source_rows.append({"split": split, "region": region,
                                        "window_id": str(part["window_id"][i]),
                                        "base_window_id": str(part["base_window_id"][i]),
                                        "mmsi": str(part["mmsi"][i]), "y": int(y[i]),
                                        "source_row": i, "source_path": str(path.resolve()),
                                        "source_sha256": source_sha})
            if base.sha(path) != source_sha:
                raise ValueError(f"Source changed while reading {path}")
            pieces[split].append(part)
    datasets = {}
    for split, parts in pieces.items():
        datasets[split] = {k: np.concatenate([part[k] for part in parts])
                           for k in ("y", "mask") + METADATA}
        datasets[split]["features"] = {f: np.concatenate([part["features"][f] for part in parts])
                                       for f in FEATURES}
        for key in ("window_id", "base_window_id"):
            if len(set(datasets[split][key])) != len(datasets[split][key]):
                raise ValueError(f"Repeated {key} in {split}")
    for key in ("window_id", "base_window_id", "mmsi"):
        if set(datasets["train"][key]) & set(datasets["val"][key]):
            raise ValueError(f"Cross-split {key}")
    return datasets, sources, source_rows


def prediction_rows(data, scores, threshold, backbone, feature, seed):
    rows = []
    for i, score in enumerate(scores):
        row = {"backbone": backbone, "feature_group": feature, "seed": seed,
               "y": int(data["y"][i]), "score": float(score), "threshold": threshold,
               "prediction": int(score >= threshold)}
        row.update({key: str(data[key][i]) for key in METADATA})
        rows.append(row)
    return rows


def run_spec(backbone, feature, seed, config_sha, protocol_sha):
    return {"backbone": backbone, "feature_group": feature, "seed": seed,
            "configuration_sha256": config_sha, "protocol_sha256": protocol_sha}


def completed_run(path, expected_spec):
    """Resume only a fully completed, hash-verified run with the exact inputs."""
    path = Path(path)
    result_path = path / "result.json"
    complete_path = path / "completion.json"
    if not result_path.is_file() or not complete_path.is_file():
        raise ValueError(f"Incomplete run preserved; choose a fresh output or inspect it: {path}")
    completion = json.loads(complete_path.read_text(encoding="utf-8"))
    if completion["result_sha256"] != base.sha(result_path):
        raise ValueError(f"Result hash mismatch in {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["run_spec"] != expected_spec:
        raise ValueError(f"Resume configuration mismatch in {path}")
    for name, wanted in result["output_hashes"].items():
        artifact = path / name
        if not artifact.is_file() or base.sha(artifact) != wanted:
            raise ValueError(f"Resume artifact missing or changed: {artifact}")
    if result["status"] != "completed_validation_only":
        raise ValueError(f"Unfinished status in {path}")
    return result


def train_one(backbone, feature, seed, datasets, normalizer, training, out,
              config_sha, protocol_sha, input_signatures):
    """No test dataset or test output exists in this function's interface."""
    if set(datasets) != {"train", "val"}:
        raise ValueError("Only train and val datasets are accepted")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    spec = run_spec(backbone, feature, seed, config_sha, protocol_sha)
    write_json_new(out / "run_configuration.json", {
        "run_spec": spec, "training": training, "input_signatures": input_signatures,
        "normalizer_policy": "This variant's training valid steps only; padding reset after normalization",
    })
    np.savez_compressed(out / "normalizer.npz", **normalizer)
    base.seed_all(seed)
    started = time.time()
    tensors = {
        split: {
            "x": torch.from_numpy(base.normalized(data["features"][feature], data["mask"], normalizer)),
            "mask": torch.from_numpy(data["mask"]),
            "y": torch.from_numpy(data["y"].astype(np.float32)),
        }
        for split, data in datasets.items()
    }
    model = base.BackboneClassifier(backbone, 16, training["hidden"], training["dropout"])
    initial_hash = state_digest(model.state_dict())
    orders = epoch_orders(len(datasets["train"]["y"]), seed, training["epochs"])
    order_hash, per_epoch_order_hash = order_digests(orders)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    history, epoch_scores = [], []
    best, best_ap, best_epoch = None, -float("inf"), None
    for epoch, order in enumerate(orders, 1):
        model.train()
        total_loss = 0.0
        tr = tensors["train"]
        for lo in range(0, len(order), training["batch_size"]):
            indices = order[lo:lo + training["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(tr["x"][indices], tr["mask"][indices])
            loss = F.binary_cross_entropy_with_logits(logits, tr["y"][indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {backbone}/{feature}/{seed}/{epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        scores = base.predict(model, tensors["val"]["x"], tensors["val"]["mask"], training["batch_size"])
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite validation predictions")
        metrics = base.region_metrics(datasets["val"]["y"], scores, datasets["val"]["region"], 0.5)
        ap = metrics["macro"]["AP"]
        history.append({"epoch": epoch, "train_loss": total_loss / len(order), "validation_macro_AP": ap})
        epoch_scores.append(scores.copy())
        if ap > best_ap:
            best_ap, best_epoch, best = ap, epoch, copy.deepcopy(model.state_dict())
        print(f"EPOCH {backbone}/{feature}/seed{seed} {epoch}/{training['epochs']} "
              f"loss={history[-1]['train_loss']:.6f} val_macro_AP={ap:.6f}", flush=True)
    if selected_epoch(history) != best_epoch:
        raise AssertionError("Checkpoint selection differs from frozen tie policy")
    model.load_state_dict(best)
    scores = base.predict(model, tensors["val"]["x"], tensors["val"]["mask"], training["batch_size"])
    if not np.array_equal(scores, epoch_scores[best_epoch - 1]):
        raise AssertionError("Saved checkpoint replay differs from selected-epoch predictions")
    threshold, pooled_f1 = base.choose_threshold(datasets["val"]["y"], scores)
    rows = prediction_rows(datasets["val"], scores, threshold, backbone, feature, seed)
    base.write_csv(out / "val_predictions.csv", rows, list(rows[0]))
    np.savez_compressed(out / "epoch_validation_predictions.npz",
                        scores=np.stack(epoch_scores), window_id=datasets["val"]["window_id"],
                        y=datasets["val"]["y"], region=datasets["val"]["region"])
    torch.save({"state_dict": best, "backbone": backbone, "input_channels": 16,
                "hidden": training["hidden"], "dropout": training["dropout"],
                "normalizer_mean": torch.from_numpy(normalizer["mean"]),
                "normalizer_std": torch.from_numpy(normalizer["std"]),
                "threshold": threshold, "selected_epoch": best_epoch,
                "protocol_sha256": protocol_sha, "configuration_sha256": config_sha}, out / "checkpoint.pt")
    result = {
        "status": "completed_validation_only", "run_spec": spec, **spec,
        "input_channels": 16, "parameters": parameter_count,
        "epochs_budget": training["epochs"], "selected_epoch": best_epoch,
        "checkpoint_rule": "maximum validation equal-region macro AP; earliest exact tie",
        "threshold": threshold, "threshold_rule": "maximum pooled validation F1; highest threshold on tie",
        "validation_selected_pooled_F1": pooled_f1,
        "validation": base.region_metrics(datasets["val"]["y"], scores, datasets["val"]["region"], threshold),
        "history": history, "initial_state_sha256": initial_hash,
        "selected_state_sha256": state_digest(best), "batch_order_sha256": order_hash,
        "epoch_batch_order_sha256": per_epoch_order_hash,
        "input_signatures": input_signatures, "normalizer_valid_steps": int(normalizer["valid_steps"]),
        "normalizer_scope": "Separate scaler per feature variant, fitted only on training valid steps",
        "elapsed_seconds": time.time() - started,
        "scope": "Validation development diagnostics; no test access, inference or claim",
    }
    result["output_hashes"] = {path.name: base.sha(path) for path in sorted(out.iterdir()) if path.is_file()}
    write_json_new(out / "history.json", history)
    result["output_hashes"]["history.json"] = base.sha(out / "history.json")
    write_json_new(out / "result.json", result)
    write_json_new(out / "completion.json", {"status": "complete", "result_sha256": base.sha(out / "result.json")})
    print("CONFIG_COMPLETE", backbone, feature, seed, canonical_json(result["validation"]["macro"]), flush=True)
    return result


def ensure_configuration(out, configuration, source_rows, resume):
    """Each backbone owns its output folder, so separate workers do not race."""
    out = Path(out)
    config_path = out / "configuration.json"
    if out.exists():
        if not resume or not config_path.is_file():
            raise ValueError(f"Output already exists; only verified --resume is supported: {out}")
        current = json.loads(config_path.read_text(encoding="utf-8"))
        if current != configuration:
            raise ValueError("Frozen worker configuration differs; refusing resume")
        manifest_path = out / "input_manifest.csv"
        if not manifest_path.is_file():
            raise ValueError("Missing input manifest on resume")
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            actual_rows = list(csv.DictReader(stream))
        expected_rows = [{key: str(value) for key, value in row.items()} for row in source_rows]
        if actual_rows != expected_rows:
            raise ValueError("Input manifest changed on resume")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json_new(config_path, configuration)
        base.write_csv(out / "input_manifest.csv", source_rows, list(source_rows[0]))
    return base.sha(config_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Shared runs root; backbone subfolder is added")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--backbone", choices=base.BACKBONES, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    protocol, protocol_sha = read_protocol(args.protocol)
    training = protocol["training"]
    if args.seeds != training["seeds"]:
        raise ValueError("All frozen seeds must be run in the declared order")
    torch.set_num_threads(training["cpu_threads_per_worker"])
    torch.set_num_interop_threads(1)
    datasets, sources, source_rows = load_train_val(args.data)
    signatures = {split: id_digest(data) for split, data in datasets.items()}
    normalizers = {feature: base.fit_normalizer(datasets["train"]["features"][feature], datasets["train"]["mask"])
                   for feature in FEATURES}
    configuration = {
        "stage": "validation_only_component_diagnostics", "backbone": args.backbone,
        "protocol_path": str(args.protocol.resolve()), "protocol_sha256": protocol_sha,
        "driver_sha256": base.sha(Path(__file__)), "vendor_sha256": VENDOR_SHA256,
        "data_path": str(args.data.resolve()), "sources": sources, "input_signatures": signatures,
        "regions": list(REGIONS), "features": list(FEATURES), "training": training,
        "architecture": "Frozen prior-pilot BackboneClassifier; masked mean plus max pooling",
        "capacity_scope": "Identical nominal parameter counts; zero input slots do not add active feature capacity",
        "normalizer_scope": "Each feature variant separately fitted on training valid steps only",
        "versions": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                     "sklearn": __import__("sklearn").__version__},
        "device": "cpu", "counts": {split: len(data["y"]) for split, data in datasets.items()},
    }
    worker_out = args.out / args.backbone
    config_sha = ensure_configuration(worker_out, configuration, source_rows, args.resume)
    if args.prepare_only:
        print("PREPARED_NO_TRAINING", worker_out, flush=True)
        return
    results = []
    for seed in training["seeds"]:
        for feature in FEATURES:
            run_out = worker_out / f"seed_{seed}" / feature
            spec = run_spec(args.backbone, feature, seed, config_sha, protocol_sha)
            if run_out.exists():
                if not args.resume:
                    raise ValueError(f"Existing run: {run_out}")
                result = completed_run(run_out, spec)
                print("RESUME_VERIFIED", args.backbone, feature, seed, flush=True)
            else:
                result = train_one(args.backbone, feature, seed, datasets, normalizers[feature],
                                   training, run_out, config_sha, protocol_sha, signatures)
            results.append(result)
    for seed in training["seeds"]:
        paired = [row for row in results if row["seed"] == seed]
        for key in ("initial_state_sha256", "batch_order_sha256", "parameters"):
            if len({row[key] for row in paired}) != 1:
                raise AssertionError(f"Unpaired {key} for seed {seed}")
    source_unchanged = all(base.sha(Path(row["path"])) == row["sha256"] for row in sources)
    if not source_unchanged:
        raise ValueError("Source features changed during training")
    summary = {"status": "completed_validation_only", "backbone": args.backbone,
               "protocol_sha256": protocol_sha, "configuration_sha256": config_sha,
               "source_data_unchanged": source_unchanged, "runs": len(results),
               "results": [{key: row[key] for key in ("backbone", "feature_group", "seed", "parameters",
                             "selected_epoch", "threshold", "validation", "initial_state_sha256",
                             "batch_order_sha256", "elapsed_seconds")} for row in results]}
    summary_path = worker_out / "worker_summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise ValueError("Existing summary differs; preserving it")
    else:
        write_json_new(summary_path, summary)
    print("BACKBONE_COMPLETE", args.backbone, len(results), flush=True)


if __name__ == "__main__":
    main()
