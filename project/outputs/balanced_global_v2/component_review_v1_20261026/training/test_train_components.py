"""Small synthetic checks of the validation-only experiment driver.

These tests never open benchmark data or launch the research experiment.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import train_components as driver


def fixture(root, region="tiny", n=4):
    directory = Path(root) / region
    directory.mkdir(parents=True)
    for split, seed in (("train", 13), ("val", 29)):
        rng = np.random.default_rng(seed)
        mask = np.zeros((n, 20), dtype=bool)
        mask[:, :10] = True
        x = rng.normal(size=(n, 20, 16)).astype(np.float32)
        x[~mask] = 0
        payload = {"y": np.array([0, 1] * (n // 2), dtype=np.int8), "mask": mask,
                   "window_id": np.array([f"{split}-w-{i}" for i in range(n)]),
                   "base_window_id": np.array([f"{split}-b-{i}" for i in range(n)]),
                   "mmsi": np.array([f"{split}-m-{i}" for i in range(n)]),
                   "region": np.repeat(region, n), "behavior_type": np.repeat("none", n),
                   "noise_type": np.repeat("none", n), "noise_scope": np.repeat("none", n),
                   "target_noise_effective": np.zeros(n, bool),
                   "neighbor_noise_effective": np.zeros(n, bool),
                   "observation_noise_effective": np.zeros(n, bool)}
        payload.update({feature: x.copy() for feature in driver.FEATURES})
        np.savez_compressed(directory / (split + ".npz"), **payload)
    # A corrupt file deliberately placed alongside the permitted files.
    (directory / "test.npz").write_bytes(b"TEST MUST NEVER BE OPENED")


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="component-driver-test-")
        self.root = Path(self.tmp.name)
        fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        return driver.load_train_val(self.root, regions=("tiny",), expected_counts={"train": 4, "val": 4})

    def test_only_explicit_train_and_val_are_opened(self):
        original = np.load
        opened = []
        def observed(path, *args, **kwargs):
            opened.append(Path(path).name)
            self.assertIn(Path(path).name, ("train.npz", "val.npz"))
            return original(path, *args, **kwargs)
        with mock.patch.object(driver.np, "load", side_effect=observed):
            data, sources, rows = self.load()
        self.assertEqual(opened, ["train.npz", "val.npz"])
        self.assertEqual(set(data), {"train", "val"})
        self.assertEqual(len(sources), 2)
        self.assertEqual(len(rows), 8)

    def test_holey_masks_fail(self):
        mask = np.array([[True, False, True]], bool)
        with self.assertRaises(ValueError):
            driver.base.validate_mask(mask)

    def test_nonzero_padding_and_cross_split_identity_fail(self):
        p = self.root / "tiny" / "val.npz"
        with np.load(p, allow_pickle=False) as z:
            values = {key: z[key].copy() for key in z.files}
        values["raw16"][0, 19, 0] = 1
        np.savez_compressed(p, **values)
        with self.assertRaisesRegex(ValueError, "padding"):
            self.load()
        values["raw16"][0, 19, 0] = 0
        values["mmsi"] = values["mmsi"].astype("U24")
        values["mmsi"][0] = "train-m-0"
        np.savez_compressed(p, **values)
        with self.assertRaisesRegex(ValueError, "Cross-split mmsi"):
            self.load()

    def test_normalization_uses_only_training_valid_steps(self):
        data, _, _ = self.load()
        train = data["train"]
        expected = train["features"]["raw16"][train["mask"]].astype(np.float64)
        scaler = driver.base.fit_normalizer(train["features"]["raw16"], train["mask"])
        np.testing.assert_allclose(scaler["mean"], expected.mean(0), atol=1e-7)
        np.testing.assert_allclose(scaler["std"], expected.std(0), atol=1e-7)
        self.assertEqual(scaler["valid_steps"], 40)
        data["val"]["features"]["raw16"][:] = 1e9
        again = driver.base.fit_normalizer(train["features"]["raw16"], train["mask"])
        np.testing.assert_array_equal(scaler["mean"], again["mean"])
        transformed = driver.base.normalized(train["features"]["raw16"], train["mask"], scaler)
        self.assertTrue(np.all(transformed[~train["mask"]] == 0))


class SelectionAndPairingTests(unittest.TestCase):
    def test_earliest_epoch_wins_exact_tie(self):
        history = [{"epoch": i + 1, "validation_macro_AP": x} for i, x in enumerate([0.4, 0.7, 0.7, 0.6])]
        self.assertEqual(driver.selected_epoch(history), 2)

    def test_threshold_matches_exhaustive_validation_f1(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        p = np.array([0.1, 0.7, 0.5, 0.7, 0.5, 0.9])
        scores = []
        for threshold in sorted(set(p)):
            pred = p >= threshold
            tp = ((y == 1) & pred).sum()
            fp = ((y == 0) & pred).sum()
            fn = ((y == 1) & ~pred).sum()
            scores.append((2 * tp / max(1, 2 * tp + fp + fn), threshold))
        expected_f1, expected_threshold = max(scores)
        threshold, f1 = driver.base.choose_threshold(y, p)
        self.assertEqual(threshold, expected_threshold)
        self.assertAlmostEqual(f1, expected_f1)

    def test_initialization_and_order_are_paired(self):
        for backbone in ("tcn", "gru", "transformer"):
            hashes = []
            for unused_feature in ("raw16", "legacy_full"):
                driver.base.seed_all(1)
                model = driver.base.BackboneClassifier(backbone, 16, 8, 0.1)
                hashes.append(driver.state_digest(model.state_dict()))
            self.assertEqual(hashes[0], hashes[1])
        a = driver.order_digests(driver.epoch_orders(41, 1, 3))
        b = driver.order_digests(driver.epoch_orders(41, 1, 3))
        c = driver.order_digests(driver.epoch_orders(41, 2, 3))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_padded_values_do_not_change_pooled_predictions(self):
        torch.set_num_threads(1)
        for backbone in ("tcn", "gru", "transformer"):
            driver.base.seed_all(0)
            model = driver.base.BackboneClassifier(backbone, 16, 8, 0.0).eval()
            mask = torch.tensor([[True] * 8 + [False] * 12])
            x = torch.randn(1, 20, 16)
            other = x.clone()
            other[~mask] = 10000
            with torch.inference_mode():
                torch.testing.assert_close(model(x, mask), model(other, mask), rtol=0, atol=0)


class EndToEndSyntheticTests(unittest.TestCase):
    def test_paired_runs_predictions_hashes_and_safe_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="component-synthetic-run-") as tmp:
            root = Path(tmp)
            fixture(root / "data")
            data, _, _ = driver.load_train_val(root / "data", regions=("tiny",),
                                               expected_counts={"train": 4, "val": 4})
            config = dict(driver.EXPECTED_TRAINING)
            config.update(epochs=2, hidden=8, dropout=0.0, batch_size=4)
            signatures = {split: driver.id_digest(value) for split, value in data.items()}
            results = []
            for feature in ("raw16", "raw_meta"):
                normalizer = driver.base.fit_normalizer(data["train"]["features"][feature], data["train"]["mask"])
                results.append(driver.train_one("tcn", feature, 0, data, normalizer, config,
                                                 root / feature, "config-hash", "protocol-hash", signatures))
            a, b = results
            for key in ("initial_state_sha256", "batch_order_sha256", "selected_state_sha256", "history"):
                self.assertEqual(a[key], b[key])
            with np.load(root / "raw16" / "epoch_validation_predictions.npz", allow_pickle=False) as z:
                self.assertEqual(z["scores"].shape, (2, 4))
                self.assertTrue(np.array_equal(z["window_id"], data["val"]["window_id"]))
            self.assertFalse(any("test" in p.name for p in (root / "raw16").iterdir()))
            spec = driver.run_spec("tcn", "raw16", 0, "config-hash", "protocol-hash")
            self.assertEqual(driver.completed_run(root / "raw16", spec)["status"], "completed_validation_only")
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                driver.completed_run(root / "raw16", {**spec, "seed": 1})
            with (root / "raw16" / "val_predictions.csv").open("a", encoding="utf-8") as stream:
                stream.write("altered\n")
            with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
                driver.completed_run(root / "raw16", spec)
            incomplete = root / "incomplete"
            incomplete.mkdir()
            with self.assertRaisesRegex(ValueError, "Incomplete run"):
                driver.completed_run(incomplete, spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
