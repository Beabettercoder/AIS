"""Behavioral contract tests: run ``python -m unittest -v`` in this folder."""
import json
import unittest
from dataclasses import replace

import numpy as np

from context_features import (BackgroundIndex, ContextConfig, KNOT_MPS,
                              build_window, masked_mean)


def records(rows):
    names = ("mmsi", "time_s", "x_m", "y_m", "sog_kn", "cog_deg")
    return {key: [row[j] for row in rows] for j, key in enumerate(names)}


def constant_track(identity, xy, speed=10., times=(0., 60.), end=60., cog=0.):
    angle = np.deg2rad(cog)
    velocity = speed * KNOT_MPS * np.array([np.sin(angle), np.cos(angle)])
    return [[str(identity), t, *(np.array(xy) + (t-end)*velocity), speed, cog] for t in times]


def scene(positions=None):
    positions = positions or [(-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000)]
    target = constant_track("target", (0, 0))
    neighbors = sum([constant_track(f"n{i}", p) for i, p in enumerate(positions)], [])
    return target, neighbors


class ContextContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = ContextConfig(min_steps=2)

    def build(self, target, background, **kwargs):
        return build_window(records(target), records(background), decision_time_s=60.,
                            config=self.cfg, **kwargs)

    def test_uniform_motion_has_zero_residual_and_correct_field(self):
        target, background = scene()
        result = self.build(target, background)
        self.assertTrue(result["context_valid"][1])
        np.testing.assert_allclose(result["background_field_mps"][1], [0, 10*KNOT_MPS], atol=1e-10)
        np.testing.assert_allclose(result["residual_mps"][1], 0, atol=1e-10)
        self.assertEqual(result["valid_neighbor_count"][1], 4)
        self.assertAlmostEqual(result["self_reliability"][1], 1.)
        self.assertEqual(result["metadata"]["steps"][1]["reason"], "valid")

    def test_self_exclusion_and_duplicate_ship_reports(self):
        target, background = scene()
        background += target + constant_track("n0", (-1000, -1000), times=(10., 20., 30.))
        background += [background[1].copy()]  # exact duplicate of n0 at 60 s
        result = self.build(target, background)
        neighbors = result["metadata"]["steps"][1]["neighbors"]
        self.assertEqual(len(neighbors), 4)
        self.assertEqual(len(set(n["mmsi"] for n in neighbors)), 4)
        self.assertNotIn("target", result["metadata"]["context_exposure_mmsi"])
        self.assertEqual(result["metadata"]["background_duplicates_removed"], 1)

    def test_future_reports_never_change_past_output(self):
        target, background = scene()
        expected = self.build(target, background)
        background += [["n0", 61., 0, 0, 90., 180.], ["future", 100., 0, 0, 10., 0.]]
        target += [["target", 61., 9000., 8000., 90., 180.]]
        result = self.build(target, background)
        for key in ("raw", "gated", "context", "self_reliability", "background_field_mps"):
            np.testing.assert_array_equal(result[key], expected[key])
        self.assertTrue(all(n["time_s"] <= step["time_s"] for step in result["metadata"]["steps"] for n in step["neighbors"]))

    def test_pollution_recomputes_neighbor_gate_and_field(self):
        target, background = scene()
        clean = self.build(target, background)
        background[1][2] += 3000
        polluted = self.build(target, background)
        self.assertNotEqual(clean["metadata"]["background_fingerprint"], polluted["metadata"]["background_fingerprint"])
        self.assertEqual(polluted["valid_neighbor_count"][1], 3)
        self.assertFalse(polluted["context_valid"][1])
        bad = [n for n in polluted["metadata"]["steps"][1]["neighbors"] if n["mmsi"] == "n0"][0]
        self.assertLess(bad["reliability"], 0.3)
        self.assertEqual(bad["rejection"], "unreliable_observation")
        np.testing.assert_array_equal(polluted["residual_mps"][1], [0, 0])
        # Ordinary B sees the same raw candidate, but does not invoke C's gate.
        self.assertEqual(polluted["plain_neighbor_count"][1], 4)
        self.assertTrue(polluted["plain_context_valid"][1])
        self.assertIn("n0", polluted["metadata"]["plain_context_exposure_mmsi"])

    def test_pollution_recomputes_target_gate(self):
        target, background = scene()
        clean = self.build(target, background)
        target[1][2] += 1000
        polluted = self.build(target, background)
        self.assertGreater(polluted["e_kin_mps"][1], 10.)
        self.assertLess(polluted["self_reliability"][1], clean["self_reliability"][1])
        self.assertLess(abs(polluted["gated"][1, 0]), abs(polluted["raw"][1, 0]))

    def test_sparse_and_collinear_context_fallback(self):
        target, background = scene()
        sparse = self.build(target, background[:6])
        self.assertFalse(sparse["context_valid"][1])
        self.assertEqual(sparse["metadata"]["steps"][1]["reason"], "sparse")
        self.assertEqual(sparse["support_fraction"][1], .75)
        target, background = scene([(-1500, 0), (-500, 0), (500, 0), (1500, 0)])
        linear = self.build(target, background)
        self.assertFalse(linear["context_valid"][1])
        self.assertEqual(linear["metadata"]["steps"][1]["reason"], "degenerate_geometry")
        np.testing.assert_array_equal(linear["background_field_mps"][1], [0, 0])

    def test_train_allowlist_and_exposure(self):
        target, background = scene()
        with self.assertRaises(ValueError):
            self.build(target, background, split="train")
        result = self.build(target, background, split="train", eligible_neighbor_mmsi={"n0", "n2"})
        self.assertEqual(result["metadata"]["context_exposure_mmsi"], ["n0", "n2"])
        self.assertEqual(result["valid_neighbor_count"][1], 2)

    def test_padding_masks_and_json_metadata(self):
        target, background = scene()
        result = self.build(target, background)
        self.assertEqual(result["raw"].shape, (20, 7))
        self.assertEqual(result["context"].shape, (20, 16))
        self.assertEqual(result["mask"].sum(), 2)
        self.assertEqual(result["kinematic_valid"].sum(), 1)
        for key in ("raw", "gated", "context", "plain_context", "plain_affine_context", "time_s", "background_field_mps"):
            np.testing.assert_array_equal(result[key][2:], np.zeros_like(result[key][2:]))
        np.testing.assert_allclose(masked_mean(result["raw"], result["mask"]), result["raw"][:2].mean(axis=0))
        json.dumps(result["metadata"], allow_nan=False)

    def test_plain_controls_have_same_dimension_and_no_reliability_leak(self):
        target, background = scene()
        result = self.build(target, background)
        for name in ("plain_context", "plain_affine_context"):
            self.assertEqual(result[name].shape, result["context"].shape)
            np.testing.assert_array_equal(result[name][:, :7], result["raw"])
            np.testing.assert_array_equal(result[name][:, 9:12], 0)
        self.assertEqual(result["metadata"]["context_exposure_mmsi"],
                         result["metadata"]["plain_context_exposure_mmsi"])

    def test_empty_background_has_masked_zero_context(self):
        target, _ = scene()
        result = self.build(target, [])
        self.assertFalse(result["context_valid"].any())
        self.assertFalse(result["plain_context_valid"].any())
        np.testing.assert_array_equal(result["residual_mps"], 0)
        json.dumps(result["metadata"], allow_nan=False)

    def test_trapezoid_speed_change_is_kinematically_compatible(self):
        v0, v1 = 5., 10.
        target = [["target", 0., 0., 0., v0 / KNOT_MPS, 0.],
                  ["target", 60., 0., (v0+v1)/2*60., v1 / KNOT_MPS, 0.]]
        result = self.build(target, [])
        self.assertAlmostEqual(result["e_kin_mps"][1], 0., places=10)
        self.assertEqual(result["self_reliability"][1], 1.)

    def test_roi_boundary_is_not_natural_sparsity(self):
        target, background = scene()
        full = self.build(target, background)
        boundary = self.build(target, background, known_coverage_bbox_m=(-2000., -2000., 2000., 2000.))
        self.assertFalse(boundary["boundary_context_valid"].any())
        self.assertFalse(boundary["context_valid"].any())
        self.assertFalse(boundary["plain_context_valid"].any())
        self.assertFalse(boundary["plain_affine_valid"].any())
        self.assertEqual(boundary["metadata"]["steps"][1]["reason"], "coverage_boundary")
        self.assertEqual(boundary["metadata"]["steps"][1]["context_reason_before_boundary_mask"], "valid")
        for name in ("raw", "gated", "valid_neighbor_count", "plain_neighbor_count"):
            np.testing.assert_array_equal(boundary[name], full[name])
        for name in ("background_mean_mps", "background_field_mps", "residual_mps",
                     "divergence_per_s", "curl_per_s", "plain_background_mean_mps"):
            np.testing.assert_array_equal(boundary[name], 0)
        self.assertEqual(boundary["metadata"]["context_exposure_mmsi"], full["metadata"]["context_exposure_mmsi"])
        self.assertEqual(boundary["metadata"]["plain_context_exposure_mmsi"], full["metadata"]["plain_context_exposure_mmsi"])

    def test_complete_roi_circle_and_unknown_coverage(self):
        target, background = scene()
        full = self.build(target, background, known_coverage_bbox_m=(-20000., -20000., 20000., 20000.))
        unknown = self.build(target, background)
        self.assertTrue(full["boundary_context_valid"][:2].all())
        self.assertTrue(full["context_valid"][1])
        self.assertEqual(full["metadata"]["coverage_status"], "known_source_roi")
        self.assertEqual(unknown["metadata"]["coverage_status"], "unknown")
        np.testing.assert_array_equal(full["context"], unknown["context"])

    def test_local_background_snapshot_preserves_causal_features(self):
        target, background = scene()
        old = constant_track("n0", (-1000, -1000), times=(-4000., -3000., -2000.))
        index = BackgroundIndex(records(background+old), self.cfg)
        local = BackgroundIndex(index.window_observations(60.), self.cfg)
        full = build_window(records(target), index, decision_time_s=60.)
        cropped = build_window(records(target), local, decision_time_s=60.)
        for name in ("context", "plain_context", "plain_affine_context", "self_reliability"):
            np.testing.assert_array_equal(full[name], cropped[name])

    def test_long_gap_and_first_observation_are_not_certified(self):
        target = constant_track("target", (0, 0), times=(-600., 60.))
        _, background = scene()
        result = self.build(target, background)
        self.assertFalse(result["kinematic_valid"].any())
        np.testing.assert_array_equal(result["self_reliability"], 0)

    def test_long_sampling_gap_remains_in_raw_features(self):
        target = constant_track("target", (0, 0), times=(-600., 60.))
        _, background = scene()
        result = self.build(target, background)
        self.assertFalse(result["kinematic_valid"][1])
        self.assertTrue(result["time_interval_valid"][1])
        self.assertTrue(result["position_increment_valid"][1])
        self.assertAlmostEqual(result["raw"][1, 4], 660/self.cfg.window_seconds)
        self.assertAlmostEqual(result["raw"][1, 1], 660*10*KNOT_MPS/(self.cfg.radius_nm*1852))
        self.assertEqual(result["self_reliability"][1], 0.)
        self.assertEqual(result["self_gate_weight"][1], 1.)
        np.testing.assert_array_equal(result["gated"][1], result["raw"][1])

    def test_first_observation_preserves_target_motion_without_certifying_it(self):
        target, background = scene()
        result = self.build(target, background)
        self.assertFalse(result["kinematic_valid"][0])
        self.assertEqual(result["self_reliability"][0], 0.)
        self.assertEqual(result["self_gate_weight"][0], 1.)
        self.assertGreater(result["raw"][0, 3], 0.)
        np.testing.assert_array_equal(result["gated"][0], result["raw"][0])
        self.assertEqual(result["valid_neighbor_count"][0], 0)
        self.assertEqual(result["plain_neighbor_count"][0], 4)

    def test_unknown_target_gate_preserves_valid_context_residual(self):
        target = constant_track("target", (0, 0), speed=8.)
        positions = [(-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000)]
        background = sum([constant_track(f"n{i}", p, times=(-60., 0., 60.)) for i, p in enumerate(positions)], [])
        result = self.build(target, background)
        self.assertTrue(result["context_valid"][0])
        self.assertFalse(result["kinematic_valid"][0])
        self.assertAlmostEqual(result["context"][0, 8], -2*KNOT_MPS/self.cfg.velocity_scale_mps)
        self.assertEqual(result["context"][0, 10], 0.)
        self.assertEqual(result["context"][0, 11], 0.)

    def test_invalid_predecessor_position_preserves_known_elapsed_time_only(self):
        target, background = scene()
        target[0][2] = float("nan")
        result = self.build(target, background)
        self.assertFalse(result["mask"][0])
        self.assertFalse(result["position_increment_valid"][1])
        self.assertFalse(result["kinematic_valid"][1])
        self.assertTrue(result["time_interval_valid"][1])
        self.assertAlmostEqual(result["raw"][1, 4], 60/self.cfg.window_seconds)
        np.testing.assert_array_equal(result["raw"][1, :2], 0.)
        np.testing.assert_array_equal(result["gated"][1], result["raw"][1])

    def test_invalid_report_is_not_replaced_by_older_valid_neighbor(self):
        target, background = scene()
        background += [["n0", 59., -1000, -1000-10*KNOT_MPS, 10., 0.]]
        background[1][4] = 102.3  # AIS speed unavailable sentinel at latest time
        result = self.build(target, background)
        self.assertEqual(result["valid_neighbor_count"][1], 3)
        bad = [n for n in result["metadata"]["steps"][1]["neighbors"] if n["mmsi"] == "n0"][0]
        self.assertEqual(bad["time_s"], 60.)
        self.assertEqual(bad["rejection"], "invalid_observation")

    def test_stale_and_heading_mismatched_neighbors_excluded(self):
        target, background = scene()
        background += constant_track("stale", (400, 400), times=(-400., -300.))
        background += constant_track("opposite", (400, 400), cog=180.)
        result = self.build(target, background)
        neighbors = result["metadata"]["steps"][1]["neighbors"]
        self.assertNotIn("stale", [n["mmsi"] for n in neighbors])
        self.assertNotIn("opposite", result["metadata"]["context_exposure_mmsi"])
        self.assertEqual(result["valid_neighbor_count"][1], 4)

    def test_index_config_and_array_immutability(self):
        target, background = scene()
        idx = BackgroundIndex(records(background), self.cfg)
        with self.assertRaises(ValueError):
            idx.data["x_m"][0] = 0
        with self.assertRaises(ValueError):
            build_window(records(target), idx, decision_time_s=60., config=replace(self.cfg, radius_nm=3.5))

    def test_affine_field_divergence_and_curl_have_physical_units(self):
        positions = np.array([[-1000, -1000], [1000, -1000], [1000, 1000], [-1000, 1000]], float)
        rows = []
        # vx = 0.0001*x - 0.0002*y; vy = 5 + 0.0003*x + 0.0004*y
        for i, xy in enumerate(positions):
            vx = .0001*xy[0] - .0002*xy[1]
            vy = 5 + .0003*xy[0] + .0004*xy[1]
            speed = np.hypot(vx, vy)/KNOT_MPS
            cog = np.rad2deg(np.arctan2(vx, vy)) % 360
            rows += constant_track(f"n{i}", xy, speed=speed, cog=cog)
        target = constant_track("target", (0, 0), speed=5/KNOT_MPS)
        cfg = replace(self.cfg, ridge=0.)
        result = build_window(records(target), records(rows), decision_time_s=60., config=cfg)
        self.assertTrue(result["context_valid"][1])
        np.testing.assert_allclose(result["background_field_mps"][1], [0, 5], atol=1e-10)
        self.assertAlmostEqual(result["divergence_per_s"][1], .0005, places=10)
        self.assertAlmostEqual(result["curl_per_s"][1], .0005, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
