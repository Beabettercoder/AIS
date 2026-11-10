import math
import pytest
from maritime_agent.detectors.sar_geometry import (
    SarPlatformGeometry,
    resolve_sar_geometry,
)


def test_default_ascending_geometry():
    geom = resolve_sar_geometry({"orbit_pass": "ASCENDING"})
    assert geom.orbit_pass == "ASCENDING"
    assert geom.heading_deg == 350.0
    assert geom.look_direction == "RIGHT"
    assert geom.ground_look_heading_deg == 80.0
    assert geom.v_sat_mps == 7590.0
    assert geom.incidence_angle_deg == 38.5
    assert geom.altitude_m == 693000.0
    # R0 = 693000 / cos(38.5 deg) ~ 885474 m
    expected_r0 = 693000.0 / math.cos(math.radians(38.5))
    assert math.isclose(geom.slant_range_m, expected_r0, rel_tol=1e-3)
    # K_shift = R0 / V_sat ~ 116.66 s
    assert math.isclose(geom.k_shift_s, expected_r0 / 7590.0, rel_tol=1e-3)


def test_descending_geometry():
    geom = resolve_sar_geometry({"orbit_pass": "DESCENDING"})
    assert geom.orbit_pass == "DESCENDING"
    assert geom.heading_deg == 190.0
    assert geom.look_direction == "RIGHT"
    assert geom.ground_look_heading_deg == 280.0


def test_custom_parameters_override():
    custom = {
        "sar_geometry": {
            "orbit_pass": "ASCENDING",
            "heading_deg": 352.5,
            "look_direction": "LEFT",
            "incidence_angle_deg": 41.2,
            "v_sat_mps": 7600.0,
            "altitude_m": 700000.0,
        }
    }
    geom = resolve_sar_geometry(custom)
    assert geom.heading_deg == 352.5
    assert geom.look_direction == "LEFT"
    # LEFT look: (352.5 - 90) = 262.5
    assert math.isclose(geom.ground_look_heading_deg, 262.5, abs_tol=1e-2)
    assert geom.incidence_angle_deg == 41.2


def test_geometry_dict_roundtrip():
    geom = resolve_sar_geometry({"orbit_pass": "ASCENDING"})
    data = geom.to_dict()
    assert isinstance(data, dict)
    restored = SarPlatformGeometry.from_dict(data)
    assert restored.orbit_pass == geom.orbit_pass
    assert restored.heading_deg == geom.heading_deg
    assert restored.k_shift_s == geom.k_shift_s
