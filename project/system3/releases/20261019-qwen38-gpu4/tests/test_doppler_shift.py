import math
import pytest
from maritime_agent.detectors.sar_geometry import resolve_sar_geometry
from maritime_agent.detectors.doppler_shift import (
    decompose_radial_velocity,
    compute_azimuth_displacement_m,
    compute_doppler_shifted_position,
    restore_ground_position_from_sar_echo,
    _geodesic_direct,
)


def test_radial_velocity_decomposition():
    geom = resolve_sar_geometry({"orbit_pass": "ASCENDING"})
    # Ascending: heading=350, look=RIGHT -> ground_look=80 deg
    # Case 1: vessel heading directly towards radar (cog=80 deg, sog=12 kn)
    sog_kn = 12.0
    cog_deg = 80.0
    decomp = decompose_radial_velocity(sog_kn, cog_deg, geom)
    v_ship = sog_kn * 0.51444444
    assert math.isclose(decomp["v_ship_mps"], v_ship, abs_tol=1e-2)
    assert math.isclose(decomp["aspect_angle_deg"], 0.0, abs_tol=1e-2)
    # v_radial_ground = v_ship
    assert math.isclose(decomp["v_radial_ground_mps"], v_ship, abs_tol=1e-2)
    # v_radial_slant = v_ship * sin(38.5 deg)
    expected_slant = v_ship * math.sin(math.radians(38.5))
    assert math.isclose(decomp["v_radial_slant_mps"], expected_slant, abs_tol=1e-2)

    # Case 2: vessel heading perpendicular to radar (cog=170 deg, aspect=90 deg)
    decomp_perp = decompose_radial_velocity(sog_kn, 170.0, geom)
    assert math.isclose(decomp_perp["v_radial_ground_mps"], 0.0, abs_tol=1e-2)
    assert math.isclose(decomp_perp["v_radial_slant_mps"], 0.0, abs_tol=1e-2)


def test_azimuth_displacement_magnitude():
    geom = resolve_sar_geometry({"orbit_pass": "ASCENDING"})
    # For a vessel moving towards radar at 15 knots (aspect=0):
    sog_kn = 15.0
    decomp = decompose_radial_velocity(sog_kn, 80.0, geom)
    v_r = decomp["v_radial_slant_mps"]
    shift_m = compute_azimuth_displacement_m(v_r, geom)
    # v_r > 0, so shift_m should be negative (backward along track)
    assert shift_m < 0
    # Magnitude: ~ 116.65 s * (15 * 0.5144 * sin(38.5)) ~ 116.65 * 4.80 ~ 560 metres
    assert 400.0 < abs(shift_m) < 700.0


def test_forward_and_inverse_closed_loop():
    geom = resolve_sar_geometry({"orbit_pass": "ASCENDING"})
    ground_lon, ground_lat = 5.23456, 59.41234
    sog_kn = 14.5
    cog_deg = 45.0

    # 1. Forward projection to SAR image plane
    fwd = compute_doppler_shifted_position(ground_lon, ground_lat, sog_kn, cog_deg, geom)
    image_lon = fwd["image_lon"]
    image_lat = fwd["image_lat"]
    shift_m = fwd["azimuth_shift_m"]
    assert abs(shift_m) > 50.0
    assert (image_lon, image_lat) != (ground_lon, ground_lat)

    # 2. Inverse restoration from SAR image echo to ground surface
    inv = restore_ground_position_from_sar_echo(image_lon, image_lat, sog_kn, cog_deg, geom)
    restored_lon = inv["ground_lon"]
    restored_lat = inv["ground_lat"]

    # Must reconstruct the original ground coordinate to sub-millimetre precision
    assert math.isclose(restored_lon, ground_lon, abs_tol=1e-6)
    assert math.isclose(restored_lat, ground_lat, abs_tol=1e-6)


def test_stationary_vessel_zero_displacement():
    geom = resolve_sar_geometry()
    fwd = compute_doppler_shifted_position(5.0, 59.0, sog_kn=0.0, cog_deg=100.0, geometry=geom)
    assert fwd["image_lon"] == 5.0
    assert fwd["image_lat"] == 59.0
    assert fwd["azimuth_shift_m"] == 0.0
