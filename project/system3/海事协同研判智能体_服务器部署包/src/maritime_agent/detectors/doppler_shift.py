"""SAR Doppler shift compensation engine for maritime moving vessel target alignment."""

from __future__ import annotations

import math
from typing import Any, Optional

from .sar_geometry import SarPlatformGeometry, resolve_sar_geometry

EARTH_RADIUS_M = 6371008.8
KNOTS_TO_MPS = 0.5144444444444445


def _geodesic_direct(lon: float, lat: float, distance_m: float, bearing_deg: float) -> tuple[float, float]:
    """Calculate target coordinates (lon, lat) given origin, distance, and bearing.

    Uses the spherical Earth direct geodesic formula, strictly valid across all latitudes.
    Negative distance moves along (bearing + 180) degrees.
    """
    if abs(distance_m) < 1e-6:
        return lon, lat

    if distance_m < 0:
        distance_m = -distance_m
        bearing_deg = (bearing_deg + 180.0) % 360.0

    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    theta = math.radians(bearing_deg % 360.0)
    delta = distance_m / EARTH_RADIUS_M

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)

    sin_phi2 = sin_phi1 * cos_delta + cos_phi1 * sin_delta * math.cos(theta)
    sin_phi2 = max(-1.0, min(1.0, sin_phi2))
    phi2 = math.asin(sin_phi2)

    y = math.sin(theta) * sin_delta * cos_phi1
    x = cos_delta - sin_phi1 * sin_phi2
    lambda2 = lambda1 + math.atan2(y, x)

    # Normalize longitude to [-180, 180]
    lon_deg = (math.degrees(lambda2) + 180.0) % 360.0 - 180.0
    lat_deg = math.degrees(phi2)
    return round(lon_deg, 7), round(lat_deg, 7)


def decompose_radial_velocity(
    sog_kn: float,
    cog_deg: float,
    geometry: SarPlatformGeometry,
) -> dict[str, float]:
    """Decompose vessel velocity into ground-range and slant-range radial components.

    Args:
        sog_kn: Vessel speed over ground in knots.
        cog_deg: Vessel course over ground in degrees clockwise from True North.
        geometry: Resolved SAR platform geometry.

    Returns:
        Dictionary containing vessel speed in m/s, ground radial velocity in m/s,
        and slant-range radial velocity in m/s (positive towards the radar).
    """
    v_ship_mps = max(0.0, float(sog_kn)) * KNOTS_TO_MPS
    cog = float(cog_deg) % 360.0

    # Angle between ship course and radar ground look direction
    aspect_angle_deg = (cog - geometry.ground_look_heading_deg) % 360.0
    aspect_rad = math.radians(aspect_angle_deg)

    # Ground-projected radial velocity (towards radar ground line of sight)
    v_radial_ground_mps = v_ship_mps * math.cos(aspect_rad)

    # Slant-range radial velocity (positive towards the satellite)
    inc_rad = math.radians(geometry.incidence_angle_deg)
    v_radial_slant_mps = v_radial_ground_mps * math.sin(inc_rad)

    return {
        "v_ship_mps": round(v_ship_mps, 3),
        "aspect_angle_deg": round(aspect_angle_deg, 2),
        "v_radial_ground_mps": round(v_radial_ground_mps, 3),
        "v_radial_slant_mps": round(v_radial_slant_mps, 3),
    }


def compute_azimuth_displacement_m(
    v_radial_slant_mps: float,
    geometry: SarPlatformGeometry,
) -> float:
    """Calculate along-track azimuth displacement caused by Doppler frequency shift.

    Formula:
        Delta_x_az = - (R0 / V_sat) * v_r = - K_shift * v_r

    When v_r > 0 (vessel moves towards radar, positive Doppler shift),
    the SAR matched filter azimuth compression shifts the echo backwards
    relative to the satellite flight direction (Delta_x_az < 0).
    """
    return -geometry.k_shift_s * v_radial_slant_mps


def compute_doppler_shifted_position(
    ground_lon: float,
    ground_lat: float,
    sog_kn: Optional[float],
    cog_deg: Optional[float],
    geometry: SarPlatformGeometry,
) -> dict[str, Any]:
    """Forward projection: compute apparent SAR image position for a moving vessel.

    Translates a moving vessel's true geographic keel coordinate to the Doppler-shifted
    apparent azimuth location in the SAR image plane.
    """
    if sog_kn is None or sog_kn <= 0.1 or cog_deg is None:
        return {
            "image_lon": ground_lon,
            "image_lat": ground_lat,
            "azimuth_shift_m": 0.0,
            "v_radial_slant_mps": 0.0,
            "v_radial_ground_mps": 0.0,
            "v_ship_mps": 0.0,
        }

    decomp = decompose_radial_velocity(sog_kn, cog_deg, geometry)
    v_r = decomp["v_radial_slant_mps"]
    shift_m = compute_azimuth_displacement_m(v_r, geometry)

    img_lon, img_lat = _geodesic_direct(
        lon=ground_lon,
        lat=ground_lat,
        distance_m=shift_m,
        bearing_deg=geometry.heading_deg,
    )

    return {
        "image_lon": img_lon,
        "image_lat": img_lat,
        "azimuth_shift_m": round(shift_m, 2),
        **decomp,
    }


def restore_ground_position_from_sar_echo(
    image_lon: float,
    image_lat: float,
    sog_kn: Optional[float],
    cog_deg: Optional[float],
    geometry: SarPlatformGeometry,
) -> dict[str, Any]:
    """Inverse restoration: restore true maritime keel coordinate from SAR echo centroid.

    Shifts the detected radar brightness peak along the negative azimuth displacement vector
    to recover the true physical coordinate on the ocean surface.
    """
    if sog_kn is None or sog_kn <= 0.1 or cog_deg is None:
        return {
            "ground_lon": image_lon,
            "ground_lat": image_lat,
            "azimuth_shift_m": 0.0,
            "v_radial_slant_mps": 0.0,
            "v_radial_ground_mps": 0.0,
            "v_ship_mps": 0.0,
        }

    decomp = decompose_radial_velocity(sog_kn, cog_deg, geometry)
    v_r = decomp["v_radial_slant_mps"]
    shift_m = compute_azimuth_displacement_m(v_r, geometry)

    # Reverse direction: shift by -shift_m along satellite heading
    ground_lon, ground_lat = _geodesic_direct(
        lon=image_lon,
        lat=image_lat,
        distance_m=-shift_m,
        bearing_deg=geometry.heading_deg,
    )

    return {
        "ground_lon": ground_lon,
        "ground_lat": ground_lat,
        "azimuth_shift_m": round(shift_m, 2),
        **decomp,
    }
