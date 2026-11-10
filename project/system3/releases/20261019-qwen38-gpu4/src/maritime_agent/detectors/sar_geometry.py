"""Radar satellite orbit and imaging geometry model for SAR doppler estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Optional


@dataclass
class SarPlatformGeometry:
    """SAR satellite orbit and observation geometry parameters.

    Attributes:
        orbit_pass: Satellite orbit pass direction ("ASCENDING" or "DESCENDING").
        heading_deg: Satellite velocity track heading in degrees clockwise from True North (0~360).
                     Default is 350.0 deg for Ascending and 190.0 deg for Descending.
        look_direction: Antenna looking direction ("RIGHT" or "LEFT"). Sentinel-1 is always "RIGHT".
        ground_look_heading_deg: Radar line-of-sight heading projected onto ground plane (0~360).
        v_sat_mps: Satellite orbital velocity along track in m/s (Sentinel-1 ~7590 m/s).
        v_ground_mps: Satellite ground-track sub-satellite point speed in m/s (~7100 m/s).
        altitude_m: Mean orbital altitude in metres (~693,000 m).
        incidence_angle_deg: Local radar beam incidence angle from surface normal in degrees (~38.5 deg).
        slant_range_m: Radar slant range R0 to target scene in metres.
        k_shift_s: Doppler shift constant K_shift = R0 / V_sat in seconds (~108~116 s).
    """

    orbit_pass: str = "ASCENDING"
    heading_deg: float = 350.0
    look_direction: str = "RIGHT"
    ground_look_heading_deg: float = 80.0
    v_sat_mps: float = 7590.0
    v_ground_mps: float = 7100.0
    altitude_m: float = 693000.0
    incidence_angle_deg: float = 38.5
    slant_range_m: float = 885400.0
    k_shift_s: float = 116.65

    def to_dict(self) -> dict[str, Any]:
        """Convert geometry parameters to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SarPlatformGeometry:
        """Create SarPlatformGeometry from dictionary, filling defaults for missing keys."""
        if not data:
            return cls()
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


def resolve_sar_geometry(
    parameters: Optional[dict[str, Any]] = None,
    default_orbit_pass: str = "ASCENDING",
) -> SarPlatformGeometry:
    """Resolve SAR platform geometry from configuration parameters.

    Supports custom parameter overrides, or automatically estimates defaults
    based on orbit direction and typical Sentinel-1 constellation parameters.
    """
    params = parameters or {}
    geo_dict = params.get("sar_geometry") or {}

    orbit_pass = str(
        geo_dict.get("orbit_pass")
        or params.get("orbit_pass")
        or default_orbit_pass
    ).upper()
    if orbit_pass not in ("ASCENDING", "DESCENDING"):
        orbit_pass = "ASCENDING"

    # Default flight track headings for Sentinel-1 near mid-latitude
    default_heading = 350.0 if orbit_pass == "ASCENDING" else 190.0
    heading_deg = float(
        geo_dict.get("heading_deg")
        or params.get("heading_deg")
        or default_heading
    ) % 360.0

    look_direction = str(
        geo_dict.get("look_direction")
        or params.get("look_direction")
        or "RIGHT"
    ).upper()
    if look_direction not in ("RIGHT", "LEFT"):
        look_direction = "RIGHT"

    # Ground look direction: perpendicular to flight track
    if look_direction == "RIGHT":
        ground_look_heading_deg = (heading_deg + 90.0) % 360.0
    else:
        ground_look_heading_deg = (heading_deg - 90.0) % 360.0

    v_sat_mps = float(geo_dict.get("v_sat_mps") or params.get("v_sat_mps") or 7590.0)
    v_ground_mps = float(geo_dict.get("v_ground_mps") or params.get("v_ground_mps") or 7100.0)
    altitude_m = float(geo_dict.get("altitude_m") or params.get("altitude_m") or 693000.0)
    incidence_angle_deg = float(
        geo_dict.get("incidence_angle_deg")
        or params.get("incidence_angle_deg")
        or 38.5
    )

    # Slant range R0 = H / cos(incidence_angle) if not explicitly set
    if geo_dict.get("slant_range_m"):
        slant_range_m = float(geo_dict["slant_range_m"])
    elif params.get("slant_range_m"):
        slant_range_m = float(params["slant_range_m"])
    else:
        theta_rad = math.radians(incidence_angle_deg)
        slant_range_m = altitude_m / max(math.cos(theta_rad), 0.1)

    # K_shift ratio: R0 / V_sat
    if geo_dict.get("k_shift_s"):
        k_shift_s = float(geo_dict["k_shift_s"])
    elif params.get("k_shift_s"):
        k_shift_s = float(params["k_shift_s"])
    else:
        k_shift_s = slant_range_m / max(v_sat_mps, 1.0)

    return SarPlatformGeometry(
        orbit_pass=orbit_pass,
        heading_deg=round(heading_deg, 3),
        look_direction=look_direction,
        ground_look_heading_deg=round(ground_look_heading_deg, 3),
        v_sat_mps=round(v_sat_mps, 2),
        v_ground_mps=round(v_ground_mps, 2),
        altitude_m=round(altitude_m, 2),
        incidence_angle_deg=round(incidence_angle_deg, 3),
        slant_range_m=round(slant_range_m, 2),
        k_shift_s=round(k_shift_s, 3),
    )
