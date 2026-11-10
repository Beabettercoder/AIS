"""Pluggable remote-sensing target detectors."""

from .sar_cfar import DETECTOR_VERSION, detect_sar_targets
from .sar_geometry import SarPlatformGeometry, resolve_sar_geometry
from .doppler_shift import (
    decompose_radial_velocity,
    compute_azimuth_displacement_m,
    compute_doppler_shifted_position,
    restore_ground_position_from_sar_echo,
)

__all__ = [
    "DETECTOR_VERSION",
    "detect_sar_targets",
    "SarPlatformGeometry",
    "resolve_sar_geometry",
    "decompose_radial_velocity",
    "compute_azimuth_displacement_m",
    "compute_doppler_shifted_position",
    "restore_ground_position_from_sar_echo",
]
