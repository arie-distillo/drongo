"""Core value types and angle helpers.

Conventions (used everywhere in drongo):
  ENU local frame, metres (x = East, y = North, z = Up).
  Yaw   : degrees clockwise from true North.
  Pitch : degrees, 0 = horizon, -90 = nadir.
  Roll  : degrees, positive = image right side tilts down.
  Time  : video-absolute milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

import numpy as np

ArrayN2 = np.ndarray   # (N, 2) float64
Mat3 = np.ndarray      # (3, 3) float64


class TelemetrySourceKind(str, Enum):
    DJI_SRT_A = "dji_srt_a"          # Mini2 / Air2: no gimbal fields
    DJI_SRT_B = "dji_srt_b"          # Mavic3 / Air2S: gb_yaw / gb_pitch / gb_roll
    DJI_SRT_C = "dji_srt_c"          # FPV / Avata: GPS(...) H ...m
    DJI_PROTOBUF = "dji_protobuf"
    EXIFTOOL = "exiftool"
    MISB_0601 = "misb_0601"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class PoseSample:
    t_ms: int
    lat_deg: float
    lon_deg: float
    alt_agl_m: float
    gimbal_yaw_deg: float
    gimbal_pitch_deg: float
    gimbal_roll_deg: float = 0.0
    alt_msl_m: Optional[float] = None
    body_yaw_deg: Optional[float] = None
    velocity_enu_ms: Optional[tuple[float, float, float]] = None
    source: TelemetrySourceKind = TelemetrySourceKind.MANUAL

    def with_(self, **kw) -> "PoseSample":
        return replace(self, **kw)


def wrap_deg(a: float) -> float:
    """Wrap to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def lerp_angle_deg(a: float, b: float, t: float) -> float:
    """Shortest-arc interpolation; result wrapped to [0, 360)."""
    return (a + wrap_deg(b - a) * t) % 360.0


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def as_n2(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, 2)
    if a.shape[-1] != 2:
        raise ValueError(f"expected (N, 2) array, got shape {a.shape}")
    return a
