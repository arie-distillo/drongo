"""PoseProvider: the seam between pose sources and geometry consumers.
Phase 1 ships TelemetryPoseProvider; Phase 2 adds a fused provider with the
same protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .camera import CameraModel
from .frames import FlatGround, GroundModel, LocalFrame
from .geometry import FrameGeometry
from .telemetry import TelemetryTrack
from .types import PoseSample


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    pose: PoseSample
    pos_sigma_m: Optional[float] = None
    yaw_sigma_deg: Optional[float] = None
    pitch_sigma_deg: Optional[float] = None
    method: str = "telemetry"


class PoseProvider(Protocol):
    def estimate(self, t_ms: float, image: Optional[np.ndarray] = None) -> Optional[PoseEstimate]: ...
    def geometry(self, t_ms: float, img_w_px: int, img_h_px: int,
                 image: Optional[np.ndarray] = None) -> Optional[FrameGeometry]: ...


class TelemetryPoseProvider:
    def __init__(self, track: TelemetryTrack, camera: CameraModel,
                 frame: Optional[LocalFrame] = None, ground: GroundModel = FlatGround(), *,
                 pos_sigma_m: Optional[float] = None, yaw_sigma_deg: Optional[float] = None,
                 pitch_sigma_deg: Optional[float] = None, extrapolate_ms: int = 0,
                 min_depression_deg: float = 1.0, max_range_m: Optional[float] = None) -> None:
        if not len(track):
            raise ValueError("empty telemetry track")
        self.track, self.camera, self.ground = track, camera, ground
        first = track[0]
        self.frame = frame or LocalFrame(first.lat_deg, first.lon_deg)
        self._sig = (pos_sigma_m, yaw_sigma_deg, pitch_sigma_deg)
        self._extrap = extrapolate_ms
        self._geom_kw = dict(min_depression_deg=min_depression_deg, max_range_m=max_range_m)

    def estimate(self, t_ms: float, image: Optional[np.ndarray] = None) -> Optional[PoseEstimate]:
        p = self.track.at(t_ms, extrapolate_ms=self._extrap)
        return None if p is None else PoseEstimate(p, *self._sig, method="telemetry")

    def geometry(self, t_ms: float, img_w_px: int, img_h_px: int,
                 image: Optional[np.ndarray] = None) -> Optional[FrameGeometry]:
        e = self.estimate(t_ms, image)
        if e is None:
            return None
        return FrameGeometry.from_pose(e.pose, self.camera, img_w_px, img_h_px,
                                       self.frame, self.ground, **self._geom_kw)
