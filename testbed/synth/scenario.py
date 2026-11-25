"""Synthetic flight scenarios with exact ground-truth poses, plus a noise
model that turns truth into realistic telemetry."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from drongo import LocalFrame, PoseSample, TelemetrySourceKind


@dataclass
class Scenario:
    name: str
    frame: LocalFrame
    poses: list[PoseSample]           # ground truth, one per video frame
    fps: float

    @property
    def frame_ms(self) -> int:
        return int(round(1000 / self.fps))


def _build(name: str, ref: tuple[float, float], fps: float, duration_s: float,
           fn: Callable[[float], tuple[float, float, float, float, float, float]]) -> Scenario:
    """fn(t) -> (east_m, north_m, alt_m, yaw_deg, pitch_deg, roll_deg)."""
    fr = LocalFrame(*ref)
    poses = []
    n = int(duration_s * fps)
    for i in range(n):
        t = i / fps
        e, nn, a, y, p, r = fn(t)
        lat, lon = fr.to_wgs84([[e, nn]])[0]
        poses.append(PoseSample(int(round(t * 1000)), float(lat), float(lon), a, y % 360.0, p, r,
                                source=TelemetrySourceKind.MANUAL))
    return Scenario(name, fr, poses, fps)


REF = (31.7683, 35.2137)


def straight_nadir(speed_ms=12.0, alt_m=85.0, heading_deg=60.0, fps=30.0, duration_s=6.0) -> Scenario:
    h = math.radians(heading_deg)
    return _build("straight_nadir", REF, fps, duration_s,
                  lambda t: (speed_ms * t * math.sin(h), speed_ms * t * math.cos(h), alt_m, heading_deg, -90.0, 0.0))


def oblique_turn(speed_ms=8.0, alt_m=85.0, pitch_deg=-45.0, turn_rate_dps=20.0, fps=30.0,
                 duration_s=6.0) -> Scenario:
    """Constant-speed coordinated turn, oblique camera along heading,
    crossing north (yaw wrap 350 -> 10 deg) mid-way."""
    y0 = 300.0

    def fn(t):
        yaw = y0 + turn_rate_dps * t
        # integrate heading analytically
        w = math.radians(turn_rate_dps)
        a0 = math.radians(y0)
        e = speed_ms / w * (math.cos(a0) - math.cos(a0 + w * t))
        n = speed_ms / w * (math.sin(a0 + w * t) - math.sin(a0))
        return e, n, alt_m, yaw, pitch_deg, 0.0

    return _build("oblique_turn", REF, fps, duration_s, fn)


def gimbal_sweep(alt_m=60.0, fps=30.0, duration_s=6.0) -> Scenario:
    """Hovering drone sweeping the gimbal from nadir to -35 deg and yawing."""
    return _build("gimbal_sweep", REF, fps, duration_s,
                  lambda t: (0.0, 0.0, alt_m, 20.0 + 10.0 * t, -90.0 + 55.0 * (t / duration_s), 0.0))


def slalom(speed_ms=10.0, alt_m=70.0, fps=30.0, duration_s=6.0) -> Scenario:
    """Weaving flight with varying turn rate and a nodding gimbal: the
    manoeuvre profile under which video/telemetry sync is observable."""
    amp, wl = 8.0, 80.0
    k = 2 * math.pi / wl

    def fn(t):
        e = speed_ms * t
        n = amp * math.sin(k * e)
        heading = math.degrees(math.atan2(1.0, amp * k * math.cos(k * e)))   # atan2(dE, dN)
        pitch = -60.0 + 12.0 * math.sin(2 * math.pi * t / 2.5)
        return e, n, alt_m, heading, pitch, 0.0

    return _build("slalom", REF, fps, duration_s, fn)


SCENARIOS = {"slalom": slalom, "straight_nadir": straight_nadir, "oblique_turn": oblique_turn, "gimbal_sweep": gimbal_sweep}


@dataclass
class TelemetryNoise:
    """Error model applied to ground truth to produce 'recorded' telemetry.

    gps_sigma_m / alt_sigma_m / angle sigmas are white noise per sample;
    *_bias are constant offsets; time_offset_ms shifts the telemetry clock
    (positive = telemetry timestamps late relative to video);
    sample_every keeps one sample in N (DJI SRT is per frame; protobuf / KLV
    may be sparser)."""
    gps_sigma_m: float = 0.0
    gps_walk_m_per_s: float = 0.0
    alt_sigma_m: float = 0.0
    alt_bias_m: float = 0.0
    yaw_sigma_deg: float = 0.0
    yaw_bias_deg: float = 0.0
    pitch_sigma_deg: float = 0.0
    pitch_bias_deg: float = 0.0
    time_offset_ms: int = 0
    sample_every: int = 1
    quantize_deg: Optional[float] = None    # e.g. 0.1 for SRT one-decimal angles
    seed: int = 0

    def apply(self, sc: Scenario) -> list[PoseSample]:
        rng = np.random.default_rng(self.seed)
        out = []
        walk = np.zeros(2)
        dt = 1.0 / sc.fps
        for i, p in enumerate(sc.poses):
            walk += rng.normal(0, self.gps_walk_m_per_s * math.sqrt(dt), 2)
            if i % self.sample_every:
                continue
            e, n = sc.frame.to_enu([[p.lat_deg, p.lon_deg]])[0] + rng.normal(0, self.gps_sigma_m, 2) + walk
            lat, lon = sc.frame.to_wgs84([[e, n]])[0]
            yaw = p.gimbal_yaw_deg + self.yaw_bias_deg + rng.normal(0, self.yaw_sigma_deg)
            pitch = p.gimbal_pitch_deg + self.pitch_bias_deg + rng.normal(0, self.pitch_sigma_deg)
            if self.quantize_deg:
                q = self.quantize_deg
                yaw, pitch = round(yaw / q) * q, round(pitch / q) * q
            out.append(p.with_(
                t_ms=p.t_ms + self.time_offset_ms,
                lat_deg=float(lat), lon_deg=float(lon),
                alt_agl_m=p.alt_agl_m + self.alt_bias_m + rng.normal(0, self.alt_sigma_m),
                gimbal_yaw_deg=yaw % 360.0, gimbal_pitch_deg=pitch,
            ))
        return out


# Rough consumer-DJI (non-RTK) profile; used as the default "realistic" case.
DJI_CONSUMER = TelemetryNoise(gps_sigma_m=0.3, gps_walk_m_per_s=0.3, alt_sigma_m=0.1,
                              yaw_sigma_deg=0.15, yaw_bias_deg=1.5, pitch_sigma_deg=0.1,
                              quantize_deg=0.1, seed=1)
