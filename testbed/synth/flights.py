"""Realistic flight profiles for demo and test clips.

Unlike the analytic scenarios in `scenario.py`, these behave like a real
multirotor under a stabilised gimbal:

  * speed along the path is planned with longitudinal and lateral
    acceleration limits, so the aircraft accelerates from rest, slows into
    turns and stops at the end;
  * the gimbal follows its target attitude through a first-order lag with a
    rate limit, as a gimbal controller does;
  * wind adds slow position and altitude sway, and the stabilised gimbal
    keeps a small residual jitter.

All of this is in the ground-truth poses the video is rendered from. Sensor
errors (GPS noise, compass bias, clock offset) are applied afterwards by
`TelemetryNoise`, exactly as for the analytic scenarios.

Every profile stays within about 250 m of the local-frame origin and keeps
its view mostly inside the default 600 m x 600 m terrain; beyond the terrain
edge the renderer mirrors the ground.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from drongo import CameraModel, LocalFrame, PoseSample, TelemetrySourceKind

from .scenario import REF, Scenario

G = 9.81


# ---------------------------------------------------------------------------- building blocks

def _resample(path_en: np.ndarray, ds: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path_en, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.arange(0.0, s[-1], ds)
    return np.stack([np.interp(t, s, path_en[:, 0]), np.interp(t, s, path_en[:, 1])], 1)


def plan_speed(path_en: np.ndarray, fps: float, duration_s: float, v_max: float,
               a_lon: float = 2.0, a_lat: float = 2.0, v_start: float = 0.0, v_end: float = 0.0,
               ds: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fly along `path_en` (N x 2 metres) as fast as the limits allow.

    Returns, per video frame: position (n x 2), course in degrees from North,
    ground speed. If the path is shorter than the clip, the aircraft holds
    its final position (hover)."""
    p = _resample(np.asarray(path_en, float), ds)
    d = np.gradient(p, axis=0)
    course = np.unwrap(np.arctan2(d[:, 0], d[:, 1]))
    kappa = np.abs(np.gradient(course)) / ds
    kappa = np.convolve(kappa, np.ones(15) / 15, mode="same")          # ~3 m smoothing
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(kappa, 1e-9)))
    v[0], v[-1] = min(v[0], v_start), min(v[-1], v_end)
    for i in range(1, len(v)):                                        # acceleration limit
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_lon * ds))
    for i in range(len(v) - 2, -1, -1):                               # braking limit
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_lon * ds))
    # each step is flown at constant acceleration, so speed is continuous in time
    v = np.maximum(v, 1e-3)
    dt = 2 * ds / (v[1:] + v[:-1])
    acc = (v[1:] - v[:-1]) / dt
    t_path = np.r_[0.0, np.cumsum(dt)]
    t = np.arange(int(round(duration_s * fps))) / fps
    k = np.clip(np.searchsorted(t_path, t, side="right") - 1, 0, len(dt) - 1)
    tau = np.clip(t - t_path[k], 0.0, dt[k])
    s_idx = k + (v[k] * tau + 0.5 * acc[k] * tau ** 2) / ds            # fractional index along the path
    idx = np.arange(len(p))
    pos = np.stack([np.interp(s_idx, idx, p[:, 0]), np.interp(s_idx, idx, p[:, 1])], 1)
    crs = np.degrees(np.interp(s_idx, idx, course))
    spd = (v[k] + acc[k] * tau) * (t <= t_path[-1])
    return pos, crs, spd


def lag(x: np.ndarray, fps: float, tau_s: float, rate_max: Optional[float] = None) -> np.ndarray:
    """First-order lag with optional rate limit (units per second)."""
    out = np.empty_like(x)
    out[0] = x[0]
    k = 1.0 - math.exp(-1.0 / (fps * tau_s))
    step = None if rate_max is None else rate_max / fps
    for i in range(1, len(x)):
        dx = k * (x[i] - out[i - 1])
        if step is not None:
            dx = max(-step, min(step, dx))
        out[i] = out[i - 1] + dx
    return out


def band_noise(n: int, fps: float, rms: float, f_lo: float, f_hi: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random signal with energy between f_lo and f_hi Hz."""
    t = np.arange(n) / fps
    k = 8
    f = np.exp(rng.uniform(math.log(f_lo), math.log(f_hi), k))
    ph = rng.uniform(0, 2 * math.pi, k)
    a = rng.uniform(0.5, 1.0, k)
    x = (a[:, None] * np.sin(2 * math.pi * f[:, None] * t[None, :] + ph[:, None])).sum(0)
    return x * rms / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _assemble(name: str, fps: float, pos: np.ndarray, alt: np.ndarray, cam_yaw: np.ndarray,
              cam_pitch: np.ndarray, body_yaw: np.ndarray, rng: np.random.Generator,
              wind: float = 1.0, ref=REF) -> Scenario:
    """Apply gimbal dynamics, wind and jitter; build PoseSamples."""
    n = len(pos)
    sway = np.stack([band_noise(n, fps, 0.25 * wind, 0.05, 0.4, rng), band_noise(n, fps, 0.25 * wind, 0.05, 0.4, rng)], 1)
    pos = pos + sway
    alt = alt + band_noise(n, fps, 0.2 * wind, 0.05, 0.3, rng)
    yaw = lag(np.unwrap(np.radians(cam_yaw)), fps, 0.35, math.radians(90)) if n else cam_yaw
    pitch = lag(np.asarray(cam_pitch, float), fps, 0.3, 40.0)
    yaw = np.degrees(yaw) + band_noise(n, fps, 0.08 * wind, 0.5, 6.0, rng)
    pitch = np.clip(pitch + band_noise(n, fps, 0.06 * wind, 0.5, 6.0, rng), -90.0, 30.0)
    roll = band_noise(n, fps, 0.05 * wind, 0.5, 6.0, rng)
    vel = np.gradient(pos, axis=0) * fps
    vz = np.gradient(alt) * fps
    fr = LocalFrame(*ref)
    ll = fr.to_wgs84(pos)
    poses = [PoseSample(int(round(i * 1000 / fps)), float(ll[i, 0]), float(ll[i, 1]), float(alt[i]),
                        float(yaw[i] % 360.0), float(pitch[i]), float(roll[i]),
                        body_yaw_deg=float(body_yaw[i] % 360.0),
                        velocity_enu_ms=(float(vel[i, 0]), float(vel[i, 1]), float(vz[i])),
                        source=TelemetrySourceKind.MANUAL)
             for i in range(n)]
    return Scenario(name, fr, poses, fps)


def _frames(fps, duration_s):
    return int(round(duration_s * fps))


# ---------------------------------------------------------------------------- profiles

def survey(camera: CameraModel = None, alt_m: float = 70.0, speed_ms: float = 8.0, area_m=(260.0, 200.0),
           sidelap: float = 0.7, angle_deg: float = 20.0, fps: float = 30.0, duration_s: float = 30.0,
           seed: int = 0, wind: float = 1.0) -> Scenario:
    """Mapping mission: nadir camera, parallel lines with turnarounds. The
    camera yaw follows the course, so the image rotates 180 deg at each end."""
    rng = np.random.default_rng(seed)
    camera = camera or CameraModel.preset("mini3pro")
    swath = alt_m * camera.sensor_w_mm / camera.focal_mm
    spacing = max(10.0, swath * (1 - sidelap))
    L, Wd = area_m
    lines = int(Wd // spacing) + 1
    pts = []
    r = spacing / 2
    for k in range(lines):
        x = -Wd / 2 + k * spacing
        y0, y1 = (-L / 2, L / 2) if k % 2 == 0 else (L / 2, -L / 2)
        pts += [[x, y0], [x, y1]]
        if k < lines - 1:                                          # semicircle to the next line
            sgn = 1 if k % 2 == 0 else -1
            a = np.linspace(math.pi, 0, 24)
            pts += list(np.stack([x + r + r * np.cos(a), y1 + sgn * r * np.sin(a)], 1))
    R = np.array([[math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))],
                  [-math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))]])
    path = np.asarray(pts) @ R.T
    pos, crs, _ = plan_speed(path, fps, duration_s, speed_ms, a_lon=1.5, a_lat=1.5)
    n = len(pos)
    return _assemble("survey", fps, pos, np.full(n, alt_m), crs, np.full(n, -90.0), crs, rng, wind)


def orbit(radius_m: float = 80.0, alt_m: float = 70.0, speed_ms: float = 6.0, poi_en=(0.0, 0.0),
          clockwise: bool = True, fps: float = 30.0, duration_s: float = 30.0, seed: int = 0,
          wind: float = 1.0) -> Scenario:
    """Point-of-interest orbit: the aircraft and camera face the centre, the
    gimbal is tilted to keep it in the middle of the frame."""
    rng = np.random.default_rng(seed)
    a0 = rng.uniform(0, 2 * math.pi)
    sgn = -1 if clockwise else 1
    turns = speed_ms * duration_s / (2 * math.pi * radius_m) + 0.2
    a = a0 + sgn * np.linspace(0, 2 * math.pi * turns, int(400 * turns) + 2)
    path = np.asarray(poi_en) + radius_m * np.stack([np.sin(a), np.cos(a)], 1)
    pos, _, _ = plan_speed(path, fps, duration_s, speed_ms, a_lon=1.5, a_lat=G, v_end=speed_ms)
    n = len(pos)
    to_poi = np.asarray(poi_en) - pos
    yaw = np.degrees(np.arctan2(to_poi[:, 0], to_poi[:, 1]))
    pitch = -np.degrees(np.arctan2(alt_m, np.linalg.norm(to_poi, axis=1)))
    return _assemble("orbit", fps, pos, np.full(n, alt_m), yaw, pitch, yaw, rng, wind)


def cruise(speed_ms: float = 11.0, alt_start_m: float = 60.0, alt_end_m: float = 85.0,
           pitch_start_deg: float = -70.0, pitch_end_deg: float = -32.0, fps: float = 30.0,
           duration_s: float = 30.0, seed: int = 0, wind: float = 1.0) -> Scenario:
    """Cinematic forward flight on a gently curving course, climbing while
    the camera tilts up from steep to shallow (a 'reveal'). The flight ends
    over the origin, so the final, farthest view looks across the terrain
    rather than past its edge."""
    rng = np.random.default_rng(seed)
    heading0 = rng.uniform(0, 360)
    length = speed_ms * duration_s * 1.1
    s = np.linspace(0, length, 2000)
    psi = math.radians(heading0) + 0.5 * np.sin(2 * math.pi * s / (length * 0.8) + rng.uniform(0, 6.28))
    d = np.stack([np.sin(psi), np.cos(psi)], 1) * (s[1] - s[0])
    path = np.cumsum(d, 0)
    pos, crs, _ = plan_speed(path, fps, duration_s, speed_ms, a_lon=2.5, a_lat=3.0)
    pos -= pos[-1]
    n = len(pos)
    u = _smoothstep(np.arange(n) / max(n - 1, 1) / 0.7)
    alt = alt_start_m + (alt_end_m - alt_start_m) * u
    pitch = pitch_start_deg + (pitch_end_deg - pitch_start_deg) * u
    return _assemble("cruise", fps, pos, alt, crs, pitch, crs, rng, wind)


def hover_pan(alt_m: float = 50.0, pan_deg: float = 140.0, pitch_deg: float = -35.0, fps: float = 30.0,
              duration_s: float = 30.0, seed: int = 0, wind: float = 1.0) -> Scenario:
    """Hovering aircraft panning the camera slowly across the scene and back,
    tilting down mid-way. The aircraft itself turns with the camera."""
    rng = np.random.default_rng(seed)
    n = _frames(fps, duration_s)
    t = np.arange(n) / fps
    y0 = rng.uniform(0, 360)
    ph = t / duration_s
    yaw = y0 + pan_deg * (_smoothstep(ph / 0.45) - _smoothstep((ph - 0.55) / 0.45) * 0.5)
    pitch = pitch_deg - 20.0 * np.sin(math.pi * _smoothstep(ph)) ** 2
    pos = np.zeros((n, 2)) + band_noise(n, fps, 0.4, 0.02, 0.15, rng)[:, None] * [1, 0.6]   # GPS hold drift
    return _assemble("hover_pan", fps, pos, np.full(n, alt_m), yaw, pitch, yaw, rng, wind)


FLIGHTS: dict[str, Callable[..., Scenario]] = {
    "survey": survey, "orbit": orbit, "cruise": cruise, "hover_pan": hover_pan,
}
