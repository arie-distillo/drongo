"""Render realistic demo / test clips with known ground truth.

    python -m testbed.synth.demo --terrain urban --flight orbit
    python -m testbed.synth.demo --all --out demo_clips             # every terrain x flight
    python -m testbed.synth.demo --terrain nature --flight cruise --telemetry embedded --offset-ms 120

Terrains: nature, urban (procedural, reproducible from --seed), or --image with
a north-up orthophoto and world file. Flights: survey (nadir mapping lines),
orbit (point of interest), cruise (forward flight with a tilt-up reveal),
hover_pan (hovering camera pan). See flights.py.

Output, for each clip <terrain>_<flight>:
    <out>/srt/<name>.mp4 + <name>.SRT     video with a DJI-style SRT sidecar
    <out>/embedded/<name>.mp4             the same video with telemetry in a DJI metadata track
    <out>/truth/<name>.json               exact per-frame poses, camera, noise settings

The video is rendered from the exact poses. Both telemetry files carry the
same recorded telemetry: one sample per video frame, with the chosen noise
profile and clock offset applied (positive offset: telemetry lags the video).
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from drongo import CameraModel, FrameGeometry, PoseSample, TelemetryTrack, write_dji_srt

from .encode import VideoWriter
from .flights import FLIGHTS
from .mp4meta import embed_telemetry
from .render import Look, MipTexture, render_rgb
from .scenario import DJI_CONSUMER, Scenario, TelemetryNoise
from .terrain import TERRAINS, load_orthophoto

DEFAULT_DURATION_S = {"survey": 40.0, "orbit": 30.0, "cruise": 25.0, "hover_pan": 25.0}
LOOKS = {"nature": Look(visibility_m=3500.0), "urban": Look(visibility_m=2500.0, haze_bgr=(215, 205, 198))}


def recorded_telemetry(sc: Scenario, noise: TelemetryNoise, ground_msl_m: float = 0.0) -> list[PoseSample]:
    """What the aircraft would log: noisy telemetry resampled at the video
    frame times, as DJI SRT and embedded tracks have one record per frame."""
    track = TelemetryTrack(noise.apply(sc))
    out = []
    for truth in sc.poses:
        p = track.at(truth.t_ms, extrapolate_ms=10_000)
        if p is not None:
            out.append(p.with_(alt_msl_m=p.alt_agl_m + ground_msl_m))
    return out


def render_clip(sc: Scenario, mip: MipTexture, camera: CameraModel, out_dir: Path, name: str,
                width: int = 1920, height: int = 1080, noise: TelemetryNoise = DJI_CONSUMER,
                telemetry: str = "both", look: Look = Look(), crf: int = 21, ground_msl_m: float = 50.0,
                meta: Optional[dict] = None, progress: bool = True) -> dict[str, Path]:
    out_dir = Path(out_dir)
    written: dict[str, Path] = {}
    if telemetry in ("srt", "both"):
        (out_dir / "srt").mkdir(parents=True, exist_ok=True)
        video = out_dir / "srt" / f"{name}.mp4"
    else:
        video = Path(tempfile.mkdtemp()) / f"{name}.mp4"
    rng = np.random.default_rng(look.seed)
    t0 = time.time()
    n = len(sc.poses)
    with VideoWriter(video, width, height, sc.fps, crf=crf) as vw:
        for i, p in enumerate(sc.poses):
            g = FrameGeometry.from_pose(p, camera, width, height, sc.frame)
            vw.write(render_rgb(mip, g, look, rng=rng))
            if progress and (i + 1) % int(sc.fps * 5) == 0:
                el = time.time() - t0
                print(f"    {name}: {i + 1}/{n} frames, {el:.0f} s elapsed, ~{el / (i + 1) * (n - i - 1):.0f} s left",
                      flush=True)
    rec = recorded_telemetry(sc, noise, ground_msl_m)
    if telemetry in ("srt", "both"):
        srt = video.with_suffix(".SRT")
        srt.write_text(write_dji_srt(rec, sc.frame_ms), encoding="utf-8")
        written.update(video_srt=video, srt=srt)
    if telemetry in ("embedded", "both"):
        (out_dir / "embedded").mkdir(parents=True, exist_ok=True)
        emb = out_dir / "embedded" / f"{name}.mp4"
        embed_telemetry(video, rec, emb)
        written["video_embedded"] = emb
        if telemetry == "embedded":
            video.unlink()
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)
    truth = out_dir / "truth" / f"{name}.json"
    truth.write_text(json.dumps({
        "name": name, "fps": sc.fps, "img_w_px": width, "img_h_px": height,
        "camera": {"name": camera.name, "sensor_w_mm": camera.sensor_w_mm, "focal_mm": camera.focal_mm},
        "local_frame_origin": [sc.frame.ref_lat_deg, sc.frame.ref_lon_deg],
        "ground_msl_m": ground_msl_m,
        "telemetry_noise": dataclasses.asdict(noise),
        **(meta or {}),
        "poses": [{"t_ms": p.t_ms, "lat_deg": p.lat_deg, "lon_deg": p.lon_deg, "alt_agl_m": p.alt_agl_m,
                   "gimbal_yaw_deg": p.gimbal_yaw_deg, "gimbal_pitch_deg": p.gimbal_pitch_deg,
                   "gimbal_roll_deg": p.gimbal_roll_deg, "body_yaw_deg": p.body_yaw_deg,
                   "velocity_enu_ms": p.velocity_enu_ms} for p in sc.poses],
    }, indent=1), encoding="utf-8")
    written["truth"] = truth
    return written


def _flight(name: str, camera: CameraModel, fps: float, duration_s: float, seed: int, wind: float) -> Scenario:
    kw = dict(fps=fps, duration_s=duration_s, seed=seed, wind=wind)
    if name == "survey":
        kw["camera"] = camera
    return FLIGHTS[name](**kw)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", choices=sorted(TERRAINS), default="urban")
    ap.add_argument("--image", type=Path, help="orthophoto with a world file, instead of a procedural terrain")
    ap.add_argument("--flight", choices=sorted(FLIGHTS), default="orbit")
    ap.add_argument("--all", action="store_true", help="every terrain x every flight")
    ap.add_argument("--telemetry", choices=["srt", "embedded", "both"], default="both")
    ap.add_argument("--duration-s", type=float, help=f"default per flight: {DEFAULT_DURATION_S}")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", default="1920x1080", help="WxH")
    ap.add_argument("--camera", default="mini3pro", help=f"preset: {CameraModel.presets()}")
    ap.add_argument("--noise", choices=["none", "dji"], default="dji")
    ap.add_argument("--offset-ms", type=int, default=0, help="telemetry clock offset to inject")
    ap.add_argument("--wind", type=float, default=1.0, help="wind / jitter scale; 0 = perfectly still")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--terrain-res-m", type=float, default=0.08)
    ap.add_argument("--crf", type=int, default=21, help="H.264 quality (lower = better, larger)")
    ap.add_argument("--out", type=Path, default=Path("demo_clips"))
    a = ap.parse_args(argv)

    w, h = (int(v) for v in a.size.lower().split("x"))
    camera = CameraModel.preset(a.camera)
    noise = DJI_CONSUMER if a.noise == "dji" else TelemetryNoise()
    noise = dataclasses.replace(noise, time_offset_ms=a.offset_ms, seed=a.seed)
    terrains = ["image"] if a.image else (sorted(TERRAINS) if a.all else [a.terrain])
    flights = sorted(FLIGHTS) if a.all else [a.flight]

    for terrain in terrains:
        print(f"terrain {terrain}: generating texture ...", flush=True)
        tex = load_orthophoto(a.image) if a.image else TERRAINS[terrain](res_m=a.terrain_res_m, seed=a.seed)
        mip = MipTexture(tex)
        del tex
        look = dataclasses.replace(LOOKS.get(terrain, Look()), seed=a.seed)
        for flight in flights:
            dur = a.duration_s or DEFAULT_DURATION_S[flight]
            sc = _flight(flight, camera, a.fps, dur, a.seed, a.wind)
            name = f"{a.image.stem if a.image else terrain}_{flight}"
            print(f"  {name}: {len(sc.poses)} frames at {w}x{h}", flush=True)
            files = render_clip(sc, mip, camera, a.out, name, w, h, noise, a.telemetry, look, a.crf,
                                meta={"terrain": terrain, "flight": flight, "seed": a.seed})
            for k, p in files.items():
                print(f"    {k:15s} {p}")
        del mip
        gc.collect()


if __name__ == "__main__":
    main()
