"""Run drongo on a video + telemetry and write a diagnostic report.

    python -m testbed.harness.run_video --video clip.mp4 [--srt clip.SRT]
        --camera mavic3_wide --out reports/clip

Outputs in --out:
    report.json         all numbers
    report.html         visual report: map with track, footprints and draped frames; plots
    report.md           human summary
    footprints.geojson  frame footprints + flight track (open in QGIS / geojson.io)
    residuals.png       telemetry-vs-vision residual over time, offset sweep
    overlay_*.png       10 m ENU grid projected onto frames (should stick to the ground)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from drongo import CameraModel, DjiSrtReader, TelemetryPoseProvider, open_telemetry, wrap_deg

from .consistency import PairResidual, estimate_homography, motion_px, pair_residual, summarize, sweep_time_offset


def telemetry_sanity(track, frame) -> dict:
    s = list(track)
    t = np.array([p.t_ms for p in s], float)
    en = frame.to_enu([[p.lat_deg, p.lon_deg] for p in s])
    dt = np.diff(t) / 1000.0
    # speed over ~1 s baselines: per-sample GPS noise makes 30 Hz differences meaningless
    k = max(1, int(round(1000.0 / max(np.median(np.diff(t)), 1.0)))) if len(t) > 1 else 1
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.linalg.norm(en[k:] - en[:-k], axis=1) / ((t[k:] - t[:-k]) / 1000.0)
        step_speed = np.linalg.norm(np.diff(en, axis=0), axis=1) / dt
    if speed.size == 0:
        speed = step_speed
    yaw = np.array([p.gimbal_yaw_deg for p in s])
    dyaw = np.array([wrap_deg(b - a) for a, b in zip(yaw[:-1], yaw[1:])])
    return {
        "samples": len(s), "source": track.source.value,
        "span_ms": [int(t[0]), int(t[-1])],
        "median_interval_ms": float(np.median(np.diff(t))) if len(t) > 1 else None,
        "speed_ms": {"median": float(np.nanmedian(speed)), "max": float(np.nanmax(speed))} if len(s) > 1 else None,
        # step longer than 40 m/s * dt plus 2 m of GPS noise allowance
        "position_jumps": int(np.sum(step_speed * dt > 40 * dt + 2.0)) if len(s) > 1 else 0,
        "alt_agl_m": [float(min(p.alt_agl_m for p in s)), float(max(p.alt_agl_m for p in s))],
        "pitch_deg": [float(min(p.gimbal_pitch_deg for p in s)), float(max(p.gimbal_pitch_deg for p in s))],
        "yaw_rate_dps_max": float(np.max(np.abs(dyaw / dt))) if len(s) > 1 else 0.0,
        "yaw_crosses_north": int(np.sum(np.abs(yaw[:-1] - yaw[1:]) > 180)),
        "has_gimbal": any(p.gimbal_pitch_deg != -90.0 or p.gimbal_yaw_deg != 0.0 for p in s),
        "zero_altitude_samples": int(sum(p.alt_agl_m <= 0 for p in s)),
    }


def draw_grid(img, g, step_m=10.0, color=(0, 255, 255)):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    fp = g.footprint()
    if fp is None:
        return out
    c = fp.corners_en
    e0, n0 = np.floor(c.min(0) / step_m) * step_m
    e1, n1 = np.ceil(c.max(0) / step_m) * step_m
    def poly(pts):
        uv = g.en_to_pixel(pts)
        ok = np.all(np.isfinite(uv), axis=1)
        for a, b, oa, ob in zip(uv[:-1], uv[1:], ok[:-1], ok[1:]):
            if oa and ob:
                cv2.line(out, tuple(np.int32(a)), tuple(np.int32(b)), color, 1, cv2.LINE_AA)
    for e in np.arange(e0, e1 + 1e-6, step_m):
        poly(np.column_stack([np.full(60, e), np.linspace(n0, n1, 60)]))
    for n in np.arange(n0, n1 + 1e-6, step_m):
        poly(np.column_stack([np.linspace(e0, e1, 60), np.full(60, n)]))
    return out


def parse_range(s: str) -> list[int]:
    a, b, st = (int(x) for x in s.split(":"))
    return list(range(a, b + 1, st))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--srt", type=Path)
    ap.add_argument("--camera", default="equiv_24mm", help=f"preset: {CameraModel.presets()}")
    ap.add_argument("--sensor-w-mm", type=float)
    ap.add_argument("--focal-mm", type=float)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=960, help="processing width (px)")
    ap.add_argument("--gap", type=int, default=3, help="frames between the two frames of a pair")
    ap.add_argument("--stride", type=int, default=5, help="frames between successive pairs")
    ap.add_argument("--max-pairs", type=int, default=300)
    ap.add_argument("--offset-sweep", default="-400:400:20", help="ms range a:b:step for sync search (write --offset-sweep=-400:400:20)")
    ap.add_argument("--overlays", type=int, default=4, help="frames to show with a ground grid and draped on the map")
    ap.add_argument("--map-range", type=float, default=600.0, help="clip footprints on the map at this ground range (m)")
    ap.add_argument("--no-html", action="store_true", help="skip report.html")
    a = ap.parse_args(argv)

    cam = (CameraModel(a.sensor_w_mm, a.focal_mm) if a.sensor_w_mm and a.focal_mm
           else CameraModel.preset(a.camera))
    if not a.video.exists():
        raise SystemExit(f"video not found: {a.video.resolve()}")
    diag: list[str] = []
    track = open_telemetry(a.video, a.srt, diagnostics=diag)
    print("telemetry readers:\n  " + "\n  ".join(diag))
    if track is None:
        raise SystemExit("no telemetry found")
    prov = TelemetryPoseProvider(track, cam, extrapolate_ms=100)
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"video": str(a.video), "camera": {"name": cam.name, "sensor_w_mm": cam.sensor_w_mm,
              "focal_mm": cam.focal_mm, "hfov_deg": cam.hfov_deg},
              "telemetry": telemetry_sanity(track, prov.frame), "telemetry_readers": diag}

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = a.width
    h = int(round(H0 * w / W0))
    report["video_info"] = {"fps": fps, "frames": n_frames, "size": [W0, H0], "processing_size": [w, h],
                            "duration_ms": n_frames * 1000 / fps}

    frames = {}
    want = set()
    for i in range(0, n_frames - a.gap, a.stride):
        want |= {i, i + a.gap}
        if len(want) >= 2 * a.max_pairs:
            break
    ov_idx = set(int(x) for x in np.linspace(0, max(n_frames - 1, 0), a.overlays + 2)[1:-1]) if a.overlays else set()
    for i in range(n_frames):
        ok, fr = cap.read()
        if not ok:
            break
        if i in want or i in ov_idx:
            frames[i] = cv2.cvtColor(cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if i > max(want | ov_idx):
            break
    cap.release()
    t_of = lambda i: i * 1000.0 / fps

    pairs_vis, rs = [], []
    for i in sorted(k for k in frames if k + a.gap in frames and (k % a.stride == 0)):
        ga = prov.geometry(t_of(i), w, h)
        gb = prov.geometry(t_of(i + a.gap), w, h)
        if ga is None or gb is None:
            continue
        clip = ga.clip_row_px()
        if clip is None:
            continue
        v = estimate_homography(frames[i], frames[i + a.gap], int(math.ceil(clip)))
        if v.H is None or v.n_inliers < 20:
            continue
        pairs_vis.append((t_of(i), t_of(i + a.gap), v.H))
        rs.append(PairResidual(t_of(i), t_of(i + a.gap), v.n_inliers, *pair_residual(ga, gb, v.H), motion_px(ga, v.H)))
    report["residual_at_offset_0"] = summarize(rs)

    def geom_at(off, t):
        p = TelemetryPoseProvider(track.with_time_offset(off), cam, prov.frame, extrapolate_ms=100)
        return p.geometry(t, w, h)

    sweep = sweep_time_offset(pairs_vis, geom_at, parse_range(a.offset_sweep)) if pairs_vis else {}
    report["time_offset_sweep"] = sweep
    best = sweep.get("best_offset_ms")
    if best is not None:
        rs_best = [PairResidual(ta, tb, 0, *pair_residual(geom_at(best, ta), geom_at(best, tb), H))
                   for ta, tb, H in pairs_vis if geom_at(best, ta) and geom_at(best, tb)]
        report["residual_at_best_offset"] = summarize(rs_best)

    # footprints + track
    feats = []
    for i in range(0, n_frames, max(1, int(fps))):
        g = prov.geometry(t_of(i), w, h)
        fp = g.footprint() if g else None
        if fp:
            feats.append(fp.to_geojson({"frame": i, "t_ms": t_of(i), "area_m2": fp.area_m2,
                                        "gsd_cm": float(g.gsd_m_per_px()[0] * 100)}))
    feats.append({"type": "Feature", "properties": {"kind": "track"}, "geometry": {
        "type": "LineString", "coordinates": [[p.lon_deg, p.lat_deg] for p in track]}})
    (a.out / "footprints.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")

    for i in sorted(ov_idx):
        g = prov.geometry(t_of(i), w, h)
        if g is not None and i in frames:
            cv2.imwrite(str(a.out / f"overlay_{i:05d}.png"), draw_grid(frames[i], g))

    _plot(a.out / "residuals.png", rs, sweep)
    (a.out / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    (a.out / "report.md").write_text(_markdown(report), encoding="utf-8")
    if not a.no_html:
        from .html_report import write_html
        p = write_html(a.out, report, track, prov, frames, fps, w, h, ov_idx, a.map_range)
        print(f"report: {p}")
    return report


def _plot(path, rs, sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    if rs:
        t = [r.t_a_ms / 1000 for r in rs]
        ax[0].plot(t, [r.median_px for r in rs], label="median")
        ax[0].plot(t, [r.p95_px for r in rs], label="p95", alpha=0.6)
        ax[0].set(xlabel="video time (s)", ylabel="px", title="Telemetry vs vision residual (offset 0)")
        ax[0].legend()
    if sweep.get("scores"):
        k = sorted(sweep["scores"])
        ax[1].plot(k, [sweep["scores"][x] for x in k])
        if sweep.get("best_offset_ms") is not None:
            ax[1].axvline(sweep["best_offset_ms"], ls="--", c="gray")
        ax[1].set(xlabel="telemetry time offset (ms)", ylabel="median px", title="Sync sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _markdown(r: dict) -> str:
    t = r["telemetry"]
    z, b = r.get("residual_at_offset_0", {}), r.get("residual_at_best_offset", {})
    sw = r.get("time_offset_sweep", {})
    lines = [f"# drongo report — {Path(r['video']).name}", "",
             f"Camera: {r['camera']['name']} (HFOV {r['camera']['hfov_deg']:.1f}°)", "",
             "## Telemetry", "",
             f"- {t['samples']} samples ({t['source']}), median interval {t['median_interval_ms']} ms",
             f"- altitude {t['alt_agl_m'][0]:.1f}–{t['alt_agl_m'][1]:.1f} m, pitch {t['pitch_deg'][0]:.1f}–{t['pitch_deg'][1]:.1f}°",
             f"- speed median {t['speed_ms']['median']:.1f} m/s, max {t['speed_ms']['max']:.1f} m/s; position jumps: {t['position_jumps']}" if t.get("speed_ms") else "- speed: n/a",
             f"- gimbal fields present: {t['has_gimbal']}; yaw crosses north {t['yaw_crosses_north']}×; zero-altitude samples: {t['zero_altitude_samples']}",
             "", "## Telemetry vs vision", "",
             f"- offset 0: {z.get('pairs', 0)} pairs, median {z.get('median_px', float('nan')):.2f} px "
             f"({z.get('median_m', float('nan')):.2f} m), p95 {z.get('p95_px', float('nan')):.2f} px; "
             f"measured image motion per pair {z.get('median_motion_px') or float('nan'):.1f} px",
             ]
    if z.get("median_motion_px") is not None and z["median_motion_px"] < 5:
        lines.append("- image motion per pair is small; increase --gap so the residual is not dominated by tracking noise")
    if sw.get("observable") is False:
        lines.append(f"- sync sweep is flat (argmin {sw.get('argmin_ms')} ms is noise): offset not observable on this clip")
    if sw.get("best_offset_ms") is not None:
        lines.append(f"- best sync offset {sw['best_offset_ms']} ms: median {b.get('median_px', float('nan')):.2f} px "
                     f"({b.get('median_m', float('nan')):.2f} m)")
    lines += ["", "Residual compares frame-to-frame motion predicted by telemetry with motion measured "
              "from the images. It is sensitive to relative errors (attitude noise, sync, camera model), "
              "not to constant GPS bias. Sync is only observable while the drone turns, accelerates or "
              "moves the gimbal non-uniformly; on a straight constant-speed leg or a steady turn the sweep is flat.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
