"""Turn per-frame detection boxes into a GeoJSON map layer.

    python examples/detections_to_geojson.py clip.mp4 detections.csv out.geojson --camera equiv_24mm

detections.csv columns: frame, x0, y0, x1, y1[, label]   (pixels, any detector)

Each detection becomes a GeoJSON Point at the bottom centre of its box (where a
vehicle or person touches the ground), with its time, size in metres and
distance from the drone. Frames outside the telemetry are skipped and counted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from drongo import CameraModel, TelemetryPoseProvider, open_telemetry


def video_info(path: Path) -> tuple[float, int, int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or not w:
        raise SystemExit(f"cannot read video {path}")
    return fps, w, h


def convert(video: Path, detections: Path, out: Path, camera: CameraModel, srt: Path | None = None) -> dict:
    diagnostics: list[str] = []
    track = open_telemetry(video, srt, diagnostics=diagnostics)
    if track is None:
        raise SystemExit("no telemetry:\n  " + "\n  ".join(diagnostics))
    provider = TelemetryPoseProvider(track, camera)
    fps, W, H = video_info(video)

    by_frame: dict[int, list[list]] = defaultdict(list)
    with open(detections, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_frame[int(row["frame"])].append(
                [float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"]), row.get("label", "")])

    features, skipped = [], 0
    for frame_idx in sorted(by_frame):
        t_ms = frame_idx * 1000.0 / fps
        g = provider.geometry(t_ms, W, H)            # one geometry per frame ...
        if g is None:
            skipped += len(by_frame[frame_idx])
            continue
        boxes = np.array([b[:4] for b in by_frame[frame_idx]])
        feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
        lat_lon = g.pixel_to_lat_lon(feet)           # ... and one call for all its boxes
        width_m = (boxes[:, 2] - boxes[:, 0]) * g.gsd_m_per_px(feet)
        range_m = g.ground_range_m(feet)
        for (lat, lon), w_m, r_m, b in zip(lat_lon, width_m, range_m, by_frame[frame_idx]):
            if not np.isfinite(lat):                 # box bottom above the horizon
                skipped += 1
                continue
            features.append({"type": "Feature",
                             "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                             "properties": {"frame": frame_idx, "t_ms": round(t_ms), "label": b[4],
                                            "width_m": round(float(w_m), 2), "range_m": round(float(r_m), 1)}})
    fc = {"type": "FeatureCollection", "features": features}
    out.write_text(json.dumps(fc), encoding="utf-8")
    return {"written": len(features), "skipped": skipped}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("detections", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--srt", type=Path)
    ap.add_argument("--camera", default="equiv_24mm", help=f"preset: {CameraModel.presets()}")
    a = ap.parse_args(argv)
    stats = convert(a.video, a.detections, a.out, CameraModel.preset(a.camera), a.srt)
    print(f"wrote {stats['written']} points to {a.out}; skipped {stats['skipped']}", file=sys.stderr)
    return stats


if __name__ == "__main__":
    main()
