"""Remove camera motion from a drone clip using telemetry alone, and show what moves.

    python examples/stabilise.py clip.mp4 out.mp4 --camera equiv_24mm --step 5

Every output frame is the current frame next to |current - previous aligned to
current|, where the alignment is the ground-plane homography predicted by
telemetry (no image matching). With good telemetry the ground cancels and what
remains is moving objects, things above the ground (parallax), and telemetry
error - so the right half is also a quick visual check of the telemetry.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from drongo import CameraModel, TelemetryPoseProvider, open_telemetry, relative_homography


def stabilise(video: Path, out: Path, camera: CameraModel, step: int = 5, srt: Path | None = None,
              max_frames: int | None = None) -> dict:
    diagnostics: list[str] = []
    track = open_telemetry(video, srt, diagnostics=diagnostics)
    if track is None:
        raise SystemExit("no telemetry:\n  " + "\n  ".join(diagnostics))
    provider = TelemetryPoseProvider(track, camera)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (2 * W, H))

    prev = prev_g = None
    idx, written, residuals = 0, 0, []
    while max_frames is None or idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = provider.geometry(idx * 1000.0 / fps, W, H)
            if g is not None and prev_g is not None:
                aligned = cv2.warpPerspective(prev, relative_homography(prev_g, g), (W, H))
                valid = (aligned > 0) & (gray > 0)
                diff = np.where(valid, cv2.absdiff(aligned, gray), 0).astype(np.uint8)
                residuals.append(float(diff[valid].mean()) if valid.any() else float("nan"))
                heat = cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=4), cv2.COLORMAP_INFERNO)
                writer.write(np.hstack([frame, heat]))
                written += 1
            prev, prev_g = gray, g
        idx += 1
    cap.release()
    writer.release()
    return {"frames_written": written, "mean_residual": float(np.nanmean(residuals)) if residuals else None}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--srt", type=Path)
    ap.add_argument("--camera", default="equiv_24mm", help=f"preset: {CameraModel.presets()}")
    ap.add_argument("--step", type=int, default=5, help="compare frames this many apart")
    a = ap.parse_args(argv)
    stats = stabilise(a.video, a.out, CameraModel.preset(a.camera), a.step, a.srt)
    print(f"wrote {stats['frames_written']} frames to {a.out}; mean residual {stats['mean_residual']:.1f} grey levels",
          file=sys.stderr)
    return stats


if __name__ == "__main__":
    main()
