"""Render a synthetic drone clip (+ DJI-style SRT) with known ground truth.

    python -m testbed.synth.make_clip --scenario slalom --out synth
    python -m testbed.synth.make_clip --scenario oblique_turn --noise dji --offset-ms 160 --out synth

The video is rendered from exact poses; the SRT carries telemetry with the
chosen noise profile, so run_video on the result shows how well drongo and
the harness recover the injected errors.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from drongo import CameraModel

from .render import make_texture, write_video
from .scenario import DJI_CONSUMER, SCENARIOS, TelemetryNoise


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="slalom")
    ap.add_argument("--duration-s", type=float, default=6.0)
    ap.add_argument("--camera", default="generic_small_sensor", help=f"preset: {CameraModel.presets()}")
    ap.add_argument("--noise", choices=["none", "dji"], default="dji")
    ap.add_argument("--offset-ms", type=int, default=0, help="telemetry clock offset to inject")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--out", type=Path, default=Path("synth"))
    a = ap.parse_args(argv)

    noise = DJI_CONSUMER if a.noise == "dji" else TelemetryNoise()
    noise = dataclasses.replace(noise, time_offset_ms=a.offset_ms)
    sc = SCENARIOS[a.scenario](duration_s=a.duration_s)
    vid, srt = write_video(sc, make_texture(), CameraModel.preset(a.camera), a.out, a.width, a.height, noise)
    print(f"wrote {vid} and {srt} ({len(sc.poses)} frames, noise={a.noise}, offset={a.offset_ms} ms)")


if __name__ == "__main__":
    main()
