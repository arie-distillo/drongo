"""Runs the scripts in examples/ on a synthetic clip and checks their output
against ground truth."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from drongo import CameraModel, FrameGeometry, LocalFrame
from testbed.synth.render import make_texture, write_video
from testbed.synth.scenario import TelemetryNoise, slalom

pytestmark = pytest.mark.slow
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
CAM = CameraModel.preset("equiv_24mm")
W, H = 960, 540


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("examples")
    sc = slalom(duration_s=3.0)
    vid, srt = write_video(sc, make_texture(seed=5), CAM, d, W, H, TelemetryNoise())
    return sc, vid, srt, d


def test_detections_to_geojson(clip):
    sc, vid, srt, d = clip
    # place "detections" at known ground points, using the true poses
    targets_en = np.array([[45.0, 3.0], [60.0, -4.0]])     # ahead of the drone on its path
    rows = []
    for f in (0, 10, 20, 30):
        g = FrameGeometry.from_pose(sc.poses[f], CAM, W, H, sc.frame)
        for k, (u, v) in enumerate(g.en_to_pixel(targets_en)):
            if 10 < u < W - 10 and 10 < v < H - 1:
                rows.append({"frame": f, "x0": u - 8, "y0": v - 12, "x1": u + 8, "y1": v, "label": f"t{k}"})
    assert len(rows) >= 4
    det = d / "detections.csv"
    with open(det, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "x0", "y0", "x1", "y1", "label"])
        w.writeheader()
        w.writerows(rows)

    out = d / "detections.geojson"
    stats = load("detections_to_geojson").main([str(vid), str(det), str(out)])
    assert stats["written"] == len(rows) and stats["skipped"] == 0

    fr = LocalFrame(sc.poses[0].lat_deg, sc.poses[0].lon_deg)
    for feat in json.loads(out.read_text(encoding="utf-8"))["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        k = int(feat["properties"]["label"][1])
        err = np.linalg.norm(fr.to_enu([[lat, lon]])[0] - targets_en[k])
        # SRT rounds angles to 0.1 deg (error <= 0.05 deg). Ground error from a pitch error is
        # (R^2 + alt^2) / alt per radian, from a yaw error R per radian; + 2 cm for positions.
        R, alt = feat["properties"]["range_m"], sc.poses[feat["properties"]["frame"]].alt_agl_m
        tol = ((R * R + alt * alt) / alt + R) * np.radians(0.05) + 0.02
        assert err < tol, (feat, err, tol)
        assert 0.5 < feat["properties"]["width_m"] < 5


def test_stabilise(clip):
    sc, vid, srt, d = clip
    out = d / "stabilised.mp4"
    stats = load("stabilise").main([str(vid), str(out), "--step", "5"])
    assert stats["frames_written"] >= 15
    assert stats["mean_residual"] < 6           # ground cancels: small grey-level residual
    assert out.exists() and out.stat().st_size > 10_000
