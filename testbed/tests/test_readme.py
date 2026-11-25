"""Runs every ```python block of the README's "Quick start" section, in order,
in one namespace, on a synthetic clip with exact telemetry. A README example
that stops working fails the build.

The checks after the run use the variables the examples define, so they also
verify that the examples produce sensible numbers, not just that they run."""
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from drongo import CameraModel
from testbed.synth.mp4meta import write_embedded
from testbed.synth.render import make_texture, write_video
from testbed.synth.scenario import TelemetryNoise, slalom

pytestmark = pytest.mark.slow
README = Path(__file__).resolve().parents[2] / "README.md"


def quickstart_blocks() -> list[str]:
    text = README.read_text(encoding="utf-8")
    m = re.search(r"^## Quick start\n(.*?)(?=^## )", text, flags=re.S | re.M)
    assert m, "README has no '## Quick start' section"
    return re.findall(r"```python\n(.*?)```", m.group(1), flags=re.S)


@pytest.fixture(scope="module")
def clip_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("quickstart")
    sc = slalom(duration_s=3.0)
    vid, srt = write_video(sc, make_texture(seed=3), CameraModel.preset("equiv_24mm"), d,
                           1920, 1080, TelemetryNoise())
    vid.rename(d / "clip.mp4")
    srt.rename(d / "clip.SRT")
    write_embedded(d / "flight.mp4", sc.poses)        # the same flight as embedded DJI packets
    return d


@pytest.fixture(scope="module")
def ns(clip_dir):
    """Namespace after running all Quick start blocks."""
    blocks = quickstart_blocks()
    assert len(blocks) >= 7
    env: dict = {"__name__": "__quickstart__"}
    cwd = os.getcwd()
    os.chdir(clip_dir)
    try:
        for i, code in enumerate(blocks, 1):
            try:
                exec(compile(code, f"README quick start block {i}", "exec"), env)
            except Exception as e:                       # point at the failing block
                raise AssertionError(f"Quick start block {i} failed: {type(e).__name__}: {e}\n{code}") from e
    finally:
        os.chdir(cwd)
    env["_dir"] = clip_dir
    return env


def test_1_geometry(ns):
    g = ns["g"]
    assert g.img_size == (1920, 1080)
    assert 60 < g.camera_enu_m[2] < 80                  # slalom flies at 70 m
    assert 0.02 < g.gsd_m_per_px()[0] < 0.5


def test_2_detections(ns):
    ll = ns["lat_lon"]
    assert ll.shape == (2, 2) and np.all(np.isfinite(ll))
    assert np.all((ns["width_m"] > 1) & (ns["width_m"] < 30))
    assert np.all(ns["range_m"] > 0)


def test_3_height_matters(ns):
    assert ns["shift_m"] > 10                           # 30 m height at 70 m altitude is a big effect


def test_4_geojson(ns):
    gj = json.loads((ns["_dir"] / "footprints.geojson").read_text(encoding="utf-8"))
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 7                    # frame + 6 tiles (no sky in this view)
    for f in gj["features"]:
        ring = f["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1] and len(ring) == 5


def test_5_alignment_removes_camera_motion(ns):
    a, b, aligned = ns["frames"][ns["i"]], ns["frames"][ns["j"]], ns["aligned"]
    both = (aligned > 0) & (b > 0) & (a > 0)
    before = float(cv2.absdiff(a, b)[both].mean())
    after = float(ns["residual"][both].mean())
    assert after < 0.4 * before, (before, after)


def test_6_ortho(ns):
    img = cv2.imread(str(ns["_dir"] / "ortho.png"), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (1000, 1000)
    assert (img > 0).mean() > 0.9                       # the area is inside the frame


def test_7_live(ns):
    assert ns["decoder"].encoding == "int"
    assert 30 <= len(ns["live"]) <= 33                  # packets up to ~1 s at 30 fps
    assert ns["g_now"] is not None and ns["g_now"].pose.t_ms == ns["newest"] + 150
