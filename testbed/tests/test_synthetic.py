"""End-to-end tests on rendered frames with exact ground truth.

1. Renderer validity: a marker painted at a known ENU point appears exactly
   where en_to_pixel says (independent check of the render path).
2. Geometry vs vision: with perfect telemetry, the visually estimated
   homography matches relative_homography to sub-pixel level.
3. Error sensitivity: injected yaw bias / time offset produce the residual
   the error budget predicts, and the sync sweep recovers the offset.
4. CLI smoke test: run_video on a synthetic clip + SRT.
"""
import json

import cv2
import numpy as np
import pytest

from drongo import CameraModel, FrameGeometry, TelemetryPoseProvider, TelemetryTrack
from testbed.harness.consistency import estimate_homography, pair_residual, sweep_time_offset
from testbed.synth.render import GroundTexture, make_texture, render, write_video
from testbed.synth.scenario import DJI_CONSUMER, TelemetryNoise, gimbal_sweep, oblique_turn, slalom, straight_nadir

pytestmark = pytest.mark.slow
CAM = CameraModel.preset("generic_small_sensor")
W, H = 960, 540


@pytest.fixture(scope="module")
def tex():
    return make_texture(half_size_m=350, res_m=0.1, seed=7)


def test_renderer_places_marker_where_geometry_says(tex):
    t = GroundTexture(np.full_like(tex.image, 128), tex.e0, tex.n1, tex.res_m)
    col, row = t.tex_px(np.array([[10.0, 17.0]]))[0].round().astype(int)
    target = (t.G_ground_from_tex @ [col, row, 1.0])[:2][None]      # exact texel centre
    cv2.circle(t.image, (int(col), int(row)), 8, 255, -1)              # 0.8 m disc
    sc = straight_nadir(duration_s=0.1)
    for pitch in (-90, -75, -65):
        p = sc.poses[0].with_(gimbal_pitch_deg=pitch, gimbal_yaw_deg=30)
        g = FrameGeometry.from_pose(p, CAM, W, H, sc.frame)
        img = render(t, g).astype(float)
        mask = img > 200
        assert mask.sum() > 3, f"marker not visible at pitch {pitch}"
        ys, xs = np.nonzero(mask)
        # centroid of a projected disc ~ projection of its centre (within 0.3 px at these ranges)
        assert np.allclose([xs.mean(), ys.mean()], g.en_to_pixel(target)[0], atol=0.25)


@pytest.mark.parametrize("scenario", [straight_nadir, oblique_turn, gimbal_sweep, slalom])
def test_vision_matches_geometry_with_perfect_telemetry(tex, scenario):
    sc = scenario(duration_s=2.0)
    prov = TelemetryPoseProvider(TelemetryTrack(sc.poses), CAM, sc.frame)
    meds = []
    for i in range(0, len(sc.poses) - 3, 6):
        ga, gb = prov.geometry(sc.poses[i].t_ms, W, H), prov.geometry(sc.poses[i + 3].t_ms, W, H)
        v = estimate_homography(render(tex, ga), render(tex, gb), int(np.ceil(ga.clip_row_px())))
        assert v.H is not None and v.n_inliers > 50
        meds.append(pair_residual(ga, gb, v.H)[0])
    assert np.median(meds) < 0.5, meds


def _residuals(tex, sc, noise, gap=3, step=6):
    truth = TelemetryPoseProvider(TelemetryTrack(sc.poses), CAM, sc.frame)
    noisy = TelemetryTrack(noise.apply(sc))
    pairs = []
    for i in range(0, len(sc.poses) - gap, step):
        ta, tb = sc.poses[i].t_ms, sc.poses[i + gap].t_ms
        ga, gb = truth.geometry(ta, W, H), truth.geometry(tb, W, H)
        v = estimate_homography(render(tex, ga), render(tex, gb), int(np.ceil(ga.clip_row_px())))
        pairs.append((ta, tb, v.H))
    return pairs, noisy


def test_yaw_bias_is_visible_only_in_absolute_position(tex):
    """A constant yaw bias rotates every frame equally, so frame-to-frame
    residual stays small: the vision check cannot see it. Absolute error at
    the frame edge is ~ range * tan(bias). This is why Phase 4 (map
    registration) is needed for absolute accuracy."""
    sc = straight_nadir(duration_s=1.0)
    pairs, noisy = _residuals(tex, sc, TelemetryNoise(yaw_bias_deg=2.0))
    prov = TelemetryPoseProvider(noisy, CAM, sc.frame)
    truth = TelemetryPoseProvider(TelemetryTrack(sc.poses), CAM, sc.frame)
    rel = [pair_residual(prov.geometry(a, W, H), prov.geometry(b, W, H), Hm)[0] for a, b, Hm in pairs]
    assert np.median(rel) < 1.0
    g_true, g_noisy = truth.geometry(0, W, H), prov.geometry(0, W, H)
    corner = [[0, 0]]
    abs_err = np.linalg.norm(g_true.pixel_to_en(corner) - g_noisy.pixel_to_en(corner))
    rng = np.linalg.norm(g_true.pixel_to_en(corner) - g_true.camera_enu_m[:2])
    assert abs_err == pytest.approx(rng * np.radians(2.0), rel=0.02)


def test_time_offset_unobservable_in_steady_turn(tex):
    """Constant speed + constant turn rate: every frame pair sees the same
    relative motion, so a clock offset changes nothing. Documents a real
    limitation of sync estimation from frame-to-frame consistency."""
    sc = oblique_turn(duration_s=2.0, turn_rate_dps=25.0)
    pairs, noisy = _residuals(tex, sc, TelemetryNoise(time_offset_ms=120), step=6)
    geom_at = lambda off, t: TelemetryPoseProvider(noisy.with_time_offset(off), CAM, sc.frame,
                                                   extrapolate_ms=200).geometry(t, W, H)
    sc_ = sweep_time_offset(pairs, geom_at, [0, 120])["scores"]
    assert abs(sc_[0] - sc_[120]) < 0.1


def test_time_offset_recovered_during_manoeuvre(tex):
    sc = slalom(duration_s=3.0)
    pairs, noisy = _residuals(tex, sc, TelemetryNoise(time_offset_ms=120), step=4)
    geom_at = lambda off, t: TelemetryPoseProvider(noisy.with_time_offset(off), CAM, sc.frame,
                                                   extrapolate_ms=200).geometry(t, W, H)
    r = sweep_time_offset(pairs, geom_at, range(0, 241, 20))
    assert r["best_offset_ms"] == pytest.approx(120, abs=20), r
    assert r["scores"][120] < 0.5 * r["scores"][0]


def test_realistic_noise_budget(tex):
    """Consumer-DJI noise profile: frame-to-frame residual is dominated by
    attitude noise + quantisation; report the number so changes are visible."""
    sc = oblique_turn(duration_s=2.0)
    pairs, noisy = _residuals(tex, sc, DJI_CONSUMER)
    prov = TelemetryPoseProvider(noisy, CAM, sc.frame)
    rel = [pair_residual(prov.geometry(a, W, H), prov.geometry(b, W, H), Hm) for a, b, Hm in pairs]
    med_px = float(np.median([r[0] for r in rel]))
    print(f"\nDJI_CONSUMER oblique_turn: median residual {med_px:.2f} px "
          f"({np.median([r[2] for r in rel]):.2f} m) at {W}x{H}")
    assert 0.5 < med_px < 40     # noisy enough to be visible, not a broken pipeline


def test_run_video_cli(tex, tmp_path):
    from testbed.harness.run_video import main
    sc = slalom(duration_s=3.0)
    vid, srt = write_video(sc, tex, CAM, tmp_path, W, H, TelemetryNoise(time_offset_ms=100, quantize_deg=0.1))
    rep = main(["--video", str(vid), "--srt", str(srt), "--camera", "generic_small_sensor", "--out", str(tmp_path / "rep"),
                "--width", str(W), "--gap", "3", "--stride", "4", "--offset-sweep=-200:300:20"])
    assert rep["telemetry"]["samples"] == len(sc.poses)
    assert rep["time_offset_sweep"]["best_offset_ms"] == pytest.approx(100, abs=40)
    html = (tmp_path / "rep" / "report.html").read_text(encoding="utf-8")
    assert "L.map(" in html and "imageOverlay" in html and "data:image/png;base64," in html
    for f in ("report.json", "report.md", "footprints.geojson", "residuals.png"):
        assert (tmp_path / "rep" / f).exists()
    gj = json.loads((tmp_path / "rep" / "footprints.geojson").read_text())
    assert len(gj["features"]) >= 3
