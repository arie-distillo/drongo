"""Demo-clip generator: colour renderer, realistic flights, H.264 + embedded
telemetry. The clips are for demos and for testing other pipelines, so what
must hold is that the pixels, the telemetry files and the ground truth agree.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from drongo import CameraModel, FrameGeometry, open_telemetry
from testbed.synth.encode import VideoWriter, find_ffmpeg
from testbed.synth.demo import main as demo_main, recorded_telemetry
from testbed.synth.flights import FLIGHTS, cruise, orbit, survey
from testbed.synth.mp4meta import embed_telemetry
from testbed.synth.render import GroundTexture, Look, MipTexture, render_rgb
from testbed.synth.scenario import DJI_CONSUMER, TelemetryNoise, straight_nadir
from testbed.synth.terrain import TERRAINS

CAM = CameraModel.preset("mini3pro")
W, H = 640, 360


def _speeds(sc):
    en = sc.frame.to_enu([[p.lat_deg, p.lon_deg] for p in sc.poses])
    return np.linalg.norm(np.diff(en, axis=0), axis=1) * sc.fps


@pytest.mark.parametrize("name", sorted(FLIGHTS))
def test_flights_start_from_rest_and_respect_limits(name):
    sc = FLIGHTS[name](duration_s=12, wind=0.0)
    v = _speeds(sc)
    assert v[0] < 0.5                                         # takes off from a hover
    a = np.diff(v) * sc.fps
    assert np.percentile(np.abs(a), 99) < 3.5                 # planner limits are 1.5-2.5 m/s^2
    pitch = np.array([p.gimbal_pitch_deg for p in sc.poses])
    assert pitch.min() >= -90.0 and pitch.max() < 0


def test_survey_slows_into_turns_and_flies_lines():
    sc = survey(camera=CAM, duration_s=60, wind=0.0)
    v = _speeds(sc)
    assert v.max() == pytest.approx(8.0, abs=0.3)
    yaw = np.unwrap(np.radians([p.gimbal_yaw_deg for p in sc.poses]))
    turned = np.abs(np.degrees(yaw - yaw[0])).max()
    assert turned > 170                                       # at least one 180 deg turnaround
    in_turn = np.abs(np.gradient(np.degrees(yaw)) * sc.fps) > 5
    assert v[in_turn[1:]].mean() < 0.8 * v.max()


def test_orbit_keeps_point_of_interest_centred():
    sc = orbit(duration_s=8, wind=0.0)
    for p in sc.poses[::40]:
        g = FrameGeometry.from_pose(p, CAM, W, H, sc.frame)
        uv = g.en_to_pixel([[0.0, 0.0]])[0]
        assert np.hypot(uv[0] - W / 2, uv[1] - H / 2) < 0.05 * W


@pytest.mark.parametrize("pitch", [-90, -60, -35])
def test_colour_render_places_marker_where_geometry_says(pitch):
    sc = straight_nadir(duration_s=0.1)
    p = sc.poses[0].with_(gimbal_pitch_deg=pitch, gimbal_yaw_deg=30.0, alt_agl_m=40.0)
    g = FrameGeometry.from_pose(p, CAM, W, H, sc.frame)
    tex = GroundTexture(np.full((3000, 3000, 3), 110, np.uint8), -150.0, 150.0, 0.1)
    seen = g.pixel_to_en([[0.6 * W, 0.7 * H]])                 # a ground point in view
    col, row = tex.tex_px(seen)[0].round().astype(int)
    target = (tex.G_ground_from_tex @ [col, row, 1.0])[:2][None]   # exact texel centre
    cv2.circle(tex.image, (int(col), int(row)), 10, (255, 255, 255), -1)
    look = Look(vignette=0.0, noise_sigma=0.0, visibility_m=1e9, contrast=1.0, haze_bgr=(0, 0, 0),
                zenith_bgr=(0, 0, 0))
    out = render_rgb(MipTexture(tex), g, look)
    m = cv2.moments((out[..., 0] > 180).astype(np.uint8))
    assert m["m00"] > 0
    found = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    expect = g.en_to_pixel(target)[0]
    assert np.linalg.norm(found - expect) < 1.0


def test_sky_above_horizon():
    tex = GroundTexture(np.full((500, 500, 3), 60, np.uint8), -50.0, 50.0, 0.2)
    sc = straight_nadir(duration_s=0.1)
    g = FrameGeometry.from_pose(sc.poses[0].with_(gimbal_pitch_deg=-5.0), CAM, W, H, sc.frame)
    out = render_rgb(MipTexture(tex), g, Look(noise_sigma=0, vignette=0))
    hz = g.horizon_row_px()
    assert hz is not None and 20 < hz < H - 20
    assert out[: int(hz) - 10].mean() > 150 > out[-5:].mean()


@pytest.mark.slow
@pytest.mark.parametrize("terrain", sorted(TERRAINS))
def test_terrains_are_reproducible(terrain):
    a = TERRAINS[terrain](half_size_m=60, seed=4).image
    b = TERRAINS[terrain](half_size_m=60, seed=4).image
    assert a.shape == (1500, 1500, 3) and np.array_equal(a, b)
    assert a.std() > 10                                       # not a flat colour


def test_recorded_telemetry_one_sample_per_frame_and_offset_is_a_lag():
    sc = cruise(duration_s=4, wind=0.0)
    clean = recorded_telemetry(sc, TelemetryNoise())
    lagged = recorded_telemetry(sc, TelemetryNoise(time_offset_ms=200))
    assert [p.t_ms for p in clean] == [p.t_ms for p in sc.poses] == [p.t_ms for p in lagged]
    i = 60
    assert lagged[i].lat_deg == pytest.approx(sc.poses[i - 6].lat_deg, abs=1e-7)   # 200 ms = 6 frames


@pytest.mark.slow
@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not available")
@pytest.mark.parametrize("faststart", [False, True])
def test_embedded_track_added_to_real_video(tmp_path, faststart):
    sc = orbit(duration_s=1.5)
    mip = MipTexture(TERRAINS["urban"](half_size_m=80, seed=2))
    src = tmp_path / "v.mp4"
    with VideoWriter(src, W, H, sc.fps) as vw:
        for p in sc.poses:
            vw.write(render_rgb(mip, FrameGeometry.from_pose(p, CAM, W, H, sc.frame)))
    if faststart:
        fs = tmp_path / "fs.mp4"
        subprocess.run([find_ffmpeg(), "-v", "error", "-y", "-i", str(src), "-c", "copy",
                        "-movflags", "+faststart", str(fs)], check=True)
        src = fs
    rec = recorded_telemetry(sc, DJI_CONSUMER)
    out = embed_telemetry(src, rec, tmp_path / "emb.mp4")
    track = open_telemetry(out)
    assert track is not None and len(track) == len(rec)
    for a, b in zip(track, rec):
        assert a.t_ms == b.t_ms
        assert a.lat_deg == pytest.approx(b.lat_deg, abs=1e-9)
        assert a.gimbal_yaw_deg == pytest.approx(round(b.gimbal_yaw_deg, 1) % 360, abs=0.051)
    c1, c2 = cv2.VideoCapture(str(src)), cv2.VideoCapture(str(out))
    n = 0
    while True:
        r1, f1 = c1.read()
        r2, f2 = c2.read()
        if not (r1 and r2):
            assert r1 == r2
            break
        assert np.array_equal(f1, f2)
        n += 1
    assert n == len(sc.poses)


@pytest.mark.slow
def test_demo_cli_writes_matching_srt_embedded_and_truth(tmp_path):
    demo_main(["--terrain", "nature", "--flight", "cruise", "--duration-s", "1", "--size", "320x180",
               "--terrain-res-m", "0.4", "--out", str(tmp_path)])
    srt = open_telemetry(tmp_path / "srt" / "nature_cruise.mp4")
    emb = open_telemetry(tmp_path / "embedded" / "nature_cruise.mp4")
    truth = json.loads((tmp_path / "truth" / "nature_cruise.json").read_text())
    assert len(srt) == len(emb) == len(truth["poses"]) == 30
    for a, b, t in zip(srt, emb, truth["poses"]):
        assert a.t_ms == b.t_ms == t["t_ms"]
        assert a.gimbal_pitch_deg == pytest.approx(b.gimbal_pitch_deg, abs=0.051)
        assert abs(a.gimbal_pitch_deg - t["gimbal_pitch_deg"]) < 1.0             # noise, not a bug


def test_buildings_and_trees_are_drawn_in_3d_where_geometry_says():
    """A box building and a tree on a plain ground: the roof centre appears where
    a point at roof height projects (not where its footprint does), the wall
    facing the camera is visible, and the crown sits at crown height."""
    from testbed.synth.scene import Building, Scene
    ground = np.full((1000, 1000, 3), 100, np.uint8)
    fp = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 22.0], [10.0, 22.0]])       # counter-clockwise
    roof_px = GroundTexture(ground, -50.0, 50.0, 0.1).tex_px(fp).round().astype(np.int32)
    cv2.fillPoly(ground, [roof_px], (0, 0, 255))                                       # red roof in the texture
    sc3 = Scene()
    sc3.buildings.append(Building(fp, 20.0, 4, (255, 0, 0)))                         # blue plain walls
    sc3.tree_en.append([-15.0, -5.0]); sc3.tree_r.append(3.0); sc3.tree_z.append(6.0)
    sc3.tree_bgr.append([0.0, 200.0, 0.0])
    tex = GroundTexture(ground, -50.0, 50.0, 0.1, sc3.freeze())
    sc = straight_nadir(duration_s=0.1)
    p = sc.poses[0]
    look = Look(vignette=0.0, noise_sigma=0.0, visibility_m=1e9, contrast=1.0)
    for yaw in (0.0, 70.0, 200.0):
        cam_en = np.array([0.0, 0.0]) - 60 * np.array([np.sin(np.radians(yaw)), np.cos(np.radians(yaw))])
        ll = sc.frame.to_wgs84(cam_en[None])[0]
        g = FrameGeometry.from_pose(p.with_(lat_deg=ll[0], lon_deg=ll[1], alt_agl_m=60.0, gimbal_yaw_deg=yaw,
                                            gimbal_pitch_deg=-45.0), CAM, W, H, sc.frame)
        out = render_rgb(MipTexture(tex), g, look)
        u, v = g.en_to_pixel(fp.mean(0)[None], h_m=20.0)[0].round().astype(int)
        assert out[v, u, 2] > 200 and out[v, u, 0] < 60                               # roof, at roof height
        near = fp.mean(0) - 4 * np.array([np.sin(np.radians(yaw)), np.cos(np.radians(yaw))])
        uw, vw = g.en_to_pixel(near[None], h_m=10.0)[0].round().astype(int)            # inside the near half
        assert out[vw, uw, 0] > out[vw, uw, 2] + 40                                    # blue wall / not red roof
        ut, vt = g.en_to_pixel([[-15.0, -5.0]], h_m=6.0)[0].round().astype(int)
        assert out[vt, ut, 1] > out[vt, ut, 0] + 30                                    # green crown at its height


@pytest.mark.slow
def test_run_all_compares_demo_telemetry_with_truth(tmp_path):
    from testbed.run_all import demo_clips, telemetry_vs_truth
    demo_main(["--terrain", "urban", "--flight", "orbit", "--duration-s", "1", "--size", "320x180",
               "--terrain-res-m", "0.4", "--terrain-half-m", "120", "--noise", "none", "--out", str(tmp_path)])
    found = demo_clips(tmp_path)
    assert [c[0] for c in found] == ["urban_orbit (srt)", "urban_orbit (embedded)"]
    for _, video, truth in found:
        r = telemetry_vs_truth(video, truth)
        assert r["ok"] and r["samples"] == r["truth_frames"] == 30
        assert r["horiz_m_p95"] < 0.05 and r["alt_m_median"] < 0.01      # noise-free: only format rounding
        assert abs(r["yaw_deg_mean"]) < 0.06 and r["pitch_deg_median"] < 0.06


@pytest.mark.slow
@pytest.mark.parametrize("telemetry", ["srt", "embedded"])
def test_composite_video_has_every_frame(tmp_path, telemetry):
    from testbed.harness.composite import compose
    demo_main(["--terrain", "nature", "--flight", "survey", "--duration-s", "1.5", "--size", "320x180",
               "--terrain-res-m", "0.4", "--terrain-half-m", "120", "--telemetry", telemetry, "--out", str(tmp_path)])
    src = tmp_path / telemetry / "nature_survey.mp4"
    out = compose(src, CAM, tmp_path / "composite.mp4", size=(960, 540), progress=False)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 45
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (960, 540)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
    ok, f = cap.read()
    assert ok
    map_panel = f[30:360, 640:]                                       # draped frame + track on the map
    assert map_panel.std() > 15
