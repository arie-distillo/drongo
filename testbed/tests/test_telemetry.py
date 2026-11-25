import pytest

from drongo import DjiSrtReader, PoseSample, TelemetrySourceKind, TelemetryTrack, lerp_angle_deg, wrap_deg, write_dji_srt
from drongo.telemetry import parse_srt_block

LAT, LON = 31.7683, 35.2137


def ps(t, yaw=0.0, roll=0.0, lat=LAT, lon=LON, alt=85.0, pitch=-90.0):
    return PoseSample(t, lat, lon, alt, yaw, pitch, roll)


@pytest.mark.parametrize("a,b,t,exp", [(359, 1, 0.5, 0), (10, 350, 0.5, 0), (0, 90, 0.5, 45),
                                       (170, -170, 0.5, 180), (350, 10, 0.25, 355)])
def test_angle_lerp_shortest_arc(a, b, t, exp):
    assert lerp_angle_deg(a, b, t) == pytest.approx(exp % 360, abs=1e-9)


def test_wrap():
    assert wrap_deg(190) == pytest.approx(-170)
    assert wrap_deg(-190) == pytest.approx(170)


def test_track_interpolates_across_north():
    tr = TelemetryTrack([ps(0, yaw=359), ps(100, yaw=1)])
    assert tr.at(50).gimbal_yaw_deg == pytest.approx(0.0, abs=1e-9)


def test_track_signed_roll():
    tr = TelemetryTrack([ps(0, roll=-2), ps(100, roll=-4)])
    assert tr.at(50).gimbal_roll_deg == pytest.approx(-3)


def test_track_position_and_bounds():
    tr = TelemetryTrack([ps(0, lat=LAT), ps(1000, lat=LAT + 0.001)], extrapolation="hold")
    assert tr.at(250).lat_deg == pytest.approx(LAT + 0.00025)
    assert tr.at(-1) is None and tr.at(1001) is None
    assert tr.at(1050, extrapolate_ms=100).lat_deg == pytest.approx(LAT + 0.001)     # held
    assert tr.at(250).t_ms == 250
    assert tr.at(-50, extrapolate_ms=100).lat_deg == pytest.approx(LAT)              # before start: held


def test_linear_extrapolation_continues_motion():
    # 10 Hz track moving north 1e-5 deg per 100 ms, yaw turning 2 deg per sample across north
    s = [ps(i * 100, lat=LAT + i * 1e-5, yaw=(356 + 2 * i) % 360, pitch=-60 - i) for i in range(6)]
    tr = TelemetryTrack(s)                                   # default: linear, 300 ms baseline
    p = tr.at(600, extrapolate_ms=200)
    assert p.lat_deg == pytest.approx(LAT + 6e-5)
    assert p.gimbal_yaw_deg == pytest.approx(8.0)            # 356 + 2*6 = 368 -> 8, across north
    assert p.gimbal_pitch_deg == pytest.approx(-66.0)
    assert tr.at(700, extrapolate_ms=200).lat_deg == pytest.approx(LAT + 7e-5)
    assert tr.at(701, extrapolate_ms=200) is None


def test_linear_extrapolation_with_repeated_gps_fixes():
    # receiver repeats each fix for 3 samples (as seen on real footage): a 300 ms baseline still sees the motion
    s = [ps(i * 33, lat=LAT + (i // 3) * 3e-6) for i in range(30)]
    tr = TelemetryTrack(s)
    true_rate = 3e-6 / 99.0                                  # deg per ms
    p = tr.at(s[-1].t_ms + 100, extrapolate_ms=200)
    assert abs(p.lat_deg - (s[-1].lat_deg + 100 * true_rate)) < 3e-6


def test_extrapolation_clamps_pitch_and_validates_mode():
    tr = TelemetryTrack([ps(0, pitch=-80), ps(100, pitch=-89)])
    assert tr.at(300, extrapolate_ms=300).gimbal_pitch_deg == -90.0
    with pytest.raises(ValueError):
        TelemetryTrack([], extrapolation="spline")


def test_time_offset_positive_means_telemetry_late():
    # telemetry stamped 200 ms late: at video t=0 the true pose is the sample stamped 200
    tr = TelemetryTrack([ps(200, yaw=10), ps(1200, yaw=20)]).with_time_offset(200)
    assert tr.at(0).gimbal_yaw_deg == pytest.approx(10)
    assert tr.at(500).gimbal_yaw_deg == pytest.approx(15)


def test_append_rejects_out_of_order():
    tr = TelemetryTrack([ps(100)])
    tr.append(ps(200))
    with pytest.raises(ValueError):
        tr.append(ps(150))


FMT_A = "<font size=\"28\">[latitude: 31.768300] [longitude: 35.213700] [altitude: 85.20]</font>"
FMT_B_SEPARATE = "[latitude : 31.7683] [longitude : 35.2137] [altitude : 85.2] [gb_yaw : 12.4] [gb_pitch : -60.5] [gb_roll : 0.3]"
FMT_B_GROUPED = ("FrameCnt: 1 [latitude: 31.7683] [longitude: 35.2137] [rel_alt: 85.200 abs_alt: 912.4] "
                 "[gb_yaw: 12.4 gb_pitch: -60.5 gb_roll: 0.3]")
FMT_C = "GPS (31.7683, 35.2137, 0), D 10.00m, H 85.20m, H.S 5.0m/s, V.S 0.00m/s"


@pytest.mark.parametrize("text,kind,pitch,yaw", [
    (FMT_A, TelemetrySourceKind.DJI_SRT_A, -90.0, 0.0),
    (FMT_B_SEPARATE, TelemetrySourceKind.DJI_SRT_B, -60.5, 12.4),
    (FMT_B_GROUPED, TelemetrySourceKind.DJI_SRT_B, -60.5, 12.4),
    (FMT_C, TelemetrySourceKind.DJI_SRT_C, -90.0, 0.0),
])
def test_srt_formats(text, kind, pitch, yaw):
    s = parse_srt_block(text, 1234)
    assert s.source == kind and s.t_ms == 1234
    assert (s.lat_deg, s.lon_deg) == pytest.approx((31.7683, 35.2137))
    assert s.alt_agl_m == pytest.approx(85.2)
    assert s.gimbal_pitch_deg == pytest.approx(pitch) and s.gimbal_yaw_deg == pytest.approx(yaw)


def test_rel_alt_not_confused_with_abs_alt():
    s = parse_srt_block(FMT_B_GROUPED, 0)
    assert s.alt_agl_m == pytest.approx(85.2) and s.alt_msl_m == pytest.approx(912.4)


def test_srt_roundtrip():
    samples = [ps(i * 33, yaw=(350 + i) % 360, pitch=-45.0 - i) for i in range(20)]
    tr = DjiSrtReader().read_text(write_dji_srt(samples, 33))
    assert len(tr) == 20
    for a, b in zip(samples, tr):
        assert b.t_ms == a.t_ms
        assert b.gimbal_yaw_deg == pytest.approx(a.gimbal_yaw_deg, abs=0.05)
        assert b.gimbal_pitch_deg == pytest.approx(a.gimbal_pitch_deg, abs=0.05)
        assert b.lat_deg == pytest.approx(a.lat_deg, abs=1e-7)


def test_srt_skips_blocks_without_gps():
    text = "1\n00:00:00,000 --> 00:00:00,033\nno gps here\n\n2\n00:00:00,033 --> 00:00:00,066\n" + FMT_A + "\n"
    tr = DjiSrtReader().read_text(text)
    assert len(tr) == 1 and tr[0].t_ms == 33


def test_srt_windows_newlines_bom_and_case(tmp_path):
    from drongo import open_telemetry
    body = write_dji_srt([ps(0), ps(33)], 33).replace("\n", "\r\n")
    (tmp_path / "clip.Srt").write_bytes(("﻿" + body).encode("utf-8"))
    (tmp_path / "CLIP.mp4").write_bytes(b"")
    diag = []
    tr = open_telemetry(tmp_path / "CLIP.mp4", diagnostics=diag)
    assert tr is not None and len(tr) == 2, diag


def test_open_telemetry_explains_failure(tmp_path):
    from drongo import open_telemetry
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "a.SRT").write_text("1\n00:00:00,000 --> 00:00:00,033\nsome unknown format\n")
    diag = []
    assert open_telemetry(tmp_path / "a.mp4", diagnostics=diag) is None
    assert "0 samples" in diag[0] and "unknown format" in diag[0]
    diag = []
    assert open_telemetry(tmp_path / "a.mp4", tmp_path / "missing.SRT", diagnostics=diag) is None
    assert "does not exist" in diag[0]
