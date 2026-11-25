"""DJI embedded telemetry: protobuf decoding, encoding detection, sanity gate,
MP4 container parsing and robustness against malformed files.

Fixtures are raw packets cut from two files: a DJI Mavic 3 Enterprise
recording (angles as protobuf int) and a simulator's output (zigzag sint).
Expected values were checked against the flight: on the M3E clip the gimbal
yaw (17.8 deg) matches the GPS track heading (17.6-18.3 deg)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from drongo import DjiEmbeddedReader, LocalFrame, PoseSample, TelemetryTrack, open_telemetry
from drongo.dji_embedded import decode_fields
from testbed.synth.mp4meta import encode_packet, write_embedded, write_meta_mp4

FIX = Path(__file__).parent / "fixtures"


def fixture_track(name):
    pk = json.loads((FIX / f"{name}_packets.json").read_text())
    r = DjiEmbeddedReader()
    tr = r.from_decoded([(p["t_ms"], decode_fields(bytes.fromhex(p["hex"]))) for p in pk])
    return r, list(tr)


def test_m3e_packets():
    r, s = fixture_track("dji_m3e")
    assert r.last_stats["angle_encoding"] == "int" and r.last_stats["rejected_implausible"] == 0
    a = s[0]
    assert (a.lat_deg, a.lon_deg) == pytest.approx((32.041041, 34.838373), abs=1e-6)
    assert a.alt_agl_m == pytest.approx(85.39, abs=0.01)
    assert a.alt_msl_m == pytest.approx(151.12, abs=0.01)
    assert (a.gimbal_pitch_deg, a.gimbal_roll_deg, a.gimbal_yaw_deg) == pytest.approx((-8.9, 0.0, 17.8))
    assert a.body_yaw_deg == pytest.approx(17.3)
    assert s[3].gimbal_pitch_deg == pytest.approx(-18.6)          # gimbal tilted down later in the clip
    assert [x.t_ms for x in s[:3]] == [0, 33, 67]                 # 30000/1001 fps sample table


def test_sim_zigzag_packets():
    r, s = fixture_track("sim_zigzag")
    assert r.last_stats["angle_encoding"] == "zigzag"
    assert (s[0].gimbal_pitch_deg, s[0].gimbal_yaw_deg) == pytest.approx((-90.0, 45.0))
    assert s[0].alt_agl_m == pytest.approx(80.0)


@pytest.mark.parametrize("zigzag", [False, True])
def test_mp4_roundtrip(tmp_path, zigzag):
    fr = LocalFrame(47.3769, 8.5417)
    truth = []
    for i in range(90):
        lat, lon = fr.to_wgs84([[i * 0.4, i * 0.3]])[0]
        truth.append(PoseSample(int(round(i * 1000 / 29.97)), float(lat), float(lon), 60.0 + i * 0.01,
                                (350.0 + i * 0.4) % 360, -45.0 - i * 0.2, 0.3, alt_msl_m=480.0))
    f = tmp_path / "flight.mp4"
    write_embedded(f, truth, zigzag=zigzag)
    diag = []
    tr = open_telemetry(f, diagnostics=diag)                    # default readers, no sidecar
    assert tr is not None, diag
    got = list(tr)
    assert len(got) == len(truth) and "dji_protobuf" in diag[-1] and "OK" in diag[-1]
    for t, g in zip(truth, got):
        assert abs(g.t_ms - t.t_ms) <= 1
        assert fr.distance_m([[t.lat_deg, t.lon_deg]], [[g.lat_deg, g.lon_deg]])[0] < 1e-6
        assert g.gimbal_pitch_deg == pytest.approx(t.gimbal_pitch_deg, abs=0.051)
        assert ((g.gimbal_yaw_deg - t.gimbal_yaw_deg + 180) % 360 - 180) == pytest.approx(0, abs=0.051)
        assert g.alt_agl_m == pytest.approx(t.alt_agl_m, abs=1e-3)


def test_sanity_gate_rejects_impossible_angles():
    good = PoseSample(0, 32.0, 35.0, 80.0, 10.0, -45.0)
    packets = [decode_fields(encode_packet(good))] * 3
    bad = decode_fields(encode_packet(good.with_(gimbal_pitch_deg=-170.0)))   # pitch beyond -120
    r = DjiEmbeddedReader()
    tr = r.from_decoded([(0, packets[0]), (33, bad), (67, packets[1]), (100, packets[2])])
    assert len(tr) == 3 and r.last_stats["rejected_implausible"] == 1


def test_reports_why_when_no_dji_track(tmp_path):
    f = tmp_path / "other.mp4"
    write_meta_mp4(f, [b"\x08\x01"], [0.0], handler_name="GoPro MET", sample_format=b"gpmd")
    diag = []
    assert open_telemetry(f, diagnostics=diag) is None
    assert "no DJI metadata track" in diag[-1] and "GoPro MET" in diag[-1]


def test_garbage_and_truncated_files(tmp_path):
    rng = np.random.default_rng(0)
    good = tmp_path / "good.mp4"
    write_embedded(good, [PoseSample(i * 33, 32.0, 35.0, 80.0, 0.0, -90.0) for i in range(10)])
    data = good.read_bytes()
    cases = {"random.mp4": rng.integers(0, 256, 5000, dtype=np.uint8).tobytes(),
             "truncated.mp4": data[: len(data) // 2],
             "empty.mp4": b""}
    for name, content in cases.items():
        f = tmp_path / name
        f.write_bytes(content)
        diag = []
        assert open_telemetry(f, diagnostics=diag) is None, name     # no exception, no samples
        assert diag
    for _ in range(300):                                             # decoder never raises
        decode_fields(rng.integers(0, 256, int(rng.integers(0, 300)), dtype=np.uint8).tobytes())


@pytest.mark.skipif(not os.environ.get("DRONGO_SAMPLES"), reason="set DRONGO_SAMPLES to a folder of real clips")
def test_real_clips_folder():
    """Optional: every .mp4/.mov in DRONGO_SAMPLES with a DJI track yields plausible telemetry."""
    found = 0
    for f in sorted(Path(os.environ["DRONGO_SAMPLES"]).iterdir()):
        if f.suffix.lower() not in (".mp4", ".mov"):
            continue
        r = DjiEmbeddedReader()
        if not r.can_read(f, None):
            continue
        tr = r.read(f, None)
        found += 1
        assert len(tr) > 0 and r.last_stats["rejected_implausible"] == 0, (f.name, r.last_stats)
    assert found, "no clip with a DJI metadata track in DRONGO_SAMPLES"
