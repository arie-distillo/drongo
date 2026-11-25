"""Live use: packets decoded one at a time, frames that arrive ahead of the
newest telemetry, and per-frame query cost."""
import os
from pathlib import Path

import numpy as np
import pytest

from drongo import (CameraModel, DjiEmbeddedReader, DjiPacketDecoder, FrameGeometry, LocalFrame, PoseSample,
                    TelemetryPoseProvider, TelemetryTrack, iter_dji_packets)
from testbed.synth.mp4meta import encode_packet, write_embedded
from testbed.synth.scenario import straight_nadir, oblique_turn

CAM = CameraModel.preset("mavic3_wide")
W, H = 3840, 2160


def replay(samples, zigzag, latency_ms, extrapolation):
    """Feed packets one by one; query each frame time with only the telemetry
    received `latency_ms` earlier. Returns ground error (m) of the image centre."""
    dec = DjiPacketDecoder()
    truth_track = TelemetryTrack(samples)                      # exact poses, for the frame time
    track = None
    errs = []
    fr = None
    for k, truth in enumerate(samples):
        for s in dec.feed(encode_packet(truth, zigzag=zigzag), truth.t_ms):
            if track is None:
                track = TelemetryTrack([s], extrapolation=extrapolation)
                fr = LocalFrame(s.lat_deg, s.lon_deg)
                prov = TelemetryPoseProvider(track, CAM, fr, extrapolate_ms=500)
            else:
                track.append(s)
        frame_t = truth.t_ms + latency_ms                       # the frame is newer than the telemetry
        true_pose = truth_track.at(frame_t)
        if track is None or true_pose is None or k < 15:
            continue
        g = prov.geometry(frame_t, W, H)
        g_true = FrameGeometry.from_pose(true_pose, CAM, W, H, fr)
        c = [[W / 2, H * 0.6]]
        errs.append(float(np.linalg.norm(g.pixel_to_en(c) - g_true.pixel_to_en(c))))
    return np.array(errs), dec


@pytest.mark.parametrize("zigzag", [False, True])
def test_stream_decoder_matches_truth(zigzag):
    sc = straight_nadir(duration_s=2.0)
    errs, dec = replay(sc.poses, zigzag, latency_ms=0, extrapolation="linear")
    assert dec.encoding == ("zigzag" if zigzag else "int")
    assert dec.stats["emitted"] == len(sc.poses) and dec.stats["rejected_implausible"] == 0
    assert errs.max() < 0.1                                     # only 0.1-deg angle rounding


@pytest.mark.parametrize("scenario,limit_m", [(straight_nadir, 0.15), (oblique_turn, 1.0)])
def test_linear_extrapolation_beats_hold(scenario, limit_m):
    sc = scenario(duration_s=4.0)
    lin, _ = replay(sc.poses, False, latency_ms=150, extrapolation="linear")
    hold, _ = replay(sc.poses, False, latency_ms=150, extrapolation="hold")
    print(f"\n{sc.name}: 150 ms ahead of telemetry - median error linear {np.median(lin):.2f} m, "
          f"hold {np.median(hold):.2f} m")
    assert np.median(lin) < limit_m
    assert np.median(lin) < 0.25 * np.median(hold)


def test_ambiguous_start_is_buffered_then_released():
    """Level camera pointing north with zero roll: the first packets decode
    plausibly under both encodings. The decoder waits, then releases them all."""
    base = PoseSample(0, 32.0, 35.0, 50.0, 0.0, 0.0)
    level = [base.with_(t_ms=i * 33) for i in range(4)]
    tilted = base.with_(t_ms=132, gimbal_pitch_deg=-30.0)
    dec = DjiPacketDecoder()
    outs = [dec.feed(encode_packet(p, zigzag=True), p.t_ms) for p in level]
    assert outs == [[], [], [], []] and dec.encoding is None
    released = dec.feed(encode_packet(tilted, zigzag=True), tilted.t_ms)
    assert dec.encoding == "zigzag" and len(released) == 5
    assert released[-1].gimbal_pitch_deg == pytest.approx(-30.0)


def test_decide_after_limit_and_flush():
    level = [PoseSample(i * 33, 32.0, 35.0, 50.0, 0.0, 0.0) for i in range(3)]
    dec = DjiPacketDecoder(decide_after=10)
    for p in level:
        assert dec.feed(encode_packet(p), p.t_ms) == []
    assert len(dec.flush()) == 3 and dec.encoding is not None
    forced = DjiPacketDecoder(encoding="zigzag")
    assert len(forced.feed(encode_packet(level[0], zigzag=True), 0)) == 1


def test_garbage_packets_are_counted_not_raised():
    dec = DjiPacketDecoder(encoding="int")
    rng = np.random.default_rng(1)
    for i in range(200):
        dec.feed(rng.integers(0, 256, int(rng.integers(0, 200)), dtype=np.uint8).tobytes(), i)
    st = dec.stats
    assert st["packets"] == 200 and st["emitted"] == 0
    assert st["undecodable"] + st["no_position"] + st["rejected_implausible"] == 200


def test_clip_row_computed_once_per_frame(monkeypatch):
    calls = []
    orig = FrameGeometry._compute_clip_row
    monkeypatch.setattr(FrameGeometry, "_compute_clip_row", lambda self: calls.append(1) or orig(self))
    fr = LocalFrame(32.0, 35.0)
    g = FrameGeometry.from_pose(PoseSample(0, 32.0, 35.0, 85.0, 17.0, -12.0), CAM, W, H, fr)
    tiles = [g.tile_footprint(x, y, x + 640, y + 540) for x in range(0, W, 640) for y in range(0, H, 540)]
    g.footprint()
    assert len(calls) == 1
    assert sum(t is None for t in tiles) > 0                  # top tiles are sky at -12 deg pitch


def test_per_frame_query_cost():
    """Generous bound; the point is that geometry never dominates a 40 ms frame budget."""
    import time
    fr = LocalFrame(32.0, 35.0)
    track = TelemetryTrack([PoseSample(i * 33, 32.0 + i * 1e-6, 35.0, 85.0, 17.0, -30.0) for i in range(3000)])
    prov = TelemetryPoseProvider(track, CAM, fr, extrapolate_ms=200)
    uv = np.random.default_rng(0).random((100, 2)) * [W, H * 0.5] + [0, H * 0.5]
    t0 = time.perf_counter()
    for k in range(50):
        g = prov.geometry(50_000 + k * 33, W, H)
        g.pixel_to_lat_lon(uv)
        for x in range(0, W, 640):
            for y in range(0, H, 540):
                g.tile_footprint(x, y, x + 640, y + 540)
    per_frame_ms = (time.perf_counter() - t0) / 50 * 1000
    print(f"\ngeometry + 100 points + 24 tile footprints: {per_frame_ms:.1f} ms per frame")
    assert per_frame_ms < 15


@pytest.mark.skipif(not os.environ.get("DRONGO_SAMPLES"), reason="set DRONGO_SAMPLES to a folder of real clips")
def test_real_clips_replayed_as_stream():
    """Every clip with a DJI track: packet-by-packet decoding equals the file reader."""
    n = 0
    for f in sorted(Path(os.environ["DRONGO_SAMPLES"]).iterdir()):
        r = DjiEmbeddedReader()
        if f.suffix.lower() not in (".mp4", ".mov") or not r.can_read(f, None):
            continue
        dec = DjiPacketDecoder()
        streamed = [s for t, p in iter_dji_packets(f) for s in dec.feed(p, t)] + dec.flush()
        assert streamed == list(r.read(f, None)), f.name
        n += 1
    assert n
