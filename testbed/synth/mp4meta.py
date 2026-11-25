"""Write DJI-style embedded telemetry: protobuf packets in a 'DJI meta' timed
metadata track of an MP4 file. Used to test DjiEmbeddedReader end to end, and
to add embedded telemetry to rendered video (demo clips)."""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Sequence

from drongo import PoseSample


# ---- protobuf encoding --------------------------------------------------------

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _key(num: int, wire: int) -> bytes:
    return _varint((num << 3) | wire)


def _int(num: int, v: int, zigzag: bool) -> bytes:
    raw = ((v << 1) ^ (v >> 63)) & ((1 << 64) - 1) if zigzag else v & ((1 << 64) - 1)
    return _key(num, 0) + _varint(raw)


def _msg(num: int, body: bytes) -> bytes:
    return _key(num, 2) + _varint(len(body)) + body


def _f64(num: int, v: float) -> bytes:
    return _key(num, 1) + struct.pack("<d", v)


def _f32(num: int, v: float) -> bytes:
    return _key(num, 5) + struct.pack("<f", v)


def encode_packet(p: PoseSample, zigzag: bool = False, alt_in_mm: bool = True) -> bytes:
    """One frame's telemetry in the field layout DjiEmbeddedReader reads."""
    d = lambda deg: int(round(deg * 10))
    yaw = ((p.gimbal_yaw_deg + 180.0) % 360.0) - 180.0
    body_yaw = p.body_yaw_deg if p.body_yaw_deg is not None else p.gimbal_yaw_deg
    body_yaw = ((body_yaw + 180.0) % 360.0) - 180.0
    aircraft = _int(1, 0, zigzag) + _int(2, 0, zigzag) + _int(3, d(body_yaw), zigzag)
    position = (_msg(1, _f64(2, math.radians(p.lat_deg)) + _f64(3, math.radians(p.lon_deg)))
                + _int(2, int(round((p.alt_msl_m or 0.0) * 1000)), False))
    rel_alt = _f32(1, p.alt_agl_m * (1000.0 if alt_in_mm else 1.0))
    frame_info = _msg(3, aircraft) + _msg(4, position) + _msg(5, rel_alt)
    gimbal = _msg(3, _int(1, d(p.gimbal_pitch_deg), zigzag) + _int(2, d(p.gimbal_roll_deg), zigzag)
                  + _int(3, d(yaw), zigzag))
    return _msg(3, _msg(1, _int(1, 0, False)) + _msg(3, frame_info) + _msg(4, gimbal))


# ---- MP4 container ------------------------------------------------------------

def _box(typ: bytes, *parts: bytes) -> bytes:
    body = b"".join(parts)
    return struct.pack(">I4s", 8 + len(body), typ) + body


def _full(typ: bytes, version_flags: int, *parts: bytes) -> bytes:
    return _box(typ, struct.pack(">I", version_flags), *parts)


def _stts_runs(times_ms: Sequence[float], timescale: int) -> tuple[list[list[int]], int]:
    deltas = [int(round((b - a) * timescale / 1000)) for a, b in zip(times_ms[:-1], times_ms[1:])]
    deltas.append(deltas[-1] if deltas else timescale // 30)
    runs: list[list[int]] = []
    for dl in deltas:
        if runs and runs[-1][1] == dl:
            runs[-1][0] += 1
        else:
            runs.append([1, dl])
    return runs, sum(deltas)


def _meta_trak(track_id: int, packets: Sequence[bytes], times_ms: Sequence[float], data_offset: int,
               handler_name: str = "DJI meta", sample_format: bytes = b"djmd", timescale: int = 1000,
               movie_timescale: int = 1000) -> bytes:
    """A timed-metadata 'trak' whose samples sit back to back at `data_offset`."""
    n = len(packets)
    runs, dur = _stts_runs(times_ms, timescale)
    t0 = int(round(times_ms[0] * timescale / 1000)) if n else 0
    stsd = _full(b"stsd", 0, struct.pack(">I", 1), _box(sample_format, bytes(6), struct.pack(">H", 1)))
    stts = _full(b"stts", 0, struct.pack(">I", len(runs)), *(struct.pack(">II", c, dl) for c, dl in runs))
    stsc = _full(b"stsc", 0, struct.pack(">I", 1), struct.pack(">III", 1, n, 1))
    stsz = _full(b"stsz", 0, struct.pack(">II", 0, n), *(struct.pack(">I", len(p)) for p in packets))
    stco = _full(b"stco", 0, struct.pack(">I", 1), struct.pack(">I", data_offset))
    stbl = _box(b"stbl", stsd, stts, stsc, stsz, stco)
    dinf = _box(b"dinf", _full(b"dref", 0, struct.pack(">I", 1), _full(b"url ", 1)))
    minf = _box(b"minf", _full(b"nmhd", 0), dinf, stbl)
    hdlr = _full(b"hdlr", 0, bytes(4), b"meta", bytes(12), handler_name.encode() + b"\0")
    mdhd = _full(b"mdhd", 0, struct.pack(">IIIIHH", 0, 0, timescale, dur, 0x55C4, 0))   # language 'und'
    mdur = int(round((dur + t0) * movie_timescale / timescale))
    tkhd = _full(b"tkhd", 3, struct.pack(">IIII", 0, 0, track_id, 0), struct.pack(">I", mdur), bytes(8),
                 struct.pack(">hhhH", 0, 0, 0, 0),
                 struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000), bytes(8))
    parts = [tkhd]
    if t0 > 0:                                            # first packet after t=0: empty edit, then media
        e_empty = int(round(t0 * movie_timescale / timescale))
        elst = _full(b"elst", 0, struct.pack(">I", 2), struct.pack(">IiI", e_empty, -1, 0x10000),
                     struct.pack(">IiI", mdur - e_empty, 0, 0x10000))
        parts.append(_box(b"edts", elst))
    return _box(b"trak", *parts, _box(b"mdia", mdhd, hdlr, minf))


def write_meta_mp4(path: Path, packets: Sequence[bytes], times_ms: Sequence[float],
                   handler_name: str = "DJI meta", timescale: int = 1000, sample_format: bytes = b"djmd") -> None:
    """MP4 with a single timed-metadata track holding `packets` (no video)."""
    ftyp = _box(b"ftyp", b"isom", struct.pack(">I", 512), b"isomiso2mp41")
    times0 = [t - times_ms[0] for t in times_ms] if times_ms else []
    _, dur = _stts_runs(times0, timescale)

    def moov(off: int) -> bytes:
        mvhd = _full(b"mvhd", 0, struct.pack(">IIII", 0, 0, timescale, dur), bytes(80))
        return _box(b"moov", mvhd, _meta_trak(2, packets, times0, off, handler_name, sample_format, timescale, timescale))

    data_offset = len(ftyp) + len(moov(0)) + 8
    Path(path).write_bytes(ftyp + moov(data_offset) + _box(b"mdat", b"".join(packets)))


# ---- adding a metadata track to an existing video MP4 -------------------------

def _iter_boxes(buf, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack_from(">I4s", buf, pos)
        head = 8
        if size == 1:
            size = struct.unpack_from(">Q", buf, pos + 8)[0]
            head = 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            raise ValueError(f"malformed box {typ!r} at {pos}")
        yield typ, pos, pos + head, pos + size
        pos += size


def _shift_chunk_offsets(moov: bytearray, delta: int) -> None:
    """Add `delta` to every stco / co64 entry inside a moov box (in place)."""
    def walk(a, b):
        for typ, _, p0, p1 in _iter_boxes(moov, a, b):
            if typ in (b"trak", b"mdia", b"minf", b"stbl"):
                walk(p0, p1)
            elif typ == b"stco":
                n = struct.unpack_from(">I", moov, p0 + 4)[0]
                for i in range(n):
                    o = p0 + 8 + 4 * i
                    v = struct.unpack_from(">I", moov, o)[0] + delta
                    if v >= 1 << 32:
                        raise ValueError("chunk offset overflows stco; file too large")
                    struct.pack_into(">I", moov, o, v)
            elif typ == b"co64":
                n = struct.unpack_from(">I", moov, p0 + 4)[0]
                for i in range(n):
                    o = p0 + 8 + 8 * i
                    struct.pack_into(">Q", moov, o, struct.unpack_from(">Q", moov, o)[0] + delta)
    walk(8, len(moov))


def add_meta_track(video: Path, packets: Sequence[bytes], times_ms: Sequence[float], out: Path | None = None,
                   handler_name: str = "DJI meta", sample_format: bytes = b"djmd") -> Path:
    """Copy `video` (an MP4 with a video track, e.g. from ffmpeg) to `out`
    with an added timed-metadata track, as a DJI camera records it. Works
    whether the moov box is before the media data ('faststart') or after it.
    Times are video-absolute milliseconds."""
    video = Path(video)
    out = Path(out) if out is not None else video
    data = video.read_bytes()
    top = list(_iter_boxes(data, 0, len(data)))
    moovs = [b for b in top if b[0] == b"moov"]
    if len(moovs) != 1:
        raise ValueError(f"{video}: expected one moov box, found {len(moovs)}")
    _, m0, mp0, m1 = moovs[0]
    if mp0 - m0 != 8:
        raise ValueError("64-bit moov size not supported")
    moov = bytearray(data[m0:m1])

    # movie timescale and next track id from mvhd
    mv = next(b for b in _iter_boxes(moov, 8, len(moov)) if b[0] == b"mvhd")
    ver = moov[mv[2]]
    if ver == 0:
        ts_off, nid_off = mv[2] + 12, mv[2] + 96
    else:
        ts_off, nid_off = mv[2] + 20, mv[2] + 108
    movie_ts = struct.unpack_from(">I", moov, ts_off)[0]
    track_id = struct.unpack_from(">I", moov, nid_off)[0]
    struct.pack_into(">I", moov, nid_off, track_id + 1)

    payload_box = _box(b"mdat", b"".join(packets))
    media_after_moov = any(t == b"mdat" and s >= m1 for t, s, _, _ in top)
    trak_len = len(_meta_trak(track_id, packets, times_ms, 0, handler_name, sample_format, 1000, movie_ts))
    if not media_after_moov:
        # moov at the end: drop it, append our mdat, then the new moov. Nothing before it moves.
        rest = data[:m0] + data[m1:]
        trak = _meta_trak(track_id, packets, times_ms, len(rest) + 8, handler_name, sample_format, 1000, movie_ts)
        new_moov = moov + trak
        struct.pack_into(">I", new_moov, 0, len(new_moov))
        out.write_bytes(rest + payload_box + bytes(new_moov))
    else:
        # moov first: it grows by the trak, so all media after it moves by that much.
        _shift_chunk_offsets(moov, trak_len)
        trak = _meta_trak(track_id, packets, times_ms, len(data) + trak_len + 8, handler_name, sample_format,
                          1000, movie_ts)
        new_moov = moov + trak
        struct.pack_into(">I", new_moov, 0, len(new_moov))
        out.write_bytes(data[:m0] + bytes(new_moov) + data[m1:] + payload_box)
    return out


def write_embedded(path: Path, samples: Sequence[PoseSample], zigzag: bool = False) -> None:
    """Metadata-only MP4 (no video) carrying `samples` as DJI packets."""
    write_meta_mp4(path, [encode_packet(s, zigzag) for s in samples], [s.t_ms for s in samples])


def embed_telemetry(video: Path, samples: Sequence[PoseSample], out: Path | None = None,
                    zigzag: bool = False) -> Path:
    """Add `samples` to a video MP4 as an embedded DJI telemetry track."""
    samples = sorted(samples, key=lambda s: s.t_ms)
    return add_meta_track(video, [encode_packet(s, zigzag) for s in samples], [s.t_ms for s in samples], out)
