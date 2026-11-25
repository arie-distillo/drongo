"""Reader for DJI telemetry embedded in MP4/MOV files ("DJI meta" timed-metadata
track, protobuf payload per video frame). Pure Python; no ffmpeg or exiftool.

Field layout (protobuf field numbers, '.'-separated path from the root):

    3.3.4.1.2   latitude, radians (double)
    3.3.4.1.3   longitude, radians (double)
    3.3.4.2     altitude above sea level (varint mm, or float m)
    3.3.5.1     altitude above takeoff (float; mm on some models, m on others)
    3.3.3.1-3   aircraft pitch, roll, yaw (deci-degrees)
    3.4.3.1-3   gimbal pitch, roll, yaw (deci-degrees, absolute)

Angles are stored as protobuf int (two's complement) on the aircraft checked
so far (Mavic 3 Enterprise), and as zigzag sint by some third-party writers.
The encoding is chosen per file: the one under which every gimbal sample is a
physically possible attitude. Samples that are impossible under the chosen
encoding are dropped and counted (TEL-11).

Verified against: DJI Mavic 3 Enterprise ("DJI M3E", 2024 firmware); files
written by simulators using the zigzag variant. Other models may use other
paths; they are reported as "no position fields" rather than guessed.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Iterator, Optional

from ._mp4 import Mp4Error, read_samples, read_tracks
from .telemetry import TelemetryTrack
from .types import PoseSample, TelemetrySourceKind

_MAX_DEPTH = 8


def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        if i >= len(b) or s > 63:
            raise ValueError("truncated varint")
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def decode_fields(buf: bytes, prefix: str = "", depth: int = 0) -> Optional[dict[str, object]]:
    """Flatten a protobuf message into {path: value}. Length-delimited fields
    are decoded as sub-messages when they parse cleanly, else kept as bytes.
    Returns None if `buf` is not a valid message. Repeated fields keep the
    first value."""
    out: dict[str, object] = {}
    i = 0
    try:
        while i < len(buf):
            key, i = _varint(buf, i)
            num, wire = key >> 3, key & 7
            if num == 0:
                return None
            path = f"{prefix}.{num}" if prefix else str(num)
            if wire == 0:
                v, i = _varint(buf, i)
            elif wire == 1:
                if i + 8 > len(buf):
                    return None
                v = struct.unpack_from("<d", buf, i)[0]
                i += 8
            elif wire == 5:
                if i + 4 > len(buf):
                    return None
                v = struct.unpack_from("<f", buf, i)[0]
                i += 4
            elif wire == 2:
                n, i = _varint(buf, i)
                if i + n > len(buf):
                    return None
                sub = buf[i:i + n]
                i += n
                inner = decode_fields(sub, path, depth + 1) if (n and depth < _MAX_DEPTH) else None
                if inner:
                    for k, val in inner.items():
                        out.setdefault(k, val)
                    continue
                v = sub
            else:
                return None
            out.setdefault(path, v)
    except ValueError:
        return None
    return out


def _as_int(v: int) -> int:          # protobuf int32/int64: two's complement in 64 bits
    return v - (1 << 64) if v >= (1 << 63) else v


def _as_sint(v: int) -> int:         # protobuf sint32/sint64: zigzag
    return (v >> 1) ^ -(v & 1)


def _plausible(pitch: float, roll: float, yaw: float) -> bool:
    return -120.0 <= pitch <= 45.0 and -95.0 <= roll <= 95.0 and -360.0 <= yaw <= 360.0


def _alt_m(v) -> Optional[float]:
    if v is None or isinstance(v, (bytes, bytearray)):
        return None
    v = float(v)
    return v / 1000.0 if abs(v) > 5000.0 else v      # mm on some models (no drone flies 5 km up)


def _gimbal(d: dict, conv) -> Optional[tuple[float, float, float]]:
    vals = [d.get(f"3.4.3.{k}", 0) for k in (1, 2, 3)]
    if not all(isinstance(v, int) for v in vals):
        return None
    return tuple(conv(v) / 10.0 for v in vals)  # type: ignore[return-value]


def _has_position(d: dict) -> bool:
    return "3.3.4.1.2" in d and "3.3.4.1.3" in d


def _encoding_scores(decoded: list[dict]) -> dict[str, int]:
    """How many packets give a physically possible gimbal attitude under each encoding."""
    out = {}
    for name, conv in (("int", _as_int), ("zigzag", _as_sint)):
        out[name] = sum(1 for d in decoded if (g := _gimbal(d, conv)) and _plausible(*g))
    return out


def packet_to_sample(t_ms: float, d: dict, encoding: str, kind=TelemetrySourceKind.DJI_PROTOBUF) -> Optional[PoseSample]:
    """One decoded packet -> PoseSample, or None if it has no position or an
    impossible attitude (the sanity gate)."""
    if not _has_position(d):
        return None
    conv = _as_int if encoding == "int" else _as_sint
    lat, lon = math.degrees(d["3.3.4.1.2"]), math.degrees(d["3.3.4.1.3"])
    g = _gimbal(d, conv)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0) or not g or not _plausible(*g):
        return None
    pitch, roll, yaw = g
    body = d.get("3.3.3.3")
    alt = _alt_m(d.get("3.3.5.1"))
    return PoseSample(
        t_ms=int(round(t_ms)), lat_deg=lat, lon_deg=lon,
        alt_agl_m=alt if alt is not None else float("nan"),
        gimbal_yaw_deg=yaw % 360.0, gimbal_pitch_deg=pitch, gimbal_roll_deg=roll,
        alt_msl_m=_alt_m(d.get("3.3.4.2")),
        body_yaw_deg=(conv(body) / 10.0) % 360.0 if isinstance(body, int) else None, source=kind)


class DjiEmbeddedReader:
    """TelemetryReader for DJI telemetry embedded in a recorded video file."""

    kind = TelemetrySourceKind.DJI_PROTOBUF
    HANDLER_NAMES = ("dji meta",)
    SAMPLE_FORMATS = ("djmd",)

    def __init__(self) -> None:
        self.last_stats: dict = {}

    def _track(self, video: Path):
        for t in read_tracks(video):
            if t.handler_name.lower() in self.HANDLER_NAMES or t.sample_format in self.SAMPLE_FORMATS:
                return t
        return None

    def can_read(self, video, sidecar) -> bool:
        if video is None or not Path(video).is_file():
            return False
        try:
            return self._track(Path(video)) is not None
        except (Mp4Error, OSError, struct.error):
            return False

    def explain(self, video, sidecar) -> str:
        if video is None or not Path(video).is_file():
            return "no video file"
        try:
            names = [t.handler_name or t.sample_format for t in read_tracks(Path(video))]
        except (Mp4Error, OSError, struct.error) as e:
            return f"not a readable MP4/MOV container ({e})"
        return f"no DJI metadata track (tracks: {', '.join(names) or 'none'})"

    def read(self, video, sidecar) -> TelemetryTrack:
        decoded = [(t, decode_fields(p) or {}) for t, p in iter_dji_packets(video)]
        return self.from_decoded(decoded)

    def from_decoded(self, decoded: list[tuple[float, dict]]) -> TelemetryTrack:
        """Build a track from (time_ms, decoded fields) pairs. The encoding is
        chosen over the whole file; also used by tests."""
        raw = [(t, d) for t, d in decoded if _has_position(d)]
        scores = _encoding_scores([d for _, d in raw])
        enc = max(scores, key=scores.get) if raw else "int"
        out = [s for t, d in raw if (s := packet_to_sample(t, d, enc, self.kind)) is not None]
        self.last_stats = {"packets": len(decoded), "with_position": len(raw), "angle_encoding": enc,
                           "plausible": scores, "rejected_implausible": len(raw) - len(out)}
        return TelemetryTrack(out)


def iter_dji_packets(video) -> Iterator[tuple[float, bytes]]:
    """(time_ms, raw packet) for each DJI telemetry packet in a recorded file,
    in order. Feeding these to DjiPacketDecoder replays a recording as a live
    stream."""
    video = Path(video)
    reader = DjiEmbeddedReader()
    track = reader._track(video)
    if track is None:
        raise ValueError(reader.explain(video, None))
    yield from read_samples(video, track)


class DjiPacketDecoder:
    """Streaming decoder: one DJI telemetry packet in, a PoseSample out.

    The angle encoding cannot be known from one packet when angles are small
    and positive (both encodings then give plausible values), so it is chosen
    from the first packets: as soon as only one encoding gives a possible
    attitude, or after `decide_after` packets by majority. Packets received
    before the decision are buffered and released with the decision, so the
    first `feed` calls may return an empty list and a later one several samples.
    Pass `encoding="int"` or `"zigzag"` to skip detection.

        dec = DjiPacketDecoder()
        for t_ms, packet in source:                 # live feed, or iter_dji_packets(path)
            for sample in dec.feed(packet, t_ms):
                track.append(sample)

    Times must be on the same clock as the video frames you query.
    """

    def __init__(self, encoding: str = "auto", decide_after: int = 10) -> None:
        if encoding not in ("auto", "int", "zigzag"):
            raise ValueError("encoding must be 'auto', 'int' or 'zigzag'")
        self.encoding: Optional[str] = None if encoding == "auto" else encoding
        self.decide_after = decide_after
        self._pending: list[tuple[float, dict]] = []
        self.stats = {"packets": 0, "undecodable": 0, "no_position": 0, "rejected_implausible": 0, "emitted": 0}

    def feed(self, packet: bytes, t_ms: float) -> list[PoseSample]:
        self.stats["packets"] += 1
        d = decode_fields(packet)
        if d is None:
            self.stats["undecodable"] += 1
            return []
        if not _has_position(d):
            self.stats["no_position"] += 1
            return []
        if self.encoding is None:
            self._pending.append((t_ms, d))
            scores = _encoding_scores([x for _, x in self._pending])
            n = len(self._pending)
            unique = [k for k, v in scores.items() if v == n] if min(scores.values()) < n else []
            if len(unique) == 1 or n >= self.decide_after:
                self.encoding = unique[0] if len(unique) == 1 else max(scores, key=scores.get)
                pending, self._pending = self._pending, []
                return self._emit(pending)
            return []
        return self._emit([(t_ms, d)])

    def flush(self) -> list[PoseSample]:
        """Decide with whatever is buffered (end of a short stream)."""
        if not self._pending:
            return []
        scores = _encoding_scores([x for _, x in self._pending])
        self.encoding = self.encoding or max(scores, key=scores.get)
        pending, self._pending = self._pending, []
        return self._emit(pending)

    def _emit(self, items) -> list[PoseSample]:
        out = []
        for t, d in items:
            s = packet_to_sample(t, d, self.encoding or "int")
            if s is None:
                self.stats["rejected_implausible"] += 1
            else:
                out.append(s)
        self.stats["emitted"] += len(out)
        return out
