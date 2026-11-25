"""Minimal ISO-BMFF / QuickTime reader: list tracks and read the samples of a
timed-metadata track. Pure Python, no external tools. Reads only box headers
and the sample tables, then the samples themselves; never the video data.

Treats the file as untrusted: box sizes are bounds-checked, nesting depth and
sample counts are capped.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta", b"dinf"}
MAX_SAMPLES = 2_000_000
MAX_SAMPLE_BYTES = 1 << 20


class Mp4Error(ValueError):
    pass


@dataclass
class Track:
    track_id: int = 0
    handler_type: str = ""          # 'vide', 'soun', 'meta', 'text', ...
    handler_name: str = ""          # e.g. 'DJI meta'
    sample_format: str = ""         # sample entry fourcc, e.g. 'djmd', 'avc1'
    timescale: int = 0
    sizes: list[int] = field(default_factory=list)
    chunk_offsets: list[int] = field(default_factory=list)
    stsc: list[tuple[int, int]] = field(default_factory=list)   # (first_chunk, samples_per_chunk)
    stts: list[tuple[int, int]] = field(default_factory=list)   # (count, delta)

    def sample_offsets(self) -> list[int]:
        out: list[int] = []
        n = len(self.sizes)
        if not self.stsc or not self.chunk_offsets:
            return out
        runs = self.stsc + [(len(self.chunk_offsets) + 1, 0)]
        s = 0
        for (first, per), (nxt, _) in zip(runs[:-1], runs[1:]):
            for chunk in range(first, nxt):
                if chunk - 1 >= len(self.chunk_offsets):
                    break
                off = self.chunk_offsets[chunk - 1]
                for _ in range(per):
                    if s >= n:
                        return out
                    out.append(off)
                    off += self.sizes[s]
                    s += 1
        return out

    def sample_times_ms(self) -> list[float]:
        t, out = 0, []
        for count, delta in self.stts:
            for _ in range(count):
                out.append(1000.0 * t / self.timescale if self.timescale else 0.0)
                t += delta
        return out[: len(self.sizes)]


def _boxes(f: BinaryIO, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """Yield (type, payload_start, payload_end) for boxes in [start, end)."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            return
        size, typ = struct.unpack(">I4s", hdr)
        head = 8
        if size == 1:
            ext = f.read(8)
            if len(ext) < 8:
                return
            size = struct.unpack(">Q", ext)[0]
            head = 16
        elif size == 0:
            size = end - pos
        if size < head or pos + size > end:
            raise Mp4Error(f"bad box size {size} for {typ!r} at {pos}")
        yield typ, pos + head, pos + size
        pos += size


def _read(f: BinaryIO, a: int, b: int) -> bytes:
    f.seek(a)
    return f.read(b - a)


def _parse_stbl(f, a, b, tr: Track) -> None:
    for typ, s, e in _boxes(f, a, b):
        d = _read(f, s, e)
        if typ == b"stsd" and len(d) >= 16:
            tr.sample_format = d[12:16].decode("latin-1").strip("\0")
        elif typ == b"stsz" and len(d) >= 12:
            fixed, n = struct.unpack(">II", d[4:12])
            if n > MAX_SAMPLES:
                raise Mp4Error("too many samples")
            tr.sizes = [fixed] * n if fixed else list(struct.unpack(f">{n}I", d[12:12 + 4 * n]))
        elif typ == b"stco" and len(d) >= 8:
            n = struct.unpack(">I", d[4:8])[0]
            tr.chunk_offsets = list(struct.unpack(f">{n}I", d[8:8 + 4 * n]))
        elif typ == b"co64" and len(d) >= 8:
            n = struct.unpack(">I", d[4:8])[0]
            tr.chunk_offsets = list(struct.unpack(f">{n}Q", d[8:8 + 8 * n]))
        elif typ == b"stsc" and len(d) >= 8:
            n = struct.unpack(">I", d[4:8])[0]
            tr.stsc = [struct.unpack(">II", d[8 + 12 * i: 16 + 12 * i]) for i in range(n)]
        elif typ == b"stts" and len(d) >= 8:
            n = struct.unpack(">I", d[4:8])[0]
            tr.stts = [struct.unpack(">II", d[8 + 8 * i: 16 + 8 * i]) for i in range(n)]


def _walk_trak(f, a, b, tr: Track, depth=0) -> None:
    if depth > 8:
        return
    for typ, s, e in _boxes(f, a, b):
        if typ == b"tkhd":
            d = _read(f, s, min(e, s + 24))
            ver = d[0] if d else 0
            off = 12 if ver == 0 else 20
            if len(d) >= off + 4:
                tr.track_id = struct.unpack(">I", d[off:off + 4])[0]
        elif typ == b"mdhd":
            d = _read(f, s, min(e, s + 32))
            ver = d[0] if d else 0
            off = 12 if ver == 0 else 20
            if len(d) >= off + 4:
                tr.timescale = struct.unpack(">I", d[off:off + 4])[0]
        elif typ == b"hdlr" and not tr.handler_type:      # mdia's hdlr; QuickTime adds a data hdlr later
            d = _read(f, s, e)
            if len(d) >= 24:
                tr.handler_type = d[8:12].decode("latin-1")
                name = d[24:]
                if name[:1] and name[0] == len(name) - 1:        # QuickTime pascal string
                    name = name[1:]
                tr.handler_name = name.split(b"\0")[0].decode("utf-8", "replace").strip()
        elif typ == b"stbl":
            _parse_stbl(f, s, e, tr)
        elif typ in _CONTAINERS:
            _walk_trak(f, s, e, tr, depth + 1)


def read_tracks(path: Path) -> list[Track]:
    tracks: list[Track] = []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        for typ, s, e in _boxes(f, 0, size):
            if typ == b"moov":
                for t2, s2, e2 in _boxes(f, s, e):
                    if t2 == b"trak":
                        tr = Track()
                        _walk_trak(f, s2, e2, tr)
                        tracks.append(tr)
    return tracks


def read_samples(path: Path, track: Track) -> Iterator[tuple[float, bytes]]:
    """(time_ms, payload) for every sample of the track."""
    offs, times = track.sample_offsets(), track.sample_times_ms()
    with open(path, "rb") as f:
        for i, (off, n) in enumerate(zip(offs, track.sizes)):
            if n > MAX_SAMPLE_BYTES:
                raise Mp4Error(f"sample {i} too large ({n} bytes)")
            f.seek(off)
            yield (times[i] if i < len(times) else float("nan")), f.read(n)
