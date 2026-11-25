"""Telemetry ingestion and time-indexed pose lookup."""
from __future__ import annotations

import bisect
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional, Protocol, Sequence

from .types import PoseSample, TelemetrySourceKind, lerp, lerp_angle_deg, wrap_deg


class TelemetryTrack:
    """Time-ordered PoseSamples with interpolation.

    `at(t)` looks up the pose at video time t + time_offset_ms. Position is
    interpolated linearly (adequate for the sub-second gaps between samples);
    yaw and roll on the circle, so 359 -> 1 passes through 0.

    Past the newest sample (live use: a frame newer than the latest telemetry)
    the pose is extrapolated, within `extrapolate_ms` passed to `at`:
      "linear" (default): continue the motion measured over the last
          `baseline_ms` of the track (position, altitude, angles; pitch
          clamped to [-90, 30] deg). A long baseline smooths over receivers
          that repeat the same GPS fix for several samples.
      "hold": repeat the newest sample.
    Before the first sample the first sample is held.
    """

    EXTRAPOLATION_MODES = ("linear", "hold")

    def __init__(self, samples: Iterable[PoseSample] = (), time_offset_ms: int = 0, *,
                 extrapolation: str = "linear", baseline_ms: int = 300) -> None:
        if extrapolation not in self.EXTRAPOLATION_MODES:
            raise ValueError(f"extrapolation must be one of {self.EXTRAPOLATION_MODES}")
        self._s: list[PoseSample] = sorted(samples, key=lambda s: s.t_ms)
        self._t: list[int] = [s.t_ms for s in self._s]
        self._offset = int(time_offset_ms)
        self.extrapolation = extrapolation
        self.baseline_ms = int(baseline_ms)

    # ---- construction / live ingest
    def append(self, sample: PoseSample) -> None:
        if self._t and sample.t_ms < self._t[-1]:
            raise ValueError(f"out-of-order sample t={sample.t_ms} < {self._t[-1]}")
        self._s.append(sample)
        self._t.append(sample.t_ms)

    def with_time_offset(self, offset_ms: int) -> "TelemetryTrack":
        return TelemetryTrack(self._s, offset_ms, extrapolation=self.extrapolation, baseline_ms=self.baseline_ms)

    # ---- properties
    @property
    def time_offset_ms(self) -> int:
        return self._offset

    @property
    def source(self) -> TelemetrySourceKind:
        return self._s[0].source if self._s else TelemetrySourceKind.MANUAL

    @property
    def span_ms(self) -> tuple[int, int]:
        if not self._s:
            raise ValueError("empty track")
        return self._t[0], self._t[-1]

    def __len__(self) -> int:
        return len(self._s)

    def __iter__(self) -> Iterator[PoseSample]:
        return iter(self._s)

    def __getitem__(self, i: int) -> PoseSample:
        return self._s[i]

    # ---- lookup
    def at(self, t_ms: float, *, extrapolate_ms: int = 0) -> Optional[PoseSample]:
        if not self._s:
            return None
        t = t_ms + self._offset
        t0, t1 = self._t[0], self._t[-1]
        if t < t0:
            return self._s[0].with_(t_ms=int(round(t_ms))) if t0 - t <= extrapolate_ms else None
        if t > t1:
            if t - t1 > extrapolate_ms:
                return None
            return self._extrapolate(t, int(round(t_ms)))
        i = bisect.bisect_right(self._t, t)
        if i >= len(self._s):
            return self._s[-1].with_(t_ms=int(round(t_ms)))
        a, b = self._s[i - 1], self._s[i]
        dt = b.t_ms - a.t_ms
        u = 0.0 if dt == 0 else (t - a.t_ms) / dt
        return _interp(a, b, u, int(round(t_ms)))


    def _extrapolate(self, t: float, t_ms: int) -> PoseSample:
        last = self._s[-1]
        if self.extrapolation == "hold" or len(self._s) < 2:
            return last.with_(t_ms=t_ms)
        j = bisect.bisect_right(self._t, last.t_ms - self.baseline_ms) - 1
        a = self._s[max(j, 0)]
        if a.t_ms >= last.t_ms:                     # all samples share the newest timestamp
            return last.with_(t_ms=t_ms)
        u = (t - a.t_ms) / (last.t_ms - a.t_ms)
        p = _interp(a, last, u, t_ms)
        return p.with_(gimbal_pitch_deg=min(30.0, max(-90.0, p.gimbal_pitch_deg)))


def _interp(a: PoseSample, b: PoseSample, u: float, t_ms: int) -> PoseSample:
    def opt(x, y, f):
        return None if x is None or y is None else f(x, y, u)

    vel = None
    if a.velocity_enu_ms is not None and b.velocity_enu_ms is not None:
        vel = tuple(lerp(x, y, u) for x, y in zip(a.velocity_enu_ms, b.velocity_enu_ms))
    return PoseSample(
        t_ms=t_ms,
        lat_deg=lerp(a.lat_deg, b.lat_deg, u),
        lon_deg=lerp(a.lon_deg, b.lon_deg, u),
        alt_agl_m=lerp(a.alt_agl_m, b.alt_agl_m, u),
        gimbal_yaw_deg=lerp_angle_deg(a.gimbal_yaw_deg, b.gimbal_yaw_deg, u),
        gimbal_pitch_deg=lerp(a.gimbal_pitch_deg, b.gimbal_pitch_deg, u),
        gimbal_roll_deg=wrap_deg(a.gimbal_roll_deg + wrap_deg(b.gimbal_roll_deg - a.gimbal_roll_deg) * u),
        alt_msl_m=opt(a.alt_msl_m, b.alt_msl_m, lerp),
        body_yaw_deg=opt(a.body_yaw_deg, b.body_yaw_deg, lerp_angle_deg),
        velocity_enu_ms=vel,
        source=a.source,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Readers
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryReader(Protocol):
    kind: TelemetrySourceKind

    def can_read(self, video: Optional[Path], sidecar: Optional[Path]) -> bool: ...
    def read(self, video: Optional[Path], sidecar: Optional[Path]) -> TelemetryTrack: ...


_TC_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_GPS_FPV_RE = re.compile(r"GPS\s*\(([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)", re.I)
_LAT_RE = re.compile(r"\[latitude\s*:\s*([+-]?\d+\.?\d*)", re.I)
_LON_RE = re.compile(r"\[longitude\s*:\s*([+-]?\d+\.?\d*)", re.I)
_ALT_RE = re.compile(r"rel_alt\s*:\s*([+-]?\d+\.?\d*)|\[altitude\s*:\s*([+-]?\d+\.?\d*)", re.I)
_ABS_ALT_RE = re.compile(r"abs_alt\s*:\s*([+-]?\d+\.?\d*)", re.I)
_H_AGL_RE = re.compile(r"(?:^|[,\s])H\s+([+-]?\d+\.?\d*)m")
# Gimbal fields appear both as "[gb_yaw : 1] [gb_pitch : 2]" and as
# "[gb_yaw: 1 gb_pitch: 2 gb_roll: 3]" depending on model/firmware.
_PITCH_RE = re.compile(r"gb_pitch\s*:\s*([+-]?\d+\.?\d*)", re.I)
_YAW_RE = re.compile(r"gb_yaw\s*:\s*([+-]?\d+\.?\d*)", re.I)
_ROLL_RE = re.compile(r"gb_roll\s*:\s*([+-]?\d+\.?\d*)", re.I)


def _tc_ms(h, m, s, ms) -> int:
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def parse_srt_block(text: str, t_ms: int, default_pitch_deg: float = -90.0) -> Optional[PoseSample]:
    """Parse the body of one DJI SRT block (formats A, B, C)."""
    text = re.sub(r"<[^>]+>", "", text)
    m = _GPS_FPV_RE.search(text)
    alt_msl = None
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        h = _H_AGL_RE.search(text)
        alt = float(h.group(1)) if h else 0.0
        kind = TelemetrySourceKind.DJI_SRT_C
    else:
        la, lo = _LAT_RE.search(text), _LON_RE.search(text)
        if not (la and lo):
            return None
        lat, lon = float(la.group(1)), float(lo.group(1))
        a = _ALT_RE.search(text)
        alt = float(a.group(1) or a.group(2)) if a else 0.0
        ab = _ABS_ALT_RE.search(text)
        alt_msl = float(ab.group(1)) if ab else None
        kind = TelemetrySourceKind.DJI_SRT_B if _YAW_RE.search(text) else TelemetrySourceKind.DJI_SRT_A
    p, y, r = _PITCH_RE.search(text), _YAW_RE.search(text), _ROLL_RE.search(text)
    return PoseSample(
        t_ms=t_ms, lat_deg=lat, lon_deg=lon, alt_agl_m=alt,
        gimbal_yaw_deg=float(y.group(1)) % 360.0 if y else 0.0,
        gimbal_pitch_deg=float(p.group(1)) if p else default_pitch_deg,
        gimbal_roll_deg=float(r.group(1)) if r else 0.0,
        alt_msl_m=alt_msl, source=kind,
    )


class DjiSrtReader:
    kind = TelemetrySourceKind.DJI_SRT_B

    def __init__(self, default_pitch_deg: float = -90.0) -> None:
        self.default_pitch_deg = default_pitch_deg

    @staticmethod
    def candidates(video: Optional[Path]) -> list[Path]:
        """Same-stem .srt files next to the video, any letter case."""
        if video is None:
            return []
        video = Path(video)
        out = [video.with_suffix(".SRT"), video.with_suffix(".srt")]
        if video.parent.is_dir():
            out += [p for p in video.parent.iterdir()
                    if p.suffix.lower() == ".srt" and p.stem.lower() == video.stem.lower()]
        return out

    def _path(self, video, sidecar) -> Optional[Path]:
        if sidecar:
            return Path(sidecar) if Path(sidecar).exists() else None
        for c in self.candidates(Path(video) if video else None):
            if c.exists():
                return c
        return None

    def explain(self, video, sidecar) -> str:
        if sidecar and not Path(sidecar).exists():
            return f"--srt file does not exist: {sidecar}"
        tried = ", ".join(str(c) for c in self.candidates(Path(video) if video else None)[:2])
        return f"no SRT sidecar found (looked for {tried})"

    def can_read(self, video, sidecar) -> bool:
        return self._path(video, sidecar) is not None

    def read(self, video, sidecar) -> TelemetryTrack:
        return self.read_text(self._path(video, sidecar).read_text(encoding="utf-8", errors="ignore"))

    def read_text(self, text: str) -> TelemetryTrack:
        out = []
        self.last_stats = {"blocks": 0, "with_timecode": 0, "with_gps": 0, "first_body": None}
        text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        for block in re.split(r"\n\s*\n", text.strip()):
            lines = block.strip().splitlines()
            if len(lines) < 3:
                continue
            self.last_stats["blocks"] += 1
            # timecode is normally line 2; tolerate a missing index line
            tc_line = next((i for i, ln in enumerate(lines[:3]) if _TC_RE.search(ln) and "-->" in ln), None)
            if tc_line is None:
                continue
            self.last_stats["with_timecode"] += 1
            body = "\n".join(lines[tc_line + 1:])
            if self.last_stats["first_body"] is None:
                self.last_stats["first_body"] = body[:400]
            s = parse_srt_block(body, _tc_ms(*_TC_RE.search(lines[tc_line]).groups()[:4]), self.default_pitch_deg)
            if s:
                self.last_stats["with_gps"] += 1
                out.append(s)
        return TelemetryTrack(out)


def open_telemetry(video: Optional[Path], sidecar: Optional[Path] = None, *,
                   readers: Optional[Sequence[TelemetryReader]] = None,
                   diagnostics: Optional[list[str]] = None) -> Optional[TelemetryTrack]:
    """Tiered detection: readers are tried in order; first non-empty track wins.
    Default: SRT sidecar, then DJI telemetry embedded in the video file.

    A failing reader does not stop the next tier, but every outcome is
    appended to `diagnostics` (if given) so a None result can be explained."""
    def note(r, msg):
        if diagnostics is not None:
            k = getattr(r, "kind", None)
            diagnostics.append(f"{getattr(k, 'value', None) or type(r).__name__:>13}: {msg}")

    if readers is None:
        from .dji_embedded import DjiEmbeddedReader
        readers = [DjiSrtReader(), DjiEmbeddedReader()]
    for r in readers:
        try:
            if not r.can_read(video, sidecar):
                note(r, r.explain(video, sidecar) if hasattr(r, "explain") else "not applicable")
                continue
            tr = r.read(video, sidecar)
            if len(tr):
                note(r, f"OK, {len(tr)} samples")
                return tr
            stats = getattr(r, "last_stats", None)
            note(r, f"read but 0 samples{f' {stats}' if stats else ''}")
        except Exception as e:   # a broken source must not block the next tier
            note(r, f"error: {type(e).__name__}: {e}")
    return None


def write_dji_srt(samples: Sequence[PoseSample], frame_ms: int) -> str:
    """Serialise samples as DJI format-B SRT (used by the synthetic testbed)."""
    def tc(ms: int) -> str:
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, s in enumerate(samples, 1):
        blocks.append(
            f"{i}\n{tc(s.t_ms)} --> {tc(s.t_ms + frame_ms)}\n"
            f"<font size=\"28\">FrameCnt: {i}, DiffTime: {frame_ms}ms\n"
            f"[latitude: {s.lat_deg:.7f}] [longitude: {s.lon_deg:.7f}] "
            f"[rel_alt: {s.alt_agl_m:.3f} abs_alt: {(s.alt_msl_m or 0):.3f}] "
            f"[gb_yaw: {s.gimbal_yaw_deg:.1f} gb_pitch: {s.gimbal_pitch_deg:.1f} gb_roll: {s.gimbal_roll_deg:.1f}] </font>"
        )
    return "\n\n".join(blocks) + "\n"
