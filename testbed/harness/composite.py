"""Side-by-side composite video: the input clip, a live map of what drongo
computes for every frame, and telemetry - all synchronised to the video.

    python -m testbed.harness.composite --video clip.mp4 --camera mini3pro
    python -m testbed.harness.composite --video DJI_0042.MP4 --srt DJI_0042.SRT --camera mavic3_wide --grid
    python -m testbed.harness.composite --video clip.mp4 --start 10 --end 25 --out promo.mp4

Layout (default 1920x1080):
    +------------------------------+-------------+
    | input video                  | map         |
    | (optional 10 m ground grid)  |             |
    +------------------------------+-------------+
    | telemetry plots + cursor     | readout     |
    +------------------------------+-------------+

The map is north-up in metres around the first telemetry position (no
basemap, so nothing about the place is implied). It shows the whole flight
path, the part flown so far, the drone and its heading, view lines to the
footprint corners, and the current frame draped onto the ground inside its
footprint, over a mosaic of earlier draped frames. Everything comes from the
telemetry and the camera model through drongo, so the mosaic lining up is a
visible check of the georeferencing.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from drongo import CameraModel, TelemetryPoseProvider, open_telemetry

from ..synth.encode import VideoWriter
from .run_video import draw_grid

BG = (24, 24, 28)
PANEL = (34, 35, 40)
TEXT = (235, 235, 235)
DIM = (150, 150, 155)
ACCENT = (40, 170, 250)          # BGR orange
TRACK = (250, 180, 60)           # BGR light blue
FOOT = (60, 220, 255)            # BGR yellow
# name, unit, colour, minimum displayed span (so sensor noise is not blown up to full height)
LANES = [("Altitude AGL", "m", (120, 200, 255), 10.0), ("Ground speed", "m/s", (120, 255, 160), 4.0),
         ("Gimbal pitch", "deg", (255, 170, 120), 20.0), ("Gimbal yaw", "deg", (200, 140, 255), 40.0)]


# ---------------------------------------------------------------------------- text

def _font_path() -> Optional[str]:
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        return str(p) if p.is_file() else None
    except Exception:
        return None


class Text:
    """Anti-aliased TrueType text (falls back to OpenCV's Hershey font)."""

    def __init__(self):
        self.path = _font_path()
        self._cache = {}

    def _font(self, size):
        if size not in self._cache:
            from PIL import ImageFont
            self._cache[size] = ImageFont.truetype(self.path, size)
        return self._cache[size]

    def draw(self, img, lines, size=18, colour=TEXT):
        """lines: list of (x, y, text) or (x, y, text, colour); y is the top of the text."""
        if not self.path:
            for ln in lines:
                x, y, s = ln[:3]
                cv2.putText(img, s, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, size / 30, ln[3] if len(ln) > 3 else colour,
                            1, cv2.LINE_AA)
            return img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(img[..., ::-1])
        d = ImageDraw.Draw(pil)
        f = self._font(size)
        for ln in lines:
            x, y, s = ln[:3]
            c = ln[3] if len(ln) > 3 else colour
            d.text((x, y), s, font=f, fill=(c[2], c[1], c[0]))
        img[:] = np.asarray(pil)[..., ::-1]
        return img


# ---------------------------------------------------------------------------- data

class Clip:
    def __init__(self, video: Path, srt: Optional[Path], camera: CameraModel, offset_ms: float = 0.0):
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {video}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        diag: list[str] = []
        self.track = open_telemetry(video, srt, diagnostics=diag)
        if self.track is None:
            raise SystemExit("no telemetry found:\n  " + "\n  ".join(diag))
        self.camera = camera
        self.offset_ms = offset_ms
        self.prov = TelemetryPoseProvider(self.track, camera, extrapolate_ms=100)
        self.frame = self.prov.frame
        s = list(self.track)
        self.t = np.array([p.t_ms for p in s], float)
        self.en = self.frame.to_enu([[p.lat_deg, p.lon_deg] for p in s])
        self.alt = np.array([p.alt_agl_m for p in s])
        self.pitch = np.array([p.gimbal_pitch_deg for p in s])
        yaw = np.degrees(np.unwrap(np.radians([p.gimbal_yaw_deg for p in s])))
        self.yaw = yaw - 360.0 * math.floor(np.mean(yaw) / 360.0)      # continuous, centred in 0..360
        dt = np.maximum(np.gradient(self.t) / 1000.0, 1e-3)
        sp = np.linalg.norm(np.gradient(self.en, axis=0), axis=1) / dt
        k = max(1, int(round(1.0 * len(sp) / max((self.t[-1] - self.t[0]) / 1000.0, 1e-3))))   # ~1 s smoothing
        self.speed = np.convolve(sp, np.ones(k) / k, mode="same")
        src = getattr(self.track.source, "name", str(self.track.source))
        self.source = {"DJI_SRT_A": "DJI SRT", "DJI_SRT_B": "DJI SRT", "DJI_SRT_C": "DJI SRT",
                       "DJI_EMBEDDED": "DJI embedded", "DJI_PROTOBUF": "DJI embedded"}.get(src, src.replace("_", " "))

    def t_video(self, i: int) -> float:
        return i * 1000.0 / self.fps

    def t_tel(self, i: int) -> float:
        return self.t_video(i) + self.offset_ms

    def geometry(self, i: int, w: int, h: int):
        return self.prov.geometry(self.t_tel(i), w, h)

    def pose(self, i: int):
        return self.track.at(self.t_tel(i), extrapolate_ms=100)


def clipped_footprint(g, max_range_m: float) -> Optional[np.ndarray]:
    """Visible-ground polygon (EN metres), cut off at `max_range_m` from the drone."""
    fp = g.footprint()
    if fp is None:
        return None
    c = np.asarray(fp.corners_en, float)
    o = g.camera_enu_m[:2]
    out = []
    for k in range(4):                                              # densify the edges, then clamp radially
        a, b = c[k], c[(k + 1) % 4]
        for t in np.linspace(0, 1, 16, endpoint=False):
            p = a + (b - a) * t
            d = np.linalg.norm(p - o)
            out.append(o + (p - o) * min(1.0, max_range_m / max(d, 1e-9)))
    return np.array(out)


# ---------------------------------------------------------------------------- panels

class MapPanel:
    def __init__(self, clip: Clip, size, frames, max_range_m: float, proc_w: int):
        self.W, self.H = size
        self.R = max_range_m
        self.proc = (proc_w, int(round(clip.h * proc_w / clip.w)))
        pts = [clip.en]
        for i in frames[:: max(1, len(frames) // 60)]:
            g = clip.geometry(i, *self.proc)
            if g is not None:
                fp = clipped_footprint(g, self.R)
                if fp is not None:
                    pts.append(fp)
        allp = np.vstack(pts)
        lo, hi = allp.min(0), allp.max(0)
        pad = 0.06 * max(hi - lo) + 5
        lo, hi = lo - pad, hi + pad
        self.s = min((self.W - 20) / (hi[0] - lo[0]), (self.H - 70) / (hi[1] - lo[1]))     # px per metre
        cx, cy = (lo + hi) / 2
        self.A = np.array([[self.s, 0, self.W / 2 - self.s * cx],
                           [0, -self.s, (self.H + 40) / 2 + self.s * cy], [0, 0, 1.0]])
        self.base = np.full((self.H, self.W, 3), PANEL, np.uint8)
        self._grid()
        self.mosaic = self.base.copy()
        self.track_px = self.px(clip.en)

    def px(self, en):
        en = np.atleast_2d(en)
        return (np.c_[en, np.ones(len(en))] @ self.A.T)[:, :2]

    def _grid(self):
        span = self.W / self.s
        step = next(s for s in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000) if span / s <= 8)
        inv = np.linalg.inv(self.A)
        e0, n1 = (inv @ [0, 0, 1])[:2]
        e1, n0 = (inv @ [self.W, self.H, 1])[:2]
        for e in np.arange(math.floor(e0 / step) * step, e1 + step, step):
            x = int(round(self.px([e, 0])[0, 0]))
            cv2.line(self.base, (x, 40), (x, self.H), (48, 49, 55), 1)
        for n in np.arange(math.floor(n0 / step) * step, n1 + step, step):
            y = int(round(self.px([0, n])[0, 1]))
            cv2.line(self.base, (0, max(y, 40)), (self.W, max(y, 40)), (48, 49, 55), 1)
        # scale bar and north arrow
        L = step * self.s
        x0, y0 = 20, self.H - 22
        cv2.line(self.base, (x0, y0), (int(x0 + L), y0), TEXT, 2, cv2.LINE_AA)
        for x in (x0, int(x0 + L)):
            cv2.line(self.base, (x, y0 - 5), (x, y0 + 5), TEXT, 2, cv2.LINE_AA)
        self.scale_label = (x0, y0 - 26, f"{step:g} m")
        nx, ny = self.W - 30, 62
        cv2.arrowedLine(self.base, (nx, ny + 28), (nx, ny), TEXT, 2, cv2.LINE_AA, tipLength=0.35)
        self.north_label = (nx - 6, ny + 30, "N")

    def drape(self, target, frame_bgr, g, fp_en, alpha=1.0):
        M = self.A @ g.H_ground_from_img()
        warped = cv2.warpPerspective(frame_bgr, M, (self.W, self.H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)
        mask = np.zeros((self.H, self.W), np.uint8)
        cv2.fillPoly(mask, [np.round(self.px(fp_en)).astype(np.int32)], 255, cv2.LINE_AA)
        mask[:40] = 0
        if alpha >= 1.0:
            np.copyto(target, warped, where=mask[..., None] > 127)
        else:
            m = (mask.astype(np.float32) / 255 * alpha)[..., None]
            target[:] = (target * (1 - m) + warped * m).astype(np.uint8)

    def render(self, i_seq: int, frame_bgr, g, text: Text, add_to_mosaic: bool) -> np.ndarray:
        img = self.mosaic.copy()
        fp = clipped_footprint(g, self.R) if g is not None else None
        small = cv2.resize(frame_bgr, self.proc, interpolation=cv2.INTER_AREA)
        if fp is not None and add_to_mosaic:
            # only the near part of each frame: far, oblique ground smears tall objects into streaks
            near = clipped_footprint(g, min(self.R, 1.5 * max(g.camera_enu_m[2], 1.0)))
            self.drape(self.mosaic, small, g, near, alpha=0.85)
        # dim the mosaic so the live frame stands out
        img = cv2.addWeighted(img, 0.7, self.base, 0.3, 0)
        if fp is not None:
            self.drape(img, small, g, fp)
        tp = np.round(self.track_px).astype(np.int32)
        cv2.polylines(img, [tp], False, (95, 95, 100), 2, cv2.LINE_AA)
        if i_seq > 1:
            cv2.polylines(img, [tp[: i_seq]], False, TRACK, 3, cv2.LINE_AA)
        if g is not None:
            c = self.px(g.camera_enu_m[:2])[0]
            if fp is not None:
                fpp = np.round(self.px(fp)).astype(np.int32)
                cv2.polylines(img, [fpp], True, FOOT, 2, cv2.LINE_AA)
                corners = self.px(clipped_footprint(g, self.R)[::16])
                for q in corners:
                    cv2.line(img, tuple(np.round(c).astype(int)), tuple(np.round(q).astype(int)), FOOT, 1, cv2.LINE_AA)
            yaw = math.radians(g.pose.body_yaw_deg if g.pose.body_yaw_deg is not None else g.pose.gimbal_yaw_deg)
            fwd = np.array([math.sin(yaw), -math.cos(yaw)])
            side = np.array([-fwd[1], fwd[0]])
            tri = np.array([c + fwd * 14, c - fwd * 8 + side * 8, c - fwd * 4, c - fwd * 8 - side * 8])
            cv2.fillPoly(img, [np.round(tri).astype(np.int32)], ACCENT, cv2.LINE_AA)
            cv2.polylines(img, [np.round(tri).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
        img[:40] = BG
        text.draw(img, [(14, 9, "Map: track, footprint, draped frames", TEXT), self.scale_label + (TEXT,),
                        self.north_label + (TEXT,)], 18)
        return img


class PlotPanel:
    def __init__(self, clip: Clip, size, t0_ms: float, t1_ms: float, text: Text):
        self.W, self.H = size
        self.t0, self.t1 = t0_ms, t1_ms
        self.left, self.right, self.top = 150, 20, 40
        lane_h = (self.H - self.top - 26) / len(LANES)
        self.lanes = []
        img = np.full((self.H, self.W, 3), PANEL, np.uint8)
        img[: self.top] = BG
        sel = (clip.t >= t0_ms) & (clip.t <= t1_ms)
        tt = clip.t[sel]
        series = [clip.alt[sel], clip.speed[sel], clip.pitch[sel], clip.yaw[sel]]
        labels = [(14, 9, "Telemetry (" + clip.source + ")", TEXT)]
        for k, ((name, unit, col, min_span), y) in enumerate(zip(LANES, series)):
            y0 = self.top + k * lane_h + 6
            y1 = self.top + (k + 1) * lane_h - 6
            lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
            if hi - lo < min_span:
                mid = (hi + lo) / 2
                lo, hi = mid - min_span / 2, mid + min_span / 2
            if name == "Gimbal pitch":                                  # keep the axis physical
                lo, hi = max(lo, -90.0), max(hi, max(lo, -90.0) + min_span)
            pad = 0.1 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            self.lanes.append((y0, y1, lo, hi, col))
            cv2.rectangle(img, (self.left, int(y0)), (self.W - self.right, int(y1)), (44, 45, 52), -1)
            pts = np.c_[self.x(tt), self.y(k, y)]
            cv2.polylines(img, [np.round(pts).astype(np.int32)], False, col, 2, cv2.LINE_AA)
            labels += [(12, int(y0) + 2, name, col), (12, int(y0) + 22, f"{lo + pad:.0f} to {hi - pad:.0f} {unit}", DIM)]
        for s in np.arange(math.ceil(t0_ms / 5000) * 5, t1_ms / 1000 + 1e-9, 5):
            x = int(round(self.x(s * 1000)))
            cv2.line(img, (x, self.top), (x, self.H - 26), (60, 61, 68), 1)
            labels.append((x - 8, self.H - 22, f"{s:g}s", DIM))
        text.draw(img, labels, 15)
        self.base = img
        self.series = (tt, series)

    def x(self, t_ms):
        return self.left + (np.asarray(t_ms) - self.t0) / max(self.t1 - self.t0, 1) * (self.W - self.left - self.right)

    def y(self, k, v):
        y0, y1, lo, hi, _ = self.lanes[k]
        return y1 - (np.asarray(v) - lo) / (hi - lo) * (y1 - y0)

    def render(self, t_ms: float) -> np.ndarray:
        img = self.base.copy()
        x = int(round(self.x(t_ms)))
        cv2.line(img, (x, self.top), (x, self.H - 26), (255, 255, 255), 1, cv2.LINE_AA)
        tt, series = self.series
        for k, ser in enumerate(series):
            v = np.interp(t_ms, tt, ser)
            cv2.circle(img, (x, int(round(self.y(k, v)))), 5, self.lanes[k][4], -1, cv2.LINE_AA)
        return img


class Readout:
    def __init__(self, clip: Clip, size, camera: CameraModel):
        self.W, self.H = size
        self.clip = clip
        self.cam = camera

    def render(self, i, g, p, text: Text) -> np.ndarray:
        """g: geometry at the video's own resolution (for GSD)."""
        img = np.full((self.H, self.W, 3), PANEL, np.uint8)
        img[:40] = BG
        t = self.clip.t_video(i) / 1000.0
        L = [(14, 9, "drongo", ACCENT), (100, 11, f"camera {self.cam.name}", DIM)]
        if p is None:
            L.append((20, 60, "no telemetry at this time", TEXT))
            return text.draw(img, L, 18)
        spd = float(np.interp(self.clip.t_tel(i), self.clip.t, self.clip.speed))
        rows = [("time", f"{t:7.2f} s   frame {i}"), ("position", f"{p.lat_deg:.6f}, {p.lon_deg:.6f}"),
                ("altitude", f"{p.alt_agl_m:6.1f} m AGL"), ("speed", f"{spd:6.1f} m/s"),
                ("gimbal", f"yaw {p.gimbal_yaw_deg % 360:5.1f}  pitch {p.gimbal_pitch_deg:5.1f}  roll {p.gimbal_roll_deg:4.1f}")]
        if g is not None:
            w, h = g.img_size
            gsd = float(g.gsd_m_per_px(np.array([[w / 2, h / 2]]))[0]) if g._row_ok(h / 2) else float("nan")
            fp = g.footprint()
            rows.append(("centre GSD", f"{gsd * 100:6.1f} cm/px" if gsd == gsd else "  -"))
            rows.append(("footprint", f"{fp.area_m2 / 1e4:7.2f} ha" if fp is not None else "  -"))
        if self.clip.offset_ms:
            rows.append(("sync", f"telemetry shifted {self.clip.offset_ms:+.0f} ms"))
        for k, (a, b) in enumerate(rows):
            L += [(18, 52 + 32 * k, a, DIM), (130, 52 + 32 * k, b, TEXT)]
        text.draw(img, L, 18)
        # compass
        cx, cy, r = self.W - 75, 105, 52
        cv2.circle(img, (cx, cy), r, (70, 72, 80), 2, cv2.LINE_AA)
        a = math.radians(p.gimbal_yaw_deg)
        tip = (int(cx + r * 0.85 * math.sin(a)), int(cy - r * 0.85 * math.cos(a)))
        cv2.arrowedLine(img, (cx, cy), tip, ACCENT, 3, cv2.LINE_AA, tipLength=0.3)
        text.draw(img, [(cx - 6, cy - r - 24, "N", DIM)], 15)
        return img


# ---------------------------------------------------------------------------- composite

def compose(video: Path, camera: CameraModel, out: Path, srt: Optional[Path] = None, size=(1920, 1080),
            start_s: float = 0.0, end_s: Optional[float] = None, offset_ms: float = 0.0,
            max_range_m: float = 600.0, mosaic_every_s: float = 0.5, grid: bool = False,
            title: Optional[str] = None, proc_w: int = 960, crf: int = 21, progress: bool = True) -> Path:
    clip = Clip(video, srt, camera, offset_ms)
    W, H = size
    vw_w = int(round(W * 2 / 3)) // 2 * 2
    vh = int(round(vw_w * 9 / 16)) // 2 * 2
    side_w, bot_h = W - vw_w, H - vh
    i0 = int(round(start_s * clip.fps))
    i1 = min(clip.n, int(round(end_s * clip.fps)) if end_s else clip.n)
    frames = list(range(i0, i1))
    text = Text()
    mp = MapPanel(clip, (side_w, vh), frames, max_range_m, proc_w)
    pp = PlotPanel(clip, (vw_w, bot_h), clip.t_tel(i0), clip.t_tel(i1 - 1), text)
    ro = Readout(clip, (side_w, bot_h), camera)
    every = max(1, int(round(mosaic_every_s * clip.fps)))
    title = title or video.name
    clip.cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    out = Path(out)
    with VideoWriter(out, W, H, clip.fps, crf=crf) as vw:
        for k, i in enumerate(frames):
            ok, fr = clip.cap.read()
            if not ok:
                break
            canvas = np.full((H, W, 3), BG, np.uint8)
            # video panel, letterboxed to 16:9
            sc = min(vw_w / clip.w, vh / clip.h)
            fw, fh = int(clip.w * sc), int(clip.h * sc)
            view = cv2.resize(fr, (fw, fh), interpolation=cv2.INTER_AREA)
            g_view = clip.geometry(i, fw, fh)
            if grid and g_view is not None:
                view = draw_grid(view, g_view)
            x0, y0 = (vw_w - fw) // 2, (vh - fh) // 2
            canvas[y0:y0 + fh, x0:x0 + fw] = view
            text.draw(canvas, [(16, 12, title, (255, 255, 255))], 20)
            g = clip.geometry(i, *mp.proc)
            seq = int(np.searchsorted(clip.t, clip.t_tel(i)))
            canvas[:vh, vw_w:] = mp.render(seq, fr, g, text, add_to_mosaic=(k % every == 0))
            canvas[vh:, :vw_w] = pp.render(clip.t_tel(i))
            canvas[vh:, vw_w:] = ro.render(i, clip.geometry(i, clip.w, clip.h), clip.pose(i), text)
            cv2.line(canvas, (vw_w, 0), (vw_w, H), BG, 2)
            cv2.line(canvas, (0, vh), (W, vh), BG, 2)
            vw.write(canvas)
            if progress and (k + 1) % int(clip.fps * 5) == 0:
                print(f"  {k + 1}/{len(frames)} frames", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--srt", type=Path, help="SRT sidecar (default: next to the video, else embedded telemetry)")
    ap.add_argument("--camera", default="equiv_24mm", help=f"preset: {CameraModel.presets()}")
    ap.add_argument("--out", type=Path, help="default: <video>_composite.mp4 next to the video")
    ap.add_argument("--size", default="1920x1080", help="output WxH")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, help="seconds")
    ap.add_argument("--offset-ms", type=float, default=0.0,
                    help="shift telemetry by this much (e.g. the sync offset from run_video)")
    ap.add_argument("--max-range", type=float, default=600.0, help="cut footprints at this ground range (m)")
    ap.add_argument("--mosaic-every", type=float, default=0.5, help="add a draped frame to the map every N s; 0 = off")
    ap.add_argument("--grid", action="store_true", help="overlay a 10 m ground grid on the video")
    ap.add_argument("--title", help="caption over the video (default: file name)")
    ap.add_argument("--crf", type=int, default=21)
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    out = a.out or a.video.with_name(a.video.stem + "_composite.mp4")
    every = a.mosaic_every if a.mosaic_every > 0 else 1e9
    compose(a.video, CameraModel.preset(a.camera), out, a.srt, (w, h), a.start, a.end, a.offset_ms,
            a.max_range, every, a.grid, a.title, crf=a.crf)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
