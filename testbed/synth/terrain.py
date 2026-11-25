"""Procedural colour ground textures for demo and test clips.

Two terrain styles, both generated from a seed so clips are reproducible and
redistributable (no third-party imagery):

  nature : crop fields with row texture, hedgerows and tree lines, woods,
           dirt tracks, a river, a few farm buildings
  urban  : rotated street grid with lane and crossing markings, building
           blocks split into lots with varied roofs and cast shadows, parks,
           street trees and parked cars

Each generator returns the ground image (with shadows cast by the 3D
objects) and a `Scene` of the objects themselves - buildings, trees, cars -
which the renderer draws in 3D on top of the ground. The ground is flat, so
every ground-truth projection stays exact.

`load_orthophoto` takes a real image with a world file instead.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .render import GroundTexture
from .scene import AMBIENT, FACADES, SUN_AZIMUTH_DEG, SUN_ELEVATION_DEG, Building, Scene

# brightness of shade relative to sunlit ground, per BGR channel (skylight is bluish)
_SHADE = AMBIENT / (AMBIENT + (1 - AMBIENT) * math.sin(math.radians(SUN_ELEVATION_DEG)))
SHADOW_STRENGTH_BGR = (1 - _SHADE * 1.10, 1 - _SHADE, 1 - _SHADE * 0.92)


class _Canvas:
    """Texture image plus a metre <-> pixel mapping (north up, east right)."""

    def __init__(self, half_size_m: float, res_m: float, rng: np.random.Generator):
        self.half = half_size_m
        self.res = res_m
        self.n = int(round(2 * half_size_m / res_m))
        self.rng = rng
        self.img = np.zeros((self.n, self.n, 3), np.float32)       # BGR, 0..255
        self.shadow = np.zeros((self.n, self.n), np.float32)        # 0..1
        self.scene = Scene()
        self.lit = []                                               # roofs: never in ground shadow
        az = math.radians(SUN_AZIMUTH_DEG + 180.0)                 # direction shadows fall
        k = 1.0 / math.tan(math.radians(SUN_ELEVATION_DEG))
        self.shadow_dir = np.array([math.sin(az) * k, math.cos(az) * k])   # metres E, N per metre of height

    # metres (east, north) -> pixel (col, row), float
    def px(self, en) -> np.ndarray:
        en = np.atleast_2d(np.asarray(en, np.float64))
        return np.stack([(en[:, 0] + self.half) / self.res, (self.half - en[:, 1]) / self.res], 1)

    def ipx(self, en) -> np.ndarray:
        return np.round(self.px(en) * 16).astype(np.int32)        # 4-bit sub-pixel for LINE_AA

    def m(self, metres: float) -> int:
        return max(1, int(round(metres / self.res)))

    def poly(self, en, colour, target: Optional[np.ndarray] = None) -> None:
        cv2.fillPoly(self.img if target is None else target, [self.ipx(en)], colour, cv2.LINE_AA, shift=4)

    def line(self, en, colour, width_m: float, target: Optional[np.ndarray] = None) -> None:
        cv2.polylines(self.img if target is None else target, [self.ipx(en)], False, colour,
                      self.m(width_m), cv2.LINE_AA, shift=4)

    def disc(self, en, r_m: float, colour, target: Optional[np.ndarray] = None) -> None:
        c = self.ipx(en)[0]
        cv2.circle(self.img if target is None else target, (int(c[0]), int(c[1])), int(round(r_m / self.res * 16)),
                   colour, -1, cv2.LINE_AA, shift=4)

    def cast(self, footprint_en: np.ndarray, height_m: float) -> None:
        """Shadow of a prism standing on `footprint_en`."""
        fp = np.asarray(footprint_en, np.float64)
        moved = fp + self.shadow_dir * height_m
        hull = cv2.convexHull(np.vstack([fp, moved]).astype(np.float32))[:, 0]
        self.poly(hull, 1.0, self.shadow)

    def cast_ellipse(self, centre_en, r_perp: float, r_along: float) -> None:
        """Shadow of a sphere: an ellipse stretched along the shadow direction."""
        c = self.px(centre_en)[0]
        ang = math.degrees(math.atan2(-self.shadow_dir[1], self.shadow_dir[0]))   # pixel rows point south
        cv2.ellipse(self.shadow, (int(round(c[0] * 16)), int(round(c[1] * 16))),
                    (max(1, int(round(r_along / self.res * 16))), max(1, int(round(r_perp / self.res * 16)))),
                    ang, 0, 360, 1.0, -1, cv2.LINE_AA, shift=4)

    def detail(self, scales_m=(6.0, 2.0, 0.6)) -> np.ndarray:
        """Noise in [0, 1] with features at the given ground scales."""
        out = np.zeros((self.n, self.n), np.float32)
        for sc in scales_m:
            cells = max(2, int(self.n * self.res / sc))
            out += cv2.resize(self.rng.random((cells, cells)).astype(np.float32), (self.n, self.n),
                              interpolation=cv2.INTER_CUBIC)
        out /= len(scales_m)
        return np.clip((out - 0.5) * 2.2 + 0.5, 0, 1)

    def finish(self, grain: float, surface: float = 0.10) -> np.ndarray:
        for poly in self.lit:                                       # roofs are above the ground shadows
            self.poly(poly, 0.0, self.shadow)
        gain = self.detail()                                        # metre-scale surface variation
        gain -= 0.5
        gain *= 2 * surface
        roofs = np.zeros(gain.shape, np.float32)                    # roofs: smooth membranes / tiles
        for poly in self.lit:
            self.poly(poly, 1.0, roofs)
        gain *= 1.0 - 0.75 * roofs
        del roofs
        gain += 1.0
        gain += self.rng.normal(0, grain / 128.0, gain.shape).astype(np.float32)
        s = cv2.GaussianBlur(self.shadow, (0, 0), max(0.5, 0.12 / self.res))     # narrow penumbra
        self.shadow = None
        np.clip(s, 0, 1, out=s)
        out = np.empty(self.img.shape, np.uint8)
        for c in range(3):
            ch = self.img[..., c]
            ch *= gain
            ch *= 1.0 - SHADOW_STRENGTH_BGR[c] * s
            out[..., c] = np.clip(ch, 0, 255)
        self.img = None
        self.scene.freeze()
        return out


def fbm(n: int, rng: np.random.Generator, octaves=((4, 0.4), (16, 0.3), (64, 0.18), (256, 0.12))) -> np.ndarray:
    """Fractal value noise in [0, 1], shape (n, n)."""
    out = np.zeros((n, n), np.float32)
    for cells, amp in octaves:
        small = rng.random((cells + 1, cells + 1)).astype(np.float32)
        out += amp * cv2.resize(small, (n, n), interpolation=cv2.INTER_CUBIC)
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-9)


def _rot(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _jitter(c, amount):
    return np.clip(np.asarray(c, np.float32) + amount, 0, 255)


# ---------------------------------------------------------------------------- nature

_FIELD_TYPES = [
    # name, base BGR, row contrast, row period m
    ("wheat", (92, 178, 214), 0.10, 0.25),
    ("stubble", (120, 170, 196), 0.18, 0.6),
    ("young_crop", (70, 150, 110), 0.22, 0.75),
    ("green_crop", (55, 128, 78), 0.12, 0.6),
    ("ploughed", (64, 92, 128), 0.25, 0.9),
    ("dry_soil", (110, 148, 176), 0.10, 1.2),
    ("fallow", (80, 130, 140), 0.05, 0.6),
]


def _crown(cv: _Canvas, en, r: float, base) -> None:
    """A tree: trunk and crown sphere in the scene, their shadows on the ground."""
    en = np.asarray(en, float)
    trunk = cv.rng.uniform(0.6, 1.6) + 0.3 * r
    zc = trunk + r
    cv.cast_ellipse(en + cv.shadow_dir * zc, r, r / math.sin(math.radians(SUN_ELEVATION_DEG)))
    cv.line(np.array([en, en + cv.shadow_dir * trunk]), 1.0, 0.35, cv.shadow)
    cv.scene.tree_en.append(en)
    cv.scene.tree_r.append(r)
    cv.scene.tree_z.append(zc)
    cv.scene.tree_bgr.append(_jitter(base, cv.rng.normal(0, 6, 3)) * cv.rng.uniform(0.9, 1.15))


def make_nature(half_size_m: float = 300.0, res_m: float = 0.08, seed: int = 0) -> GroundTexture:
    rng = np.random.default_rng(seed)
    cv = _Canvas(half_size_m, res_m, rng)
    n, H = cv.n, half_size_m

    # grass / scrub base
    g = fbm(n, rng)[..., None]
    cv.img[:] = (1 - g) * np.array([60, 125, 105], np.float32) + g * np.array([85, 150, 150], np.float32)

    # field parcels: jittered grid in a rotated frame
    angle = rng.uniform(-30, 30)
    R = _rot(angle)
    xs = np.cumsum(np.r_[-1.5 * H, rng.uniform(70, 150, 40)])
    ys = np.cumsum(np.r_[-1.5 * H, rng.uniform(60, 130, 40)])
    xs, ys = xs[: np.searchsorted(xs, 1.5 * H) + 1], ys[: np.searchsorted(ys, 1.5 * H) + 1]
    grid = np.stack(np.meshgrid(xs, ys), -1) + rng.normal(0, 8, (len(ys), len(xs), 2))
    edges = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            quad = np.array([grid[j, i], grid[j, i + 1], grid[j + 1, i + 1], grid[j + 1, i]]) @ R.T
            if np.all(np.abs(quad) > H * 1.05, axis=1).all():
                continue
            ctr = quad.mean(0)
            inset = ctr + (quad - ctr) * 0.985
            if rng.random() < 0.12:
                pass                                      # leave as grass / pasture
            else:
                _field(cv, inset, _FIELD_TYPES[rng.integers(len(_FIELD_TYPES))], angle + rng.choice([0, 90]))
            edges += [(quad[k], quad[(k + 1) % 4]) for k in (0, 3)]

    # parcel edges become dirt tracks, hedgerows or tree lines
    tracks = []
    for a, b in edges:
        u = rng.random()
        if u < 0.22:
            tracks.append((a, b))
        elif u < 0.45:
            L = np.linalg.norm(b - a)
            step = rng.uniform(4, 7)
            for t in np.arange(0, L, step) / L:
                if rng.random() < 0.85:
                    _crown(cv, a + (b - a) * t + rng.normal(0, 0.8, 2), rng.uniform(2.0, 4.0), (40, 95, 55))
        elif u < 0.6:
            cv.line(np.array([a, b]), (45, 100, 70), rng.uniform(1.5, 3.0))
    for a, b in tracks:
        cv.line(np.array([a, b]), (120, 165, 190), 4.0)
        d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        nrm = np.array([-d[1], d[0]])
        for side in (-0.9, 0.9):
            cv.line(np.array([a + nrm * side, b + nrm * side]), (100, 140, 165), 0.45)

    # woods: clusters of crowns
    for _ in range(int((2 * H) ** 2 / 90_000)):
        c = rng.uniform(-H, H, 2)
        r_wood = rng.uniform(25, 70)
        for _ in range(int(r_wood ** 2 / 10)):
            p = c + rng.normal(0, r_wood / 2, 2)
            _crown(cv, p, rng.uniform(2.5, 5.0), (rng.uniform(30, 50), rng.uniform(80, 110), rng.uniform(40, 70)))

    # river: smooth meander across the map, banks then water
    s = np.linspace(-1.5 * H, 1.5 * H, 600)
    phase = rng.uniform(0, 2 * math.pi, 3)
    off = 60 * np.sin(s / 140 + phase[0]) + 25 * np.sin(s / 47 + phase[1]) + 10 * np.sin(s / 19 + phase[2])
    river = np.stack([s, off + rng.uniform(-0.4, 0.4) * H], 1) @ _rot(rng.uniform(0, 180)).T
    width = rng.uniform(10, 16)
    cv.line(river, (110, 160, 175), width + 6)                     # sand / mud banks
    cv.line(river, (90, 100, 60), width + 1.5)
    water = np.zeros((n, n), np.float32)
    cv.line(river, 1.0, width, water)
    w_tex = fbm(n, rng, ((32, 0.6), (256, 0.4)))
    wet = water > 0
    wa, wt = water[wet][:, None], w_tex[wet][:, None]
    col = (1 - wt) * np.array([85, 90, 50], np.float32) + wt * np.array([110, 115, 70], np.float32)
    cv.img[wet] = cv.img[wet] * (1 - wa) + col * wa
    del water, w_tex, wet
    for k in range(0, len(river), 9):                               # riverside trees
        if rng.random() < 0.6:
            d = river[min(k + 1, len(river) - 1)] - river[max(k - 1, 0)]
            nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            _crown(cv, river[k] + nrm * rng.choice([-1, 1]) * (width / 2 + rng.uniform(3, 7)),
                   rng.uniform(2.5, 4.5), (45, 100, 60))

    # farmsteads along tracks
    inside = [(a, b) for a, b in tracks if np.abs((a + b) / 2).max() < H - 40]
    for a, b in inside[:: max(1, len(inside) // 5)][:5]:
        p = a + (b - a) * rng.uniform(0.3, 0.7)
        d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        nrm = np.array([-d[1], d[0]])
        for _ in range(rng.integers(1, 4)):
            c = p + nrm * rng.uniform(8, 20) + d * rng.uniform(-15, 15)
            _building(cv, c, d, rng.uniform(10, 25), rng.uniform(7, 12), rng.uniform(3.5, 6),
                      [(50, 60, 140), (150, 150, 150), (110, 115, 120), (70, 90, 120)][rng.integers(4)],
                      gable=True)

    img = cv.finish(grain=4.0)
    return GroundTexture(img, -H, H, res_m, cv.scene)


def _field(cv: _Canvas, quad_en: np.ndarray, kind, row_deg: float) -> None:
    _, base, contrast, period = kind
    px = cv.px(quad_en)
    x0, y0 = np.floor(px.min(0)).astype(int)
    x1, y1 = np.ceil(px.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, cv.n), min(y1, cv.n)
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(mask, [np.round((px - [x0, y0]) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) * cv.res
    a = math.radians(row_deg)
    u = xx * math.cos(a) + yy * math.sin(a)
    rows = 0.5 + 0.5 * np.sin(2 * math.pi * u / period)
    var = cv2.resize(cv.rng.random((6, 6)).astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    base = _jitter(base, cv.rng.normal(0, 8, 3))
    col = base[None, None, :] * (1 - contrast * rows[..., None]) * (0.88 + 0.24 * var[..., None])
    # tractor tramlines every ~20 m
    tram = (np.abs(((u + cv.rng.uniform(0, 20)) % 20) - 10) > 9.6).astype(np.float32)[..., None]
    col = col * (1 - 0.18 * tram)
    roi = cv.img[y0:y1, x0:x1]
    roi[:] = roi * (1 - mask[..., None]) + col * mask[..., None]


_WALLS = {"plaster": [(190, 215, 230), (225, 228, 230), (200, 205, 210), (185, 195, 225), (140, 190, 220)],
          "glass": [(255, 255, 255)], "brick": [(70, 90, 160), (80, 105, 170), (60, 75, 130)],
          "strip": [(200, 200, 200), (180, 185, 190)], "plain": [(200, 205, 205), (140, 150, 160), (90, 110, 150)]}


def _building(cv: _Canvas, c, d, length_m, width_m, height_m, roof, gable: bool = False,
              facade: Optional[str] = None) -> None:
    d = np.asarray(d, float) / np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    hl, hw = length_m / 2, width_m / 2
    fp = np.array([c + d * hl + nrm * hw, c - d * hl + nrm * hw, c - d * hl - nrm * hw, c + d * hl - nrm * hw])
    x, y = fp[:, 0], fp[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) < 0:     # make counter-clockwise in ENU
        fp = fp[::-1].copy()
    rng = cv.rng
    roof = _jitter(roof, rng.normal(0, 7, 3))
    rise = min(hl, hw) * math.tan(math.radians(rng.uniform(20, 32))) if gable else 0.0
    cv.cast(fp, height_m + rise * 0.6)
    facade = facade or ("plain" if gable else "plaster")
    cv.scene.buildings.append(Building(
        footprint=fp, height_m=height_m, facade=FACADES.index(facade),
        wall_bgr=tuple(_jitter(_WALLS[facade][rng.integers(len(_WALLS[facade]))], rng.normal(0, 5, 3))),
        roof="gable" if gable else "flat", roof_bgr=tuple(roof), ridge_rise_m=rise))
    cv.lit.append(fp)
    cv.poly(fp, tuple(map(float, roof * 0.72)))                     # parapet
    inner = c + (fp - c) * (1 - 0.5 / max(min(hl, hw), 1.0))
    cv.poly(inner, tuple(map(float, roof)))
    if gable:
        return
    for _ in range(rng.integers(0, 5)):                             # AC units / hatches
        p = c + d * rng.uniform(-hl, hl) * 0.75 + nrm * rng.uniform(-hw, hw) * 0.75
        s = rng.uniform(0.6, 1.4)
        box = p + np.array([d * s + nrm * s * 0.7, -d * s + nrm * s * 0.7, -d * s - nrm * s * 0.7, d * s - nrm * s * 0.7])
        cv.poly(box + cv.shadow_dir * 0.8, tuple(map(float, roof * 0.5)))    # its shadow on the roof
        cv.poly(box, (200, 200, 200) if rng.random() < 0.6 else (90, 90, 95))
    if rng.random() < 0.45:                                         # solar water heaters / panels
        rowc = c - nrm * hw * 0.5
        k = rng.integers(2, 7)
        for j in range(k):
            p = rowc + d * (j - (k - 1) / 2) * 2.2
            panel = p + np.array([d * 0.9 + nrm * 0.6, -d * 0.9 + nrm * 0.6, -d * 0.9 - nrm * 0.6, d * 0.9 - nrm * 0.6])
            cv.poly(panel, (110, 60, 40))
        if rng.random() < 0.6:
            cv.disc(rowc + nrm * 2.0, 0.55, (225, 225, 225))        # tank


# ---------------------------------------------------------------------------- urban

_ROOFS = [(185, 185, 185), (150, 152, 155), (205, 205, 200), (75, 95, 170), (95, 110, 150),
          (170, 190, 205), (120, 120, 125), (215, 220, 225)]
_CARS = [(235, 235, 235), (30, 30, 30), (160, 160, 165), (40, 40, 150), (130, 70, 30), (210, 210, 215),
         (90, 90, 95), (40, 120, 200), (20, 20, 20)]


def make_urban(half_size_m: float = 300.0, res_m: float = 0.08, seed: int = 0) -> GroundTexture:
    rng = np.random.default_rng(seed)
    cv = _Canvas(half_size_m, res_m, rng)
    n, H = cv.n, half_size_m
    g = fbm(n, rng, ((16, 0.5), (128, 0.3), (512, 0.2)))[..., None]
    cv.img[:] = 70 + 25 * g                                         # asphalt
    angle = rng.uniform(-25, 25)
    R = _rot(angle)
    to_w = lambda pts: np.asarray(pts, float) @ R.T                 # city frame -> ENU

    xs = np.cumsum(np.r_[-1.5 * H, rng.uniform(70, 130, 40)])
    ys = np.cumsum(np.r_[-1.5 * H, rng.uniform(60, 110, 40)])
    xs, ys = xs[: np.searchsorted(xs, 1.5 * H) + 1], ys[: np.searchsorted(ys, 1.5 * H) + 1]
    road_w = {"x": rng.uniform(9, 15, len(xs)), "y": rng.uniform(9, 15, len(ys))}
    for i, x in enumerate(xs):
        road_w["x"][i] = 22 if rng.random() < 0.15 else road_w["x"][i]       # occasional avenue
    for j, y in enumerate(ys):
        road_w["y"][j] = 22 if rng.random() < 0.15 else road_w["y"][j]

    # lane markings on every road, then crossings
    L = 1.5 * H
    for i, x in enumerate(xs):
        _dashes(cv, to_w([[x, -L], [x, L]]), road_w["x"][i])
    for j, y in enumerate(ys):
        _dashes(cv, to_w([[-L, y], [L, y]]), road_w["y"][j])

    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            bx0 = xs[i] + road_w["x"][i] / 2
            bx1 = xs[i + 1] - road_w["x"][i + 1] / 2
            by0 = ys[j] + road_w["y"][j] / 2
            by1 = ys[j + 1] - road_w["y"][j + 1] / 2
            if bx1 - bx0 < 20 or by1 - by0 < 20:
                continue
            ctr = to_w([[(bx0 + bx1) / 2, (by0 + by1) / 2]])[0]
            if np.any(np.abs(ctr) > H + 80):
                continue
            _block(cv, to_w, bx0, by0, bx1, by1, R)
            for (x, w) in ((xs[i], road_w["x"][i]), (xs[i + 1], road_w["x"][i + 1])):
                _zebra(cv, to_w, x, by0, w, horizontal=False)
            for (y, w) in ((ys[j], road_w["y"][j]), (ys[j + 1], road_w["y"][j + 1])):
                _zebra(cv, to_w, y, bx0, w, horizontal=True)

    img = cv.finish(grain=3.0)
    return GroundTexture(img, -H, H, res_m, cv.scene)


def _dashes(cv: _Canvas, seg, road_w):
    a, b = seg
    L = np.linalg.norm(b - a)
    d = (b - a) / L
    for t in np.arange(0, L, 9.0):
        cv.line(np.array([a + d * t, a + d * (t + 4.5)]), (215, 215, 215), 0.15)
    if road_w > 18:                                                 # avenue: median strip
        cv.line(np.array([a, b]), (70, 120, 90), 2.5)


def _zebra(cv, to_w, pos, start, road_w, horizontal):
    for k in np.arange(-road_w / 2 + 1.0, road_w / 2 - 0.5, 1.0):
        if horizontal:
            p = [[start + 0.8, pos + k], [start + 3.8, pos + k]]
        else:
            p = [[pos + k, start + 0.8], [pos + k, start + 3.8]]
        cv.line(to_w(p), (225, 225, 225), 0.5)


def _block(cv: _Canvas, to_w, x0, y0, x1, y1, R):
    rng = cv.rng
    walk = 3.0
    rect = lambda a, b, c, d: to_w([[a, b], [c, b], [c, d], [a, d]])
    cv.poly(rect(x0, y0, x1, y1), tuple(map(float, _jitter((160, 165, 168), rng.normal(0, 5, 3)))))   # sidewalk
    cv.line(to_w([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]), (185, 188, 190), 0.25)        # kerb
    ex, ey = R[:, 0], R[:, 1]

    # parked cars along the kerb, street trees along the sidewalk
    for (a, b, along) in (((x0, y0 - 1.4), (x1, y0 - 1.4), ex), ((x0, y1 + 1.4), (x1, y1 + 1.4), ex),
                          ((x0 - 1.4, y0), (x0 - 1.4, y1), ey), ((x1 + 1.4, y0), (x1 + 1.4, y1), ey)):
        a, b = to_w([a, b])
        Ls = np.linalg.norm(b - a)
        t = rng.uniform(3, 8)
        while t < Ls - 6:
            if rng.random() < 0.55:
                _car(cv, a + along * t, along, _CARS[rng.integers(len(_CARS))])
            t += rng.uniform(5.3, 6.5) if rng.random() < 0.8 else rng.uniform(10, 25)
    for (a, b) in ((to_w([[x0 + 2, y0 + 1.5]])[0], to_w([[x1 - 2, y0 + 1.5]])[0]),
                   (to_w([[x0 + 2, y1 - 1.5]])[0], to_w([[x1 - 2, y1 - 1.5]])[0])):
        if rng.random() < 0.6:
            Ls = np.linalg.norm(b - a)
            for t in np.arange(rng.uniform(0, 6), Ls, rng.uniform(8, 12)):
                _crown(cv, a + (b - a) * t / Ls, rng.uniform(2.0, 3.2), (45, 105, 60))

    ix0, iy0, ix1, iy1 = x0 + walk, y0 + walk, x1 - walk, y1 - walk
    if rng.random() < 0.1:                                          # park
        cv.poly(rect(ix0, iy0, ix1, iy1), (70, 140, 95))
        cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
        cv.line(to_w([[ix0, iy0], [ix1, iy1]]), (150, 175, 190), 2.5)
        cv.line(to_w([[ix0, iy1], [ix1, iy0]]), (150, 175, 190), 2.5)
        cv.disc(to_w([[cx, cy]])[0], 6, (150, 175, 190))
        for _ in range(int((ix1 - ix0) * (iy1 - iy0) / 120)):
            p = to_w([[rng.uniform(ix0 + 3, ix1 - 3), rng.uniform(iy0 + 3, iy1 - 3)]])[0]
            _crown(cv, p, rng.uniform(2.5, 4.5), (40, 100, 55))
        return
    if rng.random() < 0.08:                                         # surface car park
        cv.poly(rect(ix0, iy0, ix1, iy1), (95, 95, 98))
        for x in np.arange(ix0 + 1, ix1 - 2.5, 2.6):
            for y in (iy0 + 3, iy1 - 3):
                cv.line(to_w([[x, y - 2.5], [x, y + 2.5]]), (210, 210, 210), 0.12)
                if rng.random() < 0.7:
                    _car(cv, to_w([[x + 1.3, y]])[0], ey, _CARS[rng.integers(len(_CARS))])
        return
    yard = [(95, 140, 110), (120, 150, 150), (150, 160, 165)][rng.integers(3)]     # lawn, dirt, paving
    cv.poly(rect(ix0, iy0, ix1, iy1), tuple(map(float, _jitter(yard, rng.normal(0, 6, 3)))))
    # split the block into lots and put one building on each
    lots = [(ix0, iy0, ix1, iy1)]
    out = []
    while lots:
        a, b, c, d = lots.pop()
        w, h = c - a, d - b
        if max(w, h) < rng.uniform(22, 45):
            out.append((a, b, c, d))
            continue
        if w >= h:
            s = a + w * rng.uniform(0.35, 0.65)
            lots += [(a, b, s, d), (s, b, c, d)]
        else:
            s = b + h * rng.uniform(0.35, 0.65)
            lots += [(a, b, c, s), (a, s, c, d)]
    tall = rng.random() < 0.25
    for a, b, c, d in out:
        m = rng.uniform(1.0, 3.0)
        a, b, c, d = a + m, b + m, c - m, d - m
        if c - a < 5 or d - b < 5:
            continue
        if rng.random() < 0.3:
            _crown(cv, to_w([[(a + c) / 2, (b + d) / 2]])[0], rng.uniform(2, 4), (45, 105, 60))
            continue
        ctr = to_w([[(a + c) / 2, (b + d) / 2]])[0]
        floors = rng.integers(5, 10) if tall else rng.integers(1, 5)
        facade = (["glass", "strip", "plaster"] if tall else ["plaster", "plaster", "brick", "strip"])
        _building(cv, ctr, ex, c - a, d - b, 3.0 * floors + 0.6, _ROOFS[rng.integers(len(_ROOFS))],
                  facade=facade[rng.integers(len(facade))])


def _car(cv: _Canvas, c, d, colour):
    d = np.asarray(d, float)
    nrm = np.array([-d[1], d[0]])
    L, W = cv.rng.uniform(4.2, 4.9) / 2, 0.92
    body = np.array([c + d * L + nrm * W, c - d * L + nrm * W, c - d * L - nrm * W, c + d * L - nrm * W])
    cv.cast(body, 1.4)
    cv.poly(body, colour)                                           # far-away view; near: 3D box
    cv.scene.car_en.append(np.asarray(c, float))
    cv.scene.car_dir.append(d / np.linalg.norm(d))
    cv.scene.car_bgr.append(np.asarray(colour, float))
    cv.scene.car_len.append(2 * L)
    glass = np.array([c + d * L * 0.45 + nrm * W * 0.8, c - d * L * 0.55 + nrm * W * 0.8,
                      c - d * L * 0.55 - nrm * W * 0.8, c + d * L * 0.45 - nrm * W * 0.8])
    cv.poly(glass, tuple(float(v) * 0.55 + 15 for v in colour))


# ---------------------------------------------------------------------------- real imagery

def load_orthophoto(path: Path, world_file: Optional[Path] = None, centre_en=(0.0, 0.0)) -> GroundTexture:
    """A north-up orthophoto with a world file (.pgw/.jgw/.tfw/.wld) whose
    pixel size is in metres (a projected CRS). The image is placed with its
    centre at `centre_en` in the scenario's local frame."""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    if world_file is None:
        cands = [path.with_suffix(s) for s in (".pgw", ".jgw", ".tfw", ".wld", ".PGW", ".JGW", ".TFW", ".WLD")]
        world_file = next((c for c in cands if c.is_file()), None)
    if world_file is None:
        raise ValueError(f"no world file next to {path}")
    a, d, b, e, *_ = [float(v) for v in Path(world_file).read_text().split()]
    if abs(d) > 1e-9 or abs(b) > 1e-9 or abs(abs(a) - abs(e)) > 1e-6 * abs(a):
        raise ValueError("only north-up images with square pixels are supported")
    res = abs(a)
    h, w = img.shape[:2]
    return GroundTexture(img, centre_en[0] - w * res / 2, centre_en[1] + h * res / 2, res)


TERRAINS = {"nature": make_nature, "urban": make_urban}
