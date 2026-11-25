"""3D objects standing on the flat ground: buildings, trees and cars.

The ground is rendered by homography (exact, see render.py); the objects are
then drawn over it with a painter's algorithm, far to near. Each object is
convex (or drawn bottom-up), so back-face culling settles its own faces.

  buildings : extruded footprints with flat or gable roofs. Walls carry a
              tiled facade texture (windows, floors) and Lambert sun shading;
              flat roofs are the ground texture's roof patch re-projected
              onto the plane at roof height, so roof details stay in place.
  trees     : a trunk and a crown sphere, drawn as a sprite shaded for the
              current sun direction in camera coordinates.
  cars      : a body box with a cabin box on top.

Objects show parallax and occlusion as the camera moves. They are not part of
the ground truth: the truth file describes the camera, and every ground point
not hidden by an object projects exactly where drongo says.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

SUN_AZIMUTH_DEG = 225.0          # sun from the south-west: shadows fall to the north-east
SUN_ELEVATION_DEG = 45.0
AMBIENT = 0.38                   # skylight share of a sunlit horizontal surface's light


def sun_vector() -> np.ndarray:
    az, el = math.radians(SUN_AZIMUTH_DEG), math.radians(SUN_ELEVATION_DEG)
    return np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])


def light(normal_w: np.ndarray) -> float:
    """Brightness relative to a sunlit horizontal surface (the ground texture)."""
    s = sun_vector()
    lit = AMBIENT + (1 - AMBIENT) * max(0.0, float(np.dot(normal_w, s)))
    return lit / (AMBIENT + (1 - AMBIENT) * s[2])


# ---------------------------------------------------------------------------- scene description

@dataclass
class Building:
    footprint: np.ndarray            # (4, 2) ENU metres, counter-clockwise
    height_m: float                  # eaves (gable) or roof (flat) height
    facade: int                      # index into FACADES
    wall_bgr: tuple
    roof: str = "flat"               # "flat" (roof from the ground texture) or "gable"
    roof_bgr: tuple = (60, 70, 150)
    ridge_rise_m: float = 0.0


@dataclass
class Scene:
    buildings: list = field(default_factory=list)
    tree_en: list = field(default_factory=list)          # crown centre ground position
    tree_r: list = field(default_factory=list)           # crown radius
    tree_z: list = field(default_factory=list)           # crown centre height
    tree_bgr: list = field(default_factory=list)
    car_en: list = field(default_factory=list)
    car_dir: list = field(default_factory=list)
    car_bgr: list = field(default_factory=list)
    car_len: list = field(default_factory=list)

    def freeze(self) -> "Scene":
        """Lists -> arrays (call once generation is finished)."""
        for k in ("tree_en", "tree_r", "tree_z", "tree_bgr", "car_en", "car_dir", "car_bgr", "car_len"):
            setattr(self, k, np.asarray(getattr(self, k), np.float64))
        self.tree_en = self.tree_en.reshape(-1, 2)
        self.car_en = self.car_en.reshape(-1, 2)
        self.car_dir = self.car_dir.reshape(-1, 2)
        self.tree_bgr = self.tree_bgr.reshape(-1, 3)
        self.car_bgr = self.car_bgr.reshape(-1, 3)
        self._b_centre = np.array([b.footprint.mean(0) for b in self.buildings]).reshape(-1, 2)
        self._b_h = np.array([b.height_m + b.ridge_rise_m for b in self.buildings])
        return self


# ---------------------------------------------------------------------------- facade textures

PX_PER_M = 16                    # facade texture resolution; one cell = one 3 m x 3 m bay


def _facade_cell(kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 3 * PX_PER_M
    cell = np.zeros((n, n, 3), np.float32)
    if kind == "plaster":
        cell[:] = 1.0
        cv2.rectangle(cell, (12, 10), (36, 34), (0.38, 0.33, 0.30), -1)             # window
        cv2.rectangle(cell, (12, 10), (36, 34), (0.85, 0.85, 0.85), 1)              # frame
        cv2.rectangle(cell, (10, 34), (38, 37), (0.8, 0.8, 0.8), -1)                # sill
        cv2.line(cell, (0, n - 1), (n, n - 1), (0.82, 0.82, 0.82), 1)
    elif kind == "glass":
        cell[:] = (0.62, 0.48, 0.38)
        cell[: n // 2] *= 1.12                                                        # sky reflection
        cv2.rectangle(cell, (0, 0), (n - 1, n - 1), (0.55, 0.55, 0.55), 2)            # mullions
        cv2.line(cell, (n // 2, 0), (n // 2, n), (0.55, 0.55, 0.55), 1)
        cv2.rectangle(cell, (0, n - 7), (n, n), (0.42, 0.42, 0.45), -1)              # spandrel
    elif kind == "brick":
        cell[:] = 1.0
        for y in range(0, n, 4):
            cv2.line(cell, (0, y), (n, y), (0.82, 0.82, 0.82), 1)
        cv2.rectangle(cell, (14, 9), (34, 33), (0.30, 0.28, 0.27), -1)
        cv2.rectangle(cell, (12, 7), (36, 9), (0.9, 0.9, 0.9), -1)
    elif kind == "strip":                                                             # ribbon windows
        cell[:] = 1.0
        cv2.rectangle(cell, (0, 12), (n, 30), (0.36, 0.33, 0.32), -1)
        cv2.line(cell, (0, 12), (n, 12), (0.7, 0.7, 0.7), 1)
    else:                                                                             # plain / barn
        cell[:] = 1.0
        for x in range(0, n, 6):
            cv2.line(cell, (x, 0), (x, n), (0.9, 0.9, 0.9), 1)
    cell *= 1 + rng.normal(0, 0.03, (n, n, 1)).astype(np.float32)
    return cell


FACADES = ["plaster", "glass", "brick", "strip", "plain"]


@lru_cache(maxsize=None)
def facade_levels(idx: int) -> tuple:
    lv = [_facade_cell(FACADES[idx], idx)]
    while lv[-1].shape[0] > 6:
        lv.append(cv2.pyrDown(lv[-1]))
    return tuple(lv)


# ---------------------------------------------------------------------------- tree sprites

SPRITE = 96
N_SPRITES = 6


@lru_cache(maxsize=1)
def _sprite_noise():
    rng = np.random.default_rng(11)
    out = []
    yy, xx = (np.mgrid[0:SPRITE, 0:SPRITE] + 0.5) / SPRITE * 2 - 1
    ang = np.arctan2(yy, xx)
    for _ in range(N_SPRITES):
        leaf = np.zeros((SPRITE, SPRITE), np.float32)
        for cells, amp in ((4, 0.35), (9, 0.35), (20, 0.3)):
            leaf += amp * cv2.resize(rng.random((cells, cells)).astype(np.float32), (SPRITE, SPRITE),
                                     interpolation=cv2.INTER_CUBIC)
        k = rng.integers(5, 9)
        edge = 0.93 + 0.04 * np.sin(k * ang + rng.uniform(0, 6)) + 0.03 * np.sin((k + 3) * ang + rng.uniform(0, 6))
        bump = (leaf - 0.5) * 0.25
        out.append((leaf, edge, bump))
    return out, xx, yy


def tree_sprites(R_world_from_cam: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """(luminance, alpha) sprites of a leafy sphere lit by the sun, as seen
    by a camera with this orientation."""
    noise, xx, yy = _sprite_noise()
    r2 = xx ** 2 + yy ** 2
    s = sun_vector()
    s_cam = R_world_from_cam.T @ s
    out = []
    for leaf, edge, bump in noise:
        nz = -np.sqrt(np.clip(1 - r2 / edge ** 2, 0, 1))            # surface faces the camera (-z)
        n = np.stack([xx / edge, yy / edge, nz], -1)
        n[..., :2] += bump[..., None] * 0.8                          # leaf clumps perturb the normal
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        lam = np.clip(n @ s_cam, 0, 1)
        lum = (AMBIENT + (1 - AMBIENT) * lam) / (AMBIENT + (1 - AMBIENT) * s[2])
        lum *= 0.55 + 0.7 * leaf                                     # gaps and clumps
        lum *= 0.75 + 0.25 * (-nz)                                   # self-occlusion toward the rim
        rr = np.sqrt(r2)
        alpha = np.clip((edge - rr) * SPRITE / 2.5, 0, 1)                # crisp outline
        alpha *= np.where(rr < edge - 0.15, 1.0, np.clip((leaf - 0.25) * 8, 0, 1))   # ragged leafy rim
        lum *= alpha                                                 # premultiplied: no halo when resized
        out.append((lum.astype(np.float32), alpha.astype(np.float32)))
    return out


# ---------------------------------------------------------------------------- drawing

class _Cam:
    def __init__(self, geom):
        self.K = geom.intrinsics.K
        self.R = geom.R_world_from_cam
        self.C = geom.camera_enu_m
        self.w, self.h = geom.img_size
        self.f = math.sqrt(self.K[0, 0] * self.K[1, 1])
        self.geom = geom

    def project(self, X: np.ndarray):
        Xc = (np.atleast_2d(X) - self.C) @ self.R
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (Xc[:, :2] / z[:, None]) * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
        return uv, z


def _blend(img, x0, y0, patch_rgb, alpha):
    """Alpha-blend a patch at (x0, y0), clipped to the image."""
    h, w = alpha.shape
    H, W = img.shape[:2]
    ax0, ay0 = max(0, -x0), max(0, -y0)
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(W, x0 + w), min(H, y0 + h)
    if bx1 <= bx0 or by1 <= by0:
        return
    a = alpha[ay0:ay0 + by1 - by0, ax0:ax0 + bx1 - bx0, None]
    p = patch_rgb[ay0:ay0 + by1 - by0, ax0:ax0 + bx1 - bx0]
    roi = img[by0:by1, bx0:bx1]
    roi += (p - roi) * a


def _fill(img, quad_uv, colour):
    cv2.fillPoly(img, [np.round(quad_uv * 4).astype(np.int32)], tuple(float(c) for c in colour), cv2.LINE_AA, shift=2)


def _warp_into(img, src, M_img_from_src, quad_uv, factor, haze_rgb, haze, border=cv2.BORDER_WRAP):
    """Warp `src` through M into the image, inside polygon quad_uv."""
    H, W = img.shape[:2]
    x0, y0 = np.floor(quad_uv.min(0)).astype(int)
    x1, y1 = np.ceil(quad_uv.max(0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]])
    patch = cv2.warpPerspective(src, T @ M_img_from_src, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=border).astype(np.float32)
    if patch.ndim == 2:
        patch = patch[..., None]
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(mask, [np.round((quad_uv - [x0, y0]) * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
    patch = patch * factor
    patch = patch * (1 - haze) + haze_rgb * haze
    roi = img[y0:y1, x0:x1]
    roi += (patch - roi) * mask[..., None]


def draw_scene(img: np.ndarray, geom, scene: Scene, mip, haze_rgb: np.ndarray, visibility_m: float) -> None:
    """Draw the scene's objects into `img` (float32 BGR, modified in place)."""
    cam = _Cam(geom)
    items = []                                                         # (distance, kind, index)
    for kind, en, z in (("b", scene._b_centre, scene._b_h / 2), ("t", scene.tree_en, scene.tree_z),
                        ("c", scene.car_en, np.full(len(scene.car_en), 0.7))):
        if not len(en):
            continue
        X = np.column_stack([en, z])
        uv, depth = cam.project(X)
        size = {"b": 40.0, "t": 2 * scene.tree_r if len(scene.tree_r) else 0, "c": 5.0}[kind]
        rad = cam.f * np.asarray(size) / np.maximum(depth, 1e-3)
        ok = (depth > 1.0) & (uv[:, 0] > -rad) & (uv[:, 0] < cam.w + rad) & (uv[:, 1] > -rad) & (uv[:, 1] < cam.h + rad)
        if kind == "c":
            ok &= rad > 3.0                                            # smaller cars: the painted ones suffice
        dist = np.linalg.norm(X - cam.C, axis=1)
        items += [(dist[i], kind, i) for i in np.where(ok)[0]]
    items.sort(key=lambda t: -t[0])
    sprites = tree_sprites(cam.R) if len(scene.tree_en) else None
    for dist, kind, i in items:
        haze = 1 - math.exp(-dist / visibility_m)
        if kind == "b":
            _draw_building(img, cam, scene.buildings[i], mip, haze_rgb, haze)
        elif kind == "t":
            _draw_tree(img, cam, scene, i, sprites, haze_rgb, haze)
        else:
            _draw_car(img, cam, scene, i, haze_rgb, haze)


def _faces_prism(base: np.ndarray, z0: float, z1: float):
    """Vertical faces of a prism: list of (3D quad, outward normal)."""
    n = len(base)
    out = []
    for k in range(n):
        a, b = base[k], base[(k + 1) % n]
        d = b - a
        nrm = np.array([d[1], -d[0], 0.0]) / (np.linalg.norm(d) + 1e-12)   # outward for CCW footprints
        out.append((np.array([[*a, z1], [*b, z1], [*b, z0], [*a, z0]]), nrm, np.linalg.norm(d)))
    return out


def _visible(cam, quad3, normal) -> bool:
    return float(np.dot(normal, cam.C - quad3.mean(0))) > 1e-6


def _wall(img, cam, quad3, normal, length_m, height_m, facade, colour, haze_rgb, haze):
    uv, z = cam.project(quad3)
    if np.any(z < 0.5):
        return
    f = light(normal)
    col = np.asarray(colour, np.float32) / 255.0
    area = abs(cv2.contourArea(uv.astype(np.float32)))
    ppm = math.sqrt(area / max(length_m * height_m, 1e-6))             # image px per metre of wall
    levels = facade_levels(facade)
    L = int(np.clip(math.floor(math.log2(max(PX_PER_M / max(ppm, 1e-6), 1.0))), 0, len(levels) - 1))
    if area < 40 or ppm < 1.2:
        mean = levels[-1].reshape(-1, 3).mean(0) * col * 255 * f
        _fill(img, uv, mean * (1 - haze) + haze_rgb * haze)
        return
    tex = levels[L] * (col * 255)
    s = PX_PER_M / 2 ** L
    src = np.float32([[0, 0], [length_m * s, 0], [length_m * s, height_m * s], [0, height_m * s]])
    M = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
    _warp_into(img, tex, M, uv, f, haze_rgb, haze)


def _draw_building(img, cam, b: Building, mip, haze_rgb, haze):
    fp = b.footprint
    h = b.height_m
    if b.roof == "flat":
        for quad, nrm, L in _faces_prism(fp, 0.0, h):
            if _visible(cam, quad, nrm):
                _wall(img, cam, quad, nrm, L, h, b.facade, b.wall_bgr, haze_rgb, haze)
        top = np.column_stack([fp, np.full(len(fp), h)])
        uv, z = cam.project(top)
        if np.any(z < 0.5) or cam.C[2] <= h:
            return
        area = abs(cv2.contourArea(uv.astype(np.float32)))
        fp_area = abs(cv2.contourArea(fp.astype(np.float32)))
        m_per_px = math.sqrt(fp_area / max(area, 1e-6))
        L = int(np.clip(math.floor(math.log2(max(m_per_px / mip.tex.res_m, 1.0))), 0, len(mip.levels) - 1))
        M = cam.geom.H_img_from_ground(h_m=h) @ mip.G(L)
        _warp_into(img, mip.levels[L], M, uv, 1.0, haze_rgb, haze, border=cv2.BORDER_REFLECT_101)
        return
    # gable: long walls to the eaves, end walls up to the ridge, two roof planes
    d = fp[1] - fp[0]
    if np.linalg.norm(fp[2] - fp[1]) > np.linalg.norm(d):              # ridge along the longer side
        fp = np.roll(fp, -1, axis=0)
    r = h + b.ridge_rise_m
    m03 = (fp[0] + fp[3]) / 2
    m12 = (fp[1] + fp[2]) / 2
    faces = []
    for k, (quad, nrm, L) in enumerate(_faces_prism(fp, 0.0, h)):
        if k in (1, 3):                                                # end walls: pentagons
            a, bb = quad[3, :2], quad[2, :2]
            ridge = m12 if k == 1 else m03
            poly = np.array([[*a, 0], [*bb, 0], [*bb, h], [*ridge, r], [*a, h]])
            faces.append((poly, nrm, None))
        else:
            faces.append((quad, nrm, L))
    for poly, nrm, L in faces:
        if not _visible(cam, poly, nrm):
            continue
        if L is not None:
            _wall(img, cam, poly, nrm, L, h, b.facade, b.wall_bgr, haze_rgb, haze)
        else:
            uv, z = cam.project(poly)
            if np.all(z > 0.5):
                c = np.asarray(b.wall_bgr, np.float32) * 0.92 * light(nrm)
                _fill(img, uv, c * (1 - haze) + haze_rgb * haze)
    for side in (0, 1):                                                # roof planes
        if side == 0:
            poly = np.array([[*fp[0], h], [*fp[1], h], [*m12, r], [*m03, r]])
        else:
            poly = np.array([[*fp[2], h], [*fp[3], h], [*m03, r], [*m12, r]])
        e1, e2 = poly[1] - poly[0], poly[3] - poly[0]
        nrm = np.cross(e1, e2)
        nrm /= np.linalg.norm(nrm) + 1e-12
        if nrm[2] < 0:
            nrm = -nrm
        if not _visible(cam, poly, nrm):
            continue
        uv, z = cam.project(poly)
        if np.all(z > 0.5):
            c = np.asarray(b.roof_bgr, np.float32) * light(nrm)
            _fill(img, uv, c * (1 - haze) + haze_rgb * haze)
            if abs(cv2.contourArea(uv.astype(np.float32))) > 400:      # roof sheet seams
                for t in np.linspace(0.1, 0.9, 9):
                    p0, p1 = poly[0] + (poly[1] - poly[0]) * t, poly[3] + (poly[2] - poly[3]) * t
                    q, _ = cam.project(np.array([p0, p1]))
                    cv2.line(img, tuple(np.round(q[0]).astype(int)), tuple(np.round(q[1]).astype(int)),
                             tuple(float(v) for v in c * 0.85), 1, cv2.LINE_AA)


def _draw_tree(img, cam, scene, i, sprites, haze_rgb, haze):
    en, r, zc = scene.tree_en[i], scene.tree_r[i], scene.tree_z[i]
    uv, z = cam.project(np.array([[*en, zc], [*en, 0.0]]))
    rp = cam.f * r / z[0]
    col = scene.tree_bgr[i].astype(np.float32)
    if rp > 2.5:                                                       # trunk, visible in oblique views
        tw = max(1, int(round(cam.f * 0.35 / z[0])))
        trunk = np.array([[*en, 0.0], [*en, zc - 0.6 * r]])
        tu, tz = cam.project(trunk)
        if np.all(tz > 0.5):
            c = np.array([45, 60, 80], np.float32) * (1 - haze) + haze_rgb * haze
            cv2.line(img, tuple(np.round(tu[0]).astype(int)), tuple(np.round(tu[1]).astype(int)),
                     tuple(float(v) for v in c), tw, cv2.LINE_AA)
    if rp < 1.5:
        c = col * 0.85 * (1 - haze) + haze_rgb * haze
        cv2.circle(img, tuple(np.round(uv[0]).astype(int)), max(1, int(round(rp))), tuple(float(v) for v in c), -1)
        return
    lum, alpha = sprites[i % len(sprites)]
    s = max(3, int(round(2 * rp)))
    interp = cv2.INTER_AREA if s < SPRITE else cv2.INTER_LINEAR
    lum_s = cv2.resize(lum, (s, s), interpolation=interp)            # premultiplied by alpha
    a_s = cv2.resize(alpha, (s, s), interpolation=interp)
    rgb = lum_s[..., None] * col / np.maximum(a_s, 1e-3)[..., None]
    rgb = rgb * (1 - haze) + haze_rgb * haze
    x0, y0 = int(round(uv[0, 0] - s / 2)), int(round(uv[0, 1] - s / 2))
    _blend(img, x0, y0, rgb, a_s)


def _box(img, cam, c, d, L, W, z0, z1, top_bgr, side_bgr, haze_rgb, haze):
    nrm = np.array([-d[1], d[0]])
    base = np.array([c + d * L / 2 - nrm * W / 2, c + d * L / 2 + nrm * W / 2,
                     c - d * L / 2 + nrm * W / 2, c - d * L / 2 - nrm * W / 2])   # CCW
    for quad, n, _ in _faces_prism(base, z0, z1):
        if _visible(cam, quad, n):
            uv, z = cam.project(quad)
            if np.all(z > 0.5):
                col = np.asarray(side_bgr, np.float32) * light(n)
                _fill(img, uv, col * (1 - haze) + haze_rgb * haze)
    uv, z = cam.project(np.column_stack([base, np.full(4, z1)]))
    if np.all(z > 0.5) and cam.C[2] > z1:
        _fill(img, uv, np.asarray(top_bgr, np.float32) * (1 - haze) + haze_rgb * haze)


def _draw_car(img, cam, scene, i, haze_rgb, haze):
    c, d, col, L = scene.car_en[i], scene.car_dir[i], scene.car_bgr[i], scene.car_len[i]
    glass = np.array([55, 45, 40], np.float32)
    _box(img, cam, c, d, L, 1.8, 0.25, 0.85, col, col * 0.9, haze_rgb, haze)
    _box(img, cam, c - d * 0.25, d, L * 0.5, 1.6, 0.85, 1.4, col * 0.97, glass, haze_rgb, haze)
