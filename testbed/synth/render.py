"""Render synthetic drone frames from a georeferenced ground texture, using an
exact pose. The renderer uses drongo's H_img_from_ground, so rendering
correctness is established separately by unit tests (projection of known
texture points) before it is used to test anything else."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from drongo import CameraModel, FrameGeometry, write_dji_srt

from .scenario import Scenario, TelemetryNoise


@dataclass
class GroundTexture:
    image: np.ndarray            # uint8, H x W (grayscale) or H x W x 3
    e0: float                    # ENU of the texture's west edge
    n1: float                    # ENU of the texture's north edge
    res_m: float                 # metres per texture pixel
    scene: Optional[object] = None   # 3D objects standing on the ground (scene.Scene)

    @property
    def G_ground_from_tex(self) -> np.ndarray:
        r = self.res_m
        return np.array([[r, 0, self.e0 + r / 2], [0, -r, self.n1 - r / 2], [0, 0, 1.0]])

    def tex_px(self, en: np.ndarray) -> np.ndarray:
        en = np.atleast_2d(en)
        return np.stack([(en[:, 0] - self.e0) / self.res_m - 0.5, (self.n1 - en[:, 1]) / self.res_m - 0.5], 1)


def make_texture(half_size_m: float = 350.0, res_m: float = 0.1, seed: int = 0) -> GroundTexture:
    """Fractal noise + random flat 'roofs', 'roads' and markers: plenty of
    corners for feature tracking, no repetitive pattern that could alias."""
    rng = np.random.default_rng(seed)
    n = int(2 * half_size_m / res_m)
    img = np.zeros((n, n), np.float32)
    for octave, amp in ((8, 0.35), (32, 0.25), (128, 0.2), (512, 0.12), (2048, 0.08)):
        small = rng.random((octave, octave)).astype(np.float32)
        img += amp * cv2.resize(small, (n, n), interpolation=cv2.INTER_CUBIC)
    img = (img - img.min()) / (img.max() - img.min()) * 160 + 40
    img = img.astype(np.uint8)
    for _ in range(int(n * n / 150_000)):                     # roofs / yards
        w, h = rng.integers(40, 250, 2)
        x, y = rng.integers(0, n - 260, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), int(rng.integers(0, 255)), -1)
    for _ in range(int(n / 120)):                              # roads
        p1, p2 = rng.integers(0, n, 2), rng.integers(0, n, 2)
        cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), int(rng.integers(0, 60)), int(rng.integers(20, 70)))
    for _ in range(int(n * n / 20_000)):                       # small markers
        c = rng.integers(0, n, 2)
        cv2.circle(img, tuple(map(int, c)), int(rng.integers(4, 15)), int(rng.integers(0, 255)), -1)
    return GroundTexture(img, -half_size_m, half_size_m, res_m)


def render(tex: GroundTexture, geom: FrameGeometry) -> np.ndarray:
    """Ground-plane render of the texture as seen by `geom` (no distortion).
    Pixels whose rays miss the ground (sky) are 0."""
    w, h = geom.img_size
    M = geom.H_img_from_ground() @ tex.G_ground_from_tex
    out = cv2.warpPerspective(tex.image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    clip = geom.clip_row_px()
    if clip is None:
        out[:] = 0
    elif clip > 0:
        out[: int(np.ceil(clip))] = 0
    return out


def write_video(sc: Scenario, tex: GroundTexture, camera: CameraModel, out_dir: Path,
                img_w: int = 960, img_h: int = 540, noise: Optional[TelemetryNoise] = None) -> tuple[Path, Path]:
    """Render a scenario to <out>/<name>.mp4 + matching DJI-style .SRT.
    The video uses ground-truth poses; the SRT carries noisy telemetry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = out_dir / f"{sc.name}.mp4"
    srt = out_dir / f"{sc.name}.SRT"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), sc.fps, (img_w, img_h))
    for p in sc.poses:
        g = FrameGeometry.from_pose(p, camera, img_w, img_h, sc.frame)
        vw.write(cv2.cvtColor(render(tex, g), cv2.COLOR_GRAY2BGR))
    vw.release()
    samples = (noise or TelemetryNoise()).apply(sc)
    srt.write_text(write_dji_srt(samples, sc.frame_ms), encoding="utf-8")
    return vid, srt


# ---------------------------------------------------------------------------- colour rendering for demo clips

@dataclass
class Look:
    """Atmosphere and camera appearance. Affects pixels only, never geometry."""
    visibility_m: float = 3000.0                  # haze: 63 % veiled at this distance
    haze_bgr: tuple = (222, 205, 190)             # colour of the horizon haze
    zenith_bgr: tuple = (200, 140, 95)            # sky colour high above the horizon
    vignette: float = 0.22                        # brightness loss in the corners
    noise_sigma: float = 1.5                      # sensor noise, 8-bit levels
    contrast: float = 1.05
    seed: int = 0


class MipTexture:
    """A GroundTexture with a mip pyramid, so a frame samples each region
    from the level whose texels match its pixels' ground size (trilinear).
    Without this, oblique views alias badly towards the horizon."""

    def __init__(self, tex: GroundTexture, min_size: int = 256):
        self.tex = tex
        img = tex.image if tex.image.ndim == 3 else cv2.cvtColor(tex.image, cv2.COLOR_GRAY2BGR)
        self.levels = [img]
        while min(self.levels[-1].shape[:2]) > 2 * min_size:
            self.levels.append(cv2.pyrDown(self.levels[-1]))

    def G(self, level: int) -> np.ndarray:
        t = self.tex
        return GroundTexture(self.levels[level], t.e0, t.n1, t.res_m * 2 ** level).G_ground_from_tex


def render_rgb(mip: MipTexture, geom: FrameGeometry, look: Look = Look(), grid: int = 16,
               rng: Optional[np.random.Generator] = None, objects: bool = True) -> np.ndarray:
    """Colour frame (BGR uint8) as the camera of `geom` sees the texture,
    with sky above the horizon and distance haze. Ground pixels are placed
    by drongo's ground->image homography, exactly as in `render`; the
    texture's 3D objects (if any) are then drawn over the ground."""
    w, h = geom.img_size
    K = geom.intrinsics
    # per-pixel quantities on a coarse grid, upsampled afterwards
    gu = np.linspace(0, w, w // grid + 1)
    gv = np.linspace(0, h, h // grid + 1)
    uu, vv = np.meshgrid(gu, gv)
    r = geom.rays_world(np.column_stack([uu.ravel(), vv.ravel()]))
    rz = r[:, 2].reshape(uu.shape)
    alt = max(float(geom.camera_enu_m[2]), 0.1)
    down = np.clip(-rz, 1e-4, 1.0)
    dist = alt / down                                              # slant range to the ground
    pix_ang = 1.0 / np.sqrt(K.fx * K.fy)
    gsd = dist * pix_ang / np.sqrt(down)                           # geometric mean of the footprint axes
    lvl = np.clip(np.log2(np.maximum(gsd / mip.tex.res_m, 1e-6)), 0, len(mip.levels) - 1)
    ground = np.clip((-rz - 0.002) / 0.004, 0, 1)                  # soft edge at the horizon
    haze = 1 - np.exp(-dist / look.visibility_m)
    el = np.clip(rz, 0, 1)
    up = lambda a: cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    lvl_f, ground_f, haze_f, el_f = up(lvl), up(ground), up(haze), up(el)

    out = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)
    H_ig = geom.H_img_from_ground()
    has_ground = ground_f > 0
    if has_ground.any():
        rows = np.where(has_ground.any(1))[0]
        lmin, lmax = int(np.floor(lvl_f[has_ground].min())), int(np.ceil(lvl_f[has_ground].max()))
        for L in range(lmin, min(lmax, len(mip.levels) - 1) + 1):
            wt = np.clip(1 - np.abs(lvl_f - L), 0, 1) * has_ground
            band = np.where(wt.any(1))[0]
            if not len(band):
                continue
            y0, y1 = int(band[0]), int(band[-1]) + 1
            T = np.array([[1, 0, 0], [0, 1, -y0], [0, 0, 1.0]])
            img = cv2.warpPerspective(mip.levels[L], T @ H_ig @ mip.G(L), (w, y1 - y0), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
            out[y0:y1] += img.astype(np.float32) * wt[y0:y1, :, None]
            wsum[y0:y1] += wt[y0:y1]
        out /= np.maximum(wsum, 1e-6)[..., None]
        out = (out - 128) * look.contrast + 128
    hz = np.array(look.haze_bgr, np.float32)
    sky = hz + (np.array(look.zenith_bgr, np.float32) - hz) * (np.clip(el_f / 0.5, 0, 1) ** 0.6)[..., None]
    out = out * (1 - haze_f[..., None]) + hz * haze_f[..., None]
    out = out * ground_f[..., None] + sky * (1 - ground_f[..., None])
    if mip.tex.scene is not None and objects:
        from .scene import draw_scene
        draw_scene(out, geom, mip.tex.scene, mip, hz, look.visibility_m)
    if look.vignette:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rr = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        out *= (1 - look.vignette * rr ** 1.5)[..., None]
    if look.noise_sigma:
        rng = rng or np.random.default_rng(look.seed)
        out += rng.normal(0, look.noise_sigma, (h, w)).astype(np.float32)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)
