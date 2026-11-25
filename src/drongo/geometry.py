"""Per-frame geometry: the image <-> ground mapping for one camera pose."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .camera import CameraModel, Intrinsics
from .frames import FlatGround, GroundModel, LocalFrame
from .types import ArrayN2, Mat3, PoseSample, as_n2


def rotation_world_from_cam(yaw_deg: float, pitch_deg: float, roll_deg: float) -> Mat3:
    """Columns are the camera axes (OpenCV: x right, y down, z forward)
    expressed in ENU.

    yaw 0 / pitch 0  -> looking North, image-up = Up.
    pitch -90         -> nadir, image-up = heading direction.
    """
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    f = np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)])
    rt = np.array([np.cos(y), -np.sin(y), 0.0])
    d = np.cross(f, rt)
    rt_r = rt * np.cos(r) + d * np.sin(r)
    d_r = d * np.cos(r) - rt * np.sin(r)
    return np.stack([rt_r, d_r, f], axis=1)


@dataclass(frozen=True, slots=True)
class Footprint:
    corners_en: ArrayN2          # (4, 2) TL, TR, BR, BL of the visible ground region
    corners_lat_lon: ArrayN2
    clip_row_px: float           # frame row where visible ground starts (= y0 if unclipped)
    area_m2: float

    def contains_en(self, en: ArrayN2) -> np.ndarray:
        """Point-in-convex-quad test (corners are in image order)."""
        en = as_n2(en)
        c = self.corners_en
        sign = None
        inside = np.ones(len(en), bool)
        for i in range(4):
            a, b = c[i], c[(i + 1) % 4]
            cross = (b[0] - a[0]) * (en[:, 1] - a[1]) - (b[1] - a[1]) * (en[:, 0] - a[0])
            s = np.sign(np.sum(cross)) if sign is None else sign
            sign = s
            inside &= cross * s >= -1e-9
        return inside

    def to_geojson(self, properties: Optional[dict] = None) -> dict:
        ring = [[float(lo), float(la)] for la, lo in self.corners_lat_lon]
        ring.append(ring[0])
        return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": properties or {}}


def _shoelace(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


_UNSET = object()


class FrameGeometry:
    """Immutable image <-> ground mapping for one frame.

    Pixel coordinates are raw (distorted) image coordinates in all point
    methods. Homographies (H_*) are defined on undistorted pixels; with no
    distortion model the two coincide.
    """

    __slots__ = ("pose", "intrinsics", "frame", "ground", "_R", "_C", "_clip",
                 "min_depression_deg", "max_range_m")

    def __init__(self, pose: PoseSample, intrinsics: Intrinsics, frame: LocalFrame,
                 ground: GroundModel = FlatGround(), *, R_world_from_cam: Optional[Mat3] = None,
                 camera_enu_m: Optional[np.ndarray] = None,
                 min_depression_deg: float = 1.0, max_range_m: Optional[float] = None) -> None:
        self.pose = pose
        self.intrinsics = intrinsics
        self.frame = frame
        self.ground = ground
        self._R = (R_world_from_cam if R_world_from_cam is not None else
                   rotation_world_from_cam(pose.gimbal_yaw_deg, pose.gimbal_pitch_deg, pose.gimbal_roll_deg))
        if camera_enu_m is None:
            e, n = frame.to_enu([[pose.lat_deg, pose.lon_deg]])[0]
            camera_enu_m = np.array([e, n, pose.alt_agl_m])
        self._C = np.asarray(camera_enu_m, float)
        self.min_depression_deg = min_depression_deg
        self.max_range_m = max_range_m
        self._clip: object = _UNSET

    @classmethod
    def from_pose(cls, pose: PoseSample, camera: CameraModel, img_w_px: int, img_h_px: int,
                  frame: LocalFrame, ground: GroundModel = FlatGround(), **kw) -> "FrameGeometry":
        return cls(pose, camera.intrinsics(img_w_px, img_h_px), frame, ground, **kw)

    # ---- camera
    @property
    def camera_enu_m(self) -> np.ndarray:
        return self._C.copy()

    @property
    def R_world_from_cam(self) -> Mat3:
        return self._R.copy()

    @property
    def img_size(self) -> tuple[int, int]:
        return self.intrinsics.img_w_px, self.intrinsics.img_h_px

    def rays_world(self, uv: ArrayN2) -> np.ndarray:
        """Unit ray directions (N, 3) in ENU for raw pixels."""
        und = self.intrinsics.undistort_points(uv)
        h = np.column_stack([und, np.ones(len(und))])
        r = (self._R @ (self.intrinsics.K_inv @ h.T)).T
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    # ---- pixel <-> ground
    def pixel_to_en(self, uv: ArrayN2, h_m=0.0) -> ArrayN2:
        """Intersect pixel rays with the ground surface raised by h_m.
        Rays that do not hit it in front of the camera return NaN."""
        uv = as_n2(uv)
        r = self.rays_world(uv)
        h = np.broadcast_to(np.asarray(h_m, float), (len(uv),))
        out = np.full((len(uv), 2), np.nan)
        z = self.ground.height_m(np.tile(self._C[:2], (len(uv), 1))) + h
        for _ in range(1 if self.ground.is_planar else 8):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (z - self._C[2]) / r[:, 2]
            ok = (r[:, 2] < 0) & (t > 0)
            pts = self._C[:2] + t[:, None] * r[:, :2]
            out = np.where(ok[:, None], pts, np.nan)
            if self.ground.is_planar:
                break
            z = np.where(ok, self.ground.height_m(np.nan_to_num(out)) + h, z)
        return out

    def en_to_pixel(self, en: ArrayN2, h_m=0.0) -> ArrayN2:
        en = as_n2(en)
        h = np.broadcast_to(np.asarray(h_m, float), (len(en),))
        X = np.column_stack([en, self.ground.height_m(en) + h])
        xc = (self._R.T @ (X - self._C).T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (self.intrinsics.K @ (xc / xc[:, 2:3]).T).T[:, :2]
        uv[xc[:, 2] <= 0] = np.nan
        return self.intrinsics.distort_points(uv)

    def pixel_to_lat_lon(self, uv: ArrayN2, h_m=0.0) -> ArrayN2:
        return self.frame.to_wgs84(self.pixel_to_en(uv, h_m))

    def lat_lon_to_pixel(self, lat_lon: ArrayN2, h_m=0.0) -> ArrayN2:
        return self.en_to_pixel(self.frame.to_enu(lat_lon), h_m)

    # ---- homographies (planar ground at level + h_m)
    def H_img_from_ground(self, h_m: float = 0.0) -> Mat3:
        z0 = float(self.ground.height_m(self._C[:2][None])[0]) + h_m
        P = self.intrinsics.K @ np.column_stack([self._R.T, -self._R.T @ self._C])
        H = np.column_stack([P[:, 0], P[:, 1], z0 * P[:, 2] + P[:, 3]])
        return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    def H_ground_from_img(self, h_m: float = 0.0) -> Mat3:
        H = np.linalg.inv(self.H_img_from_ground(h_m))
        return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    # ---- visibility
    def _row_ok(self, v: float) -> bool:
        w = self.intrinsics.img_w_px
        uv = np.array([[0.0, v], [w / 2, v], [float(w), v]])
        r = self.rays_world(uv)
        if np.any(r[:, 2] > -np.sin(np.radians(self.min_depression_deg))):
            return False
        if self.max_range_m is not None:
            rng = np.linalg.norm(self.pixel_to_en(uv) - self._C[:2], axis=1)
            if np.any(~np.isfinite(rng)) or np.any(rng > self.max_range_m):
                return False
        return True

    def clip_row_px(self) -> Optional[float]:
        """First image row whose whole width sees ground (within the depression
        and range limits). 0 for a fully-ground frame, None if no row does.
        Computed once per FrameGeometry (the object is immutable)."""
        if self._clip is _UNSET:
            self._clip = self._compute_clip_row()
        return self._clip

    def _compute_clip_row(self) -> Optional[float]:
        h = float(self.intrinsics.img_h_px)
        if self._row_ok(0.0):
            return 0.0
        if not self._row_ok(h):
            return None
        lo, hi = 0.0, h
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if self._row_ok(mid) else (mid, hi)
        return hi

    def horizon_row_px(self) -> Optional[float]:
        """Lowest image row (over the frame width) where rays are horizontal.
        None if the horizon is above the frame (all ground)."""
        w, h = self.intrinsics.img_w_px, self.intrinsics.img_h_px
        rows = []
        Kinv = self.intrinsics.K_inv
        for u in (0.0, w / 2, float(w)):
            # ray_z(v) = a + b*v  (linear in v for fixed u, undistorted)
            a = self._R[2] @ (Kinv @ np.array([u, 0.0, 1.0]))
            b = self._R[2] @ (Kinv @ np.array([0.0, 1.0, 0.0]))
            if abs(b) < 1e-15:
                continue
            rows.append(-a / b)
        if not rows:
            return None
        v = max(rows)
        return None if v < 0 else min(v, float(h))

    # ---- footprints
    def footprint(self) -> Optional[Footprint]:
        w, h = self.img_size
        return self.tile_footprint(0, 0, w, h)

    def tile_footprint(self, x0: float, y0: float, x1: float, y1: float) -> Optional[Footprint]:
        clip = self.clip_row_px()
        if clip is None or clip >= y1:
            return None
        y0c = max(float(y0), clip)
        uv = np.array([[x0, y0c], [x1, y0c], [x1, y1], [x0, y1]], float)
        en = self.pixel_to_en(uv)
        if not np.all(np.isfinite(en)):
            return None
        return Footprint(en, self.frame.to_wgs84(en), y0c, _shoelace(en))

    # ---- derived quantities
    def gsd_m_per_px(self, uv: Optional[ArrayN2] = None) -> np.ndarray:
        if uv is None:
            uv = [[self.intrinsics.cx, self.intrinsics.cy]]
        uv = as_n2(uv)
        p = self.pixel_to_en(uv)
        dx = self.pixel_to_en(uv + [1.0, 0.0]) - p
        dy = self.pixel_to_en(uv + [0.0, 1.0]) - p
        area = np.abs(dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0])
        return np.sqrt(area)

    def ground_range_m(self, uv: ArrayN2) -> np.ndarray:
        return np.linalg.norm(self.pixel_to_en(uv) - self._C[:2], axis=1)

    def view_angle_deg(self, uv: ArrayN2) -> np.ndarray:
        """Angle between the pixel ray and straight down."""
        r = self.rays_world(uv)
        return np.degrees(np.arccos(np.clip(-r[:, 2], -1, 1)))


def relative_homography(a: FrameGeometry, b: FrameGeometry, h_m: float = 0.0) -> Mat3:
    """Ground-plane-induced homography mapping image a -> image b (undistorted px)."""
    H = b.H_img_from_ground(h_m) @ a.H_ground_from_img(h_m)
    return H / H[2, 2]


def apply_h(H: Mat3, pts: ArrayN2) -> ArrayN2:
    p = as_n2(pts)
    q = (H @ np.column_stack([p, np.ones(len(p))]).T).T
    return q[:, :2] / q[:, 2:3]
