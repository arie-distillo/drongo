"""Camera model. Intrinsics are always derived from the actual frame size, so a
16:9 crop of a 4:3 sensor never needs a separately configured sensor height."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import ArrayN2, Mat3, as_n2

# Nominal values from public specs. Calibrate before relying on them for
# accuracy work; distortion is not included.
_PRESETS: dict[str, tuple[float, float]] = {
    # name: (sensor_w_mm, focal_mm)
    "generic_small_sensor": (6.3, 4.5),   # 1/2.3"-class sensor, 4.5 mm lens
    "equiv_24mm": (36.0, 24.0),       # any camera specified as 24 mm full-frame equivalent
    "mini2": (6.17, 4.49),
    "mini3pro": (9.6, 6.72),
    "air2s": (13.2, 8.38),
    "mavic3_wide": (17.3, 12.29),
}


@dataclass(frozen=True, slots=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    img_w_px: int
    img_h_px: int
    dist_coeffs: Optional[tuple[float, ...]] = None   # k1, k2, p1, p2, k3

    @property
    def K(self) -> Mat3:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def K_inv(self) -> Mat3:
        return np.array([[1 / self.fx, 0, -self.cx / self.fx],
                         [0, 1 / self.fy, -self.cy / self.fy], [0, 0, 1.0]])

    def scaled(self, img_w_px: int, img_h_px: int) -> "Intrinsics":
        sx, sy = img_w_px / self.img_w_px, img_h_px / self.img_h_px
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                          img_w_px, img_h_px, self.dist_coeffs)

    # ---- distortion (Brown–Conrady, OpenCV coefficient order)
    def _coeffs(self):
        d = list(self.dist_coeffs or ()) + [0.0] * 5
        return d[:5]

    def _distort_norm(self, xy: np.ndarray) -> np.ndarray:
        k1, k2, p1, p2, k3 = self._coeffs()
        x, y = xy[:, 0], xy[:, 1]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd, yd], axis=1)

    def undistort_points(self, uv: ArrayN2) -> ArrayN2:
        uv = as_n2(uv)
        if not self.dist_coeffs or not any(self.dist_coeffs):
            return uv.copy()
        xd = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy], axis=1)
        x = xd.copy()
        for _ in range(20):                       # fixed-point iteration
            x = x + (xd - self._distort_norm(x))
        return np.stack([x[:, 0] * self.fx + self.cx, x[:, 1] * self.fy + self.cy], axis=1)

    def distort_points(self, uv: ArrayN2) -> ArrayN2:
        uv = as_n2(uv)
        if not self.dist_coeffs or not any(self.dist_coeffs):
            return uv.copy()
        x = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy], axis=1)
        xd = self._distort_norm(x)
        return np.stack([xd[:, 0] * self.fx + self.cx, xd[:, 1] * self.fy + self.cy], axis=1)


@dataclass(frozen=True, slots=True)
class CameraModel:
    sensor_w_mm: float
    focal_mm: float
    dist_coeffs: Optional[tuple[float, ...]] = None
    name: str = "custom"

    @classmethod
    def preset(cls, name: str) -> "CameraModel":
        try:
            w, f = _PRESETS[name]
        except KeyError:
            raise KeyError(f"unknown camera preset {name!r}; known: {sorted(_PRESETS)}") from None
        return cls(w, f, None, name)

    @staticmethod
    def presets() -> list[str]:
        return sorted(_PRESETS)

    def intrinsics(self, img_w_px: int, img_h_px: int) -> Intrinsics:
        """Square pixels; principal point at image centre. The sensor width
        maps to the full frame width, so the effective sensor height is
        sensor_w * h / w for any aspect ratio."""
        if img_w_px <= 0 or img_h_px <= 0:
            raise ValueError("frame size must be positive")
        f_px = self.focal_mm / self.sensor_w_mm * img_w_px
        return Intrinsics(f_px, f_px, img_w_px / 2.0, img_h_px / 2.0,
                          int(img_w_px), int(img_h_px), self.dist_coeffs)

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.sensor_w_mm / (2 * self.focal_mm))))
