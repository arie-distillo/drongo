"""Local metric frame (ENU tangent plane) and ground models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .types import ArrayN2, as_n2

_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def _radii(lat_rad):
    s2 = np.sin(lat_rad) ** 2
    m = _WGS84_A * (1 - _WGS84_E2) / (1 - _WGS84_E2 * s2) ** 1.5   # meridian
    n = _WGS84_A / np.sqrt(1 - _WGS84_E2 * s2)                    # prime vertical
    return m, n


@dataclass(frozen=True, slots=True)
class LocalFrame:
    """Local metric plane (E, N) anchored at a reference point.

    Mid-latitude projection on the WGS84 ellipsoid: N = dlat * M(lat_mid),
    E = dlon * N(lat_mid) * cos(lat_mid). Distances agree with the geodesic
    to < 1 mm within ~10 km of the reference (test_camera_frames.py). Earth
    curvature (ground drops ~d^2/2R, 0.3 m at 2 km) is ignored, consistent
    with the flat-ground model.
    """
    ref_lat_deg: float
    ref_lon_deg: float
    ref_alt_m: float = 0.0

    @property
    def m_per_deg(self) -> tuple[float, float]:
        """(metres per degree latitude, metres per degree longitude) at the reference."""
        phi = np.radians(self.ref_lat_deg)
        m, n = _radii(phi)
        return float(np.radians(1) * m), float(np.radians(1) * n * np.cos(phi))

    def to_enu(self, lat_lon: ArrayN2) -> ArrayN2:
        ll = np.radians(as_n2(lat_lon))
        phi0, lam0 = np.radians(self.ref_lat_deg), np.radians(self.ref_lon_deg)
        phim = (ll[:, 0] + phi0) / 2
        m, n = _radii(phim)
        return np.stack([(ll[:, 1] - lam0) * n * np.cos(phim), (ll[:, 0] - phi0) * m], axis=1)

    def to_wgs84(self, en: ArrayN2) -> ArrayN2:
        en = as_n2(en)
        phi0, lam0 = np.radians(self.ref_lat_deg), np.radians(self.ref_lon_deg)
        phi = phi0 + en[:, 1] / _radii(phi0)[0]
        for _ in range(4):                                   # converges to < 1e-12 rad
            phim = (phi + phi0) / 2
            m, n = _radii(phim)
            phi = phi0 + en[:, 1] / m
        phim = (phi + phi0) / 2
        _, n = _radii(phim)
        lam = lam0 + en[:, 0] / (n * np.cos(phim))
        return np.degrees(np.stack([phi, lam], axis=1))

    def distance_m(self, a_lat_lon: ArrayN2, b_lat_lon: ArrayN2) -> np.ndarray:
        return np.linalg.norm(self.to_enu(a_lat_lon) - self.to_enu(b_lat_lon), axis=1)


class GroundModel(Protocol):
    def height_m(self, en: ArrayN2) -> np.ndarray: ...

    @property
    def is_planar(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class FlatGround:
    """Ground plane at a fixed height relative to the takeoff point (z = 0)."""
    level_m: float = 0.0

    def height_m(self, en: ArrayN2) -> np.ndarray:
        return np.full(as_n2(en).shape[0], self.level_m)

    @property
    def is_planar(self) -> bool:
        return True
