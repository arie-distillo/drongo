"""Thin ground-plane rectification helper (requires the [cv] extra)."""
from __future__ import annotations

import numpy as np

from .geometry import FrameGeometry


def ground_from_output(roi_en: tuple[float, float, float, float], gsd_m: float) -> np.ndarray:
    """Matrix mapping output raster pixel (col, row) -> ENU, north-up."""
    e0, n0, e1, n1 = roi_en
    return np.array([[gsd_m, 0, e0 + gsd_m / 2], [0, -gsd_m, n1 - gsd_m / 2], [0, 0, 1.0]])


def rectify_roi(image: np.ndarray, geom: FrameGeometry,
                roi_en: tuple[float, float, float, float], gsd_m: float,
                h_m: float = 0.0, border_value=0) -> np.ndarray:
    """North-up ground-plane rectification of an ENU rectangle (E0, N0, E1, N1)."""
    import cv2

    e0, n0, e1, n1 = roi_en
    w = int(round((e1 - e0) / gsd_m))
    h = int(round((n1 - n0) / gsd_m))
    if w <= 0 or h <= 0:
        raise ValueError("empty ROI")
    intr = geom.intrinsics
    src = image
    if intr.dist_coeffs and any(intr.dist_coeffs):
        src = cv2.undistort(image, intr.K, np.array(intr.dist_coeffs, float))
    M = geom.H_img_from_ground(h_m) @ ground_from_output(roi_en, gsd_m)
    return cv2.warpPerspective(src, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
