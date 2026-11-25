"""Telemetry-vs-vision consistency: the main metric for judging telemetry and
geometry on footage without ground truth.

For a frame pair (a, b), vision gives H_vis (KLT feature tracking + RANSAC) and
telemetry gives H_pred
(relative_homography). If pose, camera model and sync are right, both map
ground points identically. The residual, in pixels and in metres, is what the
report shows; sweeping the telemetry time offset and picking the minimum
residual estimates video/telemetry sync."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import cv2
import numpy as np

from drongo import FrameGeometry, apply_h, relative_homography


@dataclass
class VisualH:
    H: Optional[np.ndarray]
    n_tracked: int
    n_inliers: int


def estimate_homography(a: np.ndarray, b: np.ndarray, mask_top_rows: int = 0,
                        max_corners: int = 600, ransac_px: float = 1.0) -> VisualH:
    ga = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = b if b.ndim == 2 else cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    mask = np.full(ga.shape, 255, np.uint8)
    if mask_top_rows > 0:
        mask[:mask_top_rows] = 0
    mask[ga == 0] = 0
    pts = cv2.goodFeaturesToTrack(ga, max_corners, 0.01, 8, mask=mask, blockSize=7)
    if pts is None or len(pts) < 12:
        return VisualH(None, 0 if pts is None else len(pts), 0)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, winSize=(21, 21), maxLevel=4,
                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    ok = st.ravel() == 1
    src, dst = pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
    if len(src) < 12:
        return VisualH(None, len(src), 0)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    return VisualH(H, len(src), int(inl.sum()) if inl is not None else 0)


def eval_grid(g: FrameGeometry, n: int = 9) -> np.ndarray:
    """Pixel grid over the ground-visible part of frame g."""
    w, h = g.img_size
    clip = g.clip_row_px() or 0.0
    us = np.linspace(0.08 * w, 0.92 * w, n)
    vs = np.linspace(clip + 0.08 * (h - clip), 0.92 * h, n)
    return np.array([[u, v] for v in vs for u in us])


@dataclass
class PairResidual:
    t_a_ms: float
    t_b_ms: float
    n_inliers: int
    median_px: float
    p95_px: float
    median_m: float
    motion_px: float = float("nan")      # measured image motion, for scale


def pair_residual(ga: FrameGeometry, gb: FrameGeometry, H_vis: np.ndarray) -> tuple[float, float, float]:
    grid = eval_grid(ga)
    pv = apply_h(H_vis, grid)
    pp = apply_h(relative_homography(ga, gb), grid)
    d = np.linalg.norm(pv - pp, axis=1)
    gsd = gb.gsd_m_per_px(pp)
    dm = d * np.nan_to_num(gsd, nan=0.0)
    return float(np.median(d)), float(np.percentile(d, 95)), float(np.median(dm))


def motion_px(ga: FrameGeometry, H_vis: np.ndarray) -> float:
    grid = eval_grid(ga)
    return float(np.median(np.linalg.norm(apply_h(H_vis, grid) - grid, axis=1)))


def summarize(rs: Sequence[PairResidual]) -> dict:
    if not rs:
        return {"pairs": 0}
    med = np.array([r.median_px for r in rs])
    mot = [r.motion_px for r in rs if np.isfinite(r.motion_px)]
    return {"pairs": len(rs), "median_px": float(np.median(med)),
            "median_motion_px": float(np.median(mot)) if mot else None,
            "p95_px": float(np.percentile([r.p95_px for r in rs], 95)),
            "median_m": float(np.median([r.median_m for r in rs])),
            "worst_pair_median_px": float(med.max())}


def sweep_time_offset(pairs: Sequence[tuple[float, float, np.ndarray]], geom_at, offsets_ms: Sequence[int]) -> dict:
    """pairs: (t_a, t_b, H_vis). geom_at(offset_ms, t_ms) -> FrameGeometry.
    Returns per-offset median residual and the argmin."""
    scores = {}
    for off in offsets_ms:
        vals = []
        for ta, tb, H in pairs:
            ga, gb = geom_at(off, ta), geom_at(off, tb)
            if ga is None or gb is None:
                continue
            vals.append(pair_residual(ga, gb, H)[0])
        scores[int(off)] = float(np.median(vals)) if vals else float("nan")
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    best = min(finite, key=finite.get) if finite else None
    observable = None
    if finite:
        lo, hi = min(finite.values()), max(finite.values())
        observable = bool(hi > 1.10 * lo + 0.05)     # >10% contrast across the sweep
    return {"scores": scores, "best_offset_ms": best if observable else None,
            "argmin_ms": best, "observable": observable}
