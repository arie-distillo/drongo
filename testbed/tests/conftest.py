import os

import numpy as np
import pytest

from drongo import CameraModel, FrameGeometry, LocalFrame, PoseSample

LAT, LON = 31.7683, 35.2137
W, H = 3840, 2160


@pytest.fixture
def frame():
    return LocalFrame(LAT, LON)


@pytest.fixture
def cam():
    return CameraModel.preset("generic_small_sensor")


@pytest.fixture
def make_geom(frame, cam):
    def _make(alt=85.0, yaw=0.0, pitch=-90.0, roll=0.0, e=0.0, n=0.0, w=W, h=H, camera=None, **kw):
        lat, lon = frame.to_wgs84([[e, n]])[0]
        p = PoseSample(0, float(lat), float(lon), alt, yaw, pitch, roll)
        return FrameGeometry.from_pose(p, camera or cam, w, h, frame, **kw)
    return _make


def random_poses(n=200, seed=0):
    """Random but valid poses: camera looking at least 10 deg below horizon at the top edge."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield dict(alt=float(rng.uniform(20, 200)), yaw=float(rng.uniform(0, 360)),
                   pitch=float(rng.uniform(-90, -55)), roll=float(rng.uniform(-5, 5)),
                   e=float(rng.uniform(-500, 500)), n=float(rng.uniform(-500, 500)))

