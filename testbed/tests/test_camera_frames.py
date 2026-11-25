import numpy as np
import pytest

from drongo import CameraModel, LocalFrame

LAT, LON = 31.7683, 35.2137


def test_intrinsics_aspect_derived_from_frame():
    cam = CameraModel(6.3, 4.5)
    k169 = cam.intrinsics(3840, 2160)
    k43 = cam.intrinsics(4000, 3000)
    # same horizontal FOV, square pixels, vertical FOV follows the frame shape
    assert k169.fx == pytest.approx(4.5 / 6.3 * 3840)
    assert k169.fx == k169.fy
    vfov169 = 2 * np.arctan(2160 / 2 / k169.fy)
    vfov43 = 2 * np.arctan(3000 / 2 / k43.fy)
    assert np.tan(vfov169 / 2) / np.tan(vfov43 / 2) == pytest.approx((9 / 16) / (3 / 4))


def test_intrinsics_scaling_consistent():
    cam = CameraModel(6.3, 4.5)
    assert cam.intrinsics(3840, 2160).scaled(960, 540).fx == pytest.approx(cam.intrinsics(960, 540).fx)


def test_distortion_roundtrip():
    k = CameraModel(6.3, 4.5, dist_coeffs=(-0.12, 0.05, 0.001, -0.0005, 0.0)).intrinsics(3840, 2160)
    uv = np.array([[0, 0], [3840, 2160], [1920, 1080], [100, 2000], [3500, 300]], float)
    assert np.allclose(k.distort_points(k.undistort_points(uv)), uv, atol=1e-6)


def test_presets_exist():
    for name in CameraModel.presets():
        assert CameraModel.preset(name).hfov_deg > 40
    with pytest.raises(KeyError):
        CameraModel.preset("nope")


def test_local_frame_roundtrip():
    fr = LocalFrame(LAT, LON)
    en = np.random.default_rng(0).uniform(-2000, 2000, (100, 2))
    assert np.allclose(fr.to_enu(fr.to_wgs84(en)), en, atol=1e-6)


def test_local_frame_against_geodesic():
    pyproj = pytest.importorskip("pyproj")
    geod = pyproj.Geod(ellps="WGS84")
    fr = LocalFrame(LAT, LON)
    rng = np.random.default_rng(1)
    for _ in range(200):
        e, n = rng.uniform(-3000, 3000, 2)
        lat, lon = fr.to_wgs84([[e, n]])[0]
        _, _, d = geod.inv(LON, LAT, lon, lat)
        assert abs(d - np.hypot(e, n)) < 0.01
