import numpy as np
import pytest

from drongo import CameraModel, FlatGround, apply_h, relative_homography, rotation_world_from_cam

from .conftest import H, W, random_poses


# ---------- rotation conventions

@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (37, -45, 3), (270, -90, -5), (359.9, -10, 0)])
def test_rotation_is_proper(yaw, pitch, roll):
    R = rotation_world_from_cam(yaw, pitch, roll)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_rotation_conventions():
    R = rotation_world_from_cam(0, 0, 0)            # looking north, level
    assert np.allclose(R[:, 2], [0, 1, 0]) and np.allclose(R[:, 0], [1, 0, 0]) and np.allclose(R[:, 1], [0, 0, -1])
    R = rotation_world_from_cam(90, 0, 0)           # looking east, right = south
    assert np.allclose(R[:, 2], [1, 0, 0]) and np.allclose(R[:, 0], [0, -1, 0])
    R = rotation_world_from_cam(0, -90, 0)          # nadir, image-up = north
    assert np.allclose(R[:, 2], [0, 0, -1]) and np.allclose(-R[:, 1], [0, 1, 0])
    R = rotation_world_from_cam(0, 0, 10)           # roll right: image-right goes down
    assert R[2, 0] < 0


# ---------- nadir analytic checks

def test_nadir_footprint_analytic(make_geom, cam):
    g = make_geom(alt=85, yaw=0)
    gw = 85 * cam.sensor_w_mm / cam.focal_mm
    gh = gw * H / W
    c = g.footprint().corners_en
    assert np.allclose(c, [[-gw / 2, gh / 2], [gw / 2, gh / 2], [gw / 2, -gh / 2], [-gw / 2, -gh / 2]], atol=1e-6)
    assert g.footprint().area_m2 == pytest.approx(gw * gh)
    assert g.gsd_m_per_px()[0] == pytest.approx(gw / W, rel=1e-6)
    assert g.clip_row_px() == 0 and g.horizon_row_px() is None


def test_nadir_yaw_rotates_footprint(make_geom):
    c0 = make_geom(yaw=0).footprint().corners_en
    c90 = make_geom(yaw=90).footprint().corners_en
    rot = np.array([[0, 1], [-1, 0]])               # rotate (E,N) 90 deg clockwise
    assert np.allclose(c90, c0 @ rot.T, atol=1e-6)


def test_image_top_is_heading_at_nadir(make_geom):
    g = make_geom(yaw=60)
    top = g.pixel_to_en([[W / 2, 0]])[0]
    assert np.degrees(np.arctan2(top[0], top[1])) == pytest.approx(60)


# ---------- consistency over random poses

def test_pixel_ground_roundtrip_random(make_geom):
    rng = np.random.default_rng(2)
    for kw in random_poses(150):
        g = make_geom(**kw)
        clip = g.clip_row_px()
        assert clip is not None
        uv = np.column_stack([rng.uniform(0, W, 50), rng.uniform(clip + 1, H, 50)])
        for h_m in (0.0, 12.0):
            back = g.en_to_pixel(g.pixel_to_en(uv, h_m), h_m)
            assert np.allclose(back, uv, atol=1e-6)


def test_homography_matches_ray_projection(make_geom):
    rng = np.random.default_rng(3)
    for kw in random_poses(80, seed=4):
        g = make_geom(**kw)
        uv = np.column_stack([rng.uniform(0, W, 30), rng.uniform(g.clip_row_px() + 1, H, 30)])
        for h_m in (0.0, 7.5):
            assert np.allclose(apply_h(g.H_ground_from_img(h_m), uv), g.pixel_to_en(uv, h_m), atol=1e-6)
            en = g.pixel_to_en(uv, h_m)
            assert np.allclose(apply_h(g.H_img_from_ground(h_m), en), uv, atol=1e-6)


def test_relative_homography(make_geom):
    a = make_geom(yaw=10, pitch=-70, e=0, n=0)
    b = make_geom(yaw=14, pitch=-68, e=3, n=5)
    assert np.allclose(relative_homography(a, a), np.eye(3), atol=1e-9)
    uv = np.array([[500, 1500], [2000, 1000], [3500, 2000]], float)
    assert np.allclose(apply_h(relative_homography(a, b), uv), b.en_to_pixel(a.pixel_to_en(uv)), atol=1e-6)


def test_height_corrected_projection(make_geom):
    """A point h metres above ground, seen at pixel uv, lies where the ray
    crosses z = h: closer to the camera nadir by factor (alt-h)/alt."""
    g = make_geom(alt=85, yaw=0)
    uv = [[3000, 300]]
    p0 = g.pixel_to_en(uv, 0.0)[0]
    ph = g.pixel_to_en(uv, 25.0)[0]
    assert np.allclose(ph, p0 * (85 - 25) / 85)


def test_tiles_partition_frame(make_geom):
    g = make_geom(yaw=30, pitch=-60)
    full = g.footprint().area_m2
    xs, ys = np.linspace(0, W, 5), np.linspace(0, H, 4)
    area = sum(g.tile_footprint(x0, y0, x1, y1).area_m2
               for x0, x1 in zip(xs[:-1], xs[1:]) for y0, y1 in zip(ys[:-1], ys[1:]))
    assert area == pytest.approx(full, rel=1e-9)


def test_footprint_contains_projected_points(make_geom):
    g = make_geom(yaw=200, pitch=-65, e=40)
    fp = g.footprint()
    inside = g.pixel_to_en([[100, 100], [W / 2, H / 2], [W - 5, H - 5]])
    assert fp.contains_en(inside).all()
    assert not fp.contains_en(g.camera_enu_m[:2][None] + [[5000, 0]]).any()


# ---------- oblique / horizon

def test_horizon_and_clip(make_geom, cam):
    g = make_geom(pitch=-10)                      # horizon inside the frame
    k = cam.intrinsics(W, H)
    expected = k.cy - np.tan(np.radians(10)) * k.fy   # horizon row at centre column, no roll
    assert g.horizon_row_px() == pytest.approx(expected, abs=1e-6)
    clip = g.clip_row_px()
    assert clip > g.horizon_row_px()
    fp = g.footprint()
    assert fp.clip_row_px == pytest.approx(clip)
    assert np.all(np.isfinite(fp.corners_en))
    assert np.isnan(g.pixel_to_en([[W / 2, 0]])).all()     # sky pixel


def test_max_range_clip(make_geom):
    g = make_geom(pitch=-30, max_range_m=300)
    fp = g.footprint()
    rng = np.linalg.norm(fp.corners_en - g.camera_enu_m[:2], axis=1)
    assert rng.max() <= 300 + 1e-3


def test_no_ground_visible(make_geom):
    g = make_geom(pitch=30)       # bottom edge still ~8 deg above horizon
    assert g.footprint() is None and g.clip_row_px() is None


def test_gsd_grows_with_range(make_geom):
    g = make_geom(pitch=-40)
    near, far = g.gsd_m_per_px([[W / 2, H - 1], [W / 2, g.clip_row_px() + 1]])
    assert far > near * 2
    assert g.view_angle_deg([[W / 2, H / 2]])[0] == pytest.approx(50)


def test_ground_level(make_geom):
    """Ground 10 m above takeoff == flying 10 m lower."""
    a = make_geom(alt=85, pitch=-60, ground=FlatGround(10.0))
    b = make_geom(alt=75, pitch=-60)
    uv = [[100, 900], [3000, 2000]]
    assert np.allclose(a.pixel_to_en(uv), b.pixel_to_en(uv), atol=1e-9)


def test_distortion_consistency(make_geom):
    cam = CameraModel(6.3, 4.5, dist_coeffs=(-0.1, 0.03, 0, 0, 0))
    g = make_geom(pitch=-70, camera=cam)
    uv = np.array([[50, 1900], [3800, 1200], [1920, 1080]])
    assert np.allclose(g.en_to_pixel(g.pixel_to_en(uv)), uv, atol=1e-5)
    # undistorted centre pixel is unaffected
    plain = make_geom(pitch=-70)
    assert np.allclose(g.pixel_to_en([[1920, 1080]]), plain.pixel_to_en([[1920, 1080]]))
