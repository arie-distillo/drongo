# API reference

Everything below is importable from `drongo` unless noted. Point arguments are numpy arrays of shape
(N, 2) (a single point may be given as a flat pair); results are float64 arrays. Units are in the
names: `_m` metres, `_deg` degrees, `_ms` milliseconds, `_px` pixels.

## Telemetry

### `open_telemetry(video, sidecar=None, *, readers=None, diagnostics=None) -> TelemetryTrack | None`
Tries `readers` in order (default: `DjiSrtReader()`, then `DjiEmbeddedReader()`); the first that
yields samples wins. Appends one explanatory line per reader to `diagnostics` if given.

### `class PoseSample` (frozen)
`t_ms, lat_deg, lon_deg, alt_agl_m, gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg=0.0,
alt_msl_m=None, body_yaw_deg=None, velocity_enu_ms=None, source=TelemetrySourceKind.MANUAL`.
`with_(**changes)` returns a copy.

### `class TelemetryTrack(samples=(), time_offset_ms=0, *, extrapolation="linear", baseline_ms=300)`
| Member | |
|---|---|
| `at(t_ms, *, extrapolate_ms=0) -> PoseSample \| None` | Pose at video time t (+ offset); interpolated; extrapolated up to `extrapolate_ms` past the newest sample; the first sample held up to `extrapolate_ms` before it |
| `append(sample)` | Live ingest; raises `ValueError` if older than the newest sample |
| `with_time_offset(ms) -> TelemetryTrack` | Same samples, telemetry clock shifted (positive = telemetry late) |
| `len(track)`, `track[i]`, iteration | Samples in time order |
| `span_ms`, `source`, `time_offset_ms` | Properties |

### Readers
| Class | Reads |
|---|---|
| `DjiSrtReader(default_pitch_deg=-90.0)` | DJI SRT sidecars; `read_text(text)` parses a string |
| `DjiEmbeddedReader()` | DJI metadata track in MP4/MOV; `last_stats` after `read` |
| `DjiPacketDecoder(encoding="auto", decide_after=10)` | Live: `feed(packet, t_ms) -> list[PoseSample]`, `flush()`, `encoding`, `stats` |
| `iter_dji_packets(path)` | Yields `(t_ms, packet)` from a recording |

A custom reader needs `can_read(video, sidecar) -> bool` and `read(video, sidecar) -> TelemetryTrack`;
`explain(video, sidecar) -> str` is optional.

`write_dji_srt(samples, frame_ms) -> str` writes DJI-style SRT text.

## Camera and frames

### `class CameraModel(sensor_w_mm, focal_mm, dist_coeffs=None, name="custom")`
`CameraModel.preset(name)`, `CameraModel.presets()`, `intrinsics(w, h) -> Intrinsics`, `hfov_deg`.
Presets: `generic_small_sensor`, `equiv_24mm`, `mini2`, `mini3pro`, `air2s`, `mavic3_wide` (nominal
specification values).

### `class Intrinsics`
`fx, fy, cx, cy, img_w_px, img_h_px, dist_coeffs`; `K`, `K_inv`, `scaled(w, h)`,
`distort_points(uv)`, `undistort_points(uv)`.

### `class LocalFrame(ref_lat_deg, ref_lon_deg, ref_alt_m=0.0)`
`to_enu(lat_lon) -> (E, N)`, `to_wgs84(en) -> (lat, lon)`, `distance_m(a, b)`, `m_per_deg`.

### `class FlatGround(level_m=0.0)`
Ground plane `level_m` above the takeoff altitude. Implements `GroundModel`
(`height_m(en)`, `is_planar`).

## Frame geometry

### `class TelemetryPoseProvider(track, camera, frame=None, ground=FlatGround(), *, pos_sigma_m=None, yaw_sigma_deg=None, pitch_sigma_deg=None, extrapolate_ms=0, min_depression_deg=1.0, max_range_m=None)`
`geometry(t_ms, img_w_px, img_h_px, image=None) -> FrameGeometry | None`;
`estimate(t_ms) -> PoseEstimate | None`. `frame` defaults to the first sample's position.
Implements the `PoseProvider` protocol.

### `class FrameGeometry`
Build with `FrameGeometry.from_pose(pose, camera, w, h, frame, ground=FlatGround(), **limits)` or get
one from a provider.

| Method | Returns |
|---|---|
| `pixel_to_en(uv, h_m=0)` / `pixel_to_lat_lon(uv, h_m=0)` | Ground points; NaN rows for rays that miss the ground. `h_m` scalar or per point |
| `en_to_pixel(en, h_m=0)` / `lat_lon_to_pixel(lat_lon, h_m=0)` | Pixels; NaN behind the camera |
| `H_img_from_ground(h_m=0)` / `H_ground_from_img(h_m=0)` | 3&times;3 homographies (undistorted pixels) |
| `footprint()` / `tile_footprint(x0, y0, x1, y1)` | `Footprint` or `None` if all sky |
| `horizon_row_px()` / `clip_row_px()` | Rows; `None` when not applicable |
| `gsd_m_per_px(uv=None)` | Metres per pixel (image centre if `uv` omitted) |
| `ground_range_m(uv)`, `view_angle_deg(uv)` | Distance from the point below the camera; angle from straight down |
| `rays_world(uv)` | Unit rays in ENU |
| `camera_enu_m`, `R_world_from_cam`, `img_size`, `pose`, `intrinsics`, `frame`, `ground` | Properties |

### `class Footprint` (frozen)
`corners_en`, `corners_lat_lon` (top-left, top-right, bottom-right, bottom-left), `clip_row_px`,
`area_m2`; `contains_en(en)`, `to_geojson(properties=None)`.

### Functions
`relative_homography(a, b, h_m=0)` &mdash; image a &rarr; image b through the ground plane.
`apply_h(H, pts)`. `rotation_world_from_cam(yaw, pitch, roll)`. `wrap_deg(a)`,
`lerp_angle_deg(a, b, t)`.

`drongo.rectify.rectify_roi(image, geom, roi_en, gsd_m, h_m=0, border_value=0)` (needs OpenCV):
north-up image of the ENU rectangle `(E0, N0, E1, N1)` at `gsd_m` metres per pixel.
