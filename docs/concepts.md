# Concepts

drongo answers two questions for every video frame: where was the camera, and where on the ground
does each pixel fall. This page explains the model behind the answers.

## The pieces

```
telemetry file / live packets
        │  reader / decoder
        ▼
  TelemetryTrack ── at(t) ──►  PoseSample            (position + orientation at time t)
        │
        ▼
  TelemetryPoseProvider ── geometry(t, w, h) ──►  FrameGeometry   (camera model for one frame)
                                                        │
                    pixel_to_lat_lon, footprint, H_img_from_ground, gsd_m_per_px, ...
```

| Object | What it is | Mutable? |
|---|---|---|
| `PoseSample` | One telemetry record: time, latitude, longitude, altitude above takeoff, gimbal yaw/pitch/roll (+ optional sea-level altitude, body yaw, velocity) | No |
| `TelemetryTrack` | Time-ordered samples; interpolates between them and extrapolates past the newest | Append only |
| `CameraModel` | Sensor width and focal length (+ optional lens distortion) | No |
| `LocalFrame` | A metric East-North plane around a reference point; converts to and from WGS84 | No |
| `GroundModel` | The surface rays hit: `FlatGround(level_m)` | No |
| `TelemetryPoseProvider` | Turns a time into a `FrameGeometry` | Holds the track |
| `FrameGeometry` | The camera for one frame; every query is a method on it | No |

A `FrameGeometry` is cheap to build (about 0.06 ms) and holds no precomputed results; ask it only
for what you need. Share it between all the components that process the same frame.

## Conventions

| Quantity | Convention |
|---|---|
| World frame | ENU: x = East, y = North, z = Up, in metres, from the `LocalFrame` reference; z = 0 at the takeoff altitude |
| Camera frame | x right, y down, z along the optical axis (OpenCV) |
| Yaw | Degrees clockwise from true North, stored in [0, 360) |
| Pitch | Degrees; 0 = horizon, &minus;90 = straight down |
| Roll | Degrees; positive = the image's right side tilts down |
| Pixels | (u, v) from the top-left corner, v downwards; raw (distorted) pixels in point methods, undistorted in homographies |
| Points | numpy arrays of shape (N, 2); lat/lon arrays are ordered (lat, lon) |
| Time | Milliseconds on the video's clock |
| Altitude | Above the takeoff point (what drones report); sea level kept separately when known |

Units are part of every name: `_m`, `_deg`, `_ms`, `_px`.

## The camera model

**Orientation.** From gimbal yaw &psi;, pitch &theta; and roll &phi; drongo builds the optical axis
f = (sin&psi; cos&theta;, cos&psi; cos&theta;, sin&theta;), the image-right axis r and the
image-down axis d, and the rotation R = [r d f]. At pitch &minus;90 the top of the image points
along the heading.

**Intrinsics.** The sensor width maps to the full frame width:
f<sub>px</sub> = focal<sub>mm</sub> / sensor_width<sub>mm</sub> &times; W, principal point at the
image centre, square pixels. The vertical field of view therefore follows the frame's aspect ratio
automatically &mdash; 16:9 video from a 4:3 sensor needs no separate setting.

**Pixel to ground.** A pixel defines a ray d = R K<sup>&minus;1</sup> (u, v, 1). The ground point is
where the ray meets the plane z = ground level + h:
t = (z<sub>0</sub> &minus; C<sub>z</sub>) / d<sub>z</sub>, X = C + t d. Rays that point at or above the
horizon return NaN. Ground to pixel is the exact inverse.

**Height above ground.** `h_m` raises the plane. Use it for anything that is not on the ground: a
point at ground distance R and height h, projected as if it were on the ground, lands at
R &times; alt / (alt &minus; h) &mdash; for a 20 m roof seen 100 m away from 85 m altitude, 31 m too far.
For vehicles and people, project the bottom centre of the box with `h_m=0`.

**Homographies.** For points on one plane the projection is a 3&times;3 matrix:
`H_img_from_ground(h_m)` and its inverse. The camera motion between two frames, as telemetry
predicts it, is `relative_homography(a, b)`: map image a to the ground plane, then into image b.
Use it to stabilise video or as a prior for image registration.

**Horizon and footprints.** The horizon row is solved exactly. Footprints are clipped at the first
row where every column looks at least 1&deg; below the horizon (and, optionally, within
`max_range_m`), so a footprint is always a finite quadrilateral. The clip row is computed once per
frame and cached.

**Local frame.** `LocalFrame` uses a mid-latitude projection on the WGS84 ellipsoid. Distances agree
with the geodesic to under 1 mm at 10 km. Earth curvature is ignored, consistent with a flat ground
(the ground drops about 0.3 m at 2 km).

## Time

Every `geometry(t)` call asks for the pose at video time t. The track adds `time_offset_ms`
(positive = the telemetry clock runs late) and interpolates: position linearly, yaw and roll on the
shortest arc. Past the newest sample &mdash; a live frame that arrived before its telemetry &mdash;
the track continues the motion of the last `baseline_ms` (300 ms) for up to `extrapolate_ms`. See
[telemetry.md](telemetry.md#clocks-and-sync).
