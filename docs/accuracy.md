# Accuracy and limits

drongo's geometry is exact for its model; the accuracy of a result is set by the telemetry, the
camera parameters and how well the flat-ground model fits the scene.

## What is verified

| Check | Result |
|---|---|
| Pixel &rarr; ground &rarr; pixel, 150 random poses | within 10<sup>&minus;6</sup> px |
| Homography vs ray projection | within 10<sup>&minus;6</sup> m |
| Local frame vs WGS84 geodesic | under 1 mm at 10 km |
| Telemetry-predicted vs image-measured motion, rendered video, exact telemetry | under 0.5 px median |
| Rendered marker vs projected position | within 0.25 px |

## Error budget for real flights

Typical consumer drone (no RTK), 85 m above takeoff:

| Error source | Size | Effect on a ground point |
|---|---|---|
| GPS position | 1&ndash;3 m | Same shift everywhere |
| Compass / gimbal yaw | 1&ndash;2&deg; | range &times; tan(error): 2 m at 60 m for 2&deg; |
| Gimbal pitch | ~1&deg; | (range&sup2; + alt&sup2;) / alt per radian: 3 m at &minus;45&deg; |
| Altitude above takeoff vs actual ground | terrain difference | 10 m of terrain = 12% scale error |
| Clock offset | 50 ms in a 30&deg;/s turn | 1.5&deg; of yaw: about 2 m at the frame edge |
| Lens | 70&deg; used for a 73.7&deg; camera | about 7% scale error |
| SRT rounding | angles to 0.1&deg; | under range &times; 0.0009 |

## What the diagnostics can and cannot see

- The telemetry-vs-video residual reveals attitude noise, clock offset and lens errors.
- It cannot see a **constant** GPS or compass bias: every frame moves the same way. Use the
  draped frames on the map, ground control, or registration to reference imagery.
- The clock offset is measurable only when motion changes.

## Model limits

- **Flat ground.** Everything is projected onto a plane at the ground level you set (default: the
  takeoff altitude). Terrain relief, buildings and Earth curvature are not modelled.
- **Points above the ground** need their height (`h_m`), or they land too far away:
  R &times; alt / (alt &minus; h).
- **Parallax.** Plane homographies and footprints assume one plane; tall structures move against the
  ground between frames and footprints over them are approximate.
- **Occlusion** is not modelled: a footprint is what the camera points at, not what it can see.
- **Rolling shutter** is not modelled; fast rotation skews frames slightly.
- **Camera presets** are nominal specifications; calibrate for accuracy-critical work.

A terrain (DEM) ground model, and later a surface model with buildings, are on the roadmap; the
`GroundModel` interface is already in place for them.
