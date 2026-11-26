# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break the API).

## [Unreleased]

### Added
- Demo clip generator (`python -m testbed.synth.demo`): procedural nature and urban terrains,
  realistic survey / orbit / cruise / hover-pan flights, 3D buildings, trees and cars with sun
  shadows, colour rendering with sky and haze,
  H.264 output with an SRT sidecar or embedded DJI telemetry, and a ground-truth file per clip.
- `testbed.run_all --demo-dir`: checks every demo clip and compares its telemetry with the ground
  truth; the built-in synthetic clip is now rendered over the 3D urban terrain.
- `testbed.synth.mp4meta.add_meta_track` / `embed_telemetry`: add a DJI metadata track to an
  existing video MP4.

### Fixed
- `testbed/synth/` was excluded by `.gitignore`, so the synthetic-data package the tests import was
  missing from the repository.

## [0.1.0] - 2026-11-24

### Added
- Camera model with intrinsics derived from the frame size, Brown-Conrady distortion and presets.
- `FrameGeometry`: pixel to ground and back at any height, homographies, horizon, footprints, GSD,
  ground range, view angle; one model for nadir, oblique and rolled views.
- `LocalFrame` (ENU / WGS84, under 1 mm vs the geodesic at 10 km) and `FlatGround`.
- Telemetry: `TelemetryTrack` with shortest-arc angle interpolation, clock offset and linear
  extrapolation; `open_telemetry` with per-reader diagnostics.
- Readers: DJI SRT (all known layouts); DJI metadata embedded in MP4/MOV, pure Python, with
  encoding detection and an impossible-value gate; `DjiPacketDecoder` for live streams;
  `iter_dji_packets` for replay.
- `TelemetryPoseProvider` and the `PoseProvider` / `GroundModel` / `TelemetryReader` interfaces.
- `rectify_roi` for north-up images of a ground area.
- Diagnostics report (HTML map, plots, Markdown, JSON, GeoJSON), a full-check runner, synthetic
  flight generator with rendered video and SRT or embedded telemetry.
- Examples and a README whose Quick start is executed by the test suite.
