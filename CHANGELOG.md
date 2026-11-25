# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break the API).

## [Unreleased]

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
