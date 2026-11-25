# Testing

```bash
pip install -e ".[testbed]"
pytest                    # everything, about 15 s
pytest -m "not slow"      # unit tests, about 1 s
pytest -m slow            # rendered-video, README, example and diagnostics tests
```

Optional, with your own footage (every MP4/MOV with embedded DJI telemetry in the folder must
decode cleanly, and decode identically when replayed packet by packet):

```bash
DRONGO_SAMPLES=/path/to/clips pytest testbed/tests/test_dji_embedded.py testbed/tests/test_streaming.py   # bash
$env:DRONGO_SAMPLES="D:\clips"; pytest testbed/tests/test_dji_embedded.py testbed/tests/test_streaming.py # PowerShell
```

## What each layer proves

| Layer | File | Proves |
|---|---|---|
| Conventions and maths | `test_geometry.py` | Analytic nadir footprints and GSD, rotation conventions, horizon row, height-corrected projection |
| Internal consistency | `test_geometry.py` | Round trips over 150 random poses; homography = ray projection; tiles partition the frame |
| SRT telemetry | `test_telemetry.py` | All layouts, altitude fields, yaw across North, signed roll, clock offset, extrapolation, BOM / CRLF / case |
| Embedded telemetry | `test_dji_embedded.py` | Real and simulator packets decode to verified values; MP4 round trip for both encodings; impossible angles rejected; corrupt files explained, never crash |
| Live use | `test_streaming.py` | Packet-by-packet decoding equals the file reader; ambiguous first packets buffered; extrapolation beats hold by 4&times;; per-frame cost |
| Geodesy and camera | `test_camera_frames.py` | Local frame vs geodesic; aspect-derived intrinsics; distortion round trip |
| Independent check | `test_synthetic.py` | Rendered video: image-measured motion matches telemetry; injected biases and offsets behave as predicted; diagnostics find an injected offset |
| Demo clips | `test_demo_clips.py` | Flight dynamics within limits; colour render places ground points, roofs and tree crowns where the geometry says; embedded track added to real H.264 video (moov first or last) reads back exactly and leaves the frames unchanged |
| Documentation | `test_readme.py`, `test_examples.py` | Every Quick start block and both example scripts run and give correct results |

Synthetic tests render frames from a procedural ground texture with exact poses, so the true answer
is known. Tolerances come from the physics (angle rounding, GPS noise), never from tuning to a clip.

## Adding a test clip

Clips used as fixtures must be redistributable. For a new telemetry format, a few raw packets (as
in `testbed/tests/fixtures/`) with their expected decoded values are enough; strip or shift the
location if it should not be public.

## Demo clips

`testbed/synth/demo.py` renders realistic-looking clips with known ground truth, for demos and for
testing pipelines built on drongo:

```bash
python -m testbed.synth.demo --terrain urban --flight orbit          # one clip, 1920x1080, 30 fps
python -m testbed.synth.demo --all --out demo_clips                  # nature + urban x 4 flights
python -m testbed.synth.demo --terrain nature --flight cruise --offset-ms 150 --noise dji
python -m testbed.synth.demo --image ortho.jpg --flight survey        # your orthophoto + world file
```

| Option | Values |
|---|---|
| `--terrain` | `nature` (crop fields, tree lines, woods, tracks, river) or `urban` (street grid, buildings with shadows, parked cars, parks); procedural, reproducible from `--seed` |
| `--flight` | `survey` (nadir mapping lines with turnarounds), `orbit` (point of interest, oblique), `cruise` (forward flight, climbing, camera tilting up), `hover_pan` (hover, slow pan and tilt) |
| `--telemetry` | `srt`, `embedded` or `both` (default) |
| `--noise`, `--offset-ms` | telemetry errors (`dji` = consumer GPS / compass profile) and clock offset; positive = telemetry lags the video |
| `--wind` | scale of position sway and gimbal jitter; `0` for a perfectly still aircraft |
| `--terrain-half-m`, `--terrain-res-m` | terrain extent from the origin (default 300 m) and pixel size (8 cm); beyond the edge the ground is mirrored |

Each clip `<terrain>_<flight>` produces `srt/<name>.mp4` + `srt/<name>.SRT`, `embedded/<name>.mp4`
(the same H.264 video with a `DJI meta` / `djmd` telemetry track) and `truth/<name>.json` (exact
per-frame poses, camera and noise settings). Both telemetry files hold the same record: one sample
per video frame.

Flights accelerate from rest, slow into turns and keep a gimbal lag, sway and residual jitter, all
of which is in the ground truth. Buildings (flat or gable roofs, lit walls with windows), trees and
cars are 3D objects standing on flat ground: they occlude and show parallax as the camera moves, and
cast sun shadows. Ground points that are not hidden by an object project exactly as drongo says;
an object point at height h projects as `pixel_to_en(..., h_m=h)` predicts. H.264
encoding uses ffmpeg (from `PATH`, or the `imageio-ffmpeg` package installed with `[testbed]`).
A 1080p frame takes about 0.3 s to render, and a terrain about 2.5 GB of memory to generate.
