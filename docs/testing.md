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
| Documentation | `test_readme.py`, `test_examples.py` | Every Quick start block and both example scripts run and give correct results |

Synthetic tests render frames from a procedural ground texture with exact poses, so the true answer
is known. Tolerances come from the physics (angle rounding, GPS noise), never from tuning to a clip.

## Adding a test clip

Clips used as fixtures must be redistributable. For a new telemetry format, a few raw packets (as
in `testbed/tests/fixtures/`) with their expected decoded values are enough; strip or shift the
location if it should not be public.
