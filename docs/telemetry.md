# Telemetry

## Opening a recording

```python
from drongo import open_telemetry

diagnostics = []
track = open_telemetry("DJI_0042.MP4", diagnostics=diagnostics)
if track is None:
    print("\n".join(diagnostics))
```

`open_telemetry` tries, in order:

1. **An SRT sidecar** with the same name as the video (any letter case), or the file given as
   `sidecar=`.
2. **Telemetry embedded in the video file** ("DJI meta" track in MP4/MOV).

The first source that yields samples wins. When none does, `diagnostics` holds one line per
reader saying why, for example:

```
    dji_srt_b: no SRT sidecar found (looked for .../DJI_0042.SRT, .../DJI_0042.srt)
 dji_protobuf: no DJI metadata track (tracks: VideoHandler, SoundHandler)
```

## Supported formats

### DJI SRT sidecars

Written by most DJI drones when "video subtitles" is on. drongo reads all known layouts:

| Layout | Typical models | Gimbal angles |
|---|---|---|
| `[latitude: ...] [longitude: ...] [altitude: ...]` | Mini 2, Air 2 | Not recorded: nadir assumed (`default_pitch_deg`) |
| `[latitude] [longitude] [rel_alt abs_alt] [gb_yaw gb_pitch gb_roll]` | Mavic 3, Air 2S | Both `[gb_yaw : x] [gb_pitch : y]` and grouped forms |
| `GPS (lat, lon, ...) ... H 85.20m` | FPV, Avata | Not recorded |

Byte-order marks, Windows line endings and missing block numbers are tolerated. Relative altitude
(`rel_alt`) and sea-level altitude (`abs_alt`) are kept apart.

### DJI metadata embedded in MP4/MOV

Newer DJI drones store a protobuf message per frame in a timed-metadata track. drongo reads the
container's sample table itself (no ffmpeg or exiftool) and decodes:

| Field | Meaning |
|---|---|
| position | latitude, longitude (stored in radians) |
| relative altitude | altitude above takeoff (mm or m, detected) |
| sea-level altitude | when present |
| gimbal pitch, roll, yaw | absolute, in tenths of a degree |
| aircraft yaw | kept as `body_yaw_deg` |

Angles are stored either as plain or as zigzag-encoded integers depending on the writer. drongo
picks, per file, the encoding under which every gimbal attitude is physically possible, drops any
sample that is still impossible, and reports the counts in `DjiEmbeddedReader().last_stats`.
Verified on the Mavic 3 Enterprise. Other models may place fields elsewhere; they are reported as
"no position", never guessed.

### Live streams

```python
from drongo import DjiPacketDecoder, TelemetryTrack, TelemetryPoseProvider

decoder = DjiPacketDecoder()
track = provider = None
for t_ms, packet in downlink():                 # your source of raw packets
    for sample in decoder.feed(packet, t_ms):
        if track is None:
            track = TelemetryTrack([sample])
            provider = TelemetryPoseProvider(track, camera, extrapolate_ms=300)
        else:
            track.append(sample)
```

- The decoder may hold back the first few packets until the angle encoding is clear, then
  releases them together; `decoder.flush()` releases whatever is held at the end of a stream.
- Malformed, positionless or impossible packets are counted in `decoder.stats` and never raise.
- `iter_dji_packets(path)` yields the packets of a recording in order &mdash; the easiest way to
  test a live pipeline.
- If your source already provides position and angles (an SDK, a ground station), skip the
  decoder: build `PoseSample` objects and `append` them.

Appends must be in time order; an older sample raises `ValueError`.

## Clocks and sync

Video frames and telemetry must be on the same clock. For recordings both come from the file;
for live streams stamp both on arrival with the same clock. Any constant difference between the
two clocks is the **sync offset**:

```python
track = track.with_time_offset(160)     # telemetry timestamps are 160 ms late
```

The diagnostics report estimates the offset by re-scoring the telemetry-vs-video consistency at
shifted offsets. The estimate needs non-uniform motion &mdash; turns that change rate, braking,
gimbal moves; on a straight, constant-speed leg every offset scores the same and the report says
the offset is not observable.

### Frames ahead of the telemetry

Live frames often arrive before the telemetry for that moment. Within `extrapolate_ms`, the track
continues the motion of the last `baseline_ms` (default 300 ms): for a frame 150 ms ahead this
puts the ground point within centimetres on a straight leg and under 10 cm in a turn, against
1.8&ndash;4 m when repeating the newest sample. The long baseline keeps the estimate stable when
a receiver repeats the same GPS fix for several samples. Use `TelemetryTrack(..., extrapolation="hold")`
for the old behaviour.

## Writing your own reader

A reader is any object with two methods (and, optionally, `explain` for diagnostics):

```python
from pathlib import Path
import csv
from drongo import PoseSample, TelemetryTrack, open_telemetry

class CsvReader:
    kind = "csv"

    def can_read(self, video, sidecar):
        return sidecar is not None and Path(sidecar).suffix.lower() == ".csv"

    def read(self, video, sidecar):
        with open(sidecar, newline="") as f:
            return TelemetryTrack(
                PoseSample(int(r["t_ms"]), float(r["lat"]), float(r["lon"]), float(r["alt"]),
                           float(r["yaw"]), float(r["pitch"]), float(r.get("roll", 0)))
                for r in csv.DictReader(f))

    def explain(self, video, sidecar):
        return "no .csv sidecar given"

track = open_telemetry("flight.mp4", "flight.csv", readers=[CsvReader()])
```

Rules for a reader: store **absolute** gimbal yaw (convert body-relative values); leave fields the
source does not have as `None` rather than zero; use video-clock milliseconds; never raise on a
malformed file &mdash; return an empty track and set an explanation.
