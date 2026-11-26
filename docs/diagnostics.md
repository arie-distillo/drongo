# Diagnostics

Wrong telemetry, a wrong lens setting and a clock offset all produce numbers that look plausible.
The diagnostics compare the telemetry with the video itself, so you can tell.

## One clip

```bash
python -m testbed.harness.run_video --video flight.mp4 --camera mavic3_wide --out report --gap 10 --stride 6
```

| Option | Meaning |
|---|---|
| `--camera` / `--sensor-w-mm --focal-mm` | Camera preset, or sensor width and focal length |
| `--srt PATH` | Use this telemetry file instead of searching |
| `--gap N` | Frames between the two frames of a pair; aim for 5&ndash;30 px of image motion per pair (about 1/3 s at 30 fps for typical flights) |
| `--stride N` | Frames between successive pairs |
| `--offset-sweep=A:B:STEP` | Clock offsets to try, in ms (write it with `=`) |
| `--map-range M` | Clip footprints on the map at this ground range |
| `--width` | Processing width in pixels (default 960) |

Outputs in `--out`: `report.html`, `report.md`, `report.json`, `residuals.png`,
`footprints.geojson`, `overlay_*.png`.

## Reading the report

### Summary and checks

Video and telemetry rates, altitude and gimbal ranges, speed. *Checks* flags position jumps,
missing gimbal angles (nadir assumed) and zero-altitude samples.

### Telemetry vs video (the residual)

For pairs of frames, drongo predicts the camera motion from telemetry and measures it from the
images (feature tracking + RANSAC). The **residual** is the difference, in pixels and metres,
shown next to the **measured image motion**.

| Residual / image motion | Meaning |
|---|---|
| below ~0.3 | Telemetry, lens and sync agree with the video |
| 0.3&ndash;0.7 | Usable, with an error source worth finding (sync, lens, noisy angles) |
| close to 1 | Telemetry does not describe this video (wrong file, wrong yaw convention, synthetic data) |

The residual sees **relative** errors only. A constant GPS offset or compass bias moves every frame
the same way and leaves the residual unchanged &mdash; use the map for those.

### Sync offset

The residual re-scored with the telemetry clock shifted. A clear minimum is the offset; apply it
with `track.with_time_offset(ms)`. On a straight, constant-speed flight all offsets score the same
and the report says "not observable".

### Map

Satellite imagery (or a street map) with the flight track, a footprint every second, and a few
frames rectified and draped on the map, each with its own dashed outline. If draped frames line up
with roads and buildings, absolute position, heading and lens are right. A constant shift means
GPS or compass bias; a mismatch growing towards the far edge means pitch or lens error. The map
tiles need internet access.

### Telemetry plots and grid overlays

Altitude, speed, gimbal angles, and gimbal yaw against the direction of travel (they should
agree for a forward-looking camera). Grid overlays project a 10 m ground grid onto frames; it
should stay fixed to the ground from frame to frame.

## The full check

```bash
python -m testbed.run_all --clip "flight1.mov:mavic3_wide" --clip "flight2.mp4:equiv_24mm" -- --gap 10 --stride 6
```

Runs the test suite, renders a synthetic clip (a slalom over the 3D urban demo terrain) with a
known 160 ms clock offset and checks that the diagnostics find it, then reports on each clip.
`reports/index.html` has one line per clip with a verdict ("consistent" or "telemetry does not
match video") and a link to its report.

To include demo clips (see [testing](testing.md#demo-clips)):

```bash
python -m testbed.synth.demo --all --out demo_clips
python -m testbed.run_all --demo-dir demo_clips
```

Every clip in `demo_clips/srt/` and `demo_clips/embedded/` is checked like a real clip, with the
camera preset it was rendered with and pairs 12 frames apart (orbits and surveys move too little
in the default 3 frames for the verdict; options after `--` replace this). For synthetic clips the report also compares the recorded
telemetry with the exact poses in `truth/`: position, altitude, yaw (mean and spread, next to
the injected compass bias) and pitch errors, and the injected clock offset. These errors include
the effect of that offset, since telemetry then describes where the drone was earlier.

## A clip with known answers

```bash
python -m testbed.synth.make_clip --scenario slalom --offset-ms 160 --out synth
```

Scenarios: `straight_nadir`, `oblique_turn`, `gimbal_sweep`, `slalom`. `--noise none` gives
perfect telemetry; the default adds consumer-grade GPS and attitude noise.
