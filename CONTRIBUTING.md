# Contributing to drongo

Thank you for helping. The most valuable contributions are, in order: **telemetry from drones we
do not support yet**, bug reports with a reproducible clip, and pull requests with tests.

## Report a bug

Open an issue with:
- drongo version, Python version, operating system;
- drone model and firmware, and whether telemetry is an SRT sidecar or embedded;
- the `diagnostics` list from `open_telemetry(..., diagnostics=d)`, or the console output of
  `python -m testbed.harness.run_video`;
- if you can share it, a short clip or just its telemetry (a few seconds is enough).

## Add support for a drone or telemetry format

1. Run `python -m testbed.harness.run_video --video your_clip --out report` and attach the console
   output to an issue, so we can see what each reader found.
2. For embedded telemetry, a handful of raw packets is enough:
   ```python
   from drongo import iter_dji_packets
   import json
   pk = [{"t_ms": t, "hex": p.hex()} for t, p in list(iter_dji_packets("clip.mp4"))[:5]]
   print(json.dumps(pk))
   ```
   Shift or remove the coordinates if the location should not be public.
3. A new reader implements `can_read`, `read` and `explain` (see `docs/telemetry.md`), never raises
   on malformed input, stores absolute yaw, and comes with a fixture and expected values in
   `testbed/tests/`.

## Development setup

```bash
git clone https://github.com/arie-distillo/drongo.git
cd drongo
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[testbed]"
pytest
```

## Pull requests

- One topic per pull request; describe what changes for users.
- Every behaviour change has a test. Tolerances must follow from the physics (GSD, angle rounding,
  sensor noise), not from tuning to one clip.
- Keep the core dependent on numpy only; OpenCV and other libraries go in optional extras.
- Units in names (`_m`, `_deg`, `_ms`, `_px`); type hints on public functions; docstrings that say
  what the function returns and in which units.
- If you change the README Quick start, `testbed/tests/test_readme.py` runs it: keep it passing.
- Add a line to `CHANGELOG.md` under *Unreleased*.
- `pytest` must pass on your machine; CI runs Linux, macOS and Windows.

## Releases

Maintainers bump the version in `pyproject.toml`, move *Unreleased* in `CHANGELOG.md` under the new
version, tag `vX.Y.Z` and publish to PyPI as `drongo-geo`.
