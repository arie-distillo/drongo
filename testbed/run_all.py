"""One command for a full drongo check: test suite, a synthetic clip with known
answers, demo clips, and any real clips you name. Writes reports/index.html
linking every report.

    python -m testbed.run_all
    python -m testbed.run_all --demo-dir demo_clips
    python -m testbed.run_all --clip "D:/clips/city1-25-40.mov:mavic3_wide" --clip "D:/clips/2 cars 2m.mp4:equiv_24mm"

--demo-dir takes the output folder of `python -m testbed.synth.demo`: every clip
in srt/ and embedded/ is checked against the video (as real clips are), and its
telemetry is compared with the exact poses in truth/.

--clip takes PATH or PATH:CAMERA_PRESET (default preset: equiv_24mm). Extra
harness options for every clip go after --, e.g.  -- --gap 15 --stride 6
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drongo import CameraModel, LocalFrame, open_telemetry  # noqa: E402
import numpy as np  # noqa: E402

PRESETS = set(CameraModel.presets())


def _env(extra: dict | None = None) -> dict:
    """Child processes find drongo and testbed whether or not drongo is pip-installed."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(extra or {})
    return env


def run_pytest(out: Path, samples: str | None) -> dict:
    env = _env({"DRONGO_SAMPLES": samples} if samples else None)
    junit = out / "pytest.xml"
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    (out / "pytest.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    s = ET.parse(junit).getroot()
    s = s if s.tag == "testsuite" else s.find("testsuite")
    res = {k: int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    res["seconds"] = round(time.time() - t0, 1)
    res["failed_names"] = [f"{c.get('classname')}::{c.get('name')}" for c in s.iter("testcase")
                           if c.find("failure") is not None or c.find("error") is not None]
    return res


def run_clip(video: Path, camera: str, out: Path, extra: list[str], srt: Path | None = None) -> dict:
    args = [sys.executable, "-m", "testbed.harness.run_video", "--video", str(video), "--camera", camera,
            "--out", str(out)] + (["--srt", str(srt)] if srt else []) + extra
    proc = subprocess.run(args, cwd=ROOT, env=_env(), capture_output=True, text=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "console.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    rep = out / "report.json"
    if proc.returncode or not rep.exists():
        return {"ok": False, "error": (proc.stdout + proc.stderr).strip().splitlines()[-6:]}
    r = json.loads(rep.read_text(encoding="utf-8"))
    return {"ok": True, "report": r}


SYNTH_OFFSET_MS = 160
# Orbits and surveys move only a few pixels in 3 frames, too little for the
# consistency verdict; 12 frames (0.4 s at 30 fps) gives every demo flight enough.
DEMO_HARNESS_ARGS = ["--gap", "12"]


def make_synthetic(out: Path) -> tuple[Path, Path]:
    """A 6 s slalom over the urban demo terrain (3D buildings, trees, cars), with
    DJI-like telemetry noise and a known 160 ms clock offset."""
    code = (
        "import dataclasses, sys\n"
        "from pathlib import Path\n"
        "from drongo import CameraModel\n"
        "from testbed.synth.demo import LOOKS, render_clip\n"
        "from testbed.synth.render import MipTexture\n"
        "from testbed.synth.scenario import DJI_CONSUMER, slalom\n"
        "from testbed.synth.terrain import make_urban\n"
        "out = Path(sys.argv[1])\n"
        "mip = MipTexture(make_urban(half_size_m=250.0, res_m=0.1, seed=1))\n"
        f"noise = dataclasses.replace(DJI_CONSUMER, time_offset_ms={SYNTH_OFFSET_MS})\n"
        "render_clip(slalom(duration_s=6.0), mip, CameraModel.preset('equiv_24mm'), out, 'slalom', 960, 540,\n"
        "            noise, 'srt', LOOKS['urban'], progress=False)\n")
    subprocess.run([sys.executable, "-c", code, str(out)], cwd=ROOT, env=_env(), check=True, capture_output=True)
    return out / "srt" / "slalom.mp4", out / "srt" / "slalom.SRT"


def _ang(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def telemetry_vs_truth(video: Path, truth_json: Path) -> dict:
    """Recorded telemetry of a synthetic clip against its exact poses.

    The errors are what the injected noise produced; position error includes the
    effect of any clock offset (telemetry describes where the drone was earlier)."""
    t = json.loads(truth_json.read_text(encoding="utf-8"))
    track = open_telemetry(video)
    if track is None:
        return {"ok": False, "error": "no telemetry found"}
    truth = {p["t_ms"]: p for p in t["poses"]}
    rec = [p for p in track if p.t_ms in truth]
    if not rec:
        return {"ok": False, "error": "telemetry timestamps do not match the truth file"}
    tr = [truth[p.t_ms] for p in rec]
    fr = LocalFrame(*t["local_frame_origin"])
    e_rec = fr.to_enu([[p.lat_deg, p.lon_deg] for p in rec])
    e_tru = fr.to_enu([[q["lat_deg"], q["lon_deg"]] for q in tr])
    horiz = np.linalg.norm(e_rec - e_tru, axis=1)
    dyaw = _ang([p.gimbal_yaw_deg - q["gimbal_yaw_deg"] for p, q in zip(rec, tr)])
    dpitch = np.array([p.gimbal_pitch_deg - q["gimbal_pitch_deg"] for p, q in zip(rec, tr)])
    dalt = np.array([p.alt_agl_m - q["alt_agl_m"] for p, q in zip(rec, tr)])
    n = t.get("telemetry_noise", {})
    return {"ok": True, "samples": len(rec), "truth_frames": len(t["poses"]), "source": str(track.source),
            "horiz_m_median": float(np.median(horiz)), "horiz_m_p95": float(np.percentile(horiz, 95)),
            "alt_m_median": float(np.median(np.abs(dalt))),
            "yaw_deg_mean": float(dyaw.mean()), "yaw_deg_std": float(dyaw.std()),
            "pitch_deg_median": float(np.median(np.abs(dpitch))),
            "injected_offset_ms": n.get("time_offset_ms", 0), "injected_yaw_bias_deg": n.get("yaw_bias_deg", 0.0),
            "camera": t["camera"]["name"]}


def demo_clips(demo_dir: Path) -> list[tuple[str, Path, Path]]:
    """(label, video, truth json) for every clip under a demo output folder."""
    out = []
    for truth in sorted((demo_dir / "truth").glob("*.json")):
        for kind in ("srt", "embedded"):
            v = demo_dir / kind / f"{truth.stem}.mp4"
            if v.is_file():
                out.append((f"{truth.stem} ({kind})", v, truth))
    return out


def index_html(out: Path, tests: dict, clips: list[tuple[str, str, dict]]) -> Path:
    ok_tests = tests["failures"] == 0 and tests["errors"] == 0
    rows = []
    for name, rel, c in clips:
        tv = c.get("truth")
        if tv is None:
            truth_cell = "<td>-</td>"
        elif not tv["ok"]:
            truth_cell = f"<td class='bad'>{html.escape(tv['error'])}</td>"
        else:
            truth_cell = (f"<td>{tv['horiz_m_median']:.2f} m (p95 {tv['horiz_m_p95']:.2f}); alt {tv['alt_m_median']:.2f} m; "
                          f"yaw {tv['yaw_deg_mean']:+.2f}&plusmn;{tv['yaw_deg_std']:.2f}&deg; "
                          f"(injected bias {tv['injected_yaw_bias_deg']:+.1f}&deg;); pitch {tv['pitch_deg_median']:.2f}&deg;"
                          f"<br><span style='color:#666'>injected offset {tv['injected_offset_ms']} ms, "
                          f"{tv['samples']}/{tv['truth_frames']} samples</span></td>")
        if not c["ok"]:
            rows.append(f"<tr><td>{html.escape(name)}</td><td class='bad'>failed</td><td colspan='4'><pre>"
                        f"{html.escape(chr(10).join(c['error']))}</pre></td>{truth_cell}<td><a href='{rel}/console.txt'>log</a></td></tr>")
            continue
        r = c["report"]
        z = r.get("residual_at_offset_0", {})
        sw = r.get("time_offset_sweep", {})
        sync = ("not observable" if sw.get("observable") is False else
                f"{sw.get('best_offset_ms')} ms" if sw.get("best_offset_ms") is not None else "-")
        bst = r.get("residual_at_best_offset", {})
        resid = (f"{z['median_px']:.2f} px / {z.get('median_motion_px') or 0:.1f} px motion" if z.get("pairs") else "-")
        if bst.get("pairs") and sw.get("observable"):
            resid += f"; {bst['median_px']:.2f} px at best offset"
        use = bst if (bst.get("pairs") and sw.get("observable")) else z      # after sync correction, if measurable
        ratio = (use["median_px"] / z["median_motion_px"]) if z.get("median_motion_px") else None
        verdict = ("-" if ratio is None else "<span class='ok'>consistent</span>" if ratio < 0.4
                   else "<span class='bad'>telemetry does not match video</span>")
        rows.append(f"<tr><td>{html.escape(name)}</td><td class='ok'>ok</td>"
                    f"<td>{r['telemetry']['samples']} ({html.escape(r['telemetry']['source'])})</td>"
                    f"<td>{resid}</td><td>{sync}</td><td>{verdict}</td>{truth_cell}<td><a href='{rel}/report.html'>report</a></td></tr>")
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>drongo check</title>
<style>body{{font:14px system-ui,sans-serif;margin:28px;color:#1d1d1f}} td,th{{padding:5px 14px 5px 0;text-align:left;vertical-align:top}}
th{{color:#666;font-weight:500}} .ok{{color:#1b7f3b}} .bad{{color:#b42318}} pre{{margin:0;font-size:12px}}</style></head><body>
<h1 style="font-size:20px">drongo check &mdash; {time.strftime('%Y-%m-%d %H:%M')}</h1>
<h2 style="font-size:16px">Test suite</h2>
<p class="{'ok' if ok_tests else 'bad'}">{tests['tests'] - tests['skipped'] - tests['failures'] - tests['errors']} passed,
{tests['failures'] + tests['errors']} failed, {tests['skipped']} skipped in {tests['seconds']} s
&mdash; <a href="pytest.txt">output</a></p>
{''.join(f'<div class="bad">{html.escape(n)}</div>' for n in tests['failed_names'])}
<h2 style="font-size:16px">Clips</h2>
<table><tr><th>Clip</th><th>Run</th><th>Telemetry</th><th>Residual (median / image motion)</th><th>Sync</th><th>Telemetry vs video</th><th>Telemetry vs truth (synthetic only)</th><th></th></tr>{''.join(rows)}</table>
</body></html>"""
    p = out / "index.html"
    p.write_text(page, encoding="utf-8")
    return p


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    ap.add_argument("--samples", help="folder of real clips for the optional real-clip reader test (sets DRONGO_SAMPLES)")
    ap.add_argument("--clip", action="append", default=[], help="PATH or PATH:CAMERA_PRESET (repeatable)")
    ap.add_argument("--demo-dir", type=Path, help="output folder of testbed.synth.demo: check all its clips")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    print("1/4 test suite ...", flush=True)
    tests = run_pytest(a.out, a.samples) if not a.skip_tests else {"tests": 0, "failures": 0, "errors": 0, "skipped": 0,
                                                                    "seconds": 0, "failed_names": []}
    print(f"    {tests}", flush=True)

    print(f"2/4 synthetic clip (urban slalom, {SYNTH_OFFSET_MS} ms injected sync offset) ...", flush=True)
    vid, srt = make_synthetic(a.out / "synthetic")
    c = run_clip(vid, "equiv_24mm", a.out / "synthetic" / "report", ["--offset-sweep=-100:400:20"])
    c["truth"] = telemetry_vs_truth(vid, a.out / "synthetic" / "truth" / "slalom.json")
    clips = [(f"synthetic slalom (expect sync ~{SYNTH_OFFSET_MS} ms)", "synthetic/report", c)]

    demos = demo_clips(a.demo_dir) if a.demo_dir else []
    if a.demo_dir and not demos:
        print(f"    no demo clips found under {a.demo_dir} (expected truth/*.json and srt/ or embedded/)", flush=True)
    print(f"3/4 demo clips: {len(demos)} ...", flush=True)
    for label, video, truth in demos:
        cam = json.loads(truth.read_text(encoding="utf-8"))["camera"]["name"]
        rel = f"demo/{video.parent.name}_{video.stem}"
        print(f"    {label} ({cam})", flush=True)
        c = run_clip(video, cam, a.out / rel, extra or DEMO_HARNESS_ARGS)
        c["truth"] = telemetry_vs_truth(video, truth)
        clips.append((label, rel, c))

    print(f"4/4 real clips: {len(a.clip)} ...", flush=True)
    for spec in a.clip:
        head, _, tail = spec.rpartition(":")
        path, cam = (head, tail) if head and tail in PRESETS else (spec, "equiv_24mm")
        name = Path(path).stem
        print(f"    {name} ({cam})", flush=True)
        clips.append((Path(path).name, name, run_clip(Path(path), cam, a.out / name, extra)))

    p = index_html(a.out, tests, clips)
    print(f"done: {p}")
    return 0 if tests["failures"] == 0 and tests["errors"] == 0 and all(c[2]["ok"] for c in clips) else 1


if __name__ == "__main__":
    sys.exit(main())
