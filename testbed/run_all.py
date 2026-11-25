"""One command for a full drongo check: test suite, a synthetic clip with known
answers, and any real clips you name. Writes reports/index.html linking every
report.

    python -m testbed.run_all
    python -m testbed.run_all --clip "D:/clips/city1-25-40.mov:mavic3_wide" --clip "D:/clips/2 cars 2m.mp4:equiv_24mm"

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
from drongo import CameraModel  # noqa: E402

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


def make_synthetic(out: Path) -> tuple[Path, Path]:
    subprocess.run([sys.executable, "-m", "testbed.synth.make_clip", "--scenario", "slalom", "--offset-ms", "160",
                    "--camera", "equiv_24mm", "--out", str(out)], cwd=ROOT, env=_env(), check=True, capture_output=True)
    return out / "slalom.mp4", out / "slalom.SRT"


def index_html(out: Path, tests: dict, clips: list[tuple[str, str, dict]]) -> Path:
    ok_tests = tests["failures"] == 0 and tests["errors"] == 0
    rows = []
    for name, rel, c in clips:
        if not c["ok"]:
            rows.append(f"<tr><td>{html.escape(name)}</td><td class='bad'>failed</td><td colspan='4'><pre>"
                        f"{html.escape(chr(10).join(c['error']))}</pre></td><td><a href='{rel}/console.txt'>log</a></td></tr>")
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
                    f"<td>{resid}</td><td>{sync}</td><td>{verdict}</td><td><a href='{rel}/report.html'>report</a></td></tr>")
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
<table><tr><th>Clip</th><th>Run</th><th>Telemetry</th><th>Residual (median / image motion)</th><th>Sync</th><th>Telemetry vs video</th><th></th></tr>{''.join(rows)}</table>
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
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    print("1/3 test suite ...", flush=True)
    tests = run_pytest(a.out, a.samples) if not a.skip_tests else {"tests": 0, "failures": 0, "errors": 0, "skipped": 0,
                                                                    "seconds": 0, "failed_names": []}
    print(f"    {tests}", flush=True)

    print("2/3 synthetic clip (slalom, 160 ms injected sync offset) ...", flush=True)
    vid, srt = make_synthetic(a.out / "synthetic")
    clips = [("synthetic slalom (expect sync ~160 ms)", "synthetic/report",
              run_clip(vid, "equiv_24mm", a.out / "synthetic" / "report", ["--offset-sweep=-100:400:20"]))]

    print(f"3/3 real clips: {len(a.clip)} ...", flush=True)
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
