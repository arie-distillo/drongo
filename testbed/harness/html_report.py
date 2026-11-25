"""Single-file HTML report: interactive map (flight track, frame footprints,
rectified frames draped on the map) plus telemetry and residual plots.

Everything is embedded (images as base64, data as GeoJSON). Viewing the map
needs internet access for Leaflet and the map tiles; the
plots and frames show without it. Tiles come from Esri (satellite and street map);
OpenStreetMap's tile servers block pages opened from a local file.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from drongo import FrameGeometry, TelemetryPoseProvider, wrap_deg
from drongo.rectify import rectify_roi


def _b64_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _b64_file(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def draped_frame(img_gray: np.ndarray, g: FrameGeometry, max_px: int = 1200) -> Optional[dict]:
    """North-up rectified frame with transparency outside the view, plus its
    lat/lon bounds for a map image overlay."""
    fp = g.footprint()
    if fp is None:
        return None
    lo, hi = fp.corners_en.min(0), fp.corners_en.max(0)
    span = float(max(hi - lo))
    if span <= 0:
        return None
    gsd = max(span / max_px, float(g.gsd_m_per_px()[0]))
    roi = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
    rect = rectify_roi(img_gray, g, roi, gsd)
    alpha = np.where(rect > 0, 235, 0).astype(np.uint8)
    rgba = cv2.merge([rect, rect, rect, alpha])
    sw = g.frame.to_wgs84([[roi[0], roi[1]]])[0]
    ne = g.frame.to_wgs84([[roi[2], roi[3]]])[0]
    return {"url": _b64_png(rgba), "bounds": [[float(sw[0]), float(sw[1])], [float(ne[0]), float(ne[1])]],
            "outline": fp.to_geojson()}


def telemetry_plot(track, frame) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = list(track)
    t = np.array([p.t_ms for p in s]) / 1000.0
    en = frame.to_enu([[p.lat_deg, p.lon_deg] for p in s])
    k = max(1, int(round(1.0 / max(np.median(np.diff(t)), 1e-3)))) if len(t) > 1 else 1
    sp_t = t[k:] if len(t) > k else t
    speed = (np.linalg.norm(en[k:] - en[:-k], axis=1) / (t[k:] - t[:-k])) if len(t) > k else np.zeros_like(t)
    fig, ax = plt.subplots(2, 2, figsize=(11, 5.4), sharex=True)
    ax[0, 0].plot(t, [p.alt_agl_m for p in s]); ax[0, 0].set(title="Altitude above takeoff (m)")
    ax[0, 1].plot(sp_t, speed); ax[0, 1].set(title="Ground speed, 1 s baseline (m/s)")
    ax[1, 0].plot(t, [p.gimbal_pitch_deg for p in s], label="gimbal pitch")
    ax[1, 0].plot(t, [p.gimbal_roll_deg for p in s], label="gimbal roll"); ax[1, 0].legend()
    ax[1, 0].set(title="Gimbal pitch / roll (deg)", xlabel="video time (s)")
    yaw = np.array([p.gimbal_yaw_deg for p in s])
    ref = float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(yaw))))))   # circular mean
    near = lambda a: ref + (np.asarray(a) - ref + 180.0) % 360.0 - 180.0      # plot within +-180 of it
    ax[1, 1].plot(t, near(yaw), label="gimbal yaw")
    if len(en) > k:
        hdg = np.degrees(np.arctan2(en[k:, 0] - en[:-k, 0], en[k:, 1] - en[:-k, 1])) % 360
        moving = speed > 1.0
        ax[1, 1].plot(sp_t[moving], near(hdg[moving]), ".", ms=2, label="track heading (moving)")
    ax[1, 1].legend(); ax[1, 1].set(title="Yaw vs track heading (deg)", xlabel="video time (s)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>drongo report - {title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 body {{ font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; margin: 0; color: #1d1d1f; background: #fafafa; }}
 header {{ padding: 18px 28px 6px; }} h1 {{ font-size: 20px; margin: 0 0 4px; }} h2 {{ font-size: 16px; margin: 26px 0 8px; }}
 main {{ padding: 0 28px 40px; max-width: 1280px; }}
 #map {{ height: 560px; border: 1px solid #ddd; border-radius: 8px; }}
 table {{ border-collapse: collapse; }} td, th {{ padding: 4px 12px 4px 0; text-align: left; vertical-align: top; }}
 th {{ color: #666; font-weight: 500; }} .warn {{ color: #a15c00; }} .ok {{ color: #1b7f3b; }}
 img.plot {{ max-width: 100%; border: 1px solid #eee; border-radius: 6px; background: #fff; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }}
 .grid figure {{ margin: 0; }} .grid img {{ width: 100%; border-radius: 6px; }} figcaption {{ color: #666; font-size: 12px; }}
 .note {{ color: #666; font-size: 12.5px; max-width: 900px; }}
</style></head><body>
<header><h1>drongo report &mdash; {title}</h1><div class="note">{subtitle}</div></header>
<main>
<h2>Summary</h2>{summary}
<h2>Map</h2>
<div id="map"></div>
<p class="note">Flight track (blue), frame footprints at every whole second (orange outlines, clipped at {map_range:.0f} m),
and a few rectified frames draped on the map, each with its own footprint (dashed purple; toggle them in the layer
control, top right). A draped frame is usually taken between two orange outlines, so its shape matches its dashed
outline, not the nearest orange one. If draped frames line up with
the satellite imagery, position, attitude and camera model agree with reality; a constant offset points to GPS or
compass bias, a mismatch that grows towards the far edge to pitch or lens errors.</p>
<h2>Telemetry</h2><img class="plot" src="{telemetry_png}" alt="telemetry plots">
<h2>Telemetry vs video</h2><img class="plot" src="{residual_png}" alt="residual and sync sweep">
<p class="note">Left: difference between image motion predicted from telemetry and motion measured from the video.
Right: the same, re-scored with the telemetry clock shifted; a clear minimum is the sync offset.</p>
<h2>Ground grid on frames</h2><div class="grid">{overlays}</div>
<p class="note">A 10 m ground grid projected with the telemetry. It should stay fixed to the ground from frame to frame.</p>
</main>
<script>
const track = {track};
const footprints = {footprints};
const draped = {draped};
const map = L.map('map', {{zoomSnap: 0.25}});
// OpenStreetMap's own tile servers refuse pages opened from disk (no Referer), so street maps come from Esri too
const osm = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{{z}}/{{y}}/{{x}}', {{maxZoom: 21, maxNativeZoom: 19, attribution: 'Map &copy; Esri, HERE, Garmin, OpenStreetMap contributors'}});
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{maxZoom: 21, maxNativeZoom: 19, attribution: 'Imagery &copy; Esri'}});
sat.addTo(map);
const trackLayer = L.geoJSON(track, {{style: {{color: '#2563eb', weight: 3}}}}).addTo(map);
const fpLayer = L.geoJSON(footprints, {{style: {{color: '#f97316', weight: 1, fillOpacity: 0.04}},
  onEachFeature: (f, l) => l.bindTooltip('t = ' + (f.properties.t_ms/1000).toFixed(1) + ' s, GSD ' + f.properties.gsd_cm + ' cm/px')}}).addTo(map);
map.fitBounds(fpLayer.getBounds().extend(trackLayer.getBounds()), {{padding: [20, 20]}});
const overlays = {{'Track': trackLayer, 'Footprints': fpLayer}};
draped.forEach((d, i) => {{
  const lyr = L.layerGroup([L.imageOverlay(d.url, d.bounds, {{opacity: 0.9}}),
                            L.geoJSON(d.outline, {{style: {{color: '#a21caf', weight: 2.5, dashArray: '6 4', fill: false}}}})]);
  if (i === 0) lyr.addTo(map);
  overlays['Frame at ' + d.t.toFixed(1) + ' s'] = lyr; }});
const c = track.geometry.coordinates;
L.circleMarker([c[0][1], c[0][0]], {{radius: 6, color: '#16a34a'}}).bindTooltip('start').addTo(map);
L.circleMarker([c[c.length-1][1], c[c.length-1][0]], {{radius: 6, color: '#dc2626'}}).bindTooltip('end').addTo(map);
L.control.layers({{'Satellite': sat, 'Map': osm}}, overlays, {{collapsed: false}}).addTo(map);
L.control.scale({{metric: true, imperial: false}}).addTo(map);
</script></body></html>"""


def _summary_html(report: dict) -> str:
    t = report["telemetry"]
    z = report.get("residual_at_offset_0", {})
    b = report.get("residual_at_best_offset", {})
    sw = report.get("time_offset_sweep", {})
    rows = [
        ("Video", f"{report['video_info']['size'][0]} x {report['video_info']['size'][1]}, "
                  f"{report['video_info']['fps']:.2f} fps, {report['video_info']['duration_ms'] / 1000:.1f} s"),
        ("Camera", f"{report['camera']['name']} (HFOV {report['camera']['hfov_deg']:.1f} deg)"),
        ("Telemetry", f"{t['samples']} samples ({t['source']}), every {t['median_interval_ms']:.0f} ms"),
        ("Flight", f"altitude {t['alt_agl_m'][0]:.0f}-{t['alt_agl_m'][1]:.0f} m, gimbal pitch "
                   f"{t['pitch_deg'][0]:.1f} to {t['pitch_deg'][1]:.1f} deg"
                   + (f", speed median {t['speed_ms']['median']:.1f} m/s" if t.get("speed_ms") else "")),
    ]
    flags = []
    if t.get("position_jumps"):
        flags.append(f"<span class='warn'>{t['position_jumps']} position jumps</span>")
    if not t.get("has_gimbal"):
        flags.append("<span class='warn'>no gimbal angles in telemetry (nadir assumed)</span>")
    if t.get("zero_altitude_samples"):
        flags.append(f"<span class='warn'>{t['zero_altitude_samples']} zero-altitude samples</span>")
    rows.append(("Checks", ", ".join(flags) or "<span class='ok'>no anomalies</span>"))
    if z.get("pairs"):
        mot = z.get("median_motion_px")
        ratio = (z["median_px"] / mot) if mot else None
        cls = "ok" if ratio is not None and ratio < 0.3 else "warn"
        rows.append(("Residual", f"<span class='{cls}'>{z['median_px']:.2f} px median ({z['median_m']:.2f} m)</span> "
                                 f"against {mot:.1f} px of image motion per pair; {z['pairs']} pairs"))
    if sw.get("observable") is False:
        rows.append(("Sync offset", "not observable on this clip (uniform motion)"))
    elif sw.get("best_offset_ms") is not None:
        rows.append(("Sync offset", f"{sw['best_offset_ms']} ms (residual {b.get('median_px', float('nan')):.2f} px)"))
    return "<table>" + "".join(f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"


def write_html(out_dir: Path, report: dict, track, provider: TelemetryPoseProvider,
               frames: dict, fps: float, w: int, h: int, overlay_idx, map_range_m: float) -> Path:
    from drongo import TelemetryPoseProvider as TPP
    mprov = TPP(track, provider.camera, provider.frame, extrapolate_ms=100, max_range_m=map_range_m)
    fps_ = fps or 30.0
    n = int(report["video_info"]["frames"])
    feats = []
    for i in range(0, n, max(1, int(round(fps_)))):
        g = mprov.geometry(i * 1000.0 / fps_, w, h)
        fp = g.footprint() if g else None
        if fp:
            feats.append(fp.to_geojson({"t_ms": i * 1000.0 / fps_, "gsd_cm": round(float(g.gsd_m_per_px()[0] * 100), 1)}))
    draped = []
    for i in sorted(overlay_idx):
        if i in frames:
            g = mprov.geometry(i * 1000.0 / fps_, w, h)
            d = draped_frame(frames[i], g) if g else None
            if d:
                d["t"] = i / fps_
                draped.append(d)
    track_gj = {"type": "Feature", "properties": {},
                "geometry": {"type": "LineString", "coordinates": [[p.lon_deg, p.lat_deg] for p in track]}}
    overlays = "".join(
        f"<figure><img src='{_b64_file(p)}'><figcaption>frame {int(p.stem.split('_')[1])}</figcaption></figure>"
        for p in sorted(out_dir.glob("overlay_*.png")))
    page = _TEMPLATE.format(
        title=html.escape(Path(report["video"]).name),
        subtitle=html.escape(report["video"]),
        summary=_summary_html(report), map_range=map_range_m,
        telemetry_png=telemetry_plot(track, provider.frame),
        residual_png=_b64_file(out_dir / "residuals.png"),
        overlays=overlays,
        track=json.dumps(track_gj), footprints=json.dumps({"type": "FeatureCollection", "features": feats}),
        draped=json.dumps(draped))
    path = out_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
