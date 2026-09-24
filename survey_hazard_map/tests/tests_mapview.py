#!/usr/bin/env python3
"""map.html for verified exports and ping-navigated surveys.

    python3 tests_mapview.py

Plain asserts, no test runner. Exits 1 on the first failure with the reason.

What is covered, and why each matters:

    ping navigation      _axis_aligned used to read affine coefficients off
                         every reference and crashed on a "ping" one. A strip
                         ingested from XTF is now resampled north-up, and the
                         resampled picture must put a feature where the
                         strip's own navigation says that feature is.
    verification         confidence_pct, dimensions and suppression reach the
                         map: popups show the percentage and the detector's
                         score, filtered detections sit on their own layer,
                         off by default, with their reasons.
    older exports        an export written before verification renders exactly
                         as before, with no filtered layer and no percentage.
    real surveys         render_map runs without an exception for every survey
                         in data/surveys, always into a temporary copy. Nothing
                         under data/ is written by this file.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map import hazard_mapview as mv
from survey_hazard_map.hazard_geo import PING_SIDECAR_FORMAT, Georeference, offset_latlon

ROOT = Path(__file__).resolve().parents[2]
SURVEYS = ROOT / "data" / "surveys"
PING_SURVEY = "demo-ghosttrace-mannar"
RELATIVE_SURVEYS = ("s7-submarine", "waterfall-strip", "demo-synthetic")

CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"  ok    {message}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _external_resources(html: str) -> list[str]:
    return sorted(set(
        re.findall(r'<link[^>]+href="(https?://[^"]+)"', html)
        + re.findall(r'<script[^>]+src="(https?://[^"]+)"', html)
        + re.findall(r'<img[^>]+src="(https?://[^"]+)"', html)))


def _render_copy(survey: str, workspace: Path, export: dict | None = None,
                 filename: str = "map.html") -> tuple[Path, str]:
    """Copy a survey into the workspace and render its map there, never in place."""
    source = SURVEYS / survey
    target = workspace / survey
    if not target.exists():
        shutil.copytree(source, target)
    export = export if export is not None else json.loads(
        (target / cfg.EXPORT_JSON).read_text(encoding="utf-8"))
    manifest = target / cfg.MANIFEST_JSON
    path = mv.render_map(export, target, target / "tiles",
                         manifest if manifest.is_file() else None,
                         title=export["metadata"].get("title"),
                         demo=bool(export["metadata"].get("demo")),
                         filename=filename)
    return path, path.read_text(encoding="utf-8")


def _common_page_checks(html: str, label: str) -> None:
    check(html.lstrip().lower().startswith("<!doctype html>"), f"{label}: a complete document")
    check(not _external_resources(html), f"{label}: fetches nothing from the network")
    check("$(`" not in html and "deHtml(`" in html,
          f"{label}: popups are built without jQuery, which the page does not carry")
    check("DOMContentLoaded" in html, f"{label}: the panel script waits for the map to exist")
    check("deepecho:focus" in html, f"{label}: hotspot focus by postMessage is still wired")
    check("${" not in html, f"{label}: no template placeholder survived")


def _verified_checks(html: str, export: dict, label: str) -> None:
    detections = export["detections"]
    suppressed = [d for d in detections if d.get("suppressed")]
    check(f"{mv.FILTERED_LAYER} ({len(suppressed)})" in html,
          f"{label}: the layer control offers '{mv.FILTERED_LAYER} ({len(suppressed)})'")
    check(re.search(r'"' + re.escape(mv.FILTERED_LAYER) + r' \(\d+\)"\s*:\s*feature_group_\w+',
                    html) is not None,
          f"{label}: the filtered layer is an overlay in the layer control")
    check("Filtered false positives</div>" in html,
          f"{label}: the survey summary states the filtered count")
    check(f">{len(suppressed)} filtered</span>" in html, f"{label}: the header states it too")
    for detection in detections:
        pct = detection.get("confidence_pct")
        if pct is not None:
            check(f"<b>{float(pct):.1f}%</b>" in html,
                  f"{label}: {detection['id']} shows confidence {float(pct):.1f}%")
    check("<th>Detector</th>" in html, f"{label}: the detector's own score is shown beside it")
    if any(isinstance(d.get("dimensions"), dict) for d in detections):
        check("<th>Size</th>" in html, f"{label}: popups show the object's size")


def test_axis_aligned_ping() -> None:
    print("\nPING REFERENCES")
    reference = _synthetic_ping_reference(40, 60)
    check(mv._axis_aligned(reference) == (False, True),
          "a ping reference is reported as not north-up, rather than crashing")
    check(mv._resamplable(reference), "a ping reference can place every pixel")
    along = Georeference("s", "along_track", {"origin": [0, 0], "direction": [0, 1],
                                              "t": [0, 1], "lat": [0, 1], "lon": [0, 0]})
    check(not mv._resamplable(along),
          "along-track navigation, which cannot resolve across-track, is never resampled")
    affine = Georeference("s", "affine", {"lat_coefficients": [0.0, -1e-6, 10.0],
                                          "lon_coefficients": [1e-6, 0.0, 70.0]})
    check(mv._axis_aligned(affine) == (True, True), "a north-up affine strip stays a rectangle")
    corners = Georeference.from_corners("s", {"top_left": [10.0, 70.0], "top_right": [10.001, 70.01],
                                              "bottom_left": [9.99, 70.001],
                                              "bottom_right": [9.991, 70.011]}, 100, 100)
    check(not mv._axis_aligned(corners)[0], "rotated corners are no longer drawn as a rectangle")


def _synthetic_ping_reference(width: int, height: int) -> Georeference:
    """A towfish track that turns from 60 to 120 degrees over the strip."""
    lat, lon = 9.12, 79.05
    rows = []
    for row in range(height):
        heading = 60.0 + 60.0 * row / max(1, height - 1)
        rows.append({"row": row, "lat": lat, "lon": lon, "heading_deg": heading, "quality": "ok"})
        rad = math.radians(heading)
        lat, lon = offset_latlon(lat, lon, 0.1 * math.sin(rad), 0.1 * math.cos(rad))
    sidecar = {"format": PING_SIDECAR_FORMAT, "rows": rows, "nadir_col": width / 2.0,
               "m_per_px_across": 0.1, "m_per_px_along": 0.1, "port_is_left": True,
               "width": width, "height": height}
    return Georeference.from_ping_nav("synthetic", sidecar)


def test_resample_places_features() -> None:
    print("\nRESAMPLING")
    width, height = 1000, 3000
    reference = _synthetic_ping_reference(width, height)
    grey = np.full((height, width), 30, dtype=np.uint8)
    targets = [(120, 200), (500, 1500), (880, 2800), (300, 2400), (700, 600)]
    for x, y in targets:
        grey[y - 3:y + 4, x - 3:x + 4] = 255

    started = time.perf_counter()
    warped = mv._resample_north_up(grey, reference)
    elapsed = time.perf_counter() - started
    check(warped is not None, "a 1000 x 3000 ping strip resamples")
    check(elapsed < 8.0, f"and does so in a few seconds ({elapsed:.2f} s)")
    out_w, out_h = warped["size"]
    check(max(out_w, out_h) <= mv.RESAMPLE_MAX_EDGE,
          f"the picture is capped at {mv.RESAMPLE_MAX_EDGE} px ({out_w} x {out_h})")
    check(warped["data_uri"].startswith("data:image/png;base64,"),
          "the picture is a PNG data URI, so it can be transparent")

    from PIL import Image
    import base64
    import io

    image = Image.open(io.BytesIO(base64.b64decode(warped["data_uri"].split(",", 1)[1])))
    check(image.mode == "LA" and image.size == (out_w, out_h), "luminance plus alpha, at its size")
    pixels = np.asarray(image)
    alpha = pixels[..., 1]
    check(alpha[0, 0] == 0 or alpha[-1, -1] == 0 or alpha[0, -1] == 0 or alpha[-1, 0] == 0,
          "outside the turning strip's footprint the picture is transparent")
    covered = float((alpha > 0).mean())
    check(0.2 < covered < 0.95, f"the footprint covers part of the grid, not all ({covered:.0%})")

    (south, west), (north, east) = warped["bounds"]
    merc = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))  # noqa: E731
    for x, y in targets:
        lat, lon = reference.locate(x + 0.5, y + 0.5)
        col = (lon - west) / (east - west) * out_w
        row = (merc(north) - merc(lat)) / (merc(north) - merc(south)) * out_h
        r0, c0 = min(int(row), out_h - 1), min(int(col), out_w - 1)
        value = int(pixels[r0, c0, 0])
        check(value >= 150 and int(pixels[r0, c0, 1]) == 255,
              f"the bright feature at strip pixel ({x}, {y}) lands in the output cell "
              f"its navigated position falls in (value {value})")

    # No interior pinholes: every cell whose four direct neighbours are covered
    # is covered itself.
    inner = alpha[1:-1, 1:-1] == 0
    boxed = (alpha[:-2, 1:-1] > 0) & (alpha[2:, 1:-1] > 0) & (alpha[1:-1, :-2] > 0) & (alpha[1:-1, 2:] > 0)
    check(int((inner & boxed).sum()) == 0, "no pinholes are left inside the footprint")

    footprint = warped["footprint"]
    check(len(footprint) > 8 and all(south - 1e-9 <= p[0] <= north + 1e-9 for p in footprint),
          "the strip outline follows the track and sits inside the picture's bounds")


def test_ping_survey(workspace: Path) -> None:
    print("\nPING SURVEY, " + PING_SURVEY)
    source = SURVEYS / PING_SURVEY
    if not (source / cfg.EXPORT_JSON).is_file():
        raise AssertionError(f"{source} is missing; it is the ping-navigated test survey")
    before = _digest(source / "map.html")

    path, html = _render_copy(PING_SURVEY, workspace)
    export = json.loads((workspace / PING_SURVEY / cfg.EXPORT_JSON).read_text())
    check(_digest(source / "map.html") == before, "the survey in data/ was not touched")
    check(path.parent == workspace / PING_SURVEY, "the map was written to the temporary copy")
    check(export["provenance"]["navigation"]["mode"] == "ping",
          "the survey really is ping-navigated")
    check(re.search(r'imageOverlay\(\s*"data:image/png;base64,', html) is not None,
          "the sonar strip is a data-URI PNG image overlay")
    check("crs: L.CRS.Simple" not in html, "drawn on real coordinates")
    check("resampled onto a north-up latitude/longitude grid for display only" in html,
          "the footer says the imagery is resampled for display only")
    check("L.polygon(" in html, "rotated tile and hotspot cells are drawn as their true footprint")
    _common_page_checks(html, "ping")
    _verified_checks(html, export, "ping")

    # The same survey with one detection filtered, to see the filtered path on
    # a navigated map. The export is modified in memory only.
    modified = copy.deepcopy(export)
    target = modified["detections"][0]
    target["suppressed"] = True
    target["confidence_pct"] = 12.5
    target["verification"] = dict(target.get("verification") or {}, status="checked",
                                  hard_reasons=["nadir_zone"],
                                  reasons=["sits on the nadir / water-column boundary"],
                                  suppressed=True)
    modified["hotspots"] = [h for h in modified["hotspots"]
                            if target["id"] not in h["detection_ids"]]
    for rank, hotspot in enumerate(modified["hotspots"], start=1):
        hotspot["priority_rank"] = rank
    modified["survey_summary"]["suppressed_detections"] = 1
    _, filtered_html = _render_copy(PING_SURVEY, workspace, modified, "map_filtered.html")
    check(f"{mv.FILTERED_LAYER} (1)" in filtered_html, "one filtered detection gives '(1)'")
    check(re.search(r'"' + re.escape(mv.FILTERED_LAYER) + r' \(1\)"\s*:\s*feature_group_\w+',
                    filtered_html) is not None, "and it is toggleable in the layer control")
    check("<th>Filtered because</th>" in filtered_html
          and "nadir / water-column artefact" in filtered_html
          and "sits on the nadir / water-column boundary" in filtered_html,
          "the popup explains the filter in plain English")
    check("de-filtered-chip'>filtered</span>" in filtered_html, "the popup is marked filtered")
    check('"dashArray": "3 3"' in filtered_html and f'"color": "{mv.FILTERED_COLOR}"' in filtered_html,
          "the filtered marker is grey and dashed")
    check(re.search(r'"fill": false', filtered_html) is not None, "and hollow")
    check('"className": "de-filtered-tip"' in filtered_html and '"permanent": true' in filtered_html,
          "and labelled 'filtered' on the map")
    layer_names = dict(re.findall(r'"([^"]+)"\s*:\s*(feature_group_\w+)', filtered_html))
    filtered_group = layer_names.get(f"{mv.FILTERED_LAYER} (1)")
    check(filtered_group is not None
          and f"{filtered_group}.addTo(" not in filtered_html,
          "the filtered layer is off when the map opens")


def test_relative_surveys(workspace: Path) -> None:
    print("\nRELATIVE SURVEYS")
    from survey_hazard_map import demo_survey
    generated = workspace / "generated-demo"
    demo_survey.main(["--out", str(generated), "--quiet"])
    export = json.loads((generated / cfg.EXPORT_JSON).read_text())
    html = (generated / "map.html").read_text(encoding="utf-8")
    check("suppressed_detections" in export["survey_summary"],
          "a freshly generated relative survey carries verification")
    check(re.search(r'imageOverlay\(\s*"data:image/jpeg;base64,', html) is not None,
          "relative: the sonar strip is a data-URI image overlay")
    check("crs: L.CRS.Simple" in html, "relative: drawn on a pixel plane")
    check("<th>Latitude</th>" not in html, "relative: no latitude anywhere")
    check("resampled onto a north-up" not in html, "relative: nothing is resampled")
    _common_page_checks(html, "relative")
    _verified_checks(html, export, "relative")
    suppressed = [d for d in export["detections"] if d.get("suppressed")]
    if suppressed:
        check("<th>Filtered because</th>" in html, "relative: filtered detections explain why")
        heat = re.search(r"L\.heatLayer\(\s*(\[\[.*?\]\])\s*,", html, re.S)
        weights = sorted(round(float(w), 4) for w in re.findall(r",\s*([0-9.]+)\]", heat.group(1)))
        expected = sorted(round(float(d["severity"]), 4) for d in export["detections"]
                          if not d.get("suppressed"))
        check(weights == expected, "relative: filtered detections add no heat")

    # An export from before verification: every new field removed.
    old = copy.deepcopy(export)
    old["survey_summary"].pop("suppressed_detections", None)
    for detection in old["detections"]:
        for key in ("confidence_pct", "confidence_pct_basis", "suppressed",
                    "dimensions", "verification"):
            detection.pop(key, None)
    old_path = mv.render_map(old, generated, generated / "tiles", generated / cfg.MANIFEST_JSON,
                             title="old export", demo=True, filename="map_old.html")
    old_html = old_path.read_text(encoding="utf-8")
    check(mv.FILTERED_LAYER not in old_html and " filtered</span>" not in old_html,
          "an older export gets no filtered layer and no filtered count")
    check("%</b>" not in old_html and "<th>Confidence</th>" in old_html,
          "and shows the detector confidence as it always did")
    check(old_html.count("L.circleMarker(") >= len(old["detections"]),
          "and still draws every detection")
    check("SYNTHETIC DEMO DATA" in old_html, "the synthetic banner is kept")

    for survey in RELATIVE_SURVEYS:
        if not (SURVEYS / survey / cfg.EXPORT_JSON).is_file():
            print(f"  skip  {survey}: not present")
            continue
        before = _digest(SURVEYS / survey / "map.html")
        _, page = _render_copy(survey, workspace)
        loaded = json.loads((workspace / survey / cfg.EXPORT_JSON).read_text())
        check(_digest(SURVEYS / survey / "map.html") == before, f"{survey}: data/ untouched")
        check(re.search(r'imageOverlay\(\s*"data:image/jpeg;base64,', page) is not None,
              f"{survey}: renders with its imagery as a data URI")
        _common_page_checks(page, survey)
        if "suppressed_detections" in loaded["survey_summary"]:
            _verified_checks(page, loaded, survey)
        if loaded["metadata"].get("demo"):
            check("SYNTHETIC DEMO DATA" in page, f"{survey}: the synthetic banner is kept")


def main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    workspace = Path(tempfile.mkdtemp(prefix="deepecho-mapview-"))
    try:
        test_axis_aligned_ping()
        test_resample_places_features()
        test_ping_survey(workspace)
        test_relative_surveys(workspace)
    except AssertionError as exc:
        print(f"\nFAILED  {exc}")
        return 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    print(f"\nPASSED  {CHECKS}/{CHECKS} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
