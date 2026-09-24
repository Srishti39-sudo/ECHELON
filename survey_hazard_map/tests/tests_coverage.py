#!/usr/bin/env python3
"""Coverage, blind spots, re-look lines and mission replay (hazard_coverage.py).

    python3 tests_coverage.py

Plain asserts, no test runner. Exits 1 on the first failure with the reason.
Every survey under data/surveys is read from a temporary copy, so nothing under
data/ is written by this file (coverage.json included).

What is covered, and why each matters:

    analytic swath     a straight synthetic line of known length, altitude and
                       range: footprint, nadir strip and imaged area must match
                       the closed-form areas within 5%, for a ground-range
                       (sonar_ingest) sidecar and a slant-range (geotag) one.
    degraded rows      rows flagged dropout never count as imaged, on the
                       synthetic line and on the Gulf of Mannar demo.
    Mannar demo        imaged area > 0 and no larger than the survey's hull or
                       footprint; the nadir strip is non-zero and imaged +
                       nadir + gaps accounts for the footprint.
    re-look lines      exist for a low-confidence, an unidentified and a
                       filtered-but-severe contact, run perpendicular to the
                       track, and keep the contact out of the new nadir strip.
    exports            GPX parses as XML with waypoints and routes; CSV parses.
    no navigation      an image survey answers available: false with a reason,
                       over HTTP too, and its downloads are 404.
    cache              coverage.json is reused, and recomputed once export.json
                       is newer.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from survey_hazard_map import hazard_coverage as hc
from survey_hazard_map.hazard_geo import PING_SIDECAR_FORMAT, offset_latlon

ROOT = Path(__file__).resolve().parents[2]
SURVEYS = ROOT / "data" / "surveys"
PING_SURVEY = "demo-ghosttrace-mannar"
IMAGE_SURVEY = "s7-submarine"
GPX_NS = "{http://www.topografix.com/GPX/1/1}"

CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"  ok    {message}")


def close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance * abs(expected)


def copy_survey(name: str, into: Path) -> Path:
    target = into / name
    shutil.copytree(SURVEYS / name, target, ignore=shutil.ignore_patterns(
        "tiles", "strips", "map.html", "*.wc.npz", hc.CACHE_FILE))
    return target


# --- synthetic straight line --------------------------------------------------------

LENGTH_M = 200.0
ALONG = 0.1
ALTITUDE = 10.0
GROUND_MAX = 45.0
DS = 0.1
LAT0, LON0 = 9.0, 79.0


def straight_survey(root: Path, *, kind: str, dropout: tuple[int, int] | None = None) -> Path:
    """A due-east line: LENGTH_M long, one row per ALONG metres, constant altitude."""
    survey = root / f"straight-{kind}-{'drop' if dropout else 'clean'}"
    (survey / "nav").mkdir(parents=True)
    n = int(round(LENGTH_M / ALONG)) + 1
    rows = []
    for i in range(n):
        lat, lon = offset_latlon(LAT0, LON0, i * ALONG, 0.0)
        bad = dropout is not None and dropout[0] <= i <= dropout[1]
        row = {"row": i, "time": f"2026-01-01T00:{(i * 0.05) // 60:02.0f}:{(i * 0.05) % 60:06.3f}+00:00",
               "lat": lat, "lon": lon, "heading_deg": 90.0,
               "altitude_m": None if bad else ALTITUDE, "quality": "dropout" if bad else "ok"}
        if kind == "slant":
            row.update({"slant_range_m": 50.0, "samples_per_side": 500})
        rows.append(row)
    sidecar = {"format": PING_SIDECAR_FORMAT, "width": 1000, "height": n, "nadir_col": 500.0,
               "m_per_px_along": ALONG, "port_is_left": True, "synthetic": True, "rows": rows,
               "degraded_rows": [[dropout[0], dropout[1], "dropout"]] if dropout else []}
    if kind == "ground":
        sidecar.update({"m_per_px_across": 0.1, "processing": {"slant_range_correction": {
            "ground_range_max_m": GROUND_MAX, "native_slant_m_per_sample": DS}}})
        mode = "ping"
    else:
        sidecar.update({"m_per_px_across": None, "slant_range_corrected": False,
                        "slant_m_per_sample": DS})
        mode = "geotag_slant_range"
    (survey / "nav" / "line.nav.json").write_text(json.dumps(sidecar), encoding="utf-8")
    (survey / "manifest.json").write_text(json.dumps({"survey": {"navigation": {
        "mode": mode, "sidecars": {"line": "nav/line.nav.json"}}}, "tiles": []}), encoding="utf-8")
    (survey / "export.json").write_text(json.dumps({
        "metadata": {"survey_id": survey.name, "demo": True}, "detections": []}), encoding="utf-8")
    return survey


def test_analytic(tmp: Path) -> None:
    print("analytic straight line")
    g0 = math.sqrt((ALTITUDE + DS) ** 2 - ALTITUDE ** 2)

    survey = straight_survey(tmp, kind="ground")
    metrics = hc.compute_coverage(survey)["metrics"]
    footprint = LENGTH_M * 2 * GROUND_MAX
    nadir = LENGTH_M * 2 * g0
    check(close(metrics["swath_footprint_km2"] * 1e6, footprint, 0.05),
          f"ground-range footprint {metrics['swath_footprint_km2'] * 1e6:.0f} m² ≈ {footprint:.0f} m²")
    check(close(metrics["nadir_blind_km2"] * 1e6, nadir, 0.05),
          f"nadir strip {metrics['nadir_blind_km2'] * 1e6:.0f} m² ≈ {nadir:.0f} m² "
          f"(2 × {g0:.3f} m × {LENGTH_M:g} m)")
    check(close(metrics["imaged_km2"] * 1e6, footprint - nadir, 0.05),
          f"imaged {metrics['imaged_km2'] * 1e6:.0f} m² ≈ {footprint - nadir:.0f} m²")
    check(close(metrics["track_length_km"] * 1000, LENGTH_M, 0.01), "track length 200 m")
    check(metrics["imaged_pct_of_hull"] <= 100.0, "imaged share of hull is a percentage")

    slant = straight_survey(tmp, kind="slant")
    doc = hc.compute_coverage(slant)
    half = math.sqrt(50.0 ** 2 - ALTITUDE ** 2)
    check(close(doc["metrics"]["swath_footprint_km2"] * 1e6, LENGTH_M * 2 * half, 0.05),
          f"slant-range footprint uses sqrt(slant² − altitude²) = {half:.2f} m per side")
    check(doc["method"]["strips"][0]["geometry"] == "slant_range", "slant-range geometry recorded")

    dropped = straight_survey(tmp, kind="ground", dropout=(800, 899))
    doc = hc.compute_coverage(dropped)
    lost = 10.0 * (2 * GROUND_MAX - 2 * g0)
    check(close(doc["metrics"]["imaged_km2"] * 1e6, footprint - nadir - lost, 0.05),
          f"100 dropout rows remove ≈ {lost:.0f} m² from the imaged area")
    check(close(doc["metrics"]["degraded_gap_by_reason_km2"].get("dropout", 0) * 1e6,
                10.0 * 2 * GROUND_MAX, 0.1),
          "dropout gap area reported by reason (10 m × full swath)")
    from shapely.geometry import Point, shape
    imaged = shape(doc["polygons"]["features"][0]["geometry"])
    mid_lat, mid_lon = offset_latlon(LAT0, LON0, 85.0, 20.0)
    check(not imaged.contains(Point(mid_lon, mid_lat)),
          "a point inside the dropout rows' swath is not imaged")
    gaps = doc["gaps"]["features"]
    check(len(gaps) == 1 and gaps[0]["properties"]["relook"],
          "the dropout gap is a gap with re-look lines")
    lines = [f for f in doc["relook_lines"]["features"] if f["properties"]["kind"] == "coverage_gap"]
    check(len(lines) >= 1 and all(abs(f["properties"]["heading_deg"] - 90.0) < 0.5 for f in lines),
          f"{len(lines)} gap re-look line(s) on the original heading")


# --- the Gulf of Mannar demo ----------------------------------------------------------


def test_mannar(tmp: Path) -> dict:
    print("demo-ghosttrace-mannar")
    survey = copy_survey(PING_SURVEY, tmp)
    doc = hc.coverage_for_survey(survey)
    check(doc["available"] is True, "coverage available for a ping-navigated survey")
    m = doc["metrics"]
    check(m["imaged_km2"] > 0, f"imaged area {m['imaged_km2']} km² > 0")
    check(m["imaged_km2"] <= m["hull_km2"], "imaged area does not exceed the convex hull")
    check(m["imaged_km2"] <= m["swath_footprint_km2"], "imaged area does not exceed the footprint")
    check(0 < m["imaged_pct_of_hull"] <= 100, f"imaged {m['imaged_pct_of_hull']}% of the hull")
    check(m["nadir_blind_km2"] > 0, f"nadir blind strip {m['nadir_blind_km2'] * 1e6:.0f} m²")
    accounted = m["imaged_km2"] + m["nadir_blind_km2"] + m["degraded_gap_km2"]
    check(close(accounted, m["swath_footprint_km2"], 0.02),
          "imaged + nadir strip + degraded gaps accounts for the footprint (±2%)")
    check(m["rows_degraded"] == 262, "every degraded sidecar row is counted")
    check(m["degraded_gap_by_reason_km2"].get("dropout", 0) > 0, "dropout rows are a gap")

    from shapely.geometry import shape
    imaged = shape(doc["polygons"]["features"][0]["geometry"])
    for gap in doc["gaps"]["features"]:
        poly = shape(gap["geometry"])
        check(imaged.intersection(poly).area <= 0.01 * poly.area,
              f"gap {gap['properties']['id']} ({gap['properties']['reason']}) is not imaged")
    for line in doc["relook_lines"]["features"]:
        check(line["geometry"]["type"] == "LineString" and line["properties"]["reason"],
              f"{line['properties']['id']} is a LineString with a reason")

    replay = hc.replay_for_survey(survey)
    track = replay["track"]
    check(replay["available"] and len(track["t"]) <= hc.REPLAY_MAX_POINTS,
          f"replay has {len(track['t'])} points")
    check(all(b >= a for a, b in zip(track["t"], track["t"][1:])), "replay is in time order")
    check(all(len(v) == len(track["t"]) for v in track.values()), "columnar track is rectangular")
    check(any(q == "dropout" for q in track["quality"]), "dropout rows reach the replay track")
    check(any(d["reason"] == "dropout" for d in replay["degraded"]), "degraded intervals listed")
    check(len(replay["detections"]) == 6 and all(0 <= d["t"] <= replay["duration_s"]
                                                  for d in replay["detections"]),
          "every detection has a replay time inside the survey")
    check(close(replay["totals"]["area_km2"], m["imaged_km2"], 0.01),
          "replay's final area matches the coverage imaged area (±1%)")
    check(close(replay["totals"]["distance_km"], m["track_length_km"], 0.01),
          "replay's final distance matches the track length")

    gpx = hc.relook_gpx(doc)
    root = ET.fromstring(gpx.split("\n", 1)[1])
    waypoints = root.findall(f"{GPX_NS}wpt")
    routes = root.findall(f"{GPX_NS}rte")
    check(root.tag == f"{GPX_NS}gpx" and len(waypoints) >= 2 * m["relook_lines"],
          f"GPX parses: {len(waypoints)} waypoints")
    check(len(routes) == m["relook_lines"], f"GPX has one route per line ({len(routes)})")
    rows = list(csv.DictReader(io.StringIO(hc.relook_csv(doc))))
    check(len(rows) == m["relook_lines"] and set(hc.RELOOK_CSV_COLUMNS) <= set(rows[0]),
          "CSV parses with one row per line")

    # Cache: reused, then recomputed when export.json is newer.
    cache = survey / hc.CACHE_FILE
    check(cache.is_file(), "coverage.json written beside export.json")
    first = json.loads(cache.read_text())["generated_at"]
    time.sleep(1.1)
    check(hc.coverage_for_survey(survey)["generated_at"] == first, "cache reused while fresh")
    future = time.time() + 5
    os.utime(survey / "export.json", (future, future))
    check(hc.coverage_for_survey(survey)["generated_at"] != first,
          "cache recomputed once export.json is newer")
    return doc


def test_contacts(tmp: Path) -> None:
    print("re-look lines for contacts")
    survey = copy_survey(PING_SURVEY, tmp / "contacts")
    export_path = survey / "export.json"
    export = json.loads(export_path.read_text())
    base = export["detections"]
    low = copy.deepcopy(base[1])        # a net: its action does not ask for identification
    low.update({"id": "TEST_low", "confidence_pct": 45.0, "recommended_action": "Schedule cleanup"})
    unknown = copy.deepcopy(base[2])
    unknown.update({"id": "TEST_unknown", "object_class": "unknown", "class_normalized": "unknown",
                    "confidence_pct": 82.0, "recommended_action": "Flag navigation hazard"})
    filtered = copy.deepcopy(base[4])
    filtered.update({"id": "TEST_filtered", "suppressed": True, "severity_tier": "critical",
                     "confidence_pct": 70.0, "recommended_action": "Deploy EOD team"})
    plain = copy.deepcopy(base[5])
    plain.update({"id": "TEST_plain", "confidence_pct": 90.0, "recommended_action": "Schedule cleanup"})
    export["detections"] = [low, unknown, filtered, plain]
    export_path.write_text(json.dumps(export))

    doc = hc.compute_coverage(survey)
    by_target = {f["properties"]["target_id"]: f for f in doc["relook_lines"]["features"]
                 if f["properties"]["kind"] == "contact"}
    check("TEST_low" in by_target and "low confidence" in by_target["TEST_low"]["properties"]["reason"],
          "a low-confidence contact gets a re-look line")
    check("TEST_unknown" in by_target
          and "unidentified" in by_target["TEST_unknown"]["properties"]["reason"],
          "an unidentified contact gets a re-look line")
    check("TEST_filtered" in by_target
          and "filtered" in by_target["TEST_filtered"]["properties"]["reason"],
          "a filtered but critical contact gets a re-look line")
    check("TEST_plain" not in by_target, "a confident, identified contact gets none")

    analysis = hc.analyse(survey)
    track = analysis.tracks[0]
    for detection in (low, unknown, filtered):
        line = by_target[detection["id"]]
        row = int(round(detection["global_y"] - 0.5))
        diff = (line["properties"]["heading_deg"] - float(track.heading[row])) % 180.0
        check(abs(diff - 90.0) < 1.0, f"{detection['id']} line is perpendicular to the track "
                                      f"({diff:.1f}°)")
        from shapely.geometry import LineString, Point
        (x0, x1), (y0, y1) = analysis.projection.to_xy(
            [c[0] for c in line["geometry"]["coordinates"]],
            [c[1] for c in line["geometry"]["coordinates"]])
        px, py = analysis.projection.to_xy(detection["longitude"], detection["latitude"])
        distance = LineString([(x0, y0), (x1, y1)]).distance(Point(px, py))
        check(distance > float(track.nadir_w[row]) and distance < float(track.stbd_w[row]),
              f"{detection['id']} passes abeam at {distance:.1f} m: outside the nadir strip, "
              f"inside the swath")


def test_unavailable(tmp: Path) -> None:
    print("image survey without navigation")
    survey = copy_survey(IMAGE_SURVEY, tmp)
    doc = hc.coverage_for_survey(survey)
    check(doc == {"available": False, "reason": doc["reason"], "version": hc.VERSION}
          and "navigation" in doc["reason"], "coverage unavailable, with a reason")
    replay = hc.replay_for_survey(survey)
    check(replay["available"] is False and replay["reason"], "replay unavailable, with a reason")


def test_routes(tmp: Path) -> None:
    print("HTTP routes")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except ImportError:
        print("  skip  fastapi.testclient not installed")
        return
    from survey_hazard_map.routes import survey as routes

    root = tmp / "http"
    root.mkdir()
    copy_survey(PING_SURVEY, root)
    copy_survey(IMAGE_SURVEY, root)
    previous = routes.SURVEYS_DIR
    routes.SURVEYS_DIR = root
    try:
        app = FastAPI()
        app.include_router(routes.router)
        client = TestClient(app)
        response = client.get(f"/survey/{PING_SURVEY}/coverage")
        check(response.status_code == 200 and response.json()["available"], "GET coverage 200")
        response = client.get(f"/survey/{PING_SURVEY}/replay")
        check(response.status_code == 200 and response.json()["track"]["t"], "GET replay 200")
        response = client.get(f"/survey/{PING_SURVEY}/relook.gpx")
        check(response.status_code == 200 and "gpx" in response.headers["content-type"]
              and "attachment" in response.headers.get("content-disposition", ""),
              "GET relook.gpx is a GPX attachment")
        ET.fromstring(response.text.split("\n", 1)[1])
        response = client.get(f"/survey/{PING_SURVEY}/relook.csv")
        check(response.status_code == 200 and response.text.startswith("line_id,"),
              "GET relook.csv 200")
        response = client.get(f"/survey/{IMAGE_SURVEY}/coverage")
        check(response.status_code == 200 and response.json()["available"] is False,
              "image survey coverage is 200 with available: false")
        check(client.get(f"/survey/{IMAGE_SURVEY}/relook.gpx").status_code == 404,
              "image survey relook.gpx is 404")
        check(client.get("/survey/..%2Fetc/coverage").status_code in (400, 404),
              "a path-escaping id is refused")
    finally:
        routes.SURVEYS_DIR = previous


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="deepecho-coverage-") as name:
        tmp = Path(name)
        try:
            test_analytic(tmp)
            test_mannar(tmp)
            test_contacts(tmp)
            test_unavailable(tmp)
            test_routes(tmp)
        except AssertionError as exc:
            print(f"  FAIL  {exc}")
            return 1
    print(f"\n{CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
