#!/usr/bin/env python3
"""Tests for the GhostTrace router (backend/app/routes/ghosttrace.py).

    .venv/bin/python tests_ghosttrace_api.py

Plain functions and asserts, like tests_jobs.py, with an exit code. The router
is mounted on a bare FastAPI app, so nothing else in the application (the
index, the detector, storage) is loaded. Surveys and habitat layers live in a
temporary directory: DEEPECHO_SURVEYS_DIR and DEEPECHO_GHOSTTRACE_DATA_DIR are
set before the router is imported, and data/ is never touched.

run_ghosttrace.py is replaced by small stand-in scripts for the run route. What
is under test is the route's contract (gating, one-at-a-time, timeout, failure
passthrough, summary), not the engine, which has its own tests.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WORK = Path(tempfile.mkdtemp(prefix="deepecho-ghosttrace-"))
SURVEYS = WORK / "surveys"
LAYERS = WORK / "ghosttrace"
os.environ["DEEPECHO_SURVEYS_DIR"] = str(SURVEYS)
os.environ["DEEPECHO_GHOSTTRACE_DATA_DIR"] = str(LAYERS)
os.environ.pop("DEEPECHO_ENABLE_GHOSTTRACE_RUN", None)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ghosttrace.routes import api as gt  # noqa: E402

app = FastAPI()
app.include_router(gt.router)
client = TestClient(app)

DOC = {
    "format": "deepecho-ghosttrace/1",
    "survey_id": "alpha",
    "generated_at": "2026-09-13T00:00:00+00:00",
    "demo": True,
    "synthetic_inputs": True,
    "data_sources": [],
    "caveats": ["test document"],
    "targets": [{
        "detection_id": "alpha_d0",
        "object_class": "net",
        "latitude": 9.2, "longitude": 79.1,
        "alert": {
            "authorities": [{"name": "Test Authority", "role": "coastal", "contact_basis": "x"}],
            "subject": "Ghost net near reef",
            "draft_text": "A suspected ghost net was detected.",
            "generated_by": "template",
            "citations": ["kb/example.md"],
            "basis": "template fill",
        },
    }, {
        "detection_id": "alpha_d1", "object_class": "debris", "latitude": 9.3, "longitude": 79.2,
        "alert": {},
    }],
    "summary": {"targets": 2, "urgent": 1},
}


def feature(name, lon, lat, size=0.01):
    return {"type": "Feature", "properties": {"name": name},
            "geometry": {"type": "Polygon", "coordinates": [[
                [lon, lat], [lon + size, lat], [lon + size, lat + size], [lon, lat + size], [lon, lat]]]}}


def setup():
    (SURVEYS / "alpha").mkdir(parents=True)
    (SURVEYS / "alpha" / "export.json").write_text("{}")
    (SURVEYS / "alpha" / "ghosttrace.json").write_text(json.dumps(DOC))
    (SURVEYS / "alpha" / "ghosttrace.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}))
    (SURVEYS / "bravo").mkdir()
    (SURVEYS / "bravo" / "export.json").write_text("{}")
    (WORK / "outside").mkdir()
    (WORK / "outside" / "ghosttrace.json").write_text(json.dumps(DOC))

    LAYERS.mkdir()
    # Convention: stem names the kind.
    (LAYERS / "reefs_example.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "source": "synthetic",
        "features": [feature("near", 79.10, 9.20), feature("far", 72.0, 19.0),
                     {"type": "Feature", "properties": {"name": "pt"},
                      "geometry": {"type": "Point", "coordinates": [79.15, 9.25]}}]}))
    # Manifest: a file whose name does not reveal its kind.
    (LAYERS / "layer_07.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": [feature("mpa one", 79.0, 9.0)]}))
    (LAYERS / "manifest.json").write_text(json.dumps({
        "layers": [{"kind": "protected_area", "file": "layer_07.geojson"},
                   {"kind": "dugong", "file": "../outside/ghosttrace.json"}]}))
    (LAYERS / "broken_harbours.geojson").write_text("{not json")


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def reads_ghosttrace_json():
    r = client.get("/ghosttrace/alpha")
    assert r.status_code == 200, r.text
    assert r.json() == DOC


@case
def survey_without_ghosttrace_says_how_to_generate():
    r = client.get("/ghosttrace/bravo")
    assert r.status_code == 404, r.text
    detail = r.json()["detail"]
    assert "not been generated" in detail and "run_ghosttrace.py" in detail, detail


@case
def missing_survey_is_404():
    r = client.get("/ghosttrace/nope")
    assert r.status_code == 404, r.text


@case
def ids_that_escape_are_refused():
    for bad in (".hidden", "..", "..%5Coutside", "a%5Cb"):
        r = client.get(f"/ghosttrace/{bad}")
        assert r.status_code in (400, 404), (bad, r.status_code)
        assert r.status_code != 200
    assert gt._survey_dir.__doc__
    try:
        gt._survey_dir("../outside")
    except gt.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("../outside was accepted")
    try:
        gt._survey_dir("a\\b")
    except gt.HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("backslash was accepted")


@case
def serves_geojson():
    r = client.get("/ghosttrace/alpha/geojson")
    assert r.status_code == 200, r.text
    assert r.json()["type"] == "FeatureCollection"
    r = client.get("/ghosttrace/bravo/geojson")
    assert r.status_code == 404 and "not been generated" in r.json()["detail"]


@case
def alert_text_download():
    r = client.get("/ghosttrace/alpha/alerts/alpha_d0.txt")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/plain")
    assert "attachment" in r.headers.get("content-disposition", "")
    body = r.text
    assert body.startswith("DRAFT"), body
    assert "A suspected ghost net was detected." in body
    assert "Subject: Ghost net near reef" in body
    assert "Generated by: template" in body
    assert "Test Authority (coastal)" in body

    assert client.get("/ghosttrace/alpha/alerts/alpha_d1.txt").status_code == 404
    assert client.get("/ghosttrace/alpha/alerts/zzz.txt").status_code == 404
    assert client.get("/ghosttrace/alpha/alerts/..txt").status_code in (400, 404)
    assert client.get("/ghosttrace/alpha/alerts/-x.txt").status_code == 400


@case
def layer_whitelist_and_absence():
    r = client.get("/ghosttrace/layers/volcano")
    assert r.status_code == 404 and "unknown layer" in r.json()["detail"], r.text
    r = client.get("/ghosttrace/layers/turtle_nesting")
    assert r.status_code == 404 and "no turtle_nesting layer" in r.json()["detail"], r.text


@case
def layer_by_file_name_convention_with_bbox():
    r = client.get("/ghosttrace/layers/reef")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "reef" and body["total_features"] == 3
    assert len(body["features"]) == 3 and body["source"] == "synthetic"

    r = client.get("/ghosttrace/layers/reef", params={"bbox": "79.0,9.0,79.5,9.5"})
    names = [f["properties"]["name"] for f in r.json()["features"]]
    assert names == ["near", "pt"], names
    assert r.json()["bbox_filter"] == [79.0, 9.0, 79.5, 9.5]

    for bad in ("1,2,3", "a,b,c,d", "10,0,0,10", "0,0,200,10"):
        r = client.get("/ghosttrace/layers/reef", params={"bbox": bad})
        assert r.status_code == 400, (bad, r.text)


@case
def layer_from_manifest_and_containment():
    r = client.get("/ghosttrace/layers/protected_area")
    assert r.status_code == 200, r.text
    assert r.json()["features"][0]["properties"]["name"] == "mpa one"
    # The manifest points dugong outside the data directory; it must not be served.
    r = client.get("/ghosttrace/layers/dugong")
    assert r.status_code == 404, r.text


@case
def fetcher_manifest_merges_regions_and_withholds_restricted_licences():
    data = WORK / "fetched"
    (data / "layers").mkdir(parents=True, exist_ok=True)

    def layer(name, lon, lat):
        (data / "layers" / name).write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"name": name},
             "geometry": {"type": "Point", "coordinates": [lon, lat]}}]}))

    layer("reef_osm_a.geojson", 79.1, 9.1)
    layer("reef_osm_b.geojson", 86.9, 20.5)
    layer("reef_wcmc_a.geojson", 79.2, 9.2)
    (data / "manifest.json").write_text(json.dumps({"sources": [
        {"id": "osm", "name": "OSM reefs", "redistributable": True,
         "files": ["layers/reef_osm_a.geojson", "layers/reef_osm_b.geojson"]},
        {"id": "wcmc", "name": "UNEP-WCMC reefs", "redistributable": False,
         "files": ["layers/reef_wcmc_a.geojson"]},
    ]}))
    saved = gt.DATA_DIR
    gt.DATA_DIR = data
    try:
        body = client.get("/ghosttrace/layers/reef").json()
        names = sorted(f["properties"]["name"] for f in body["features"])
        # Both regions of the open source are served; the restricted one is not.
        assert names == ["reef_osm_a.geojson", "reef_osm_b.geojson"], names
        assert body["withheld_by_licence"] == ["UNEP-WCMC reefs"], body
        gt.SERVE_RESTRICTED_LAYERS = True
        names = sorted(f["properties"]["name"]
                       for f in client.get("/ghosttrace/layers/reef").json()["features"])
        assert "reef_wcmc_a.geojson" in names, names
    finally:
        gt.DATA_DIR = saved
        gt.SERVE_RESTRICTED_LAYERS = False


@case
def broken_layer_is_500_not_a_crash():
    r = client.get("/ghosttrace/layers/harbour")
    assert r.status_code == 500 and "could not be read" in r.json()["detail"], r.text


@case
def capabilities_lists_present_layers():
    os.environ["DEEPECHO_ENABLE_GHOSTTRACE_RUN"] = "0"
    r = client.get("/ghosttrace/capabilities")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_enabled"] is False
    assert set(body["layers"]) == {"reef", "protected_area", "harbour"}, body["layers"]


def fake_script(source: str) -> Path:
    path = WORK / f"fake_{len(list(WORK.glob('fake_*')))}.py"
    path.write_text(textwrap.dedent(source))
    return path


@case
def run_is_gated_by_flag():
    os.environ["DEEPECHO_ENABLE_GHOSTTRACE_RUN"] = "0"
    r = client.post("/ghosttrace/bravo/run")
    assert r.status_code == 403 and "DEEPECHO_ENABLE_GHOSTTRACE_RUN" in r.json()["detail"], r.text

    # Unset: follows whether the script exists.
    os.environ.pop("DEEPECHO_ENABLE_GHOSTTRACE_RUN")
    original = gt.SCRIPT
    try:
        gt.SCRIPT = WORK / "does-not-exist.py"
        assert gt._run_enabled() is False
        gt.SCRIPT = fake_script("pass\n")
        assert gt._run_enabled() is True
    finally:
        gt.SCRIPT = original


@case
def run_writes_and_returns_summary():
    os.environ["DEEPECHO_ENABLE_GHOSTTRACE_RUN"] = "1"
    original = gt.SCRIPT
    try:
        gt.SCRIPT = fake_script("""
            import json, sys
            from pathlib import Path
            args = sys.argv[1:]
            assert args[0] == "--survey" and args[2] == "--quiet", args
            survey = Path(args[1])
            doc = {"format": "deepecho-ghosttrace/1", "survey_id": survey.name,
                   "generated_at": "now", "targets": [], "summary": {"targets": 0, "urgent": 0}}
            (survey / "ghosttrace.json").write_text(json.dumps(doc))
        """)
        r = client.post("/ghosttrace/bravo/run")
        assert r.status_code == 200, r.text
        assert r.json() == {"survey_id": "bravo", "generated_at": "now",
                            "summary": {"targets": 0, "urgent": 0}}
        assert client.get("/ghosttrace/bravo").status_code == 200

        assert client.post("/ghosttrace/nope/run").status_code == 404
        (SURVEYS / "charlie").mkdir()
        r = client.post("/ghosttrace/charlie/run")
        assert r.status_code == 404 and "export.json" in r.json()["detail"], r.text
    finally:
        gt.SCRIPT = original


@case
def run_failure_timeout_and_concurrency():
    os.environ["DEEPECHO_ENABLE_GHOSTTRACE_RUN"] = "1"
    original, original_timeout = gt.SCRIPT, gt.RUN_TIMEOUT_SECONDS
    try:
        gt.SCRIPT = fake_script("import sys\nsys.stderr.write('engine said no')\nsys.exit(2)\n")
        r = client.post("/ghosttrace/alpha/run")
        assert r.status_code == 500 and "engine said no" in r.json()["detail"], r.text

        gt.SCRIPT = fake_script("import time\ntime.sleep(10)\n")
        gt.RUN_TIMEOUT_SECONDS = 1
        r = client.post("/ghosttrace/alpha/run")
        assert r.status_code == 504, r.text
        assert gt._RUN_LOCK.acquire(blocking=False), "lock leaked after timeout"
        try:
            r = client.post("/ghosttrace/alpha/run")
            assert r.status_code == 409, r.text
        finally:
            gt._RUN_LOCK.release()
    finally:
        gt.SCRIPT, gt.RUN_TIMEOUT_SECONDS = original, original_timeout


@case
def frontend_fixture_is_contract_shaped():
    """The UI's synthetic fixture must stay a valid, labelled contract document."""
    fixture = ROOT / "frontend" / "src" / "ghosttrace" / "fixtures" / "example.json"
    if not fixture.is_file():
        return
    doc = json.loads(fixture.read_text(encoding="utf-8"))
    assert doc.get("synthetic_example") is True
    assert doc["format"] == "deepecho-ghosttrace/1"
    for key in ("survey_id", "generated_at", "demo", "synthetic_inputs", "data_sources",
                "caveats", "targets", "removed_since_previous", "change_summary",
                "recovery_plan", "summary"):
        assert key in doc, key
    for target in doc["targets"]:
        for key in ("detection_id", "object_class", "latitude", "longitude", "activity",
                    "habitat", "drift", "people", "change", "priority", "alert"):
            assert key in target, (target.get("detection_id"), key)
        assert target["priority"]["tier"] in {"urgent", "high", "routine"}


@case
def panel_copy_of_engine_fixture_is_current():
    """The fixture preview bundles a copy of the engine fixture; say when it is stale."""
    source = ROOT / "tests" / "fixtures" / "ghosttrace_example.json"
    copy = ROOT / "frontend" / "src" / "ghosttrace" / "fixtures" / "engine_example.json"
    if not (source.is_file() and copy.is_file()):
        return
    if json.loads(source.read_text()) != json.loads(copy.read_text()):
        print("  note  frontend/src/ghosttrace/fixtures/engine_example.json differs from "
              "tests/fixtures/ghosttrace_example.json; re-copy it to refresh the preview")
    doc = json.loads(copy.read_text())
    assert doc["format"] == "deepecho-ghosttrace/1" and doc.get("synthetic_example") is True


def main() -> int:
    setup()
    failures = 0
    try:
        for fn in CASES:
            try:
                fn()
                print(f"  ok    {fn.__name__}")
            except Exception:
                failures += 1
                print(f"  FAIL  {fn.__name__}")
                traceback.print_exc()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
