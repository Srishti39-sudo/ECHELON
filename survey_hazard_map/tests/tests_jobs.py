#!/usr/bin/env python3
"""Tests for survey jobs: the engine's event hooks, the worker and the routes.

    .venv/bin/python tests_jobs.py            everything, including a real run
    .venv/bin/python tests_jobs.py --fast     skip the end-to-end model run

Plain functions and asserts, like unit_tests.py and smoke_test.py, with an exit
code so it can gate a commit.

The end-to-end case is not mocked. It uploads the S-7 submarine record from
samples/ with four corners, and the worker genuinely runs models/*.pt over its
tiles in a subprocess. That is slow (tens of seconds, more on a first load),
and it is the point: what is being tested is that a real run reports itself
in the right order, not that a stand-in does.

Everything is written to a temporary directory. DEEPECHO_SURVEYS_DIR and
DEEPECHO_SURVEY_UPLOADS_DIR are set before the application is imported, and
the worker inherits them, so data/surveys is never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WORK = Path(tempfile.mkdtemp(prefix="deepecho-jobs-"))
os.environ["DEEPECHO_SURVEYS_DIR"] = str(WORK / "surveys")
os.environ["DEEPECHO_SURVEY_UPLOADS_DIR"] = str(WORK / "uploads")
os.environ.setdefault("DEEPECHO_ENABLE_SURVEY_JOBS", "1")
os.environ["DEEPECHO_SSE_HEARTBEAT"] = "1"

SAMPLE = ROOT / "survey_hazard_map" / "samples" / "sidescan-s7-submarine.jpg"
CORNERS = {"top_left": [12.9200, 74.8500], "top_right": [12.9200, 74.8560],
           "bottom_left": [12.9170, 74.8500], "bottom_right": [12.9170, 74.8560]}
JOB_TIMEOUT = float(os.environ.get("DEEPECHO_TEST_JOB_TIMEOUT", "900"))

CASES = []


def case(name, slow=False):
    def register(fn):
        CASES.append((name, fn, slow))
        return fn
    return register


# --- engine hooks, no model ----------------------------------------------------

def _tiny_survey(workspace: Path):
    """A small strip with bright blobs, tiled for real, with corners."""
    import numpy as np
    from PIL import Image

    from survey_hazard_map.survey_preparation import prepare_survey

    rng = np.random.default_rng(7)
    array = (rng.normal(90, 25, (900, 1300))).clip(0, 255).astype("uint8")
    for cx, cy in ((300, 200), (700, 600), (1100, 300)):
        array[cy - 20:cy + 20, cx - 30:cx + 30] = 250
    workspace.mkdir(parents=True, exist_ok=True)
    strip = workspace / "tiny.png"
    Image.fromarray(array).save(strip)
    nav = {"mode": "corners", "strips": {"tiny": CORNERS}}
    return prepare_survey([strip], workspace / "prep", nav=nav)


class _FakeDetector:
    """Deterministic boxes wherever a tile has bright pixels. No torch."""

    name = "fake"
    classes = ["mine", "debris"]
    conf = 0.3

    def __call__(self, image_path):
        import numpy as np
        from PIL import Image

        array = np.asarray(Image.open(image_path).convert("L"))
        ys, xs = np.nonzero(array > 240)
        if not len(xs):
            return []
        return [{"class": "mine", "confidence": 0.77,
                 "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                 "model": "fake"}]


def _strip_volatile(export):
    export = json.loads(json.dumps(export))
    export["metadata"].pop("processed_at", None)
    export["metadata"].pop("processing_seconds", None)
    return export


@case("on_event=None leaves build_hazard_map's output identical")
def _():
    from survey_hazard_map.hazard_map import build_hazard_map

    workspace = WORK / "identical"
    tiles, manifest = _tiny_survey(workspace)
    plain = build_hazard_map("fake.pt", tiles, workspace / "a", manifest,
                             detector=_FakeDetector())
    seen = []
    observed = build_hazard_map("fake.pt", tiles, workspace / "b", manifest,
                                detector=_FakeDetector(), on_event=seen.append)
    assert plain["detections"], "the fake detector should find the blobs"
    a, b = _strip_volatile(plain), _strip_volatile(observed)
    a["metadata"].pop("survey_id"); b["metadata"].pop("survey_id")
    a["metadata"].pop("title"); b["metadata"].pop("title")
    assert a["detections"] == b["detections"]
    assert a["hotspots"] == b["hotspots"]
    assert a["survey_summary"] == b["survey_summary"]

    types = [e["type"] for e in seen]
    stages = [e["stage"] for e in seen if e["type"] == "stage"]
    assert stages[:6] == ["detect", "dedup", "geo", "verify", "hotspots", "export"], stages
    assert types[-1] == "final_detections"
    progress = [e for e in seen if e["type"] == "progress"]
    assert progress[-1]["tiles_done"] == progress[-1]["tiles_total"] == len(list(tiles.iterdir()))
    provisional = [e for e in seen if e["type"] == "detection"]
    assert provisional and all(e["provisional"] for e in provisional)
    # Georeferenced by corners, so provisional positions exist and are real.
    assert all(e["latitude"] is not None and 12.91 < e["latitude"] < 12.93 for e in provisional)
    assert seen[-1]["detections"] == observed["detections"]


@case("a provisional detection on a strip without navigation has null lat/lon")
def _():
    import numpy as np
    from PIL import Image

    from survey_hazard_map.hazard_map import build_hazard_map
    from survey_hazard_map.survey_preparation import prepare_survey

    workspace = WORK / "relative"
    array = np.full((700, 700), 80, dtype="uint8")
    array[::7, :] = 20
    array[300:340, 300:360] = 250
    workspace.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(workspace / "rel.png")
    tiles, manifest = prepare_survey([workspace / "rel.png"], workspace / "prep")
    seen = []
    build_hazard_map("fake.pt", tiles, workspace / "out", manifest,
                     detector=_FakeDetector(), on_event=seen.append)
    provisional = [e for e in seen if e["type"] == "detection"]
    assert provisional
    assert all(e["latitude"] is None and e["longitude"] is None for e in provisional)


@case("a raising callback does not break the survey")
def _():
    from survey_hazard_map.hazard_map import build_hazard_map

    workspace = WORK / "raising"
    tiles, manifest = _tiny_survey(workspace)
    calls = []

    def explode(event):
        calls.append(event["type"])
        raise RuntimeError("observer bug")

    import logging
    logging.getLogger("deepecho.hazard").disabled = True  # keep the output readable
    try:
        export = build_hazard_map("fake.pt", tiles, workspace / "out", manifest,
                                  detector=_FakeDetector(), on_event=explode)
    finally:
        logging.getLogger("deepecho.hazard").disabled = False
    assert export["detections"]
    assert (workspace / "out" / "export.json").is_file()
    assert "final_detections" in calls and "progress" in calls


@case("run_detector's on_tile reaches the total even when a tile fails")
def _():
    from survey_hazard_map.hazard_detect import run_detector

    workspace = WORK / "on_tile"
    tiles, _ = _tiny_survey(workspace)
    paths = sorted(tiles.iterdir())
    broken = paths[0].name

    def detector(path):
        if path.name == broken:
            raise OSError("unreadable")
        return _FakeDetector()(path)

    ticks = []
    run_detector(detector, paths, {}, on_tile=lambda i, n, name, boxes: ticks.append((i, n, name, len(boxes))))
    assert [t[0] for t in ticks] == list(range(1, len(paths) + 1))
    assert ticks[0][2] == broken and ticks[0][3] == 0


# --- routes ------------------------------------------------------------------------

def _client():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    return TestClient(app)


@case("traversal in a job id is refused")
def _():
    client = _client()
    for bad in ("...", ".hidden", "a..b", "a%2F..%2Fb", "x" * 80, "jobs", "a b"):
        response = client.get(f"/survey/jobs/{bad}")
        assert response.status_code in (400, 404), (bad, response.status_code)
        assert response.status_code == 400 or bad in ("a%2F..%2Fb",), (bad, response.text)
    response = client.post("/survey/jobs", files={"files": ("s.png", b"x", "image/png")},
                           data={"survey_id": "../escape"})
    assert response.status_code == 400, response.text
    assert not (WORK / "escape").exists()


@case("traversal and bad types in filenames are refused")
def _():
    client = _client()
    for name, status in (("../../evil.png", 400), ("..\\evil.png", 400), (".bashrc.png", 400),
                         ("payload.exe", 415), ("strip.svg", 415)):
        response = client.post("/survey/jobs", files={"files": (name, b"data", "image/png")})
        assert response.status_code == status, (name, response.status_code, response.text)
    response = client.post("/survey/jobs",
                           files={"files": ("s.png", b"data", "image/png"),
                                  "nav": ("../nav.csv", b"a,b", "text/csv")})
    assert response.status_code == 400, response.text
    assert not any((WORK / "uploads").glob("*")) if (WORK / "uploads").exists() else True


@case("oversize uploads are refused and leave nothing behind")
def _():
    from survey_hazard_map.routes import jobs

    client = _client()
    original = jobs.MAX_TOTAL_BYTES
    jobs.MAX_TOTAL_BYTES = 1000
    try:
        # Over by Content-Length, refused before the body is parsed ...
        response = client.post("/survey/jobs",
                               files={"files": ("big.png", b"0" * (3 * 1024 * 1024), "image/png")})
        assert response.status_code == 413, response.text
        # ... and over by counted bytes, with a declared length inside the slack.
        response = client.post("/survey/jobs", data={"survey_id": "oversize-counted"},
                               files={"files": ("big.png", b"0" * 5000, "image/png")})
        assert response.status_code == 413, response.text
        assert not (WORK / "uploads" / "oversize-counted").exists()
        assert not (WORK / "surveys" / "oversize-counted").exists()
    finally:
        jobs.MAX_TOTAL_BYTES = original


@case("bad corners are refused")
def _():
    client = _client()
    for corners in ("not json", json.dumps({"top_left": [1, 2]}),
                    json.dumps({**CORNERS, "top_left": [123, 2]})):
        response = client.post("/survey/jobs", data={"corners": corners},
                               files={"files": ("s.png", b"data", "image/png")})
        assert response.status_code == 400, (corners, response.text)


@case("strip preview names cannot escape the strips directory")
def _():
    client = _client()
    (WORK / "surveys" / "probe" / "strips").mkdir(parents=True, exist_ok=True)
    (WORK / "surveys" / "secret.png").write_bytes(b"no")
    for name in ("..%2Fsecret", "...", ".x", "a%5Cb"):
        response = client.get(f"/survey/probe/strips/{name}.png")
        assert response.status_code in (400, 404), (name, response.status_code)
    assert client.get("/survey/probe/report.csv").status_code == 404


@case("end to end: upload, real models, ordered events, SSE, downloads", slow=True)
def _():
    client = _client()
    with SAMPLE.open("rb") as handle:
        response = client.post(
            "/survey/jobs",
            data={"title": "S-7 test", "corners": json.dumps(CORNERS), "survey_id": "e2e-s7"},
            files={"files": (SAMPLE.name, handle, "image/jpeg")})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    assert job_id == "e2e-s7"

    # A second job while the first runs is refused (default cap is one).
    with SAMPLE.open("rb") as handle:
        second = client.post("/survey/jobs", files={"files": (SAMPLE.name, handle, "image/jpeg")})
    assert second.status_code == 429, second.text

    deadline = time.time() + JOB_TIMEOUT
    status = {}
    while time.time() < deadline:
        status = client.get(f"/survey/jobs/{job_id}").json()
        if status["state"] in ("done", "failed"):
            break
        time.sleep(2)
    worker_log = WORK / "surveys" / job_id / "worker.log"
    assert status.get("state") == "done", (
        status, worker_log.read_text()[-3000:] if worker_log.is_file() else "")

    lines = (WORK / "surveys" / job_id / "events.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    types = [e["type"] for e in events]

    def first(kind, **match):
        return next(i for i, e in enumerate(events)
                    if e["type"] == kind and all(e.get(k) == v for k, v in match.items()))

    order = [first("stage", stage="ingest"), first("strip"), first("stage", stage="tile"),
             first("stage", stage="detect"), first("progress"), first("detection"),
             first("stage", stage="dedup"), first("stage", stage="export"),
             first("final_detections"), first("stage", stage="report"), first("done")]
    assert order == sorted(order), order
    assert types[-1] == "done" and "error" not in types

    strip = events[first("strip")]
    assert strip["width"] == 1676 and strip["height"] == 871
    assert client.get(strip["image_url"]).status_code == 200
    footprinted = [e for e in events if e["type"] == "strip" and e.get("footprint")]
    assert footprinted, "corners were given, so the strip should have a footprint"

    final = events[first("final_detections")]["detections"]
    assert final and all(d["latitude"] is not None for d in final)

    done = events[-1]
    assert {"export.json", "actions.csv"} <= set(done["downloads"])
    for name in done["downloads"]:
        assert client.get(f"/survey/{job_id}/{name}").status_code == 200, name

    # The stream replays everything and then ends on its own.
    with client.stream("GET", f"/survey/jobs/{job_id}/events") as stream:
        body = "".join(stream.iter_text())
    data = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert [d["seq"] for d in data] == list(range(1, len(events) + 1))
    assert data[-1]["type"] == "done"

    # Resuming skips what was already seen, by query and by header.
    with client.stream("GET", f"/survey/jobs/{job_id}/events?after={len(events) - 2}") as stream:
        tail = [line for line in "".join(stream.iter_text()).splitlines() if line.startswith("data: ")]
    assert len(tail) == 2, tail
    with client.stream("GET", f"/survey/jobs/{job_id}/events",
                       headers={"Last-Event-ID": str(len(events) - 1)}) as stream:
        tail = [line for line in "".join(stream.iter_text()).splitlines() if line.startswith("data: ")]
    assert len(tail) == 1 and json.loads(tail[0][6:])["type"] == "done"

    # The finished survey is now an ordinary survey.
    listed = [s["survey_id"] for s in client.get("/survey").json()["surveys"]]
    assert job_id in listed

    # An existing id is not overwritten.
    with SAMPLE.open("rb") as handle:
        again = client.post("/survey/jobs", data={"survey_id": job_id},
                            files={"files": (SAMPLE.name, handle, "image/jpeg")})
    assert again.status_code == 409, again.text


@case("a survey that cannot be read fails with the engine's message", slow=True)
def _():
    client = _client()
    response = client.post("/survey/jobs", data={"survey_id": "broken-strip"},
                           files={"files": ("broken.png", b"this is not an image", "image/png")})
    assert response.status_code == 202, response.text
    deadline = time.time() + 120
    status = {}
    while time.time() < deadline:
        status = client.get("/survey/jobs/broken-strip").json()
        if status["state"] in ("done", "failed"):
            break
        time.sleep(1)
    assert status["state"] == "failed", status
    assert "could be opened" in (status.get("error") or ""), status
    with client.stream("GET", "/survey/jobs/broken-strip/events") as stream:
        body = "".join(stream.iter_text())
    data = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert data[-1]["type"] == "error" and data[-1]["message"] == status["error"]


# --- pipeline selection and the team pipeline ------------------------------------
#
# No best.pt exists yet, so the team pipeline is exercised with a FAKE
# sonar_detector.py in a temporary modules folder: the blob threshold geotag's
# own selftest uses, with the trained model's class names. geotag.py and
# sonar_pipeline.py are COPIED from the teammate's folder (never modified,
# never imported from there with bytecode on), so what runs is their real
# pipeline around a stand-in detector.

TEAMMATE_SOURCE = ROOT.parent / "models"

FAKE_DETECTOR = '''\
# FAKE sonar_detector.py for tests_jobs.py: bright blobs, not a model.
import json, sys
import numpy as np, cv2

LABELS = ["fishing_gear", "mine_like_object", "shipwreck", "pipeline"]


class SonarDetector:
    def __init__(self, weights, calib):
        with open(weights, "rb") as handle:
            assert handle.read(), "empty weights"
        self.calib = json.load(open(calib))

    def detect(self, bgr, raw_conf=0.10, min_calibrated=0.25):
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = (grey > 235).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = sorted([s for s in stats[1:] if s[4] > 20], key=lambda s: s[1])
        detections = [dict(label=LABELS[i % len(LABELS)], confidence_pct=88.0 - i,
                           box_xyxy_px=[float(x), float(y), float(x + w), float(y + h)])
                      for i, (x, y, w, h, _a) in enumerate(blobs)]
        print(f"fake detector: {len(detections)} blob(s) at raw_conf={raw_conf}", flush=True)
        return {"detections": detections, "tiles": 1}
'''

TEAMMATE_OBJECTS = [
    dict(ping=80, side="stbd", ground_range_m=22.0, len_pings=12, width_m=2.5),
    dict(ping=200, side="port", ground_range_m=35.0, len_pings=40, width_m=6.0),
    dict(ping=330, side="stbd", ground_range_m=41.0, len_pings=8, width_m=1.5),
]

_TEAM: dict = {}


def _geotag():
    from survey_hazard_map import import_geotag
    return import_geotag.load_geotag(TEAMMATE_SOURCE)


def _fake_modules(name: str, *, detector: bool = True, extra: dict | None = None) -> Path:
    folder = WORK / name
    folder.mkdir(parents=True, exist_ok=True)
    for script in ("geotag.py", "sonar_pipeline.py"):
        shutil.copyfile(TEAMMATE_SOURCE / script, folder / script)
    if detector:
        (folder / "sonar_detector.py").write_text(FAKE_DETECTOR)
    (folder / "best.pt").write_bytes(b"FAKE WEIGHTS for tests_jobs.py, not a checkpoint")
    (folder / "calibration.json").write_text(json.dumps({"A": 1.0, "B": 0.0, "note": "fake"}))
    for file_name, text in (extra or {}).items():
        (folder / file_name).write_text(text)
    return folder


def _synthetic_xtf() -> Path:
    if "xtf" in _TEAM:
        return _TEAM["xtf"]
    from survey_hazard_map.tools.make_synthetic_xtf import ORIGIN_LAT, ORIGIN_LON

    folder = WORK / "team-inputs"
    folder.mkdir(parents=True, exist_ok=True)
    xtf = folder / "team_line.xtf"
    _geotag().write_synthetic_xtf(xtf, n_pings=400, lat0=ORIGIN_LAT, lon0=ORIGIN_LON,
                                  objects=TEAMMATE_OBJECTS)
    _TEAM["xtf"] = xtf
    return xtf


class _env:
    """Set environment variables for a block, restoring the previous values."""

    def __init__(self, **values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)

    def __exit__(self, *_):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _wait(client, job_id, timeout=600):
    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        status = client.get(f"/survey/jobs/{job_id}").json()
        if status["state"] in ("done", "failed"):
            return status
        time.sleep(1)
    return status


def _events(job_id):
    lines = (WORK / "surveys" / job_id / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _strict(survey: Path):
    from survey_hazard_map.validate_output import Report, validate

    report = Report()
    validate(survey, report)
    assert not report.failures and not report.warnings, (report.failures, report.warnings)


@case("pipeline selection: auto, forced, invalid, and what is missing")
def _():
    from survey_hazard_map import survey_job

    complete = _fake_modules("sel-complete")
    no_detector = _fake_modules("sel-no-detector", detector=False)
    with _env(DEEPECHO_PIPELINE="auto", DEEPECHO_TEAMMATE_MODULES=complete,
              DEEPECHO_TEAMMATE_WEIGHTS=None, DEEPECHO_TEAMMATE_CALIB=None):
        choice = survey_job.select_pipeline()
        assert choice["pipeline"] == "teammate" and not choice["error"], choice
        assert choice["label"] == "team detector (best.pt)"
        assert not choice["teammate"]["shadow"] and choice["teammate"]["anomaly"] is None
        assert survey_job.select_pipeline("echelon")["pipeline"] == "echelon"
    with _env(DEEPECHO_PIPELINE=None, DEEPECHO_TEAMMATE_MODULES=no_detector,
              DEEPECHO_TEAMMATE_WEIGHTS=None, DEEPECHO_TEAMMATE_CALIB=None):
        choice = survey_job.select_pipeline()
        assert choice["requested"] == "auto" and choice["pipeline"] == "echelon", choice
        assert "sonar_detector.py not delivered yet" in choice["reason"], choice["reason"]
        assert choice["teammate"]["missing"] == ["sonar_detector.py"]
        forced = survey_job.select_pipeline("teammate")
        assert forced["error"] and "sonar_detector.py" in forced["error"], forced
    assert survey_job.select_pipeline("gpu")["error"]

    # Optional stages are offered only when their files exist.
    extras = _fake_modules("sel-extras", extra={"shadow_check.py": "# stub", "bg_embed.pkl": "x"})
    with _env(DEEPECHO_TEAMMATE_MODULES=extras, DEEPECHO_TEAMMATE_WEIGHTS=None,
              DEEPECHO_TEAMMATE_CALIB=None):
        team = survey_job.teammate_status()
        assert team["shadow"] and team["anomaly"] is None and team["notes"], team
        (extras / "anomaly.py").write_text("# stub")
        assert survey_job.teammate_status()["anomaly"].endswith("bg_embed.pkl")
    # Weights and calibration can live elsewhere; the calibration defaults beside the weights.
    elsewhere = WORK / "sel-weights"
    elsewhere.mkdir(exist_ok=True)
    (elsewhere / "marine.pt").write_bytes(b"x")
    with _env(DEEPECHO_TEAMMATE_MODULES=no_detector, DEEPECHO_TEAMMATE_WEIGHTS=elsewhere / "marine.pt",
              DEEPECHO_TEAMMATE_CALIB=None):
        team = survey_job.teammate_status()
        assert team["calib"] == str(elsewhere / "calibration.json")
        assert set(team["missing"]) == {"sonar_detector.py", "calibration.json"}, team
        assert team["label"] == "team detector (marine.pt)"


@case("plan_teammate: one source per job, nothing dropped silently")
def _():
    from survey_hazard_map import survey_job

    folder = WORK / "plan"
    folder.mkdir(exist_ok=True)
    a, b, jsf = folder / "a.xtf", folder / "b.xtf", folder / "c.jsf"
    png = folder / "d.png"
    navtable = folder / "nav.csv"
    navtable.write_text("ping,time,lat,lon,heading_deg,altitude_m,slant_range_m,samples_per_side\n")
    fixes = folder / "fixes.csv"
    fixes.write_text("x,y,lat,lon\n")
    plan = survey_job.plan_teammate([a, b, jsf], [png], None, None)
    assert plan["primary"] == a and plan["kind"] == "xtf"
    assert [s["file"] for s in plan["skipped"]] == ["b.xtf", "c.jsf", "d.png"], plan
    assert all(s["reason"].startswith("not processed") for s in plan["skipped"])
    plan = survey_job.plan_teammate([], [png], navtable, None)
    assert plan["primary"] == png and plan["kind"] == "image" and plan["nav"] == navtable
    for nav, corners in ((None, None), (fixes, None), (None, {"top_left": [0, 0]})):
        plan = survey_job.plan_teammate([], [png], nav, corners)
        assert plan["primary"] is None and "DeepEcho engine" in plan["fallback_reason"], plan
    assert survey_job.plan_teammate([jsf], [], None, None)["primary"] is None


@case("capabilities route reports the selection before any upload")
def _():
    client = _client()
    no_detector = _fake_modules("cap-no-detector", detector=False)
    with _env(DEEPECHO_PIPELINE="auto", DEEPECHO_TEAMMATE_MODULES=no_detector,
              DEEPECHO_TEAMMATE_WEIGHTS=None, DEEPECHO_TEAMMATE_CALIB=None):
        body = client.get("/survey/jobs/capabilities").json()
    assert body["selected"] == "echelon", body
    assert "sonar_detector.py not delivered yet" in body["reason"]
    assert body["pipelines"]["teammate"]["available"] is False
    assert body["pipelines"]["teammate"]["missing"] == ["sonar_detector.py"]
    assert body["pipelines"]["echelon"]["label"].startswith("DeepEcho engine (")
    complete = _fake_modules("cap-complete")
    with _env(DEEPECHO_PIPELINE="auto", DEEPECHO_TEAMMATE_MODULES=complete,
              DEEPECHO_TEAMMATE_WEIGHTS=None, DEEPECHO_TEAMMATE_CALIB=None):
        body = client.get("/survey/jobs/capabilities").json()
    assert body["selected"] == "teammate" and body["label"] == "team detector (best.pt)", body
    # "capabilities" can never be taken as a job id.
    response = client.post("/survey/jobs", data={"survey_id": "capabilities"},
                           files={"files": ("s.png", b"data", "image/png")})
    assert response.status_code == 400, response.text


@case("auto -> echelon when sonar_detector.py is missing, reason recorded in job.json")
def _():
    client = _client()
    no_detector = _fake_modules("job-no-detector", detector=False)
    with _env(DEEPECHO_PIPELINE="auto", DEEPECHO_TEAMMATE_MODULES=no_detector,
              DEEPECHO_TEAMMATE_WEIGHTS=None, DEEPECHO_TEAMMATE_CALIB=None):
        # An unreadable strip makes the DeepEcho run fail fast after the
        # selection is recorded; the selection is what is under test here
        # (tests_e2e.py runs the DeepEcho engine to completion).
        response = client.post("/survey/jobs", data={"survey_id": "auto-echelon"},
                               files={"files": ("broken.png", b"not an image", "image/png")})
        assert response.status_code == 202, response.text
        status = _wait(client, "auto-echelon", timeout=180)
    assert status["state"] == "failed", status
    assert status["pipeline"] == "echelon", status
    assert "sonar_detector.py not delivered yet" in status["pipeline_reason"], status
    events = _events("auto-echelon")
    chosen = next(e for e in events if e["type"] == "pipeline")
    assert chosen["pipeline"] == "echelon" and chosen["reason"] == status["pipeline_reason"]
    ingest = next(e for e in events if e["type"] == "stage")
    assert ingest["stage"] == "ingest" and ingest["pipeline"] == "echelon"


@case("forced teammate with files missing fails with the reason, never swaps engines")
def _():
    client = _client()
    no_detector = _fake_modules("job-forced", detector=False)
    with _env(DEEPECHO_PIPELINE="teammate", DEEPECHO_TEAMMATE_MODULES=no_detector,
              DEEPECHO_TEAMMATE_WEIGHTS=None, DEEPECHO_TEAMMATE_CALIB=None):
        with _synthetic_xtf().open("rb") as handle:
            response = client.post("/survey/jobs", data={"survey_id": "forced-teammate"},
                                   files={"files": ("line.xtf", handle, "application/octet-stream")})
        assert response.status_code == 202, response.text
        status = _wait(client, "forced-teammate", timeout=120)
    assert status["state"] == "failed", status
    assert "DEEPECHO_PIPELINE=teammate requested" in status["error"], status
    assert "sonar_detector.py" in status["error"]
    assert not (WORK / "surveys" / "forced-teammate" / "export.json").exists()


@case("teammate pipeline end to end: fake detector, real sonar_pipeline + geotag + import")
def _():
    import csv

    client = _client()
    modules = _fake_modules("job-teammate")
    xtf = _synthetic_xtf()
    with _env(DEEPECHO_PIPELINE="auto", DEEPECHO_TEAMMATE_MODULES=modules,
              DEEPECHO_TEAMMATE_WEIGHTS=None, DEEPECHO_TEAMMATE_CALIB=None):
        with xtf.open("rb") as first, xtf.open("rb") as second:
            response = client.post(
                "/survey/jobs", data={"survey_id": "team-xtf", "title": "Team pipeline test"},
                files=[("files", ("team_line.xtf", first, "application/octet-stream")),
                       ("files", ("team_line_repeat.xtf", second, "application/octet-stream"))])
        assert response.status_code == 202, response.text
        status = _wait(client, "team-xtf")
    survey = WORK / "surveys" / "team-xtf"
    worker_log = survey / "worker.log"
    assert status["state"] == "done", (status, worker_log.read_text()[-4000:]
                                       if worker_log.is_file() else "")
    assert status["pipeline"] == "teammate", status
    assert status["pipeline_reason"].startswith("auto: team pipeline complete"), status
    assert [s["file"] for s in status["inputs_not_processed"]] == ["team_line_repeat.xtf"]

    events = _events("team-xtf")
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["ingest", "detect", "geotag", "verify", "export", "report", "ghosttrace"], stages
    assert all(e["pipeline"] == "teammate" for e in events if e["type"] == "stage")
    messages = [e.get("message", "") for e in events if e["type"] == "log"]
    assert any(m.startswith("sonar_pipeline: fake detector: 3 blob(s)") for m in messages), messages
    assert any("team_line_repeat.xtf not processed" in m for m in messages)
    strip = next(e for e in events if e["type"] == "strip")
    assert strip["track"] and strip["m_per_px_across"] is None
    assert client.get(strip["image_url"]).status_code == 200
    final = next(e for e in events if e["type"] == "final_detections")["detections"]
    assert len(final) == 3 and all(d["latitude"] is not None for d in final), final
    done = events[-1]
    assert done["type"] == "done" and done["pipeline"] == "teammate", done
    assert done["pipeline_label"] == "team detector (best.pt)"
    assert done["inputs_not_processed"][0]["file"] == "team_line_repeat.xtf"
    for name in ("export.json", "report.csv", "report.geojson", "actions.csv", "map.html",
                 "ghosttrace.json"):
        assert name in done["downloads"], (name, done["downloads"])
        assert client.get(f"/survey/team-xtf/{name}").status_code == 200, name

    _strict(survey)
    assert client.get("/ghosttrace/team-xtf").status_code == 200
    with (survey / "report.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lat_key = next(k for k in rows[0] if k.lower() in ("lat", "latitude"))
    lon_key = next(k for k in rows[0] if k.lower() in ("lon", "longitude"))
    assert rows and all(r[lat_key] and r[lon_key] for r in rows), rows
    assert all(8.5 < float(r[lat_key]) < 9.6 for r in rows)
    export = json.loads((survey / "export.json").read_text())
    assert export["metadata"]["survey_id"] == "team-xtf"
    assert export["metadata"]["detector"].startswith("sonar_detector.py")
    # The staging folder never leaks into the survey, and job files survived the move.
    for path in survey.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".csv", ".geojson", ".html"):
            assert "teammate/import" not in path.read_text(errors="ignore"), path
    assert (survey / "job.json").is_file() and (survey / "events.jsonl").is_file()
    assert not (WORK / "uploads" / "team-xtf" / "teammate" / "import").exists()
    listed = [s["survey_id"] for s in client.get("/survey").json()["surveys"]]
    assert "team-xtf" in listed
    # Nothing was written into the teammate's own folder.
    assert not (TEAMMATE_SOURCE / "hazards.json").exists()


@case("teammate pipeline on an image with a NavTable CSV; images without one fall back")
def _():
    import cv2

    client = _client()
    gt = _geotag()
    image, nav = gt.read_xtf(_synthetic_xtf())
    folder = WORK / "team-image"
    folder.mkdir(exist_ok=True)
    png = folder / "team_strip.png"
    cv2.imwrite(str(png), image)
    nav_csv = folder / "team_strip_nav.csv"
    nav.to_csv(nav_csv)
    modules = _fake_modules("job-teammate-image")
    with _env(DEEPECHO_PIPELINE="auto", DEEPECHO_TEAMMATE_MODULES=modules,
              DEEPECHO_TEAMMATE_WEIGHTS=None, DEEPECHO_TEAMMATE_CALIB=None):
        with png.open("rb") as strip, nav_csv.open("rb") as table:
            response = client.post("/survey/jobs", data={"survey_id": "team-png"},
                                   files={"files": ("team_strip.png", strip, "image/png"),
                                          "nav": ("team_strip_nav.csv", table, "text/csv")})
        assert response.status_code == 202, response.text
        status = _wait(client, "team-png")
        survey = WORK / "surveys" / "team-png"
        assert status["state"] == "done", (status, (survey / "worker.log").read_text()[-4000:])
        assert status["pipeline"] == "teammate" and not status["inputs_not_processed"], status
        _strict(survey)
        export = json.loads((survey / "export.json").read_text())
        assert len(export["detections"]) == 3
        assert all(d["latitude"] is not None for d in export["detections"])

        # The same image with no navigation: the DeepEcho engine, and the job says why.
        response = client.post("/survey/jobs", data={"survey_id": "team-png-fallback"},
                               files={"files": ("broken.png", b"not an image", "image/png")})
        assert response.status_code == 202, response.text
        status = _wait(client, "team-png-fallback", timeout=180)
    assert status["pipeline"] == "echelon", status
    assert "image uploads need a navigation CSV" in status["pipeline_reason"], status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="skip the real model run")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = parser.parse_args()

    failed = 0
    run = 0
    for name, fn, slow in CASES:
        if slow and args.fast:
            print(f"  skip  {name}")
            continue
        run += 1
        started = time.time()
        try:
            fn()
            print(f"  ok    {name} ({time.time() - started:.1f}s)")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()

    if args.keep:
        print(f"\nworkspace kept at {WORK}")
    else:
        shutil.rmtree(WORK, ignore_errors=True)
    print(f"\n{'PASSED' if not failed else 'FAILED'}  {run - failed}/{run} job tests")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
