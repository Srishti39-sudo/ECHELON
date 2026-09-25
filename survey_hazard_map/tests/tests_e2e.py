#!/usr/bin/env python3
"""The whole system, end to end, in one process: upload to assistant.

    .venv/bin/python tests_e2e.py                    DeepEcho engine (known.pt + anomaly.pt)
    DEEPECHO_PIPELINE=auto .venv/bin/python tests_e2e.py
                                                     whichever pipeline auto selects
    .venv/bin/python tests_e2e.py --keep             keep the temporary workspace

What runs, against the real FastAPI application through TestClient (no server
is started and no port is opened):

    1. GET  /survey/jobs/capabilities    the pipeline this run will use, and why
    2. POST /survey/jobs                 samples/synthetic_xtf/SYNTHETIC_mannar_line01.xtf,
                                         processed by the real worker subprocess
    3. GET  /survey/jobs/{id}/events     the SSE stream, followed live to "done";
                                         stages in order, ending with ghosttrace
    4. every file the done event offers, downloaded
    5. GET  /survey, /survey/{id}/export, /ghosttrace/{id}, /ghosttrace/layers/reef
    6. POST /detect                      one sample tile (skipped when the route is off)
    7. POST /chat                        with a GhostTrace context built from this
                                         survey's ghosttrace.json. A model answer or,
                                         when every provider is rate-limited or
                                         unreachable, the labelled retrieval_only
                                         fallback; rate limits are never a failure.

Every path is temporary: DEEPECHO_SURVEYS_DIR, DEEPECHO_SURVEY_UPLOADS_DIR,
DEEPECHO_DB_PATH and DEEPECHO_UPLOAD_DIR are set before the application is
imported, and the worker inherits them. data/ is never written.

Prints a summary table and exits 1 if any step failed.
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
os.chdir(ROOT)

WORK = Path(tempfile.mkdtemp(prefix="deepecho-e2e-"))
os.environ["DEEPECHO_SURVEYS_DIR"] = str(WORK / "surveys")
os.environ["DEEPECHO_SURVEY_UPLOADS_DIR"] = str(WORK / "uploads")
os.environ["DEEPECHO_DB_PATH"] = str(WORK / "deepecho.db")
os.environ["DEEPECHO_UPLOAD_DIR"] = str(WORK / "scan-uploads")
os.environ.setdefault("DEEPECHO_ENABLE_SURVEY_JOBS", "1")
os.environ.setdefault("DEEPECHO_SSE_HEARTBEAT", "5")
# The DeepEcho engine unless the caller asks otherwise, so this test means the
# same thing on the day the team's best.pt lands. DEEPECHO_PIPELINE=auto runs
# the same checks on whatever the auto-selection picks.
os.environ.setdefault("DEEPECHO_PIPELINE", "echelon")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

SAMPLE = ROOT / "survey_hazard_map" / "samples" / "synthetic_xtf" / "SYNTHETIC_mannar_line01.xtf"
TILE_DIR = ROOT / "survey_hazard_map" / "samples" / "tiles"
JOB_TIMEOUT = float(os.environ.get("DEEPECHO_E2E_JOB_TIMEOUT", "1800"))
SURVEY_ID = "e2e-mannar-line01"
RATE_LIMIT_MARKERS = ("rate", "limit", "quota", "429", "overloaded", "unavailable", "timeout",
                      "timed out", "connection", "api key", "api_key", "credit", "billing",
                      "not configured", "no provider")


class Skip(Exception):
    """A step that cannot run here, for a reason that is not a failure."""


RESULTS: list[tuple[str, str, float, str]] = []
STATE: dict = {}


def step(name: str, needs: tuple[str, ...] = ()):
    def register(fn):
        def run(client):
            missing = [key for key in needs if key not in STATE]
            if missing:
                RESULTS.append((name, "FAIL", 0.0, f"not run: an earlier step did not produce "
                                                    f"{', '.join(missing)}"))
                return
            started = time.time()
            try:
                note = fn(client) or ""
                RESULTS.append((name, "ok", time.time() - started, note))
            except Skip as exc:
                RESULTS.append((name, "skip", time.time() - started, str(exc)))
            except Exception as exc:
                traceback.print_exc()
                RESULTS.append((name, "FAIL", time.time() - started,
                                f"{type(exc).__name__}: {exc}"[:300]))
        run.step_name = name
        return run
    return register


# --- steps ------------------------------------------------------------------------


@step("capabilities: pipeline selection before upload")
def capabilities(client):
    response = client.get("/survey/jobs/capabilities")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["selected"] in ("echelon", "teammate") and not body["error"], body
    assert body["reason"], body
    STATE["capabilities"] = body
    return f"{body['label']} ({body['reason'][:90]})"


@step("upload SYNTHETIC_mannar_line01.xtf to POST /survey/jobs", needs=("capabilities",))
def upload(client):
    assert SAMPLE.is_file(), f"missing sample {SAMPLE}"
    with SAMPLE.open("rb") as handle:
        response = client.post("/survey/jobs",
                               data={"survey_id": SURVEY_ID, "title": "E2E Mannar line 01"},
                               files={"files": (SAMPLE.name, handle, "application/octet-stream")})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["job_id"] == SURVEY_ID and body["events_url"].endswith("/events")
    STATE["job"] = body
    return f"job {SURVEY_ID}, {SAMPLE.stat().st_size / 1e6:.1f} MB"


@step("SSE stream followed to done; stage order", needs=("job",))
def stream(client):
    deadline = time.time() + JOB_TIMEOUT
    events: list[dict] = []
    with client.stream("GET", f"/survey/jobs/{SURVEY_ID}/events") as response:
        assert response.status_code == 200, response.status_code
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if time.time() > deadline:
                raise TimeoutError(f"no done event within {JOB_TIMEOUT:.0f} s")
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            events.append(event)
            if event["type"] in ("done", "error"):
                break
    assert events, "the stream produced no events"
    if events[-1]["type"] == "error":
        log = WORK / "surveys" / SURVEY_ID / "worker.log"
        raise AssertionError(f"job failed: {events[-1]['message']}\n"
                             + (log.read_text()[-3000:] if log.is_file() else ""))
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    pipeline = events[-1]["pipeline"]
    expected = (["ingest", "tile", "detect", "export", "report", "ghosttrace"]
                if pipeline == "echelon" else
                ["ingest", "detect", "geotag", "verify", "export", "report", "ghosttrace"])
    positions = [stages.index(s) for s in expected]
    assert positions == sorted(positions), stages
    assert stages[-1] == "ghosttrace", stages
    assert pipeline == STATE["capabilities"]["selected"], (pipeline, STATE["capabilities"])
    status = client.get(f"/survey/jobs/{SURVEY_ID}").json()
    assert status["state"] == "done" and status["pipeline"] == pipeline, status
    assert status["pipeline_reason"] == events[-1]["pipeline_reason"]
    STATE["events"] = events
    STATE["done"] = events[-1]
    detections = next((e["detections"] for e in events if e["type"] == "final_detections"), [])
    return (f"{len(events)} events, {len(detections)} detection(s), pipeline {pipeline}; "
            f"stages {' > '.join(dict.fromkeys(stages))}")


@step("every download in the done event", needs=("done",))
def downloads(client):
    names = STATE["done"]["downloads"]
    for required in ("export.json", "report.csv", "report.geojson", "actions.csv", "map.html"):
        assert required in names, (required, names)
    sizes = []
    for name in names:
        response = client.get(f"/survey/{SURVEY_ID}/{name}")
        assert response.status_code == 200, (name, response.status_code, response.text[:200])
        assert response.content, name
        sizes.append(f"{name} {len(response.content) / 1024:.0f} KB")
    return ", ".join(sizes)


@step("GET /survey lists it; GET /survey/{id}/export", needs=("done",))
def survey_routes(client):
    listed = client.get("/survey")
    assert listed.status_code == 200, listed.text
    assert SURVEY_ID in [s["survey_id"] for s in listed.json()["surveys"]]
    export = client.get(f"/survey/{SURVEY_ID}/export")
    assert export.status_code == 200, export.text[:300]
    body = export.json()
    assert body["metadata"]["survey_id"] == SURVEY_ID
    assert body["metadata"]["coordinate_mode"] == "Geo-referenced", body["metadata"]
    from survey_hazard_map.validate_output import Report, validate

    report = Report()
    validate(WORK / "surveys" / SURVEY_ID, report)
    assert not report.failures and not report.warnings, (report.failures, report.warnings)
    return (f"{len(body['detections'])} detection(s), {len(body['hotspots'])} hotspot(s); "
            f"validate_output --strict: {report.passed} checks")


@step("GET /ghosttrace/{id} and /ghosttrace/layers/reef", needs=("done",))
def ghosttrace(client):
    response = client.get(f"/ghosttrace/{SURVEY_ID}")
    assert response.status_code == 200, response.text[:300]
    doc = response.json()
    assert doc.get("survey_id") == SURVEY_ID, doc.get("survey_id")
    STATE["ghosttrace"] = doc
    layer = client.get("/ghosttrace/layers/reef")
    assert layer.status_code == 200, layer.text[:300]
    features = layer.json().get("features") or []
    assert layer.json().get("type") == "FeatureCollection"
    return f"{len(doc.get('targets') or [])} target(s); reef layer {len(features)} feature(s)"


@step("POST /detect with a sample tile")
def detect(client):
    tiles = sorted(TILE_DIR.glob("*.jpg"))
    if not tiles:
        raise Skip(f"no sample tiles in {TILE_DIR}")
    with tiles[0].open("rb") as handle:
        response = client.post("/detect", files={"file": (tiles[0].name, handle, "image/jpeg")})
    if response.status_code == 404:
        raise Skip("the detector route is disabled on this backend (DEEPECHO_ENABLE_UPLOAD)")
    assert response.status_code == 200, response.text[:300]
    body = response.json()
    assert "detections" in body and "stub" in body, body
    assert body["stub"] is False, "a real detector was expected; got the labelled stub"
    return f"{tiles[0].name}: {len(body['detections'])} detection(s), models {body.get('models')}"


def _ghosttrace_context(doc: dict) -> tuple[dict, str]:
    """This survey's GhostTrace context: its first target, or the survey itself."""
    from rag_assistant.eval.ghosttrace_context import context_from_target

    targets = doc.get("targets") or []
    if targets:
        return context_from_target(doc, targets[0]), "first target"
    # Nothing to recover in this survey (the DeepEcho engine has no net class).
    # The context still carries the survey's identity and caveats, as the
    # rescue queue would, so the assistant is asked about this survey.
    return ({"kind": "ghosttrace_target", "survey_id": doc.get("survey_id"),
             "survey_title": doc.get("title"),
             "synthetic": bool(doc.get("demo") or doc.get("synthetic_inputs")),
             "caveats": list(doc.get("caveats") or [])}, "survey-level (no targets)")


@step("POST /chat with this survey's GhostTrace context", needs=("ghosttrace",))
def chat(client):
    context, basis = _ghosttrace_context(STATE["ghosttrace"])
    payload = {"message": "A derelict gill net was found in the Gulf of Mannar survey. "
                          "Who should be told and how should recovery be approached?",
               "ghosttrace_context": context}
    response = client.post("/chat", json=payload)
    if response.status_code == 503:
        detail = str(response.json().get("detail", ""))
        if any(marker in detail.lower() for marker in RATE_LIMIT_MARKERS):
            raise Skip(f"providers unavailable and no fallback text: {detail[:160]}")
    assert response.status_code == 200, response.text[:400]
    body = response.json()
    assert body["answer"].strip(), body
    assert body["generated_by"] in ("model", "retrieval_only", "none"), body["generated_by"]
    assert body["ghosttrace_citation"], "a turn with a GhostTrace context names its citation"
    if body["generated_by"] == "retrieval_only":
        # Rate-limited or offline: the labelled fallback, never an error.
        assert body["provider_errors"], body
        assert body["sources"] and body["answer"].lstrip().startswith("**Offline"), body["answer"][:200]
        assert body["provider"] == "" and body["model"] == ""
        return (f"retrieval_only fallback ({basis}); {len(body['sources'])} source(s); "
                f"providers: {'; '.join(body['provider_errors'])[:120]}")
    if body["generated_by"] == "model":
        assert body["provider"], body
    return (f"{body['generated_by']} via {body['provider'] or '-'} ({basis}); "
            f"{len(body['sources'])} source(s), grounded={body['grounded']}")


STEPS = [capabilities, upload, stream, downloads, survey_routes, ghosttrace, detect, chat]


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepEcho end-to-end test")
    parser.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    args = parser.parse_args()

    from fastapi.testclient import TestClient

    from backend.app.main import app

    print(f"workspace {WORK}; DEEPECHO_PIPELINE={os.environ['DEEPECHO_PIPELINE']}")
    started = time.time()
    try:
        with TestClient(app) as client:
            for run in STEPS:
                print(f"  ... {run.step_name}", flush=True)
                run(client)
    finally:
        width = max(len(name) for name, *_ in RESULTS) if RESULTS else 10
        print(f"\n{'step':<{width}}  result  seconds  note")
        print(f"{'-' * width}  ------  -------  ----")
        for name, result, seconds, note in RESULTS:
            print(f"{name:<{width}}  {result:<6}  {seconds:7.1f}  {note}")
        if args.keep:
            print(f"\nworkspace kept at {WORK}")
        else:
            shutil.rmtree(WORK, ignore_errors=True)

    failed = [r for r in RESULTS if r[1] == "FAIL"]
    skipped = [r for r in RESULTS if r[1] == "skip"]
    print(f"\n{'PASSED' if not failed else 'FAILED'}  {len(RESULTS) - len(failed) - len(skipped)} ok, "
          f"{len(skipped)} skipped, {len(failed)} failed, {time.time() - started:.0f} s")
    return 1 if failed or len(RESULTS) != len(STEPS) else 0


if __name__ == "__main__":
    sys.exit(main())
