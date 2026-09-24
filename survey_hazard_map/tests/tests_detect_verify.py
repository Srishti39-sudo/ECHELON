#!/usr/bin/env python3
"""Tests for verification on the single-tile upload path (POST /detect).

    .venv/bin/python tests_detect_verify.py

The survey pipeline scores every detection 0-100 against the image and flags
likely false positives (hazard_verify). These cases check that an uploaded tile
gets the same treatment, that it is stored and shown, and that a suppressed
detection is kept but raises no alert and counts apart.

No torch is loaded. detect._ask, the only function that talks to the detector
worker, is replaced with fixed boxes, so the merge, verification, storage and
routes all run for real on real sample imagery. The store is a SQLite file in
a temporary directory, set through DEEPECHO_DB_PATH before the application is
imported, so data/deepecho.db is never touched.

Plain functions and asserts, like unit_tests.py; exit code 1 on any failure.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WORK = Path(tempfile.mkdtemp(prefix="deepecho-detect-verify-"))
os.environ["DEEPECHO_DB_PATH"] = str(WORK / "deepecho.db")
os.environ["DEEPECHO_UPLOAD_DIR"] = str(WORK / "uploads")
os.environ["DEEPECHO_ENABLE_UPLOAD"] = "1"
# Storage must be the temporary SQLite file even where Supabase is configured.
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_KEY"] = ""

# S-7: a real record with a water column, so the nadir is estimable (col ~804).
S7 = ROOT / "survey_hazard_map" / "samples" / "sidescan-s7-submarine.jpg"
TILE = ROOT / "survey_hazard_map" / "samples" / "tiles" / "sidescan-waterfall-strip_2048_1024.jpg"

# A box squarely on S-7's water column, and the box known.pt really reports on the wreck.
NADIR_BOX = [780.0, 300.0, 48.0, 160.0]
WRECK_BOX = [24.4, 41.7, 267.1, 495.3]

CASES = []


def case(name):
    def register(fn):
        CASES.append((name, fn))
        return fn
    return register


def _fake_worker(boxes):
    """Replace the worker round trip with fixed boxes, as detector_worker returns them."""
    from survey_hazard_map import detect

    detect._ask = lambda _bytes: [dict(b) for b in boxes]
    detect._status = {"ready": True, "models": ["anomaly", "known"], "classes": {}}


def _no_worker():
    from survey_hazard_map import detect

    detect._ask = lambda _bytes: None
    detect._status = {"ready": False, "models": [], "classes": {}}


BOXES = [
    {"model": "anomaly", "cls": "shipwreck", "confidence": 0.55, "bbox": NADIR_BOX},
    {"model": "known", "cls": "ship", "confidence": 0.78, "bbox": WRECK_BOX},
]


# --- survey_hazard_map.detect directly ---------------------------------------------------

@case("every record from a real tile carries confidence_pct in [0, 100] and a basis")
def _():
    from survey_hazard_map import detect

    boxes = [{"model": "anomaly", "cls": "shipwreck", "confidence": 0.4966,
              "bbox": [300.0, 200.0, 120.0, 90.0]},
             {"model": "known", "cls": "human", "confidence": 0.2988,
              "bbox": [40.0, 400.0, 60.0, 60.0]}]
    _fake_worker(boxes)
    records, stub, models = detect.run_detector(TILE.read_bytes(), TILE.name)
    assert not stub and models == ["anomaly", "known"], (stub, models)
    assert len(records) == 2, records
    for record in records:
        pct = record["confidence_pct"]
        assert isinstance(pct, float) and 0.0 <= pct <= 100.0, pct
        assert isinstance(record["suppressed"], bool)
        assert record["verification"]["status"] == "checked", record["verification"]
        assert "fused with image evidence" in record["confidence_pct_basis"]
        # The raw detector score is not overwritten.
        assert record["confidence"] in (0.4966, 0.2988)
        assert record["verification"]["detector_confidence"] == round(record["confidence"], 4)
        assert "terms" in record["verification"] and "reasons" in record["verification"]
    # The response and the JSON column both need plain JSON.
    json.dumps(records, allow_nan=False)


@case("a box on the nadir / water column is suppressed with a plain-English reason")
def _():
    from survey_hazard_map import detect

    _fake_worker(BOXES)
    records, _, _ = detect.run_detector(S7.read_bytes(), S7.name)
    by_class = {r["detector_class"]: r for r in records}
    nadir, wreck = by_class["shipwreck"], by_class["ship"]

    assert nadir["suppressed"] is True, nadir["verification"]
    assert "nadir_zone" in nadir["verification"]["hard_reasons"]
    assert any("nadir" in reason for reason in nadir["verification"]["reasons"])
    assert nadir["confidence_pct"] < 35.0, nadir["confidence_pct"]
    assert nadir["confidence"] == 0.55

    assert wreck["suppressed"] is False, wreck["verification"]["reasons"]
    assert wreck["confidence_pct"] > nadir["confidence_pct"]


@case("a verification failure degrades to the detector confidence and never fails")
def _():
    from survey_hazard_map import hazard_verify
    from survey_hazard_map import detect

    _fake_worker(BOXES)
    original = hazard_verify.verify_survey

    def broken(*_args, **_kwargs):
        raise RuntimeError("simulated verification failure")

    import logging

    # The failure is expected and logged with a traceback; keep the output clean.
    logger = logging.getLogger("deepecho")
    level, logger.disabled = logger.level, True
    hazard_verify.verify_survey = broken
    try:
        records, stub, _ = detect.run_detector(S7.read_bytes(), S7.name)
        junk, _, _ = detect.run_detector(b"not an image", "junk.png")
    finally:
        hazard_verify.verify_survey = original
        logger.disabled = False
        logger.setLevel(level)
    assert not stub and len(records) == 2
    for record in records:
        assert record["confidence_pct"] == round(record["confidence"] * 100, 1), record
        assert record["confidence_pct_basis"] == "detector confidence, not verified"
        assert record["suppressed"] is False
        assert record["verification"]["status"] == "not_checked"
        assert "simulated verification failure" in record["verification"]["reason"]

    # An image that cannot be decoded is the same: unverified, not an error.
    assert junk and all(r["confidence_pct_basis"] == "detector confidence, not verified"
                        and r["suppressed"] is False for r in junk)


@case("the stub path is labelled a stub and never presented as verified")
def _():
    from survey_hazard_map import detect

    _no_worker()
    records, stub, models = detect.run_detector(S7.read_bytes(), S7.name)
    assert stub and models == []
    for record in records:
        assert record["confidence_pct_basis"] == "stub, not verified", record
        assert record["verification"]["status"] == "not_checked"
        assert record["suppressed"] is False
        assert "evidence" not in record["verification"]


@case("a record without a height gets no dimensions; with no box it is not checked")
def _():
    from survey_hazard_map import detect

    _fake_worker([{"model": "anomaly", "cls": "other", "confidence": 0.5,
                   "bbox": [5000.0, 5000.0, 10.0, 10.0]}])
    records, _, _ = detect.run_detector(S7.read_bytes(), S7.name)
    record = records[0]
    assert "dimensions" not in record, record.get("dimensions")
    assert record["verification"]["status"] == "not_checked"
    assert record["confidence_pct_basis"].startswith("detector confidence, not verified")
    assert record["suppressed"] is False


# --- routes, with an isolated SQLite store ---------------------------------------

def _client():
    from fastapi.testclient import TestClient

    from survey_hazard_map import store
    from backend.app.main import app

    assert str(store.DB_PATH).startswith(str(WORK)), store.DB_PATH
    if not isinstance(store._store, store.SqliteStore):
        store._store = store.SqliteStore(WORK / "deepecho.db", WORK / "uploads")
    assert store._store.db_path == WORK / "deepecho.db", store._store.db_path
    return TestClient(app)


@case("POST /detect returns and stores confidence_pct, suppressed and the reasons")
def _():
    client = _client()
    _fake_worker(BOXES)
    response = client.post("/detect", files={"file": (S7.name, S7.read_bytes(), "image/jpeg")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stored"] is True, body.get("store_error")
    assert body["filtered"] == 1, body["filtered"]
    assert {d["suppressed"] for d in body["detections"]} == {True, False}
    for saved in body["saved"]:
        assert 0 <= saved["confidence_pct"] <= 100
        assert saved["record"]["verification"]["status"] == "checked"

    detail = client.get(f"/history/{body['scan_id']}").json()
    assert detail["total_detected"] == 2 and detail["total_filtered"] == 1, detail
    assert detail["total_objects"] == 1, detail
    rows = {d["record"]["detector_class"]: d for d in detail["detections"]}
    nadir = rows["shipwreck"]
    assert nadir["suppressed"] is True and nadir["verified"] is True
    assert nadir["verification_reasons"], nadir
    assert nadir["anomaly"] is False
    assert rows["ship"]["suppressed"] is False
    assert rows["ship"]["confidence_pct"] == rows["ship"]["record"]["confidence_pct"]

    scans = client.get("/history").json()["scans"]
    listed = next(s for s in scans if s["id"] == body["scan_id"])
    assert listed["total_filtered"] == 1 and listed["total_objects"] == 1


@case("a suppressed unidentified contact raises no anomaly, and counts apart in stats")
def _():
    client = _client()
    before = client.get("/stats").json()
    # A withheld class is unidentified, so without verification it would be an
    # anomaly and an alert. On the water column it is suppressed.
    _fake_worker([{"model": "known", "cls": "human", "confidence": 0.30, "bbox": NADIR_BOX}])
    body = client.post("/detect", files={"file": (S7.name, S7.read_bytes(),
                                                  "image/jpeg")}).json()
    record = body["detections"][0]
    assert record["object_class"] == "unknown" and record["suppressed"] is True, record
    assert body["saved"][0]["anomaly"] is False

    after = client.get("/stats").json()
    assert after["filtered_detections"] == before["filtered_detections"] + 1
    assert after["total_detections"] == before["total_detections"]
    assert after["total_anomalies"] == before["total_anomalies"]

    listing = client.get("/detections").json()
    ids = {d["id"] for d in listing["detections"]}
    assert body["saved"][0]["id"] in ids, "filtered rows are listed by default"
    assert listing["filtered"] == after["filtered_detections"]
    # What Alerts.jsx raises: anomaly or high, never a suppressed row.
    listed = next(d for d in listing["detections"] if d["id"] == body["saved"][0]["id"])
    assert listed["unidentified"] is True and listed["suppressed"] is True, listed
    alerts = [d for d in listing["detections"]
              if (d["anomaly"] or d["severity"] == "high") and not d["suppressed"]]
    assert body["saved"][0]["id"] not in {a["id"] for a in alerts}
    anomalies = client.get("/detections?anomaly_only=true").json()["detections"]
    assert body["saved"][0]["id"] not in {a["id"] for a in anomalies}

    hidden = client.get("/detections?include_filtered=false").json()["detections"]
    assert all(not d["suppressed"] for d in hidden)
    assert len(hidden) == len(listing["detections"]) - listing["filtered"]


@case("an old-style stored row with no verification fields still serialises, unverified")
def _():
    import uuid

    from survey_hazard_map import store as store_module

    client = _client()
    store = store_module.get_store()
    scan_id = str(uuid.uuid4())
    store.create_scan({"id": scan_id, "filename": "old.png", "models": ["known"]})
    old_id = str(uuid.uuid4())
    store.insert_detections(scan_id, [{
        "id": old_id, "class": "unknown", "confidence": 0.42, "bbox": [1, 2, 3, 4],
        "anomaly": True, "severity": "unknown",
        "record": {"object_class": "unknown", "confidence": 0.42, "bbox": [1, 2, 2, 2]},
    }])
    # And one with no record at all, as a Supabase row arrives.
    with store._lock:
        store._conn.execute("UPDATE detections SET record = NULL WHERE id = ?", (old_id,))
        store._conn.commit()

    detail = client.get(f"/history/{scan_id}")
    assert detail.status_code == 200, detail.text
    row = detail.json()["detections"][0]
    assert row["confidence_pct"] == 42.0 and row["suppressed"] is False
    assert row["verified"] is False and row["verification_reasons"] == []
    assert "not verified" in row["confidence_pct_basis"]
    assert detail.json()["total_anomalies"] == 1 and detail.json()["total_filtered"] == 0

    listing = client.get("/detections")
    assert listing.status_code == 200
    assert old_id in {d["id"] for d in listing.json()["detections"]}
    assert client.get("/stats").status_code == 200


def main() -> int:
    failures = 0
    try:
        for name, fn in CASES:
            try:
                fn()
                print(f"  ok    {name}")
            except Exception:
                failures += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    finally:
        try:
            from survey_hazard_map import store

            if isinstance(store._store, store.SqliteStore):
                store._store._conn.close()
        except Exception:
            pass
        shutil.rmtree(WORK, ignore_errors=True)
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
