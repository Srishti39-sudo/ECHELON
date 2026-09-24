#!/usr/bin/env python3
"""Tests for the field-kit telemetry (ghosttrace/telemetry.py, backend/app/routes/telemetry.py).

    .venv/bin/python tests_telemetry.py

Plain functions and asserts with an exit code. The telemetry router is mounted
on a bare FastAPI app (it is not registered in main.py by this change). Storage,
surveys and the recovery overlay live in a temporary directory. Drift forecasts
run over a SYNTHETIC analytic current field. Nothing touches data/.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WORK = Path(tempfile.mkdtemp(prefix="deepecho-telemetry-"))
os.environ["DEEPECHO_TELEMETRY_DIR"] = str(WORK / "telemetry")
os.environ["DEEPECHO_SURVEYS_DIR"] = str(WORK / "surveys")
os.environ["DEEPECHO_GHOSTTRACE_DATA_DIR"] = str(WORK / "ghosttrace")
os.environ.pop("DEEPECHO_TELEMETRY_TOKEN", None)
(WORK / "surveys").mkdir(parents=True)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ghosttrace.routes import api as gt_route  # noqa: E402
from ghosttrace.routes import telemetry as route  # noqa: E402
from ghosttrace import changes, telemetry as tm  # noqa: E402
from ghosttrace.currents import AnalyticField  # noqa: E402
from ghosttrace.layers import LayerSet  # noqa: E402

app = FastAPI()
app.include_router(route.router)
app.include_router(gt_route.router)
LOCAL = TestClient(app, client=("127.0.0.1", 50000))
REMOTE = TestClient(app, client=("192.168.1.50", 50000))

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


NOW = datetime.now(timezone.utc).replace(microsecond=0)


def payload(**over):
    p = {"device_id": "TAG-1", "device_type": "drifter_tag", "t": iso(NOW - timedelta(hours=1)),
         "lat": 9.12, "lon": 79.05, "battery_v": 4.1, "rssi": -98.0, "snr": 7.5}
    p.update(over)
    return p


def rejects(p, fragment):
    try:
        tm.validate_payload(p)
    except tm.TelemetryError as exc:
        assert fragment in str(exc), (fragment, str(exc))
        return
    raise AssertionError(f"accepted a payload that should fail ({fragment}): {p}")


# --- validation -----------------------------------------------------------------------------------


@case
def ingest_validation_rules():
    rec = tm.validate_payload(payload())
    assert rec["device_type"] == "drifter_tag" and rec["simulated"] is False and rec["event"] is None
    assert rec["t"].endswith("Z") and rec["heading"] is None
    rejects(payload(device_type="buoy"), "device_type")
    rejects(payload(device_id="bad id with spaces"), "device_id")
    rejects(payload(t="2026-09-15T04:30:00"), "timezone")
    rejects(payload(t="yesterday"), "ISO-8601")
    rejects(payload(t=iso(NOW + timedelta(hours=2))), "future")
    rejects(payload(lat=0.0, lon=0.0), "no-fix")
    rejects(payload(lat=95.0), "lat must be")
    rejects(payload(lat=float("nan")), "finite")
    rejects(payload(lat="9.1"), "finite")
    rejects(payload(heading=360.0), "heading")
    rejects(payload(rssi=12.0), "rssi")
    rejects(payload(colour="red"), "unknown field")
    rejects(payload(simulated="yes"), "simulated")
    rejects(payload(event="recovered", target={"survey_id": "s", "detection_id": "d"}), "not valid for drifter_tag")
    rejects(payload(event="deployed"), "needs target")
    rejects(payload(target={"survey_id": "s", "detection_id": "d"}), "only accepted with an event")
    rejects(payload(event="deployed", target={"survey_id": "../x", "detection_id": "d"}), "survey_id")
    rejects(payload(device_type="net_finder", event="recovered", target={"survey_id": "s"}), "detection_id")
    nav = tm.validate_payload(payload(device_type="nav_logger", heading=270.5, roll=-4.0, pitch=1.5, heave=0.3))
    assert nav["roll"] == -4.0 and nav["heave"] == 0.3
    rejects(payload(device_type="nav_logger", event="arrived", target={"survey_id": "s", "detection_id": "d"}),
            "not valid for nav_logger")


@case
def auth_rule_loopback_or_token():
    os.environ.pop("DEEPECHO_TELEMETRY_TOKEN", None)
    route.reset_rate_limits()
    r = REMOTE.post("/telemetry/ingest", json=payload(device_id="AUTH-1"))
    assert r.status_code == 403 and "DEEPECHO_TELEMETRY_TOKEN" in r.json()["detail"], r.text
    r = LOCAL.post("/telemetry/ingest", json=payload(device_id="AUTH-1"))
    assert r.status_code == 200 and r.json()["accepted"] == 1, r.text
    os.environ["DEEPECHO_TELEMETRY_TOKEN"] = "s3cret-token"
    try:
        assert REMOTE.post("/telemetry/ingest", json=payload(device_id="AUTH-2")).status_code == 401
        assert LOCAL.post("/telemetry/ingest", json=payload(device_id="AUTH-2")).status_code == 401  # token wins
        r = REMOTE.post("/telemetry/ingest", json=payload(device_id="AUTH-2"),
                        headers={"X-DeepEcho-Token": "s3cret-token-not"})
        assert r.status_code == 401
        r = REMOTE.post("/telemetry/ingest", json=payload(device_id="AUTH-2"), headers={"X-DeepEcho-Token": "s3cret-token"})
        assert r.status_code == 200, r.text
    finally:
        os.environ.pop("DEEPECHO_TELEMETRY_TOKEN", None)
    # GET routes are read-only and open, like the rest of the API.
    assert REMOTE.get("/telemetry/devices").status_code == 200


@case
def ingest_http_errors_and_batch():
    route.reset_rate_limits()
    r = LOCAL.post("/telemetry/ingest", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    r = LOCAL.post("/telemetry/ingest", json=payload(device_id="ERR-1", lat=123))
    assert r.status_code == 422 and "lat" in r.json()["detail"]
    batch = [payload(device_id="BATCH-1", t=iso(NOW - timedelta(minutes=30 - i))) for i in range(3)]
    batch.append(payload(device_id="BATCH-1", device_type="toaster"))
    r = LOCAL.post("/telemetry/ingest", json=batch)
    body = r.json()
    assert r.status_code == 200 and body["accepted"] == 3 and body["rejected"] == 1, body
    assert body["results"][3]["status"] == 422
    assert LOCAL.post("/telemetry/ingest", json=[]).status_code == 422
    tracks = LOCAL.get("/telemetry/tracks", params={"device_id": "BATCH-1"}).json()["tracks"]
    assert len(tracks) == 1 and len(tracks[0]["points"]) == 3
    since = iso(NOW - timedelta(minutes=28, seconds=30))
    tracks = LOCAL.get("/telemetry/tracks", params={"device_id": "BATCH-1", "since": since}).json()["tracks"]
    assert len(tracks[0]["points"]) == 1, tracks
    assert LOCAL.get("/telemetry/tracks", params={"since": "garbage"}).status_code == 400
    devices = {d["device_id"]: d for d in LOCAL.get("/telemetry/devices").json()["devices"]}
    assert devices["BATCH-1"]["fixes"] == 3 and devices["BATCH-1"]["simulated"] is False


@case
def rate_limit():
    limiter = tm.RateLimiter(per_minute=60, burst=2)
    assert limiter.allow("k", now=0.0)[0] and limiter.allow("k", now=0.0)[0]
    ok, wait = limiter.allow("k", now=0.0)
    assert not ok and close(wait, 1.0, 0.01)
    assert limiter.allow("k", now=1.01)[0]
    os.environ["DEEPECHO_TELEMETRY_RATE_PER_DEVICE"] = "6"   # burst 1
    try:
        route.reset_rate_limits()
        assert LOCAL.post("/telemetry/ingest", json=payload(device_id="RATE-1")).status_code == 200
        r = LOCAL.post("/telemetry/ingest", json=payload(device_id="RATE-1"))
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1, (r.status_code, r.headers)
        assert LOCAL.post("/telemetry/ingest", json=payload(device_id="RATE-2")).status_code == 200
    finally:
        os.environ.pop("DEEPECHO_TELEMETRY_RATE_PER_DEVICE", None)
        route.reset_rate_limits()


def close(a, b, tol):
    return abs(a - b) <= tol


@case
def server_sent_events():
    route.reset_rate_limits()
    start_id = route.store().max_fix_id()
    for i in range(3):
        LOCAL.post("/telemetry/ingest", json=payload(device_id="SSE-1", t=iso(NOW - timedelta(minutes=10 - i)),
                                                    simulated=True))
    with LOCAL.stream("GET", "/telemetry/stream", params={"since_id": start_id, "limit": 3}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    blocks = [b for b in text.split("\n\n") if b.strip()]
    assert blocks[0].startswith("retry:") and "event: hello" in blocks[0]
    fixes = [b for b in blocks if "event: fix" in b]
    assert len(fixes) == 3, text
    data = [json.loads(next(line[6:] for line in b.splitlines() if line.startswith("data: "))) for b in fixes]
    assert [d["device_id"] for d in data] == ["SSE-1"] * 3 and all(d["simulated"] is True for d in data)
    ids = [int(next(line[4:] for line in b.splitlines() if line.startswith("id: "))) for b in fixes]
    assert ids == sorted(ids) and ids[0] > start_id
    # A timeout closes an idle stream.
    with LOCAL.stream("GET", "/telemetry/stream", params={"timeout_s": 0.6}) as r:
        text = "".join(r.iter_text())
    assert "event: hello" in text and "event: fix" not in text


# --- drifter linkage -------------------------------------------------------------------------------

U_EAST = 0.2


def synthetic_field(start: datetime) -> AnalyticField:
    return AnalyticField(bounds=(8.5, 78.5, 9.8, 79.8), t_start=start - timedelta(hours=1), hours=80,
                         velocity=(U_EAST, 0.0), label="SYNTHETIC uniform eastward test current")


@case
def drifter_linkage_stats_maths():
    start = NOW - timedelta(hours=10)
    field = synthetic_field(start)
    fc = tm.deployment_forecast(9.12, 79.05, start.timestamp(), field=field, layers=LayerSet([]),
                                horizon_hours=12, n_particles=300, snapshot_every_hours=2)
    assert fc["available"] and fc["current_source"]["synthetic"] is True
    assert [s["t_hours"] for s in fc["snapshots"]] == [0, 2, 4, 6, 8, 10, 12]
    # A tag that moved exactly with the current for 6 h (4.32 km east): on the mean, inside both cones.
    from ghosttrace.validation import displace
    lat6, lon6 = displace(9.12, 79.05, U_EAST * 6 * 3600, 0.0)
    cmp = tm.compare_fix(fc, 6.0, lat6, lon6)
    assert cmp["available"] and cmp["separation_km"] < 0.3 and cmp["inside_cone50"] and cmp["inside_cone90"], cmp
    assert cmp["cone_t_hours"] == 6
    # Interpolated mean at 5 h lies between the 4 h and 6 h means.
    lat5, lon5 = displace(9.12, 79.05, U_EAST * 5 * 3600, 0.0)
    assert tm.compare_fix(fc, 5.0, lat5, lon5)["separation_km"] < 0.3
    # 15 km north of where it should be: separated, outside both cones.
    far_lat, far_lon = displace(lat6, lon6, 0.0, 15000.0)
    cmp = tm.compare_fix(fc, 6.0, far_lat, far_lon)
    assert close(cmp["separation_km"], 15.0, 0.4) and cmp["inside_cone90"] is False, cmp
    assert tm.compare_fix(fc, 13.0, lat6, lon6)["available"] is False
    assert tm.compare_fix(fc, -1.0, lat6, lon6)["available"] is False
    at0 = tm.compare_fix(fc, 0.0, 9.12, 79.05)
    assert at0["inside_cone90"] is None and at0["cone_note"]


@case
def drifter_linkage_through_the_api():
    route.reset_rate_limits()
    start = NOW - timedelta(hours=8)
    field = synthetic_field(start)
    original = tm.deployment_forecast

    def fake_forecast(lat, lon, t_s, **kw):
        return original(lat, lon, t_s, field=field, layers=LayerSet([]), horizon_hours=12, n_particles=300,
                        snapshot_every_hours=1)

    tm.deployment_forecast = fake_forecast
    try:
        from ghosttrace.validation import displace
        deploy = payload(device_id="LINK-TAG", t=iso(start), lat=9.12, lon=79.05, simulated=True, event="deployed",
                         target={"survey_id": "alpha", "detection_id": "alpha_d0"})
        r = LOCAL.post("/telemetry/ingest?wait_forecast=1", json=deploy)
        assert r.status_code == 200 and r.json()["link"]["detection_id"] == "alpha_d0", r.text
        for h in (1, 2, 3):
            lat, lon = displace(9.12, 79.05, U_EAST * h * 3600, 0.0)
            assert LOCAL.post("/telemetry/ingest", json=payload(device_id="LINK-TAG", t=iso(start + timedelta(hours=h)),
                                                               lat=lat, lon=lon, simulated=True)).status_code == 200
        # a fix before the deployment belongs to no link
        LOCAL.post("/telemetry/ingest", json=payload(device_id="LINK-TAG", t=iso(start - timedelta(hours=1)),
                                                    simulated=True))
    finally:
        tm.deployment_forecast = original
    body = LOCAL.get("/telemetry/links", params={"survey_id": "alpha"}).json()
    assert len(body["links"]) == 1, body
    link = body["links"][0]
    assert link["forecast_status"] == "ready" and link["simulated"] is True
    assert len(link["track"]) == 4, link["track"]
    latest = link["latest"]
    assert latest["available"] and close(latest["elapsed_hours"], 3.0, 1e-6)
    assert latest["separation_km"] < 0.3 and latest["inside_cone90"] is True, latest
    assert link["cone_now"]["t_hours"] == 3 and link["cone_now"]["cone90"]["type"] in ("Polygon", "MultiPolygon")
    assert len(link["series"]) == 4 and any("SIMULATED DEVICE" in n for n in link["notes"])
    assert LOCAL.get("/telemetry/links", params={"survey_id": "nobody"}).json()["links"] == []
    devices = {d["device_id"]: d for d in LOCAL.get("/telemetry/devices").json()["devices"]}
    assert devices["LINK-TAG"]["link"]["detection_id"] == "alpha_d0" and devices["LINK-TAG"]["simulated"]


# --- recovery -> overlay -> changes ------------------------------------------------------------------


def write_prior_and_current(root: Path):
    """Prior survey p1 (one net, located) and current survey c2 covering it with no detection there."""
    import tests_ghosttrace_core as core

    prior = core.previous_detections()
    p1 = core.write_survey(root, "p1", prior, processed_at="2026-06-01T00:00:00+00:00", nav=False)
    cur = core.current_detections()
    c2 = core.write_survey(root, "c2", cur, processed_at="2026-09-02T00:00:00+00:00")
    targets = core._targets_from(cur)
    return p1, c2, targets, {d["id"]: d for d in prior}


@case
def recovered_event_to_overlay_to_changes():
    route.reset_rate_limits()
    root = Path(os.environ["DEEPECHO_SURVEYS_DIR"])
    p1, c2, targets, prior = write_prior_and_current(root)
    net = prior["S1_P3"]                       # the one change tracking calls removed
    # the prior survey's ghosttrace.json lets the server measure distance to the target
    (p1 / "ghosttrace.json").write_text(json.dumps({"survey_id": "p1", "targets": [
        {"detection_id": d["id"], "object_class": d["object_class"], "latitude": d["latitude"],
         "longitude": d["longitude"], "dimensions": d.get("dimensions")} for d in prior.values()]}))

    baseline = changes.compare_with_previous(c2, targets, surveys_root=root, recoveries={})
    rem = baseline["removed_since_previous"]
    assert [r["previous_detection_id"] for r in rem] == ["S1_P3"] and rem[0]["status"] == "removed"
    assert baseline["change_summary"]["confirmed_recovered"] == 0

    finder = {"device_id": "FINDER-1", "device_type": "net_finder", "lat": net["latitude"] + 0.0001,
              "lon": net["longitude"], "simulated": True, "target": {"survey_id": "p1", "detection_id": "S1_P3"}}
    r = LOCAL.post("/telemetry/ingest", json={**finder, "t": "2026-08-01T05:00:00Z", "event": "arrived"})
    assert r.status_code == 200 and r.json()["recovery"]["status"] == "arrived", r.text
    r = LOCAL.post("/telemetry/ingest", json={**finder, "t": "2026-08-01T06:00:00Z", "event": "recovered"})
    rec = r.json()["recovery"]
    assert rec["status"] == "recovered" and rec["recovered_at"] == "2026-08-01T06:00:00Z" and rec["device_id"] == "FINDER-1"
    assert rec["simulated"] is True and close(rec["distance_to_target_m"], 11.1, 1.0), rec
    assert [h["event"] for h in rec["history"]] == ["arrived", "recovered"]
    assert any("SIMULATED DEVICE" in f for f in rec["flags"])

    overlay = tm.load_recoveries()
    assert overlay["p1|S1_P3"]["status"] == "recovered"
    listed = LOCAL.get("/telemetry/recoveries", params={"survey_id": "p1"}).json()["records"]
    assert len(listed) == 1 and listed[0]["detection_id"] == "S1_P3"

    # changes.py with the default overlay (data dir from DEEPECHO_TELEMETRY_DIR)
    res = changes.compare_with_previous(c2, targets, surveys_root=root)
    removed = res["removed_since_previous"]
    assert len(removed) == 1 and removed[0]["status"] == "removed: confirmed recovered", removed
    assert removed[0]["recovered"] is True and removed[0]["recovery"]["device_id"] == "FINDER-1"
    assert "SIMULATED DEVICE" in removed[0]["basis"]
    assert res["change_summary"]["removed"] == 1 and res["change_summary"]["confirmed_recovered"] == 1
    # the other statuses are unaffected
    assert res["changes"]["S2_D1"]["status"] == "persistent" and res["changes"]["S2_D2"]["status"] == "moved"

    # A recovery reported AFTER the current survey does not explain its absence.
    later = {"p1|S1_P3": {**overlay["p1|S1_P3"], "recovered_at": "2026-09-10T00:00:00Z"}}
    res = changes.compare_with_previous(c2, targets, surveys_root=root, recoveries=later)
    removed = res["removed_since_previous"]
    assert removed[0]["status"] == "removed" and "recovery_reported_later" in removed[0], removed
    assert res["change_summary"]["confirmed_recovered"] == 0

    # A net reported recovered but still matched in the later survey is flagged, not hidden.
    flagged = {"p1|S1_P1": {"status": "recovered", "recovered_at": "2026-08-01T00:00:00Z", "device_id": "FINDER-1"}}
    res = changes.compare_with_previous(c2, targets, surveys_root=root, recoveries=flagged)
    assert res["changes"]["S2_D1"]["recovery_reported"]["device_id"] == "FINDER-1"
    assert any("verify" in n for n in res["notes"])

    # The ghosttrace router re-reads a stored document's removed list against the overlay.
    doc = {"survey_id": "c2", "targets": [], "removed_since_previous": baseline["removed_since_previous"]}
    (c2 / "ghosttrace.json").write_text(json.dumps(doc))
    body = LOCAL.get("/ghosttrace/c2/recoveries").json()
    assert body["confirmed_recovered"] == 1 and body["removed_since_previous"][0]["status"] == "removed: confirmed recovered"
    body = LOCAL.get("/ghosttrace/p1/recoveries").json()
    assert body["records"][0]["status"] == "recovered" and body["removed_since_previous"] == []


@case
def recovery_far_from_target_is_flagged():
    store = route.store()
    rec = tm.validate_payload({"device_id": "FINDER-2", "device_type": "net_finder", "t": iso(NOW - timedelta(minutes=5)),
                               "lat": 9.2, "lon": 79.2, "event": "recovered",
                               "target": {"survey_id": "zeta", "detection_id": "z1"}})
    out = store.record_target_event(rec, (9.12, 79.05))
    assert out["status"] == "recovered" and out["distance_to_target_m"] > 10000
    assert any("verify" in f for f in out["flags"]) and not any("SIMULATED" in f for f in out["flags"])
    out = store.record_target_event({**rec, "survey_id": "zeta", "detection_id": "z2"}, None)
    assert any("distance not checked" in f for f in out["flags"])


@case
def storage_is_git_ignored():
    ignore = Path(os.environ["DEEPECHO_TELEMETRY_DIR"]) / ".gitignore"
    assert ignore.is_file() and "*" in ignore.read_text().split()


def main() -> int:
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
