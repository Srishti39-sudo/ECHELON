"""Field kit telemetry over HTTP: LoRa gateway in, live tracks out.

    POST /telemetry/ingest               one payload object, or a list of up to 50 (store-and-forward)
    GET  /telemetry/devices              every device seen, its last fix and its target link
    GET  /telemetry/tracks?since=ISO     fixes per device (optionally ?device_id=)
    GET  /telemetry/stream               Server-Sent Events: one `fix` event per stored fix
    GET  /telemetry/links?survey_id=     drifter tags linked to that survey's targets, with live
                                         forecast-vs-actual statistics
    GET  /telemetry/recoveries?survey_id= the recovery overlay (net_finder arrived / recovered)

The contract, the store and the maths live in ghosttrace/telemetry.py; this
module is HTTP only. Register it in backend/app/main.py with

    from .routes.telemetry import router as telemetry_router
    app.include_router(telemetry_router)

WHO MAY POST
    DEEPECHO_TELEMETRY_TOKEN set     every ingest must carry header
                                     X-DeepEcho-Token: <the token> (compared in
                                     constant time); 401 otherwise.
    DEEPECHO_TELEMETRY_TOKEN unset   ingest is accepted only from the loopback
                                     interface (127.0.0.1 / ::1), i.e. a gateway
                                     bridge on the same machine; 403 otherwise.
    The GET routes are read-only and, like the rest of this API, unauthenticated;
    the server binds to 127.0.0.1 by default.

RATE LIMITS (429 with Retry-After)
    DEEPECHO_TELEMETRY_RATE_PER_DEVICE   messages per minute per device_id (default 120)
    DEEPECHO_TELEMETRY_RATE_PER_CLIENT   messages per minute per client address (default 1200)
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend import config
from ghosttrace import telemetry as tm

log = logging.getLogger("deepecho")

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

SURVEYS_DIR = Path(os.environ.get("DEEPECHO_SURVEYS_DIR", config.ROOT / "data" / "surveys"))
TOKEN_ENV = "DEEPECHO_TELEMETRY_TOKEN"
TOKEN_HEADER = "X-DeepEcho-Token"
MAX_BATCH = 50
LOOPBACK = {"127.0.0.1", "::1", "localhost"}

_per_device = tm.RateLimiter(float(os.environ.get("DEEPECHO_TELEMETRY_RATE_PER_DEVICE", "120")))
_per_client = tm.RateLimiter(float(os.environ.get("DEEPECHO_TELEMETRY_RATE_PER_CLIENT", "1200")))

_store: tm.TelemetryStore | None = None
_store_lock = threading.Lock()


def store() -> tm.TelemetryStore:
    """The store for the directory configured NOW (tests point DEEPECHO_TELEMETRY_DIR elsewhere)."""
    global _store
    with _store_lock:
        wanted = tm.telemetry_dir()
        if _store is None or _store.dir.resolve() != wanted.resolve():
            _store = tm.TelemetryStore(wanted)
        return _store


def reset_rate_limits() -> None:
    """For tests."""
    global _per_device, _per_client
    _per_device = tm.RateLimiter(float(os.environ.get("DEEPECHO_TELEMETRY_RATE_PER_DEVICE", "120")))
    _per_client = tm.RateLimiter(float(os.environ.get("DEEPECHO_TELEMETRY_RATE_PER_CLIENT", "1200")))


def _authorise(request: Request) -> None:
    token = os.environ.get(TOKEN_ENV, "")
    if token:
        given = request.headers.get(TOKEN_HEADER, "")
        if not hmac.compare_digest(given.encode("utf-8"), token.encode("utf-8")):
            raise HTTPException(status_code=401, detail=f"missing or wrong {TOKEN_HEADER} header")
        return
    host = request.client.host if request.client else ""
    if host not in LOOPBACK:
        raise HTTPException(status_code=403,
                            detail=f"{TOKEN_ENV} is not set, so telemetry is accepted only from this machine "
                                   f"(loopback); set {TOKEN_ENV} and send {TOKEN_HEADER} to post from elsewhere")


# --- target lookup for links and recoveries -------------------------------------------------------


def _target_position(survey_id: str, detection_id: str) -> tuple[float, float] | None:
    if "/" in survey_id or "\\" in survey_id or survey_id.startswith("."):
        return None
    path = (SURVEYS_DIR / survey_id / "ghosttrace.json").resolve()
    try:
        path.relative_to(SURVEYS_DIR.resolve())
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for t in doc.get("targets") or []:
        if str(t.get("detection_id")) == detection_id and t.get("latitude") is not None \
                and t.get("longitude") is not None:
            return float(t["latitude"]), float(t["longitude"])
    return None


_FORECAST_THREADS: list[threading.Thread] = []


def _start_forecast(s: tm.TelemetryStore, link_id: int, record: dict[str, Any], synchronous: bool) -> None:
    def work() -> None:
        try:
            fc = tm.deployment_forecast(record["lat"], record["lon"], record["t_s"])
            if fc.get("available"):
                s.set_link_forecast(link_id, "ready", fc)
            else:
                s.set_link_forecast(link_id, "unavailable", fc, fc.get("reason"))
        except Exception as exc:  # recorded on the link; ingest has already succeeded
            log.warning("telemetry forecast failed: %s", exc)
            s.set_link_forecast(link_id, "failed", None, f"{type(exc).__name__}: {str(exc)[:200]}")

    if synchronous:
        work()
        return
    thread = threading.Thread(target=work, name=f"telemetry-forecast-{link_id}", daemon=True)
    _FORECAST_THREADS.append(thread)
    thread.start()


def _ingest_one(s: tm.TelemetryStore, payload: Any, client: str, synchronous_forecast: bool) -> dict[str, Any]:
    try:
        record = tm.validate_payload(payload)
    except tm.TelemetryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    ok, wait = _per_device.allow(f"device:{record['device_id']}")
    if not ok:
        raise HTTPException(status_code=429, detail=f"rate limit for device {record['device_id']}",
                            headers={"Retry-After": str(max(1, int(wait + 0.999)))})
    fix_id = s.insert_fix(record)
    out: dict[str, Any] = {"id": fix_id, "device_id": record["device_id"], "t": record["t"],
                           "simulated": record["simulated"]}
    if record["event"] == "deployed":
        link_id = s.create_link(record)
        _start_forecast(s, link_id, record, synchronous_forecast)
        out["link"] = {"id": link_id, "survey_id": record["survey_id"], "detection_id": record["detection_id"],
                       "forecast": "computing" if not synchronous_forecast else "done"}
    elif record["event"] == "retrieved":
        s.end_link(record["device_id"], record["t_s"])
    elif record["event"] in ("arrived", "recovered"):
        pos = _target_position(record["survey_id"], record["detection_id"])
        out["recovery"] = s.record_target_event(record, pos)
    return out


@router.post("/ingest")
async def ingest(request: Request) -> JSONResponse:
    """Store one payload (or a batch). 422 names the offending field."""
    _authorise(request)
    client = request.client.host if request.client else "unknown"
    ok, wait = _per_client.allow(f"client:{client}")
    if not ok:
        raise HTTPException(status_code=429, detail="rate limit for this client",
                            headers={"Retry-After": str(max(1, int(wait + 0.999)))})
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="body is not JSON")
    batch = isinstance(body, list)
    items = body if batch else [body]
    if batch and not 1 <= len(items) <= MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"a batch holds 1 to {MAX_BATCH} payloads")
    synchronous = request.query_params.get("wait_forecast") in ("1", "true")
    s = store()
    results = []
    for index, item in enumerate(items):
        try:
            results.append(await asyncio.to_thread(_ingest_one, s, item, client, synchronous))
        except HTTPException as exc:
            if not batch:
                raise
            results.append({"index": index, "error": exc.detail, "status": exc.status_code})
    if batch:
        accepted = sum(1 for r in results if "error" not in r)
        return JSONResponse({"accepted": accepted, "rejected": len(results) - accepted, "results": results},
                            status_code=200 if accepted else 422)
    return JSONResponse({"accepted": 1, **results[0]})


@router.get("/devices")
def devices() -> dict[str, Any]:
    return {"devices": store().devices()}


@router.get("/tracks")
def tracks(since: str | None = Query(default=None, description="ISO-8601 time with timezone"),
           device_id: str | None = Query(default=None)) -> dict[str, Any]:
    since_s = None
    if since:
        try:
            since_s = tm.parse_time(since, now_s=float("inf"))
        except tm.TelemetryError as exc:
            raise HTTPException(status_code=400, detail=f"since: {exc}")
    return {"since": since, "tracks": store().tracks(since_s, device_id)}


@router.get("/links")
def links(survey_id: str = Query(..., min_length=1, max_length=128)) -> dict[str, Any]:
    s = store()
    views = [tm.link_view(s, link) for link in s.links_for_survey(survey_id)]
    return {"survey_id": survey_id, "links": views}


@router.get("/recoveries")
def recoveries(survey_id: str | None = Query(default=None)) -> dict[str, Any]:
    records = [r for r in tm.load_recoveries(store().dir).values()
               if survey_id is None or r.get("survey_id") == survey_id]
    return {"survey_id": survey_id, "records": records}


@router.get("/stream")
async def stream(request: Request,
                 since_id: int | None = Query(default=None, ge=0),
                 limit: int | None = Query(default=None, ge=1, le=100000,
                                           description="close after this many fix events"),
                 timeout_s: float | None = Query(default=None, gt=0, le=86400,
                                                 description="close after this many seconds")) -> StreamingResponse:
    """Server-Sent Events. Starts after the newest stored fix unless since_id / Last-Event-ID says otherwise."""
    s = store()
    header_id = request.headers.get("last-event-id")
    if since_id is not None:
        last = since_id
    elif header_id and header_id.isdigit():
        last = int(header_id)
    else:
        last = await asyncio.to_thread(s.max_fix_id)

    async def events():
        nonlocal last
        loop = asyncio.get_running_loop()
        started = loop.time()
        sent = 0
        idle = 0.0
        yield f"retry: 3000\nevent: hello\ndata: {json.dumps({'last_id': last})}\n\n"
        while True:
            if timeout_s is not None and loop.time() - started >= timeout_s:
                return
            if await request.is_disconnected():
                return
            rows = await asyncio.to_thread(s.fixes_after, last, 200)
            for row in rows:
                last = row["id"]
                payload = {k: row[k] for k in ("id", "device_id", "device_type", "t", "lat", "lon", "heading", "roll",
                                               "pitch", "heave", "battery_v", "rssi", "snr", "event", "survey_id",
                                               "detection_id", "simulated")}
                yield f"id: {row['id']}\nevent: fix\ndata: {json.dumps(payload)}\n\n"
                sent += 1
                if limit is not None and sent >= limit:
                    return
            if rows:
                idle = 0.0
                continue
            await asyncio.sleep(0.5)
            idle += 0.5
            if idle >= 15.0:
                idle = 0.0
                yield ": keep-alive\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
