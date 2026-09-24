"""Survey jobs: upload a sonar log, watch it being processed, collect the reports.

    POST /survey/jobs                  multipart upload, starts a worker, 202
    GET  /survey/jobs/capabilities     which pipeline an upload would run, and why
    GET  /survey/jobs/{id}             state, event count, last event
    GET  /survey/jobs/{id}/events      Server-Sent Events, replay then tail

This is the path the mission console (/mission in the frontend) uses. It exists
beside POST /survey/process rather than replacing it, because the two answer
different needs: /survey/process holds one request open for a small survey on
files already on the server, while this accepts the operator's own files and
reports progress as the run happens.

HOW IT RUNS
    The route writes the uploads and a job.json, starts
    `python -m survey_hazard_map.survey_job --job <id>` with a fixed argv and no shell,
    and returns. The worker appends events to data/surveys/<id>/events.jsonl
    and this module tails that file. See backend/survey_job.py for why the
    worker is a subprocess (torch and faiss cannot share a process on macOS)
    and why the channel is a file.

WHY IT IS GATED THE WAY IT IS
    Mirrors /detect. The route exists only when the process could genuinely run
    a detector: config.ENABLE_UPLOAD defaults to "torch and ultralytics are
    importable and a checkpoint is on disk", so a full local checkout has it
    and the serve container, which ships no torch, does not advertise a route
    it would only fail. DEEPECHO_ENABLE_SURVEY_JOBS overrides that in either
    direction, independently of DEEPECHO_ENABLE_UPLOAD.

    Like /detect, nothing here authenticates. The API binds to 127.0.0.1 by
    default and is an operator console, not a public service. What this module
    does defend against is the input itself: filenames and ids are validated
    as path segments, extensions are allow-listed, the total size is capped,
    concurrent jobs are capped, and nothing from the client reaches a shell.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile

from backend import config
from survey_hazard_map import survey_job

log = logging.getLogger("deepecho")

router = APIRouter(prefix="/survey/jobs", tags=["survey jobs"])

_flag = os.environ.get("DEEPECHO_ENABLE_SURVEY_JOBS", "").strip().lower()


def _teammate_ready() -> bool:
    try:
        return bool(survey_job.teammate_status()["available"])
    except Exception:  # a broken teammate folder must not take the API down
        return False


# The team pipeline counts as "a detector can really run" too, so a machine
# that has only the team's files still gets the route.
ENABLED = (True if _flag in {"1", "true", "yes"} else
           False if _flag in {"0", "false", "no"} else
           bool(config.ENABLE_UPLOAD) or _teammate_ready())

MAX_TOTAL_BYTES = int(os.environ.get("DEEPECHO_MAX_SURVEY_UPLOAD_BYTES", str(512 * 1024 * 1024)))
MAX_CONCURRENT = max(1, int(os.environ.get("DEEPECHO_MAX_SURVEY_JOBS", "1")))
MAX_FILES = int(os.environ.get("DEEPECHO_MAX_SURVEY_FILES", "32"))
HEARTBEAT_SECONDS = float(os.environ.get("DEEPECHO_SSE_HEARTBEAT", "15"))
POLL_SECONDS = 0.25

STRIP_EXTENSIONS = (".xtf", ".jsf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
NAV_EXTENSIONS = (".csv",)
RAW_EXTENSIONS = tuple(s.lower() for s in survey_job.SONAR_SUFFIXES)

# Ids are path segments and URL segments. "jobs" is reserved because
# /survey/jobs/... would otherwise be ambiguous with /survey/{survey_id}/...
JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
RESERVED_IDS = {"jobs", "capabilities"}
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

TERMINAL_TYPES = {"done", "error"}

# Workers started by THIS server process, so their exit can be reaped and the
# concurrency cap counts real, running processes rather than stale files.
_processes: dict[str, subprocess.Popen] = {}


# --- validation --------------------------------------------------------------

def validate_job_id(job_id: str) -> str:
    """The same refusal survey.py's _survey_dir applies, made stricter.

    No separators, no parent references, no leading dot, a bounded length, and
    after resolution the path must still be inside SURVEYS_DIR.
    """
    if (not job_id or not JOB_ID.fullmatch(job_id) or job_id in RESERVED_IDS):
        raise HTTPException(status_code=400, detail=f"invalid job id {job_id!r}")
    for root in (survey_job.SURVEYS_DIR, survey_job.UPLOADS_DIR):
        resolved = (root / job_id).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid job id {job_id!r}")
    return job_id


def safe_filename(name: str | None, allowed: tuple[str, ...]) -> str:
    """A client filename, accepted only as a plain name with an allowed type.

    A name carrying a directory is refused outright rather than stripped to its
    last segment: a browser never sends one, so a client that does is not a
    browser, and silently "fixing" it would hide that. What remains is reduced
    to a conservative character set.
    """
    if not name or "\x00" in name:
        raise HTTPException(status_code=400, detail="a file was uploaded without a name")
    if "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise HTTPException(status_code=400, detail=f"invalid filename {name!r}")
    cleaned = SAFE_NAME.sub("_", name)[-120:].lstrip("._-")
    suffix = Path(cleaned).suffix.lower()
    if not cleaned or suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"{name!r} is not an accepted type. Accepted: {', '.join(allowed)}")
    return cleaned


def parse_corners(raw: str | None) -> dict[str, list[float]] | None:
    """Four strip corners as {"top_left": [lat, lon], ...}, checked, or None."""
    if raw is None or not str(raw).strip():
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="corners must be a JSON object")
    keys = ("top_left", "top_right", "bottom_left", "bottom_right")
    if not isinstance(payload, dict) or any(k not in payload for k in keys):
        raise HTTPException(status_code=400,
                            detail=f"corners must give {', '.join(keys)} as [lat, lon]")
    corners: dict[str, list[float]] = {}
    for key in keys:
        value = payload[key]
        try:
            lat, lon = (float(v) for v in value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"corners.{key} must be [lat, lon]")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise HTTPException(status_code=400, detail=f"corners.{key} is out of range")
        corners[key] = [lat, lon]
    return corners


# --- process bookkeeping -------------------------------------------------------

def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _running_count() -> int:
    """Workers that are still doing survey work.

    A worker that has recorded done or failed is not counted even if its
    process has not exited yet: tearing down torch can take a few seconds
    after the last event, and refusing the next upload for that would be a
    limit on interpreter shutdown, not on processing.
    """
    running = 0
    for job_id, process in list(_processes.items()):
        if process.poll() is not None:
            _processes.pop(job_id, None)
            continue
        status = survey_job.read_json(survey_job.job_paths(job_id)["status"], {}) or {}
        if status.get("state") not in {"done", "failed"}:
            running += 1
    return running


def _append_event(job_id: str, event: dict[str, Any]) -> None:
    """Write an event on the worker's behalf, used only once the worker is gone."""
    events = survey_job.EventLog(survey_job.job_paths(job_id)["events"])
    try:
        events.emit(event)
    finally:
        events.close()


def _reconcile(job_id: str, status: dict[str, Any]) -> dict[str, Any]:
    """Notice a worker that died without saying so, and record it.

    A worker that segfaults, is OOM-killed or loses its interpreter never
    writes its own error event, and a stream waiting for one would wait
    forever. If the job still claims to be queued or running but its process
    is gone, it failed, and the event log is told so in the same format the
    worker would have used.
    """
    if status.get("state") not in {"queued", "running"}:
        return status
    process = _processes.get(job_id)
    exit_note = ""
    if process is not None:
        code = process.poll()
        if code is None:
            return status
        _processes.pop(job_id, None)
        # A negative code is the signal that killed it. Naming it turns "exited
        # without a result" into something a reader can act on: -9 is an
        # external kill, -11 a crash in a native library.
        exit_note = (f" (killed by signal {-code})" if code < 0 else f" (exit code {code})")
    elif _pid_alive(status.get("pid")):
        return status
    elif status.get("state") == "queued" and time.time() - status.get("created_ts", 0) < 30:
        return status  # a worker that has not written its pid yet

    # Re-read: the worker may have finished between the two checks.
    status = survey_job.read_json(survey_job.job_paths(job_id)["status"], status) or status
    if status.get("state") not in {"queued", "running"}:
        return status
    message = f"the survey worker exited without reporting a result{exit_note}; see worker.log"
    _append_event(job_id, {"type": "error", "message": message})
    return survey_job.update_status(job_id, state="failed", error=message,
                                    finished_at=survey_job.utc_now())


def _load_status(job_id: str) -> dict[str, Any]:
    validate_job_id(job_id)
    status = survey_job.read_json(survey_job.job_paths(job_id)["status"])
    if not isinstance(status, dict):
        raise HTTPException(status_code=404, detail=f"no survey job {job_id!r}")
    return _reconcile(job_id, status)


# --- routes ----------------------------------------------------------------------

@router.post("", status_code=202)
async def create_job(request: Request) -> dict[str, Any]:
    """Accept a survey upload and start processing it.

    Multipart fields:
        files      one or more .xtf, .jsf, .png, .jpg, .jpeg, .tif, .tiff
        nav        optional .csv of pixel-to-lat/lon fixes (image strips)
        corners    optional JSON {"top_left": [lat, lon], "top_right": ...,
                   "bottom_left": ..., "bottom_right": ...}, applied to every
                   image strip; ignored for raw logs, which carry their own
                   navigation
        title      optional human label
        survey_id  optional id; generated when absent
        conf       optional confidence threshold override, 0..1

    The body is parsed by hand rather than through declared parameters so the
    size cap can be applied from Content-Length before any of it is spooled.
    """
    if _running_count() >= MAX_CONCURRENT:
        raise HTTPException(status_code=429,
                            detail=f"{MAX_CONCURRENT} survey job(s) already running; "
                                   "wait for it to finish and try again")

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_TOTAL_BYTES + 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"upload is {int(declared)} bytes; the limit is "
                                   f"{MAX_TOTAL_BYTES} bytes in total")

    try:
        form = await request.form(max_files=MAX_FILES + 1, max_fields=16)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not read the upload: {exc}")

    try:
        uploads = [f for f in form.getlist("files") if isinstance(f, UploadFile)]
        if not uploads:
            raise HTTPException(status_code=400,
                                detail="no files: send one or more sonar logs or strip images "
                                       "in the 'files' field")
        if len(uploads) > MAX_FILES:
            raise HTTPException(status_code=400, detail=f"at most {MAX_FILES} files per survey")

        nav_upload = form.get("nav")
        nav_upload = nav_upload if isinstance(nav_upload, UploadFile) and nav_upload.filename else None
        corners = parse_corners(form.get("corners") if isinstance(form.get("corners"), str) else None)
        if nav_upload and corners:
            raise HTTPException(status_code=400,
                                detail="nav and corners are two ways to say the same thing; pick one")

        title = form.get("title")
        title = title.strip()[:120] if isinstance(title, str) and title.strip() else None

        conf = form.get("conf")
        if isinstance(conf, str) and conf.strip():
            try:
                conf = float(conf)
            except ValueError:
                raise HTTPException(status_code=400, detail="conf must be a number")
            if not 0.0 <= conf <= 1.0:
                raise HTTPException(status_code=400, detail="conf must be between 0 and 1")
        else:
            conf = None

        requested = form.get("survey_id")
        if isinstance(requested, str) and requested.strip():
            job_id = validate_job_id(requested.strip())
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            job_id = f"mission-{stamp}-{uuid.uuid4().hex[:6]}"

        names: list[str] = []
        for upload in uploads:
            name = safe_filename(upload.filename, STRIP_EXTENSIONS)
            base, suffix, n = Path(name).stem, Path(name).suffix, 2
            while name in names:  # two files that sanitise to one name
                name, n = f"{base}-{n}{suffix}", n + 1
            names.append(name)
        nav_name = safe_filename(nav_upload.filename, NAV_EXTENSIONS) if nav_upload else None
        if nav_name in names:
            nav_name = f"nav-{nav_name}"

        if any(Path(n).suffix.lower() in RAW_EXTENSIONS for n in names) \
                and not (config.ROOT / "survey_hazard_map" / "sonar_ingest.py").is_file():
            raise HTTPException(
                status_code=415,
                detail="raw XTF/JSF ingest is not installed on this backend; upload strip "
                       "images with a navigation CSV or corners instead")

        paths = survey_job.job_paths(job_id)
        if paths["survey"].exists() or paths["upload"].exists():
            raise HTTPException(status_code=409, detail=f"survey {job_id!r} already exists")

        # --- write, counting every byte against the cap --------------------------
        paths["upload"].mkdir(parents=True)
        total = 0
        try:
            for upload, name in [*zip(uploads, names),
                                 *([(nav_upload, nav_name)] if nav_upload else [])]:
                with (paths["upload"] / name).open("wb") as handle:
                    while chunk := await upload.read(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_TOTAL_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"upload exceeds the {MAX_TOTAL_BYTES}-byte limit")
                        handle.write(chunk)
                if (paths["upload"] / name).stat().st_size == 0:
                    raise HTTPException(status_code=400, detail=f"{name} is empty")
        except BaseException:
            shutil.rmtree(paths["upload"], ignore_errors=True)
            raise
    finally:
        await form.close()

    created = survey_job.utc_now()
    spec = {"job_id": job_id, "title": title, "inputs": names, "nav": nav_name,
            "corners": corners, "options": {"conf": conf}, "created_at": created,
            "bytes": total}
    survey_job.write_json_atomic(paths["spec"], spec)
    paths["survey"].mkdir(parents=True)
    paths["events"].touch()
    survey_job.write_json_atomic(paths["status"], {
        "job_id": job_id, "title": title, "state": "queued", "created_at": created,
        "created_ts": time.time(), "started_at": None, "finished_at": None, "error": None,
        "inputs": names, "nav": nav_name, "corners": bool(corners)})

    # A fixed argv, no shell. The only client-derived value is the job id, which
    # has already been held to [A-Za-z0-9][A-Za-z0-9_-]{0,63}.
    command = ["--job", job_id]  # for survey_hazard_map.survey_job, started by procs.spawn_python
    if _running_count() >= MAX_CONCURRENT:  # another upload finished writing first
        shutil.rmtree(paths["upload"], ignore_errors=True)
        shutil.rmtree(paths["survey"], ignore_errors=True)
        raise HTTPException(status_code=429,
                            detail=f"{MAX_CONCURRENT} survey job(s) already running; "
                                   "wait for it to finish and try again")
    worker_log = (paths["survey"] / "worker.log").open("ab")
    try:
        # posix_spawn, not fork: see backend/procs.py for the crash this avoids.
        # setsid=True keeps the worker in its own session, as before.
        from backend.procs import spawn_python

        process = spawn_python("survey_hazard_map.survey_job", command, setsid=True,
                               stdout=worker_log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL)
    except OSError as exc:
        survey_job.update_status(job_id, state="failed", error=f"worker did not start: {exc}")
        raise HTTPException(status_code=500, detail=f"the survey worker could not start: {exc}")
    finally:
        worker_log.close()

    _processes[job_id] = process
    survey_job.update_status(job_id, pid=process.pid)
    log.info("survey job %s started (pid %s, %d file(s), %d bytes)",
             job_id, process.pid, len(names), total)

    return {"job_id": job_id, "state": "queued", "title": title, "inputs": names,
            "nav": nav_name, "corners": corners,
            "events_url": f"/survey/jobs/{job_id}/events",
            "status_url": f"/survey/jobs/{job_id}"}


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    """Which pipeline the next upload runs, and why, before anything is uploaded.

    Declared before /{job_id} so "capabilities" is never read as a job id. The
    answer is computed per request from the files on disk and the environment,
    exactly as the worker computes it, so dropping the team's files in place
    changes it on the next request with no restart.
    """
    choice = survey_job.select_pipeline()
    team, engine = choice["teammate"], choice["echelon"]

    def public(block: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {k: block.get(k) for k in keys}

    return {
        "enabled": ENABLED,
        "requested": choice["requested"],
        "selected": choice["pipeline"],
        "label": choice["label"],
        "reason": choice["reason"],
        "error": choice["error"],
        "env": survey_job.PIPELINE_ENV,
        "pipelines": {
            "echelon": public(engine, ("available", "label", "reason", "models")),
            "teammate": {
                **public(team, ("available", "label", "reason", "found", "missing",
                                "shadow", "notes")),
                "anomaly": team["anomaly"] is not None,
                "modules": team["modules"],
                "weights": team["weights"],
                "calib": team["calib"],
            },
        },
        "notes": [
            "The team pipeline runs one source per job: the first .xtf, else a georeferenced "
            "GeoTIFF mosaic, else a .png/.jpg with a NavTable navigation CSV. Other files "
            "are listed as not processed.",
            "Images without a NavTable CSV run on the DeepEcho engine.",
        ],
    }


@router.get("/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    """Where a job is, without opening a stream."""
    status = _load_status(job_id)
    events_path = survey_job.job_paths(job_id)["events"]
    last = None
    count = 0
    try:
        with events_path.open("rb") as handle:
            for line in handle:
                if line.endswith(b"\n"):
                    count += 1
                    last = line
    except OSError:
        pass
    last_event = None
    if last:
        try:
            parsed = json.loads(last)
            last_event = {k: parsed.get(k) for k in ("seq", "ts", "type", "stage", "message")
                          if parsed.get(k) is not None}
        except ValueError:
            pass
    status = {k: v for k, v in status.items() if k not in {"pid", "created_ts"}}
    return {**status, "events": count, "last_event": last_event}


@router.get("/{job_id}/events")
async def job_events(job_id: str, request: Request, after: int | None = None):
    """The job's events as Server-Sent Events: everything so far, then live.

    Resumes after `?after=N` or the Last-Event-ID header (which EventSource
    sends by itself on reconnect), whichever is larger. Each event's SSE id is
    its sequence number. The stream ends after a done or error event. A comment
    line is sent every HEARTBEAT_SECONDS so proxies and the browser keep an
    idle connection open while a long tile is being detected.
    """
    _load_status(job_id)
    path = survey_job.job_paths(job_id)["events"]

    start = max(0, after or 0)
    header = request.headers.get("last-event-id", "")
    if header.strip().isdigit():
        start = max(start, int(header.strip()))

    async def stream():
        yield "retry: 3000\n\n"
        seq = 0
        buffer = b""
        last_sent = time.monotonic()
        last_check = time.monotonic()
        handle = None
        finished_pending = finished = False
        try:
            while True:
                if handle is None and path.exists():
                    handle = path.open("rb")
                chunk = handle.read() if handle else b""
                if chunk:
                    buffer += chunk
                    *lines, buffer = buffer.split(b"\n")
                    for line in lines:
                        if not line.strip():
                            continue
                        seq += 1
                        if seq <= start:
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        yield f"id: {seq}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                        last_sent = time.monotonic()
                        if event.get("type") in TERMINAL_TYPES:
                            return
                    continue

                if finished:
                    # The job is over and the file has been read to its end
                    # without a terminal event (an old or hand-edited log).
                    # Ending is better than a stream that waits for nothing.
                    return
                if await request.is_disconnected():
                    return
                now = time.monotonic()
                if now - last_sent >= HEARTBEAT_SECONDS:
                    yield ": heartbeat\n\n"
                    last_sent = now
                if now - last_check >= 2.0:
                    last_check = now
                    status = survey_job.read_json(survey_job.job_paths(job_id)["status"], {}) or {}
                    # A worker that died silently gets its error written by
                    # _reconcile; the next read picks that line up.
                    status = _reconcile(job_id, status)
                    if status.get("state") in {"done", "failed"}:
                        finished_pending = True
                        await asyncio.sleep(POLL_SECONDS)
                        continue
                if finished_pending:
                    finished = True
                    continue
                await asyncio.sleep(POLL_SECONDS)
        finally:
            if handle:
                handle.close()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
