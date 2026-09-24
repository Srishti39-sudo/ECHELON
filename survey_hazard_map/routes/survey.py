"""Survey hazard maps, over HTTP.

The engine at the repository root does the work. This module is a thin adapter
over it, in the same spirit as the rest of this package: routes validate,
delegate, and serialise, and nothing here re-implements anything the engine
already does.

WHY IT IS THIN
    build_hazard_map() is a plain function and this process can call it. There
    is no queue, no worker and no second service, because none is needed. What
    the frontend gets is the export.json the engine already writes, served as
    it is. The React page never learns anything about tiling, deduplication or
    severity arithmetic; it reads a document.

WHY PROCESSING IS NOT THE DEFAULT PATH
    Running a detector over a survey's tiles takes as long as it takes, and a
    synchronous HTTP request is the wrong place to spend it. The surveys under
    data/surveys are generated ahead of time by run_survey.py, and the routes
    below read them. POST /survey/process exists for small surveys and for
    development; it is off unless DEEPECHO_ENABLE_SURVEY_PROCESSING is set,
    and the interface does not depend on it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from backend import config

log = logging.getLogger("deepecho")

router = APIRouter(prefix="/survey", tags=["survey"])

# Where run_survey.py and demo_survey.py leave their output. One directory per
# survey, each holding export.json, actions.csv, map.html and its tiles.
SURVEYS_DIR = Path(os.environ.get("DEEPECHO_SURVEYS_DIR", config.ROOT / "data" / "surveys"))

# Where POST /survey/process is allowed to read imagery from. A client names a
# strip by path, so without this it names any file on the server. Confined to
# the repository's own sample and upload directories, and checked after
# resolution so neither ".." nor a symlink gets out.
STRIP_ROOTS = tuple(
    Path(p).resolve() for p in os.environ.get(
        "DEEPECHO_STRIP_ROOTS",
        f"{config.ROOT / 'survey_hazard_map' / 'samples'},{config.ROOT / 'data' / 'strips'}").split(",")
    if p.strip())

# Turning this on lets the API run a detector. See the module docstring for why
# it is off by default.
ENABLE_PROCESSING = os.environ.get(
    "DEEPECHO_ENABLE_SURVEY_PROCESSING", "").lower() in {"1", "true", "yes"}

# How long a single processing request may run before it is given up on. A
# survey that needs longer than this needs a shell, not an HTTP request.
PROCESS_TIMEOUT_SECONDS = int(os.environ.get("DEEPECHO_SURVEY_TIMEOUT", "900"))


def _survey_dir(survey_id: str, require_export: bool = True) -> Path:
    """Resolve a survey id to its directory, refusing anything that escapes.

    The id arrives from the client and is used as a path segment, so it is
    checked rather than trusted: no separators, no parent references, and the
    resolved path must still sit inside SURVEYS_DIR.

    require_export=False is for the strip previews a survey job writes before
    its export exists; the live mission view shows them while the run is still
    going. The same containment check applies either way.
    """
    if not survey_id or "/" in survey_id or "\\" in survey_id or survey_id.startswith("."):
        raise HTTPException(status_code=400, detail=f"invalid survey id {survey_id!r}")

    path = (SURVEYS_DIR / survey_id).resolve()
    try:
        path.relative_to(SURVEYS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid survey id {survey_id!r}")

    if require_export and not (path / "export.json").is_file():
        raise HTTPException(status_code=404, detail=f"no survey {survey_id!r}")
    if not require_export and not path.is_dir():
        raise HTTPException(status_code=404, detail=f"no survey {survey_id!r}")
    return path


def _read_export(path: Path) -> dict[str, Any]:
    try:
        return json.loads((path / "export.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500,
                            detail=f"{path.name}/export.json could not be read: {exc}")


@router.get("")
def list_surveys() -> dict[str, Any]:
    """Every processed survey, with just enough to populate a picker.

    The summary comes out of each export rather than from a separate index, so
    a survey regenerated on disk is immediately correct here with nothing to
    rebuild. `demo` is carried through so the interface can mark synthetic data
    as synthetic; that flag is set by the engine and never inferred.
    """
    if not SURVEYS_DIR.is_dir():
        return {"surveys": []}

    surveys = []
    for directory in sorted(p for p in SURVEYS_DIR.iterdir() if p.is_dir()):
        if not (directory / "export.json").is_file():
            continue
        try:
            export = json.loads((directory / "export.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("survey %s has an unreadable export.json; skipped", directory.name)
            continue
        summary = export.get("survey_summary", {})
        metadata = export.get("metadata", {})
        surveys.append({
            "survey_id": directory.name,
            "title": metadata.get("title") or metadata.get("survey_id") or directory.name,
            "processed_at": metadata.get("processed_at"),
            "demo": bool(metadata.get("demo")),
            "detections": summary.get("total_deduplicated_detections", 0),
            "hotspots": summary.get("total_hotspots", 0),
            "coordinate_mode": summary.get("coordinate_mode"),
            "georeferenced": bool(summary.get("georeferenced")),
            "has_map": (directory / "map.html").is_file(),
            "has_ghosttrace": (directory / "ghosttrace.json").is_file(),
        })
    return {"surveys": surveys}


@router.get("/{survey_id}/export")
def get_export(survey_id: str) -> dict[str, Any]:
    """The whole export document, exactly as the engine wrote it.

    Served unmodified on purpose. The contract the engine publishes is the
    contract the interface consumes, so there is no place here for the two to
    drift apart.
    """
    return _read_export(_survey_dir(survey_id))


@router.get("/{survey_id}/actions.csv")
def get_actions(survey_id: str) -> FileResponse:
    """The prioritised worklist, for an operator who wants it in a spreadsheet."""
    path = _survey_dir(survey_id) / "actions.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="this survey has no actions.csv")
    return FileResponse(path, media_type="text/csv", filename=f"{survey_id}-actions.csv")


@router.get("/{survey_id}/map")
def get_map(survey_id: str) -> FileResponse:
    """The standalone map.html, for opening full screen or saving.

    The dashboard embeds this in a frame. The file carries its own Leaflet and
    its own imagery, so it renders with no network and nothing beside it.
    """
    path = _survey_dir(survey_id) / "map.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="this survey has no map.html")
    return FileResponse(path, media_type="text/html")


@router.get("/{survey_id}/export.json")
def download_export(survey_id: str) -> FileResponse:
    """export.json as a file download, byte for byte what the engine wrote.

    GET /{id}/export serves the same document as JSON for the interface to read;
    this one sets a filename so a browser saves it instead of rendering it.
    """
    path = _survey_dir(survey_id) / "export.json"
    return FileResponse(path, media_type="application/json",
                        filename=f"{survey_id}-export.json")


@router.get("/{survey_id}/map.html")
def download_map(survey_id: str) -> FileResponse:
    """map.html as a download, for opening offline on a machine with no server.

    GET /{id}/map serves the same file inline for the embedded frame. The file
    carries its own Leaflet and imagery, so the saved copy works on its own.
    """
    path = _survey_dir(survey_id) / "map.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="this survey has no map.html")
    return FileResponse(path, media_type="text/html", filename=f"{survey_id}-map.html")


@router.get("/{survey_id}/report.csv")
def get_report_csv(survey_id: str) -> FileResponse:
    """One row per detected hazard: position, dimensions, class, confidence.

    Written by the engine beside export.json when that stage is present. A
    survey processed before it existed simply has none, and says so with a 404
    rather than a file generated here that the engine never produced.
    """
    path = _survey_dir(survey_id) / "report.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="this survey has no report.csv")
    return FileResponse(path, media_type="text/csv", filename=f"{survey_id}-report.csv")


@router.get("/{survey_id}/report.geojson")
def get_report_geojson(survey_id: str) -> FileResponse:
    """The same hazards as GeoJSON, for GIS tools. See get_report_csv."""
    path = _survey_dir(survey_id) / "report.geojson"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="this survey has no report.geojson")
    return FileResponse(path, media_type="application/geo+json",
                        filename=f"{survey_id}-report.geojson")


@router.get("/{survey_id}/ghosttrace.json")
def download_ghosttrace(survey_id: str) -> FileResponse:
    """ghosttrace.json as a download, beside the survey's other files.

    A job's `downloads` list names every file by its name in the survey
    directory, and a client following that list asks for /survey/{id}/{name}.
    /ghosttrace/{id} serves the same document for the GhostTrace page to read.
    """
    path = _survey_dir(survey_id) / "ghosttrace.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="this survey has no ghosttrace.json")
    return FileResponse(path, media_type="application/json",
                        filename=f"{survey_id}-ghosttrace.json")


@router.get("/{survey_id}/ghosttrace.geojson")
def download_ghosttrace_geojson(survey_id: str) -> FileResponse:
    """GhostTrace targets, drift cones and recovery route as GeoJSON. See above."""
    path = _survey_dir(survey_id) / "ghosttrace.geojson"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="this survey has no ghosttrace.geojson")
    return FileResponse(path, media_type="application/geo+json",
                        filename=f"{survey_id}-ghosttrace.geojson")


# A strip preview name is a strip name: the characters a job's sanitised upload
# filenames can produce, nothing that could name a directory.
_STRIP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")


@router.get("/{survey_id}/strips/{name}.png")
def get_strip_preview(survey_id: str, name: str) -> FileResponse:
    """A browser-sized PNG of one strip, written by the survey job worker.

    Served while the job is still running, so the export is not required. The
    name is held to a plain-filename pattern and the resolved file must sit
    inside this survey's own strips directory, so neither ".." nor a symlink
    reaches anything else.
    """
    if not _STRIP_NAME.fullmatch(name) or ".." in name:
        raise HTTPException(status_code=400, detail=f"invalid strip name {name!r}")
    strips = (_survey_dir(survey_id, require_export=False) / "strips").resolve()
    path = (strips / f"{name}.png").resolve()
    if path.parent != strips or not path.is_file():
        raise HTTPException(status_code=404, detail=f"no strip preview {name!r}")
    return FileResponse(path, media_type="image/png")


# --- coverage, re-look lines and mission replay ------------------------------------
#
# hazard_coverage.py at the repository root does the geometry from the survey's
# navigation sidecars. A survey without them is answered with 200 and
# {"available": false, "reason": ...}, because "this survey has no navigation"
# is a fact about the survey, not a failed request. The downloads, which have
# nothing to contain in that case, answer 404 with the same reason.


def _coverage(survey_id: str) -> dict[str, Any]:
    from survey_hazard_map import hazard_coverage
    path = _survey_dir(survey_id)
    try:
        return hazard_coverage.coverage_for_survey(path)
    except (OSError, ValueError) as exc:
        log.warning("coverage for %s failed: %s", survey_id, exc)
        raise HTTPException(status_code=500, detail=f"coverage could not be computed: {exc}")


@router.get("/{survey_id}/replay")
def get_replay(survey_id: str) -> dict[str, Any]:
    """The survey in ping-time order: track, swath widths, detections, degraded rows."""
    from survey_hazard_map import hazard_coverage
    path = _survey_dir(survey_id)
    try:
        return hazard_coverage.replay_for_survey(path)
    except (OSError, ValueError) as exc:
        log.warning("replay for %s failed: %s", survey_id, exc)
        raise HTTPException(status_code=500, detail=f"replay could not be computed: {exc}")


@router.get("/{survey_id}/coverage")
def get_coverage(survey_id: str) -> dict[str, Any]:
    """Imaged seabed, nadir strip, gaps, metrics and re-look lines, cached as coverage.json."""
    return _coverage(survey_id)


def _relook_download(survey_id: str, suffix: str, media_type: str) -> Response:
    from survey_hazard_map import hazard_coverage
    coverage = _coverage(survey_id)
    if not coverage.get("available"):
        raise HTTPException(status_code=404, detail=coverage.get("reason") or
                            "this survey has no coverage analysis")
    body = (hazard_coverage.relook_gpx(coverage) if suffix == "gpx"
            else hazard_coverage.relook_csv(coverage))
    return Response(content=body, media_type=media_type, headers={
        "Content-Disposition": f'attachment; filename="{survey_id}-relook.{suffix}"'})


@router.get("/{survey_id}/relook.gpx")
def get_relook_gpx(survey_id: str) -> Response:
    """Re-look lines as GPX 1.1: start/end waypoints, contacts, one route per line."""
    return _relook_download(survey_id, "gpx", "application/gpx+xml")


@router.get("/{survey_id}/relook.csv")
def get_relook_csv(survey_id: str) -> Response:
    """Re-look lines as CSV, one row per line."""
    return _relook_download(survey_id, "csv", "text/csv")


if ENABLE_PROCESSING:
    from pydantic import BaseModel, Field

    class ProcessRequest(BaseModel):
        """Run the engine over a strip already on disk beside the server."""

        survey_id: str = Field(min_length=1, max_length=64)
        strips: list[str] = Field(min_length=1)
        conf: float | None = Field(default=None, ge=0.0, le=1.0)

    @router.post("/process")
    def process(request: ProcessRequest) -> dict[str, Any]:
        """Tile a survey and build its hazard map, in a subprocess.

        A subprocess, not this process, and for a hard reason rather than a
        stylistic one. faiss and torch each ship their own copy of libomp, and
        on macOS whichever initialises second aborts with OMP Error #15. This
        application has already loaded faiss for the assistant, so importing
        ultralytics here kills the server mid-request. backend/detector_worker.py
        keeps the two apart for /detect and this does the same for a survey.

        It runs run_survey.py, which is the same entry point a person would use
        from a shell. The command is fixed and every argument is validated
        above; nothing a client sends reaches a shell, and shell=True is not
        used.

        Still synchronous, and still honest about it: a survey of any size holds
        the request open for as long as inference takes. That is why this route
        is behind a flag and why the interface reads pre-generated surveys.
        """
        import subprocess
        import sys

        out_dir = SURVEYS_DIR / request.survey_id
        if (out_dir / "export.json").is_file():
            raise HTTPException(status_code=409,
                                detail=f"survey {request.survey_id!r} already exists")

        # A path from a client is a request, not an instruction. Each one is
        # resolved and then required to sit inside an allowed root, so "..",
        # an absolute path and a symlink all fail the same way.
        strips: list[Path] = []
        for entry in request.strips:
            candidate = Path(entry)
            resolved = (candidate if candidate.is_absolute()
                        else (config.ROOT / candidate)).resolve()
            if not any(resolved == root or root in resolved.parents
                       for root in STRIP_ROOTS):
                raise HTTPException(
                    status_code=400,
                    detail=f"{entry!r} is outside the allowed strip directories "
                           f"({', '.join(r.name for r in STRIP_ROOTS)})")
            strips.append(resolved)

        missing = [str(s) for s in strips if not s.is_file()]
        if missing:
            raise HTTPException(status_code=400,
                                detail=f"not found: {', '.join(missing)}")

        models = [p for p in config.DETECTOR_MODELS.values() if p.exists()]
        if not models:
            raise HTTPException(status_code=503,
                                detail="no detector checkpoint is available")

        command = [sys.executable, "-m", "survey_hazard_map.run_survey",
                   "--strips", *[str(s) for s in strips],
                   "--model", *[str(m) for m in models],
                   "--out", str(out_dir), "--quiet"]
        if request.conf is not None:
            command += ["--conf", str(request.conf)]

        try:
            completed = subprocess.run(command, cwd=str(config.ROOT), capture_output=True,
                                       text=True, timeout=PROCESS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail=f"processing exceeded {PROCESS_TIMEOUT_SECONDS}s. Run "
                       f"run_survey.py from a shell for a survey this size.")

        if completed.returncode != 0:
            # The engine's own message is the useful one; passing it through
            # beats replacing it with something vaguer.
            detail = (completed.stderr or completed.stdout or "no output").strip()
            log.warning("survey processing failed: %s", detail[-500:])
            raise HTTPException(status_code=500,
                                detail=f"processing failed: {detail[-400:]}")

        export = _read_export(out_dir)
        return {"survey_id": request.survey_id,
                "summary": export["survey_summary"]}
