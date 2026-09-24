"""GhostTrace, over HTTP: the rescue decision for each ghost net, as documents.

    GET  /ghosttrace/capabilities                 run flag, which habitat layers exist
    GET  /ghosttrace/layers/{kind}?bbox=...       one bundled habitat layer, bbox-filtered
    GET  /ghosttrace/{survey_id}                  ghosttrace.json, as the engine wrote it
    GET  /ghosttrace/{survey_id}/geojson          ghosttrace.geojson
    GET  /ghosttrace/{survey_id}/alerts/{id}.txt  one drafted alert, as a text download
    POST /ghosttrace/{survey_id}/run              run_ghosttrace.py over one survey
    GET  /ghosttrace/validation                   "Model trust": drift vs real GDP drifters (summary.json)
    GET  /ghosttrace/validation/segments/{i}      one validation segment's tracks and cones
    GET  /ghosttrace/{survey_id}/recoveries       field-device recovery overlay for this survey

Same shape as routes/survey.py and for the same reasons: the engine at the
repository root (run_ghosttrace.py) does the work and writes its output beside
the survey's export.json. This module validates, reads and serialises. It does
not recompute an activity score, a drift cone or a priority, so the interface
and the archived document can never disagree about one.

WHY THE RUN ROUTE IS GATED THE WAY IT IS
    Mirrors routes/jobs.py, which mirrors /detect: the capability decides the
    default and an environment variable overrides it in either direction.

        DEEPECHO_ENABLE_GHOSTTRACE_RUN=1|true|yes   on
        DEEPECHO_ENABLE_GHOSTTRACE_RUN=0|false|no   off
        unset                                       on when run_ghosttrace.py is
                                                    on disk beside the server

    So a full local checkout can press "Run GhostTrace" with no configuration,
    and the serve container (whose Dockerfile does not copy run_ghosttrace.py)
    does not offer a button that could only fail. The route is always registered
    and answers 403 with the variable's name when off, rather than vanishing,
    so the interface can say why instead of guessing; GET /capabilities reports
    the flag so the button is not shown at all.

    The run is less exposed than POST /survey/process, which is off by default:
    the only client input is a survey id, validated exactly like survey.py's
    _survey_dir, and it must already name a processed survey. The argv is fixed,
    no shell is used, one run happens at a time (409 otherwise) and it is killed
    after DEEPECHO_GHOSTTRACE_TIMEOUT seconds. Like the rest of this API nothing
    authenticates; it binds to 127.0.0.1 by default and is an operator console.

HABITAT LAYERS
    Bundled GeoJSON under data/ghosttrace/ (DEEPECHO_GHOSTTRACE_DATA_DIR). The
    kind is whitelisted. The file for a kind is found at request time, first
    from a manifest if one exists (manifest.json, layers.json, sources.json,
    SOURCES.json, index.json), then by file name convention (reef*.geojson,
    protected_area*.geojson, mpa*.geojson, ...). Nothing is hard-coded to a file
    name that another stage might choose differently. A bbox filter keeps a
    national layer from crossing the wire for one survey; it compares feature
    bounding boxes, in pure Python, which is cheap and deliberately conservative
    (a feature whose bounds touch the box is kept even if its shape does not).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from backend import config

log = logging.getLogger("deepecho")

router = APIRouter(prefix="/ghosttrace", tags=["ghosttrace"])

SURVEYS_DIR = Path(os.environ.get("DEEPECHO_SURVEYS_DIR", config.ROOT / "data" / "surveys"))
DATA_DIR = Path(os.environ.get("DEEPECHO_GHOSTTRACE_DATA_DIR", config.ROOT / "data" / "ghosttrace"))
SCRIPT = config.ROOT / "ghosttrace" / "run_ghosttrace.py"
VALIDATION_DIR = Path(os.environ.get("DEEPECHO_GHOSTTRACE_VALIDATION_DIR", DATA_DIR / "validation"))

GHOSTTRACE_FILE = "ghosttrace.json"
GEOJSON_FILE = "ghosttrace.geojson"

RUN_TIMEOUT_SECONDS = int(os.environ.get("DEEPECHO_GHOSTTRACE_TIMEOUT", "600"))


def _run_enabled() -> bool:
    """Read at request time so a test, or an operator, can flip it without a restart."""
    flag = os.environ.get("DEEPECHO_ENABLE_GHOSTTRACE_RUN", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    return SCRIPT.is_file()


# One run at a time. The route is a plain `def`, so FastAPI runs it in a worker
# thread and a threading lock is the right primitive.
_RUN_LOCK = threading.Lock()

# The whitelist, and the file-name spellings each kind is recognised by. The
# first entry is the canonical kind the interface asks for.
LAYER_ALIASES: dict[str, tuple[str, ...]] = {
    "reef": ("reef", "reefs", "coral_reef", "coral_reefs", "coral"),
    "protected_area": ("protected_area", "protected_areas", "mpa", "mpas",
                       "marine_protected_area", "marine_protected_areas", "wdpa"),
    "turtle_nesting": ("turtle_nesting", "turtle_nesting_beaches", "turtle_nesting_sites",
                       "turtles", "turtle", "nesting_beaches"),
    "dugong": ("dugong", "dugongs", "dugong_habitat", "seagrass_dugong"),
    "harbour": ("harbour", "harbours", "harbor", "harbors", "port", "ports",
                "fishing_harbour", "fishing_harbours", "landing_centres"),
}
LAYER_KINDS = tuple(LAYER_ALIASES)
MANIFEST_NAMES = ("manifest.json", "layers.json", "sources.json", "SOURCES.json", "index.json")
LAYER_SUFFIXES = (".geojson", ".json")

# Survey ids: identical rules to survey.py. Detection ids are a filename-ish
# token and nothing that could name a directory.
_DETECTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")


# --- survey documents ---------------------------------------------------------

def _survey_dir(survey_id: str) -> Path:
    """Resolve a survey id to its directory, refusing anything that escapes.

    The same checks as routes/survey.py _survey_dir: no separators, no parent
    references, and the resolved path must still sit inside SURVEYS_DIR. Kept as
    a copy rather than an import so this router mounts on a bare app in tests
    and cannot be broken by an unrelated edit to the survey router.
    """
    if not survey_id or "/" in survey_id or "\\" in survey_id or survey_id.startswith("."):
        raise HTTPException(status_code=400, detail=f"invalid survey id {survey_id!r}")

    path = (SURVEYS_DIR / survey_id).resolve()
    try:
        path.relative_to(SURVEYS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid survey id {survey_id!r}")

    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"no survey {survey_id!r}")
    return path


def _not_generated(survey_id: str) -> HTTPException:
    how = (f"Press Run GhostTrace, or run `python run_ghosttrace.py --survey "
           f"data/surveys/{survey_id}` from the repository root."
           if _run_enabled() else
           f"Run `python run_ghosttrace.py --survey data/surveys/{survey_id}` from "
           f"the repository root (the run route is off: set "
           f"DEEPECHO_ENABLE_GHOSTTRACE_RUN=1 to enable it).")
    return HTTPException(
        status_code=404,
        detail=f"GhostTrace has not been generated for survey {survey_id!r} yet. {how}")


def _read_ghosttrace(path: Path) -> dict[str, Any]:
    target = path / GHOSTTRACE_FILE
    if not target.is_file():
        raise _not_generated(path.name)
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500,
                            detail=f"{path.name}/{GHOSTTRACE_FILE} could not be read: {exc}")


# --- habitat layers -----------------------------------------------------------

def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _kind_for(name: str) -> str | None:
    """Which whitelisted kind a free-text label or file stem refers to, if any.

    A stem matches an alias exactly, or starts with one followed by a separator
    ("reefs_gulf_of_mannar"), or ends with one ("india_mpa"). Longest alias
    first so "protected_areas" is not read as a shorter spelling of something.
    """
    stem = _normalise(name)
    pairs = sorted(((alias, kind) for kind, aliases in LAYER_ALIASES.items()
                    for alias in aliases), key=lambda p: -len(p[0]))
    for alias, kind in pairs:
        if stem == alias or stem.startswith(alias + "_") or stem.endswith("_" + alias):
            return kind
    return None


def _inside_data_dir(candidate: Path) -> Path | None:
    try:
        resolved = candidate.resolve()
        resolved.relative_to(DATA_DIR.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _manifest_entries(document: Any) -> Iterable[tuple[str, str]]:
    """(label, file) pairs out of whatever shape a manifest happens to have.

    Accepts {"layers": [...]} or a bare list of objects carrying a kind-ish key
    and a file-ish key, or a mapping of kind -> file / kind -> {file: ...}.
    Anything unrecognised is skipped; the file-name convention still applies.
    """
    if isinstance(document, dict):
        for key in ("layers", "files", "sources", "datasets"):
            if isinstance(document.get(key), (list, dict)):
                yield from _manifest_entries(document[key])
        for label, value in document.items():
            if isinstance(value, str) and value.endswith(LAYER_SUFFIXES):
                yield label, value
            elif isinstance(value, dict):
                file = next((value[k] for k in ("file", "path", "geojson", "filename")
                             if isinstance(value.get(k), str)), None)
                if file:
                    yield str(value.get("kind") or value.get("layer") or label), file
    elif isinstance(document, list):
        for item in document:
            if not isinstance(item, dict):
                continue
            file = next((item[k] for k in ("file", "path", "geojson", "filename")
                         if isinstance(item.get(k), str)), None)
            label = next((item[k] for k in ("kind", "layer", "id", "name")
                          if isinstance(item.get(k), str)), None)
            if file and label:
                yield label, file


# The fetcher's manifest (tools/fetch_ghosttrace_data.py) names files by a
# kind prefix and records whether each source may be redistributed.
MANIFEST_KIND_PREFIX = {
    "reef": "reef_",
    "protected_area": "protected_",
    "turtle_nesting": "turtle_nesting_",
    "dugong": "dugong_",
    "harbour": "harbour_",
}

# UNEP-WCMC's licence forbids redistribution and allows publication only in
# non-downloadable form. Its layers inform the analysis inside the engine; this
# route does not serve their geometry unless the operator of a private, local
# deployment explicitly says the use is covered.
SERVE_RESTRICTED_LAYERS = os.environ.get(
    "DEEPECHO_SERVE_RESTRICTED_LAYERS", "").lower() in {"1", "true", "yes"}


def _layer_files(kind: str) -> tuple[list[Path], list[str], list[dict[str, Any]]]:
    """(files to serve, sources withheld by licence, attribution) for one kind.

    Every region and every source for the kind is served together, so a map
    over Odisha is not given the Gulf of Mannar file only. Falls back to the
    single-file naming convention when the data directory has no fetcher
    manifest.
    """
    manifest_path = DATA_DIR / "manifest.json"
    prefix = MANIFEST_KIND_PREFIX.get(kind)
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = document.get("sources") if isinstance(document, dict) else None
    except (OSError, ValueError):
        sources = None

    if not isinstance(sources, list) or prefix is None:
        single = _layer_file(kind)
        return ([single] if single else []), [], []

    served: list[Path] = []
    withheld: list[str] = []
    attribution: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        files = [f for f in source.get("files") or []
                 if isinstance(f, str) and Path(f).name.startswith(prefix)
                 and f.endswith(LAYER_SUFFIXES)]
        if not files:
            continue
        if source.get("redistributable") is False and not SERVE_RESTRICTED_LAYERS:
            withheld.append(str(source.get("name") or source.get("id")))
            continue
        for file in files:
            found = _inside_data_dir(DATA_DIR / file)
            if found:
                served.append(found)
        attribution.append({"name": source.get("name"), "licence": source.get("licence"),
                            "url": source.get("url")})
    return served, withheld, attribution


def _layer_file(kind: str) -> Path | None:
    if not DATA_DIR.is_dir():
        return None

    for manifest_name in MANIFEST_NAMES:
        manifest = DATA_DIR / manifest_name
        if not manifest.is_file():
            continue
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("ghosttrace manifest %s is unreadable; ignored", manifest)
            continue
        for label, file in _manifest_entries(document):
            if _kind_for(label) == kind or _kind_for(Path(file).stem) == kind:
                found = _inside_data_dir(DATA_DIR / file)
                if found:
                    return found

    # Convention: a GeoJSON file whose stem names the kind, at the top level or
    # one directory down (data/ghosttrace/layers/reefs.geojson).
    candidates = sorted(
        [*DATA_DIR.glob("*"), *DATA_DIR.glob("*/*")],
        key=lambda p: (len(p.relative_to(DATA_DIR).parts), p.name))
    for candidate in candidates:
        if (candidate.suffix.lower() in LAYER_SUFFIXES and candidate.name not in MANIFEST_NAMES
                and _kind_for(candidate.stem) == kind):
            found = _inside_data_dir(candidate)
            if found:
                return found
    return None


# Parsed layers keyed by path, invalidated by mtime and size, so a regenerated
# layer is served fresh with nothing to restart.
_LAYER_CACHE: dict[Path, tuple[tuple[float, int], dict[str, Any], list]] = {}
_LAYER_CACHE_LOCK = threading.Lock()


def _coordinates_bounds(coordinates: Any, bounds: list[float]) -> None:
    if (isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2
            and all(isinstance(c, (int, float)) for c in coordinates[:2])):
        lon, lat = float(coordinates[0]), float(coordinates[1])
        bounds[0] = min(bounds[0], lon)
        bounds[1] = min(bounds[1], lat)
        bounds[2] = max(bounds[2], lon)
        bounds[3] = max(bounds[3], lat)
        return
    if isinstance(coordinates, (list, tuple)):
        for child in coordinates:
            _coordinates_bounds(child, bounds)


def _geometry_bounds(geometry: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(geometry, dict):
        return None
    bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
    if geometry.get("type") == "GeometryCollection":
        for child in geometry.get("geometries") or []:
            inner = _geometry_bounds(child)
            if inner:
                _coordinates_bounds([[inner[0], inner[1]], [inner[2], inner[3]]], bounds)
    else:
        _coordinates_bounds(geometry.get("coordinates"), bounds)
    return None if bounds[0] == float("inf") else tuple(bounds)  # type: ignore[return-value]


def _load_layer(path: Path) -> tuple[dict[str, Any], list]:
    stat = path.stat()
    signature = (stat.st_mtime, stat.st_size)
    with _LAYER_CACHE_LOCK:
        cached = _LAYER_CACHE.get(path)
        if cached and cached[0] == signature:
            return cached[1], cached[2]
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"layer {path.name} could not be read: {exc}")

    if isinstance(document, dict) and document.get("type") == "FeatureCollection":
        features = document.get("features") or []
    elif isinstance(document, dict) and document.get("type") == "Feature":
        features = [document]
    else:
        raise HTTPException(status_code=500,
                            detail=f"layer {path.name} is not a GeoJSON FeatureCollection")

    indexed = [(feature, _geometry_bounds(feature.get("geometry")))
               for feature in features if isinstance(feature, dict)]
    header = {k: v for k, v in document.items() if k not in {"features", "type"}} \
        if document.get("type") == "FeatureCollection" else {}
    with _LAYER_CACHE_LOCK:
        _LAYER_CACHE[path] = (signature, header, indexed)
    return header, indexed


def _parse_bbox(bbox: str | None) -> tuple[float, float, float, float] | None:
    if bbox is None or not bbox.strip():
        return None
    try:
        parts = [float(p) for p in bbox.split(",")]
    except ValueError:
        parts = []
    if len(parts) != 4 or not all(abs(p) != float("inf") and p == p for p in parts):
        raise HTTPException(status_code=400,
                            detail="bbox must be minLon,minLat,maxLon,maxLat as four numbers")
    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon > max_lon or min_lat > max_lat:
        raise HTTPException(status_code=400,
                            detail="bbox must be minLon,minLat,maxLon,maxLat with min <= max")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180
            and -90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise HTTPException(status_code=400, detail="bbox is outside longitude/latitude range")
    return min_lon, min_lat, max_lon, max_lat


def _intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


# --- routes -------------------------------------------------------------------
# Literal paths are declared before /{survey_id} so "capabilities" and "layers"
# are never read as survey ids.

@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    """What the interface may offer: the run button, and which layers exist."""
    return {
        "run_enabled": _run_enabled(),
        "run_flag": "DEEPECHO_ENABLE_GHOSTTRACE_RUN",
        "layers": [kind for kind in LAYER_KINDS if _layer_files(kind)[0]],
        "layer_kinds": list(LAYER_KINDS),
    }


@router.get("/layers/{kind}")
def get_layer(kind: str,
              bbox: str | None = Query(default=None,
                                       description="minLon,minLat,maxLon,maxLat")) -> Response:
    """One bundled habitat layer as GeoJSON, optionally narrowed to a bbox."""
    if kind == "geojson":
        # A survey literally named "layers" still has its geojson reachable.
        return get_geojson("layers")
    if kind not in LAYER_ALIASES:
        raise HTTPException(status_code=404,
                            detail=f"unknown layer {kind!r}; expected one of {', '.join(LAYER_KINDS)}")

    box = _parse_bbox(bbox)
    paths, withheld, attribution = _layer_files(kind)
    if not paths:
        raise HTTPException(
            status_code=404,
            detail=(f"no {kind} layer is bundled in {DATA_DIR.name}/ on this server that may be "
                    f"served" + (f"; withheld by licence: {', '.join(withheld)}" if withheld else "")))

    kept: list = []
    total = 0
    header: dict[str, Any] = {}
    for path in paths:
        layer_header, indexed = _load_layer(path)
        header = header or layer_header
        total += len(indexed)
        kept += [feature for feature, bounds in indexed
                 if box is None or (bounds is not None and _intersects(bounds, box))]

    body = {
        "type": "FeatureCollection",
        **header,
        "kind": kind,
        "files": [p.name for p in paths],
        "attribution": attribution,
        # Layers the engine used for its analysis but whose licence forbids
        # handing the geometry out. Named so the map can say what it is not
        # drawing, rather than looking as if the data did not exist.
        "withheld_by_licence": withheld,
        "total_features": total,
        "bbox_filter": list(box) if box else None,
        "features": kept,
    }
    return Response(content=json.dumps(body, separators=(",", ":")),
                    media_type="application/geo+json")


@router.get("/validation")
def get_validation() -> dict[str, Any]:
    """The drift validation summary written by tools/validate_drift_gdp.py. Never computed here."""
    from ghosttrace import validation

    summary = validation.load_summary(VALIDATION_DIR)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="Drift validation has not been run on this server. Run "
                   "`.venv/bin/python tools/validate_drift_gdp.py` from the repository root.")
    return summary


@router.get("/validation/segments/{index}")
def get_validation_segment(index: int) -> dict[str, Any]:
    from ghosttrace import validation

    if index < 0:
        raise HTTPException(status_code=400, detail="segment index must be >= 0")
    segment = validation.load_segment(index, VALIDATION_DIR)
    if segment is None:
        raise HTTPException(status_code=404, detail=f"no validation segment {index}")
    return segment


@router.get("/{survey_id}")
def get_ghosttrace(survey_id: str) -> dict[str, Any]:
    """ghosttrace.json exactly as the engine wrote it."""
    return _read_ghosttrace(_survey_dir(survey_id))


@router.get("/{survey_id}/geojson")
def get_geojson(survey_id: str) -> FileResponse:
    path = _survey_dir(survey_id) / GEOJSON_FILE
    if not path.is_file():
        if not (path.parent / GHOSTTRACE_FILE).is_file():
            raise _not_generated(survey_id)
        raise HTTPException(status_code=404,
                            detail=f"survey {survey_id!r} has no {GEOJSON_FILE}")
    return FileResponse(path, media_type="application/geo+json",
                        filename=f"{survey_id}-ghosttrace.geojson")


@router.get("/{survey_id}/alerts/{detection_id}.txt")
def get_alert_text(survey_id: str, detection_id: str) -> Response:
    """One drafted alert as plain text, with the draft status stated in the file.

    The text is the engine's draft_text. A header says it is a draft and who
    produced it, because a saved file loses the page that said so.
    """
    if not _DETECTION_ID.fullmatch(detection_id) or ".." in detection_id:
        raise HTTPException(status_code=400, detail=f"invalid detection id {detection_id!r}")

    document = _read_ghosttrace(_survey_dir(survey_id))
    target = next((t for t in document.get("targets") or []
                   if str(t.get("detection_id")) == detection_id), None)
    if target is None:
        raise HTTPException(status_code=404,
                            detail=f"no target {detection_id!r} in survey {survey_id!r}")
    alert = target.get("alert") or {}
    if not alert.get("draft_text"):
        raise HTTPException(status_code=404,
                            detail=f"target {detection_id!r} has no drafted alert")

    lines = [
        "DRAFT - verify every detail before sending. Not an official notice.",
        f"Generated by: {alert.get('generated_by') or 'not stated'}",
        f"Survey: {survey_id}   Detection: {detection_id}",
    ]
    authorities = alert.get("authorities") or []
    if authorities:
        lines.append("To: " + "; ".join(
            f"{a.get('name')}" + (f" ({a.get('role')})" if a.get("role") else "")
            for a in authorities if isinstance(a, dict)))
    if alert.get("subject"):
        lines.append(f"Subject: {alert['subject']}")
    lines += ["", str(alert["draft_text"]).rstrip(), ""]
    citations = alert.get("citations") or []
    if citations:
        lines.append("Sources:")
        lines += [f"  - {c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)}"
                  for c in citations]
    if alert.get("basis"):
        lines.append(f"Basis: {alert['basis']}")

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", f"{survey_id}-{detection_id}-alert.txt")
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{safe_name}"'})


@router.get("/{survey_id}/recoveries")
def get_recoveries(survey_id: str) -> dict[str, Any]:
    """Field-device reports for this survey's targets, and its removed contacts re-read against them.

    `records`: net_finder arrived / recovered reports naming a target of this
    survey. `removed_since_previous`: this document's removed list passed
    through ghosttrace.changes.apply_recoveries, the same function the engine
    uses, so a recovery reported after the document was generated shows here
    exactly as a re-run would write it.
    """
    from ghosttrace import changes, telemetry

    path = _survey_dir(survey_id)
    overlay = telemetry.load_recoveries()
    records = [r for r in overlay.values() if r.get("survey_id") == survey_id]
    removed: list[dict[str, Any]] = []
    target = path / GHOSTTRACE_FILE
    if target.is_file():
        try:
            document = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            document = {}
        removed = [dict(r) for r in document.get("removed_since_previous") or [] if isinstance(r, dict)]
        when, _basis = changes.survey_time(path)
        changes.apply_recoveries(removed, overlay, changes._parse_time(when))
    return {"survey_id": survey_id, "records": records, "removed_since_previous": removed,
            "confirmed_recovered": sum(1 for r in removed if r.get("recovered"))}


@router.post("/{survey_id}/run")
def run_ghosttrace(survey_id: str) -> dict[str, Any]:
    """Run run_ghosttrace.py over one processed survey, then return its summary."""
    if not _run_enabled():
        raise HTTPException(
            status_code=403,
            detail="Running GhostTrace from the API is off on this server. Set "
                   "DEEPECHO_ENABLE_GHOSTTRACE_RUN=1, or run run_ghosttrace.py from a shell.")

    path = _survey_dir(survey_id)
    if not (path / "export.json").is_file():
        raise HTTPException(status_code=404,
                            detail=f"survey {survey_id!r} has not been processed (no export.json)")
    if not SCRIPT.is_file():
        raise HTTPException(status_code=503,
                            detail="run_ghosttrace.py is not present on this server")

    if not _RUN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409,
                            detail="a GhostTrace run is already in progress; try again when it finishes")
    try:
        # posix_spawn, not fork: see backend/procs.py for the crash this avoids.
        from backend.procs import spawn_python

        process = spawn_python(SCRIPT, ["--survey", str(path), "--quiet"], module=False,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = process.communicate(timeout=RUN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise HTTPException(status_code=504,
                                detail=f"GhostTrace exceeded {RUN_TIMEOUT_SECONDS}s. Run "
                                       f"run_ghosttrace.py from a shell for this survey.")
        completed = subprocess.CompletedProcess(process.args, process.returncode, out, err)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no output").strip()
            log.warning("ghosttrace run failed: %s", detail[-500:])
            raise HTTPException(status_code=500, detail=f"GhostTrace failed: {detail[-400:]}")
    finally:
        _RUN_LOCK.release()

    document = _read_ghosttrace(path)
    return {"survey_id": survey_id,
            "generated_at": document.get("generated_at"),
            "summary": document.get("summary") or {}}
