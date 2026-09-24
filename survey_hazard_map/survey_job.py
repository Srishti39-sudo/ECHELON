"""One survey, processed in its own process, reporting as it goes.

    python -m survey_hazard_map.survey_job --job <id>

This is the worker behind POST /survey/jobs. The route saves the uploaded
files and a job.json under data/uploads/<id>/, starts this module with a fixed
argv, and returns. From then on this process owns the job, and the only thing
it shares with the API is the filesystem:

    data/uploads/<id>/job.json         what to run (written by the route)
    data/surveys/<id>/job.json         state: queued|running|done|failed
    data/surveys/<id>/events.jsonl     one JSON event per line, append-only
    data/surveys/<id>/strips/*.png     browser-sized previews of each strip
    data/surveys/<id>/export.json ...  the engine's normal output

WHY A SUBPROCESS AND NOT A THREAD
    faiss and torch each ship their own libomp, and on macOS whichever one
    initialises second aborts the whole process (see backend/detector_worker.py).
    The API has faiss loaded for the assistant, so the detector cannot run
    inside it. A subprocess also means a crash in the detector costs one job,
    not the server.

WHY A FILE OF EVENTS AND NOT A QUEUE
    The file is the simplest thing that is also correct. It survives a browser
    reconnect (the stream replays from any line), a server restart (the events
    are still there), and it is an audit trail of what the run actually did,
    in order, with timestamps. The API tails it; nothing else is needed.

WHAT "LIVE" MEANS HERE
    Every event is written by the real run at the moment the real thing
    happens: a strip decoded, a tile finished, a box returned by the detector.
    Nothing is simulated, paced or replayed for effect. A provisional detection
    is labelled provisional because that is what it is -- one detector call on
    one tile, before merging -- and the final, deduplicated list replaces it.

TWO PIPELINES, CHOSEN PER JOB
    echelon    this repository's engine: sonar_ingest, tiling, marine.pt over
               every tile, build_hazard_map. Runs for plain strip images.
    teammate   models/marine/sonar_pipeline.py (marine.pt through
               sonar_detector.py, geotag.py and shadow_check.py) in a
               subprocess, then import_geotag.py turns its hazards.json into
               the same survey directory. Runs for .xtf logs and for images
               with a NavTable CSV, which is where geotag.py has navigation to
               position every contact.
    DEEPECHO_PIPELINE=auto (the default) runs "teammate" only when every file
    it needs is on disk, and "echelon" otherwise; select_pipeline() is the
    one place that decides, and the job records what ran and why. See
    docs/E2E.md.

EVENT SCHEMA (every line also carries "seq", 1-based, and "ts", UTC ISO)
    {"type": "pipeline", "pipeline": "echelon|teammate", "label", "reason", "requested"}
    {"type": "stage", "stage": "ingest|tile|detect|dedup|verify|geo|geotag|hotspots|export|report|ghosttrace",
        "message", "pipeline"}
    {"type": "log", "level": "info|warning|error", "message"}
    {"type": "strip", "strip", "width", "height", "source", "synthetic", "nadir_col",
        "m_per_px_across", "m_per_px_along", "track": [[lat, lon], ...] | null (<= 500),
        "image_url", "preview_scale", "degraded_rows": [[first, last, reason], ...] | null,
        "footprint"?: [[lat, lon] x 4], "georef_mode"?}
        Fields a plain image cannot supply (nadir_col, m_per_px_*, track,
        degraded_rows) are null for image uploads, never estimated.
        A strip may be sent twice: once when it is read, and again with a
        footprint once its navigation has been fitted. Merge by "strip".
    {"type": "progress", "tiles_done", "tiles_total", "tile"}
    {"type": "detection", "provisional": true, "id", "class", "confidence",
        "strip", "bbox_global", "latitude", "longitude", ...}
    {"type": "final_detections", "detections": [export detections]}
    {"type": "done", "summary", "downloads": [...], "survey_id", "pipeline",
        "pipeline_label", "pipeline_reason", "inputs_not_processed": [{"file", "reason"}]}
    {"type": "error", "message"}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend import config

ROOT = config.ROOT

# The engine lives at the repository root, not in a package. `python -m` from
# the root already puts it on the path; this makes it true from anywhere.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Shared with the API, which imports these rather than re-deriving them, so the
# route and the worker cannot disagree about where a job lives.
SURVEYS_DIR = Path(os.environ.get("DEEPECHO_SURVEYS_DIR", ROOT / "data" / "surveys"))
UPLOADS_DIR = Path(os.environ.get("DEEPECHO_SURVEY_UPLOADS_DIR", ROOT / "data" / "uploads"))

SPEC_NAME = "job.json"
STATUS_NAME = "job.json"
EVENTS_NAME = "events.jsonl"
STRIPS_DIRNAME = "strips"

# Raw sonar formats, if the ingest module is present. Kept as a fallback tuple
# so this module still imports on a checkout that does not have it yet.
try:  # pragma: no cover - depends on a sibling module landing
    from survey_hazard_map.sonar_ingest import SONAR_SUFFIXES  # type: ignore
except Exception:  # ImportError, or anything the module does at import
    SONAR_SUFFIXES = (".xtf", ".jsf")

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Files the done event may offer, in the order the interface lists them. Only
# the ones that actually exist are offered; see _downloads.
DOWNLOADABLE = ("export.json", "report.csv", "report.geojson", "actions.csv", "map.html",
                "ghosttrace.json", "ghosttrace.geojson")

# A browser does not need a 20 000-row strip to draw boxes over. The preview
# is scaled down and the boxes are placed in fractions of the ORIGINAL size,
# which the strip event carries, so scaling never moves a box.
PREVIEW_MAX_SIDE = int(os.environ.get("DEEPECHO_STRIP_PREVIEW_MAX", "2048"))
TRACK_MAX_POINTS = 500

log = logging.getLogger("deepecho.job")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def job_paths(job_id: str) -> dict[str, Path]:
    """Every path one job uses. The id is validated by the caller."""
    survey = SURVEYS_DIR / job_id
    return {
        "upload": UPLOADS_DIR / job_id,
        "spec": UPLOADS_DIR / job_id / SPEC_NAME,
        "survey": survey,
        "status": survey / STATUS_NAME,
        "events": survey / EVENTS_NAME,
        "strips": survey / STRIPS_DIRNAME,
    }


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    """Replace a small JSON file in one step, so a reader never sees half of it.

    The API reads job.json while this process writes it. Writing a temporary
    file and renaming it over the old one means the reader gets the old
    document or the new one, never a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def update_status(job_id: str, **changes: Any) -> dict[str, Any]:
    paths = job_paths(job_id)
    status = read_json(paths["status"], {}) or {}
    status.update(changes)
    write_json_atomic(paths["status"], status)
    return status


def count_events(path: Path) -> int:
    """Complete lines in an events file. A partial last line is not an event."""
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.endswith(b"\n"))
    except OSError:
        return 0


class EventLog:
    """Append-only JSON lines, flushed per event, numbered from 1.

    The sequence number is the line number. The stream endpoint uses it as the
    SSE id, so a browser that reconnects with Last-Event-ID resumes at exactly
    the next line with nothing skipped and nothing repeated.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = count_events(path)
        self.handle = path.open("a", encoding="utf-8")

    def emit(self, event: dict[str, Any]) -> dict[str, Any]:
        self.seq += 1
        record = {"seq": self.seq, "ts": utc_now(), **event}
        self.handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.handle.flush()
        return record

    def close(self) -> None:
        try:
            self.handle.close()
        except OSError:
            pass


class _EventLogHandler(logging.Handler):
    """The engine's own log lines, forwarded into the live log.

    The engine already narrates what it is doing ("strip x: 3600x2758 px, 7 x 6
    tile grid", "loaded detector known with 5 classes"). Forwarding those lines
    is more honest than writing a second, friendlier narration beside them,
    which could drift from what actually ran.
    """

    def __init__(self, events: EventLog) -> None:
        super().__init__(level=logging.INFO)
        self.events = events
        self._busy = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._busy:  # an error inside emit must not recurse into emit
            return
        self._busy = True
        try:
            self.events.emit({"type": "log", "level": record.levelname.lower(),
                              "message": record.getMessage()})
        except Exception:
            pass
        finally:
            self._busy = False


# --- strips ----------------------------------------------------------------

def _downsample_track(points: list[list[float]], limit: int = TRACK_MAX_POINTS) -> list[list[float]]:
    """At most `limit` points, always keeping the first and the last."""
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    return [points[round(i * step)] for i in range(limit)]


def _track_from_sidecar(nav_path: Path | None) -> list[list[float]] | None:
    """The towfish track from a {stem}.nav.json sidecar, if it has one.

    The sidecar format belongs to sonar_ingest. This reads it permissively --
    a list of fixes under "pings", "fixes", "track" or "navigation", each a
    dict with lat/lon keys or a [lat, lon] pair -- and returns None rather than
    guessing when it recognises nothing. Rows without a position are dropped,
    never filled in.
    """
    if nav_path is None or not Path(nav_path).is_file():
        return None
    payload = read_json(Path(nav_path))
    if payload is None:
        return None

    rows: Any = payload
    if isinstance(payload, dict):
        for key in ("track", "pings", "fixes", "navigation", "nav", "rows"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
    if not isinstance(rows, list):
        return None

    track: list[list[float]] = []
    for row in rows:
        lat = lon = None
        if isinstance(row, dict):
            lat = next((row[k] for k in ("lat", "latitude") if row.get(k) is not None), None)
            lon = next((row[k] for k in ("lon", "lng", "longitude") if row.get(k) is not None), None)
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            lat, lon = row[0], row[1]
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            track.append([round(lat, 7), round(lon, 7)])
    return _downsample_track(track) if track else None


def _write_preview(source: Path, destination: Path) -> tuple[int, int, float]:
    """A PNG the browser can show. Returns (width, height, scale) of the ORIGINAL."""
    from PIL import Image

    from survey_hazard_map import survey_preparation
    Image.MAX_IMAGE_PIXELS = getattr(survey_preparation, "MAX_STRIP_PIXELS", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        width, height = image.size
        if image.mode not in ("L", "RGB"):
            image = image.convert("L" if image.mode in ("1", "I", "I;16", "F", "LA") else "RGB")
        scale = min(1.0, PREVIEW_MAX_SIDE / float(max(width, height)))
        if scale < 1.0:
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        image.save(destination, "PNG", optimize=False)
    return width, height, round(scale, 6)


def _summary_value(summary: Any, *keys: str) -> Any:
    if not isinstance(summary, dict):
        return None
    for key in keys:
        if summary.get(key) is not None:
            return summary[key]
    return None


# --- the run ---------------------------------------------------------------

def _build_nav(spec: dict[str, Any], upload_dir: Path,
               image_strip_names: list[str]) -> Any:
    """The navigation argument prepare_survey takes, from what was uploaded.

    A nav CSV is passed through as a path. A single set of four corners is
    applied to every uploaded IMAGE strip, the same rule run_survey.py's
    --corners follows. Raw XTF/JSF strips are never given corners: they carry
    their own per-ping navigation, and overriding a recorded fix with a typed
    rectangle would replace evidence with an assumption.
    """
    if spec.get("nav"):
        return str(upload_dir / spec["nav"])
    corners = spec.get("corners")
    if corners and image_strip_names:
        return {"mode": "corners", "strips": {name: corners for name in image_strip_names}}
    return None


def _make_detector(conf: float | None, events: EventLog):
    """The detector for this job, or a clear failure. Never a stub.

    A survey is a record other people act on. A stub detector's placeholder
    boxes are fine on a single labelled upload and wrong here, where they would
    be merged, ranked and written into a report with a latitude on them.
    """
    from survey_hazard_map import hazard_detect
    models = [path for path in config.DETECTOR_MODELS.values() if Path(path).is_file()]
    if not models:
        raise RuntimeError(
            "no detector checkpoint is available (looked for "
            + ", ".join(str(Path(p).relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
                        for p in config.DETECTOR_MODELS.values())
            + "). A survey is not run without a real model.")

    backend = os.environ.get("DEEPECHO_DETECTOR_BACKEND", "").strip().lower()
    if backend == "onnx":
        try:
            from survey_hazard_map import hazard_detect_onnx  # type: ignore
            make = getattr(hazard_detect_onnx, "make_detector", None)
        except ImportError:
            make = None
        if make is not None:
            # make_detector picks the runtime by extension, so the exported
            # sibling of each checkpoint is named where one exists. A model
            # with no export keeps its .pt and still runs.
            exported = [path.with_suffix(".onnx") if path.with_suffix(".onnx").is_file()
                        else path for path in models]
            events.emit({"type": "log", "level": "info",
                         "message": "DEEPECHO_DETECTOR_BACKEND=onnx: "
                                    + ", ".join(p.name for p in exported)})
            return make(exported, conf=conf), exported
        events.emit({"type": "log", "level": "warning",
                     "message": "DEEPECHO_DETECTOR_BACKEND=onnx was requested but "
                                "hazard_detect_onnx.make_detector is not available; "
                                "using the Ultralytics checkpoints"})

    events.emit({"type": "log", "level": "info",
                 "message": "loading detector weights: " + ", ".join(p.name for p in models)})
    return hazard_detect.UltralyticsDetector(models if len(models) > 1 else models[0],
                                             conf=conf), models


def _downloads(out_dir: Path) -> list[str]:
    return [name for name in DOWNLOADABLE if (out_dir / name).is_file()]


# --- pipeline selection ------------------------------------------------------

PIPELINE_CHOICES = ("auto", "echelon", "teammate")
PIPELINE_ENV = "DEEPECHO_PIPELINE"
TEAMMATE_SCRIPTS = ("sonar_pipeline.py", "geotag.py", "sonar_detector.py")
TEAMMATE_TIMEOUT = float(os.environ.get("DEEPECHO_TEAMMATE_TIMEOUT", "3600"))
TEAMMATE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
# A GeoTIFF mosaic carries its own georeference, so geotag.GeoRaster positions
# every box from the raster transform with no navigation file at all.
TEAMMATE_GEOTIFF_SUFFIXES = (".tif", ".tiff")
# The columns geotag.NavTable.from_csv reads. An image runs on the team
# pipeline only with a CSV that has them; a pixel-to-lat/lon fix CSV (the
# format the DeepEcho engine takes) is a different thing and is not guessed at.
NAVTABLE_COLUMNS = ("lat", "lon", "heading_deg", "altitude_m", "slant_range_m", "samples_per_side")
RAW_JSF = (".jsf",)
LOG_LINE_MAX = 2000


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def teammate_locations() -> dict[str, Path]:
    """Where the team pipeline's files are expected, from the environment or the defaults.

    modules   DEEPECHO_TEAMMATE_MODULES, else the kit committed in
              models/marine/ (config.MARINE_KIT_DIR) when it exists, else
              <repo>/../models when that folder exists, else <repo>/models
    weights   DEEPECHO_TEAMMATE_WEIGHTS, else the first of marine.pt, best.pt
              in <modules>, else the same names under <repo>/models
    calib     DEEPECHO_TEAMMATE_CALIB, else calibration.json beside the weights
    """
    modules = _env_path("DEEPECHO_TEAMMATE_MODULES")
    if modules is None:
        kit = Path(config.MARINE_KIT_DIR)
        modules = kit if kit.is_dir() else ROOT / "survey_hazard_map" / "models"
    weights = _env_path("DEEPECHO_TEAMMATE_WEIGHTS")
    if weights is None:
        candidates = [folder / name for folder in (modules, ROOT / "survey_hazard_map" / "models")
                      for name in ("marine.pt", "best.pt")]
        weights = next((p for p in candidates if p.is_file()), modules / "marine.pt")
    calib = _env_path("DEEPECHO_TEAMMATE_CALIB") or weights.parent / "calibration.json"
    return {"modules": modules, "weights": weights, "calib": calib}


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return str(path)


def teammate_status() -> dict[str, Any]:
    """Whether the team pipeline can run, and exactly what is missing if not.

    sonar_pipeline.py puts --modules and the weights' folder on its import
    path, so geotag.py and sonar_detector.py count when they are in either.
    The optional stages are offered only when their files exist: --shadow
    needs shadow_check.py, --anomaly needs anomaly.py and a background
    embedding (bg_embed.pkl or bg_embed.npz) beside the modules.
    """
    where = teammate_locations()
    modules, weights, calib = where["modules"], where["weights"], where["calib"]
    search = [modules] + ([weights.parent] if weights.parent != modules else [])

    def find(name: str) -> Path | None:
        return next((folder / name for folder in search if (folder / name).is_file()), None)

    found: dict[str, str] = {}
    missing: list[str] = []
    script = modules / "sonar_pipeline.py"
    for name in TEAMMATE_SCRIPTS:
        # The script is run from --modules; the two imports resolve from either folder.
        if name == "sonar_pipeline.py":
            path = script if script.is_file() else None
        else:
            path = find(name)
        if path is None:
            missing.append(name)
        else:
            found[name] = _display(path)
    for path in (weights, calib):
        if path.is_file():
            found[path.name] = _display(path)
        else:
            missing.append(path.name)

    shadow = find("shadow_check.py")
    anomaly_module = find("anomaly.py")
    background = next((folder / name for folder in search
                       for name in ("bg_embed.pkl", "bg_embed.npz") if (folder / name).is_file()),
                      None)
    notes: list[str] = []
    if background is not None and anomaly_module is None:
        notes.append(f"{background.name} is present but anomaly.py is not, so --anomaly is not passed")
    anomaly = background if background is not None and anomaly_module is not None else None

    available = not missing
    if available:
        reason = (f"team pipeline complete: {', '.join(found)} found in {_display(modules)}")
    else:
        reason = (f"teammate pipeline unavailable: {', '.join(missing)} not delivered yet "
                  f"(looked in {_display(modules)}"
                  + (f" and {_display(weights.parent)}" if weights.parent != modules else "") + ")")
    return {
        "available": available,
        "reason": reason,
        "label": f"team detector ({weights.name})",
        "modules": str(modules),
        "weights": str(weights),
        "calib": str(calib),
        "found": found,
        "missing": missing,
        "shadow": shadow is not None,
        "anomaly": str(anomaly) if anomaly is not None else None,
        "notes": notes,
    }


def echelon_status() -> dict[str, Any]:
    models = [Path(p) for p in config.DETECTOR_MODELS.values()]
    present = [p for p in models if p.is_file()]
    if present:
        reason = "DeepEcho engine checkpoints on disk: " + ", ".join(p.name for p in present)
    else:
        reason = ("no DeepEcho checkpoint on disk (looked for "
                  + ", ".join(_display(p) for p in models) + ")")
    return {"available": bool(present), "reason": reason,
            "label": f"DeepEcho engine ({' + '.join(p.name for p in present) or 'no checkpoint'})",
            "models": [p.name for p in present]}


def select_pipeline(requested: str | None = None) -> dict[str, Any]:
    """Which pipeline a job will run, and why. The one place that decides.

    auto      teammate when every file it needs exists, else echelon
    echelon   always the DeepEcho engine
    teammate  the team pipeline, or an error naming what is missing; an
              explicit request is never quietly swapped for the other engine
    """
    raw = os.environ.get(PIPELINE_ENV, "auto") if requested is None else requested
    requested = (raw or "auto").strip().lower()
    team = teammate_status()
    engine = echelon_status()
    out: dict[str, Any] = {"requested": requested, "teammate": team, "echelon": engine,
                           "error": None}
    if requested not in PIPELINE_CHOICES:
        out.update(pipeline=None, label=None,
                   reason=f"{PIPELINE_ENV}={raw!r} is not one of {', '.join(PIPELINE_CHOICES)}",
                   error=f"{PIPELINE_ENV}={raw!r} is not one of {', '.join(PIPELINE_CHOICES)}")
    elif requested == "echelon":
        out.update(pipeline="echelon", label=engine["label"],
                   reason=f"{PIPELINE_ENV}=echelon: DeepEcho engine requested"
                          + ("" if team["available"] else f"; {team['reason']}"))
    elif requested == "teammate":
        if team["available"]:
            out.update(pipeline="teammate", label=team["label"],
                       reason=f"{PIPELINE_ENV}=teammate: {team['reason']}")
        else:
            out.update(pipeline="teammate", label=team["label"],
                       reason=f"{PIPELINE_ENV}=teammate requested, but {team['reason']}",
                       error=f"{PIPELINE_ENV}=teammate requested, but {team['reason']}")
    elif team["available"]:
        out.update(pipeline="teammate", label=team["label"], reason=f"auto: {team['reason']}")
    else:
        out.update(pipeline="echelon", label=engine["label"],
                   reason=f"auto: {team['reason']}; running the DeepEcho engine")
    return out


def _navtable_csv(path: Path | None) -> bool:
    """True when a CSV has the columns geotag.NavTable.from_csv needs."""
    if path is None or not path.is_file():
        return False
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = handle.readline()
    except OSError:
        return False
    columns = {c.strip().lower() for c in header.split(",")}
    return all(c in columns for c in NAVTABLE_COLUMNS)


def _georeferenced(path: Path) -> bool:
    """True when a .tif carries a CRS and transform rasterio can read.

    A plain TIFF strip is an image like any other and takes the image path;
    only a real mosaic goes to GeoRaster. Without rasterio nothing is
    georeferenced, and the file runs on the DeepEcho engine with a reason.
    """
    try:
        import rasterio
    except ImportError:
        return False
    try:
        with rasterio.open(path) as ds:
            return ds.crs is not None and ds.transform is not None and not ds.transform.is_identity
    except Exception:
        return False


def plan_teammate(raw_inputs: list[Path], image_inputs: list[Path], nav: Path | None,
                  corners: Any) -> dict[str, Any]:
    """What the team pipeline can take from this upload, and what it cannot.

    sonar_pipeline.py writes one hazards.json for one source, and import_geotag
    builds one survey from one hazards.json, so one job runs one source on this
    path: the first .xtf, else the first .png/.jpg that has a NavTable CSV.
    Every other file is listed with the reason it was not processed; nothing
    is dropped without saying so. When no upload suits the team pipeline, the
    job falls back to the DeepEcho engine and says why.
    """
    xtfs = [p for p in raw_inputs if p.suffix.lower() == ".xtf"]
    images = [p for p in image_inputs if p.suffix.lower() in TEAMMATE_IMAGE_SUFFIXES]
    geotiffs = [p for p in image_inputs if p.suffix.lower() in TEAMMATE_GEOTIFF_SUFFIXES
                and _georeferenced(p)]
    if xtfs:
        primary, kind = xtfs[0], "xtf"
    elif geotiffs:
        primary, kind = geotiffs[0], "geotiff"
    elif images and _navtable_csv(nav):
        primary, kind = images[0], "image"
    else:
        if image_inputs and not raw_inputs:
            why = ("image uploads need a navigation CSV with geotag NavTable columns ("
                   + ", ".join(NAVTABLE_COLUMNS) + ") on the team pipeline"
                   + ("; four corners are not a NavTable" if corners else "")
                   + ", and none was given, so the images run on the DeepEcho engine")
        else:
            why = ("the team pipeline reads .xtf logs, georeferenced GeoTIFF mosaics, and "
                   ".png/.jpg with a NavTable CSV; nothing in this upload is one, so it runs "
                   "on the DeepEcho engine")
        return {"primary": None, "fallback_reason": why, "skipped": []}

    skipped = []
    for path in [*raw_inputs, *image_inputs]:
        if path == primary:
            continue
        if path.suffix.lower() == ".xtf":
            reason = ("not processed: the team pipeline and import_geotag build one survey per "
                      f"hazards.json, and {primary.name} is this job's; upload it as its own survey")
        elif path.suffix.lower() in RAW_JSF:
            reason = "not processed: sonar_pipeline.py does not read .jsf; upload it as its own survey"
        else:
            reason = (f"not processed: this job runs {primary.name} on the team pipeline, which "
                      "makes one survey per source; upload the image as its own survey")
        skipped.append({"file": path.name, "reason": reason})
    return {"primary": primary, "kind": kind, "nav": nav if kind == "image" else None,
            "skipped": skipped, "fallback_reason": None}


def _synthetic_marker(path: Path) -> bool:
    """The same rule sonar_ingest applies: 'SYNTHETIC' in the XTF NoteString or a notes packet.

    Not a byte search of the file: a sonar named "Synthetic Aperture ..." is
    real data. Only notes packets are decoded (pyxtf skips the pings).
    """
    if path.suffix.lower() != ".xtf":
        return False
    try:
        import pyxtf

        header, packets = pyxtf.xtf_read(str(path), types=[pyxtf.XTFHeaderType.notes])
        texts = [header.NoteString.decode("latin-1", "replace")]
        texts += [note.NotesText.decode("latin-1", "replace")
                  for note in packets.get(pyxtf.XTFHeaderType.notes, [])]
    except Exception:
        return False
    return any("SYNTHETIC" in text.upper() for text in texts)


def _footprints(manifest_json: Path) -> dict[str, dict[str, Any]]:
    """Each strip's four corners on the ground, from its fitted navigation.

    Only for a transform that resolves across-track position. An along-track
    fit knows where the towfish was, not how wide the swath is, so drawing a
    rectangle for it would draw a width nobody measured.
    """
    from survey_hazard_map.hazard_geo import references_from_manifest
    from survey_hazard_map.survey_preparation import load_manifest

    rows, survey = load_manifest(manifest_json)
    references, _ = references_from_manifest(rows)
    sizes = {s.get("strip"): (s.get("width"), s.get("height"))
             for s in survey.get("strips", []) if isinstance(s, dict)}
    out: dict[str, dict[str, Any]] = {}
    for strip, reference in references.items():
        if not reference.detail.get("across_track_resolved", False):
            continue
        width, height = sizes.get(strip, (None, None))
        if not width or not height:
            continue
        corners = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        points = [reference.locate(x, y) for x, y in corners]
        if any(lat is None or lon is None for lat, lon in points):
            continue
        out[strip] = {"footprint": [[lat, lon] for lat, lon in points],
                      "georef_mode": reference.mode}
    return out


def _run_echelon(job_id: str, spec: dict[str, Any], events: EventLog, stage,
                 raw_inputs: list[Path], image_inputs: list[Path]) -> dict[str, Any]:
    """The DeepEcho engine: ingest, tile, the checkpoint over every tile, build_hazard_map, map."""
    paths = job_paths(job_id)
    title = spec.get("title") or job_id
    conf = (spec.get("options") or {}).get("conf")
    upload_dir = paths["upload"]
    out_dir = paths["survey"]

    # (path handed to prepare_survey, strip-event extras for it)
    strip_sources: list[tuple[Path, dict[str, Any]]] = []
    had_sidecar: list[bool] = []

    if raw_inputs:
        try:
            from survey_hazard_map import sonar_ingest  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "raw XTF/JSF ingest is not available in this build "
                f"(sonar_ingest could not be imported: {exc}). Upload strip "
                "images with a navigation CSV or corners instead.") from exc

        ingest_dir = out_dir / "ingest"
        for raw in raw_inputs:
            events.emit({"type": "log", "level": "info",
                         "message": f"decoding {raw.name}"})
            for ingested in sonar_ingest.ingest(raw, ingest_dir):
                image_path = Path(ingested.image_path)
                nav_path = Path(ingested.nav_path) if getattr(ingested, "nav_path", None) else None
                # prepare_survey looks for {stem}.nav.json beside the image.
                # Make sure that is where the sidecar is, whatever layout the
                # ingest module chose, so the recorded navigation is used.
                if nav_path is not None and nav_path.is_file():
                    expected = image_path.with_name(f"{image_path.stem}.nav.json")
                    if nav_path.resolve() != expected.resolve():
                        expected.write_bytes(nav_path.read_bytes())
                summary = getattr(ingested, "summary", None) or {}
                had_sidecar.append(nav_path is not None and nav_path.is_file())
                strip_sources.append((image_path, {
                    "source": raw.name,
                    # A log generated by tools/make_synthetic_xtf.py says so in
                    # its own summary, and the interface must say so too.
                    "synthetic": bool(summary.get("synthetic")),
                    "rows_with_position": summary.get("rows_with_position"),
                    "nadir_col": _summary_value(summary, "nadir_col", "nadir_column"),
                    "m_per_px_across": _summary_value(summary, "m_per_px_across",
                                                      "across_track_m_per_px"),
                    "m_per_px_along": _summary_value(summary, "m_per_px_along",
                                                     "along_track_m_per_px"),
                    "degraded_rows": _summary_value(summary, "degraded_rows",
                                                    "degraded_pings"),
                    "track": _track_from_sidecar(nav_path),
                    "_declared_strip": getattr(ingested, "strip", None),
                }))

    for image in image_inputs:
        strip_sources.append((image, {"source": image.name, "synthetic": False,
                                      "nadir_col": None,
                                      "m_per_px_across": None, "m_per_px_along": None,
                                      "degraded_rows": None, "track": None}))

    # Strip names exactly as prepare_survey will assign them, so a preview,
    # a manifest row and a detection all agree on what a strip is called.
    from survey_hazard_map import survey_preparation
    all_paths = [p for p, _ in strip_sources]
    try:
        names = survey_preparation._strip_names(survey_preparation._resolve_strips(all_paths))
    except Exception:
        names = {p: p.stem for p in all_paths}

    image_strip_names = []
    for path, extras in strip_sources:
        strip = names.get(path, path.stem)
        if path in image_inputs:
            image_strip_names.append(strip)
        extras.pop("_declared_strip", None)
        try:
            width, height, scale = _write_preview(path, paths["strips"] / f"{strip}.png")
        except Exception as exc:
            # Not fatal here: prepare_survey records an unreadable strip
            # with its reason and carries on with the rest.
            events.emit({"type": "log", "level": "warning",
                         "message": f"{path.name} could not be read for preview: "
                                    f"{type(exc).__name__}: {exc}"})
            continue
        events.emit({"type": "strip", "strip": strip, "width": width, "height": height,
                     "preview_scale": scale,
                     "image_url": f"/survey/{job_id}/strips/{strip}.png", **extras})

    # --- tile -----------------------------------------------------------
    nav = _build_nav(spec, upload_dir, image_strip_names)
    nav_label = ("navigation CSV" if spec.get("nav") else
                 "four corners" if isinstance(nav, dict) else
                 "per-ping navigation" if raw_inputs else "no navigation")
    stage("tile", f"cutting strips into overlapping tiles ({nav_label})")
    tiles_dir, manifest_csv = survey_preparation.prepare_survey(all_paths, out_dir, nav=nav)
    manifest_json = out_dir / "manifest.json"
    _, survey_meta = survey_preparation.load_manifest(manifest_json)
    events.emit({"type": "log", "level": "info",
                 "message": f"{survey_meta.get('tiles_written', 0)} tiles written, "
                            f"{survey_meta.get('tiles_skipped_low_content', 0)} skipped "
                            f"as featureless; coordinates: "
                            f"{survey_meta.get('coordinate_mode')}"})
    if any(had_sidecar) and survey_meta.get("coordinate_mode") != "Geo-referenced":
        # Said out loud rather than left for the operator to infer from a
        # map that stays empty: the log carried navigation and the tiler
        # did not attach it (an older survey_preparation, or a sidecar
        # whose rows have no usable position).
        events.emit({"type": "log", "level": "warning",
                     "message": "the raw log's navigation sidecar was not applied to the "
                                "tiles; results are in relative coordinates"})
    for bad in survey_meta.get("strips_unreadable", []) or []:
        events.emit({"type": "log", "level": "warning",
                     "message": f"{bad.get('source_image')} skipped: {bad.get('error')}"})

    try:
        for strip, extra in _footprints(manifest_json).items():
            events.emit({"type": "strip", "strip": strip, **extra})
    except Exception as exc:
        events.emit({"type": "log", "level": "warning",
                     "message": f"strip footprints unavailable: {exc}"})

    # --- detect, dedup, geo, hotspots, export (the engine) ----------------
    detector, models = _make_detector(conf, events)

    from survey_hazard_map.hazard_map import build_hazard_map

    def on_event(event: dict[str, Any]) -> None:
        if event.get("type") == "stage":
            event = {**event, "pipeline": "echelon"}
        events.emit(event)

    export = build_hazard_map(models if len(models) > 1 else models[0], tiles_dir, out_dir,
                              manifest_json, detector=detector, conf=conf, title=title,
                              on_event=on_event)

    # --- report ---------------------------------------------------------
    stage("report", "rendering the offline map and collecting reports")
    try:
        from survey_hazard_map.hazard_mapview import render_map

        render_map(export, out_dir, tiles_dir, manifest_json, title=title)
    except Exception as exc:
        # The map is a convenience over export.json, not the record itself.
        events.emit({"type": "log", "level": "warning",
                     "message": f"map.html was not written: {type(exc).__name__}: {exc}"})
    return export


def _run_subprocess(argv: list[str], cwd: Path, events: EventLog, timeout: float,
                    env: dict[str, str]) -> tuple[int, list[str]]:
    """Run argv with no shell, streaming each output line as a log event.

    Returns (exit code, the last lines). A run that outlives `timeout` has its
    whole process group killed and is reported as a failure, never waited on.
    """
    process = subprocess.Popen(argv, cwd=str(cwd), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               env=env, start_new_session=True)
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        assert process.stdout is not None
        for raw in iter(process.stdout.readline, b""):
            lines.put(raw.decode("utf-8", errors="replace").rstrip())
        lines.put(None)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    tail: list[str] = []
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"sonar_pipeline.py did not finish within {timeout:.0f} s "
                                       "(DEEPECHO_TEAMMATE_TIMEOUT); it was stopped")
                continue
            if line is None:
                break
            if not line.strip():
                continue
            tail = (tail + [line])[-40:]
            level = "warning" if line.lstrip().startswith(("Traceback", "WARNING", "Warning")) else "info"
            events.emit({"type": "log", "level": level,
                         "message": "sonar_pipeline: " + line[:LOG_LINE_MAX]})
        code = process.wait(timeout=max(1.0, deadline - time.monotonic()))
        return code, tail
    except BaseException:
        # Timeout, SIGTERM of this worker, or anything else: the child goes too.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        process.wait()
        raise
    finally:
        reader.join(timeout=5)


def _merge_into(source: Path, destination: Path) -> None:
    """Move a staged survey into the job's directory without touching job files."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir() and target.is_dir():
            _merge_into(item, target)
            item.rmdir()
        else:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(item), str(target))


def _run_teammate(job_id: str, spec: dict[str, Any], events: EventLog, stage,
                  plan: dict[str, Any], team: dict[str, Any]) -> dict[str, Any]:
    """The team pipeline on one source, imported as this job's survey.

    1. sonar_pipeline.py in a subprocess (fixed argv, no shell, timeout),
       writing hazards.json and <name>_waterfall.png under the upload folder.
    2. import_geotag.import_geotag into a staging folder named after the job,
       then moved into data/surveys/<id>/. Not straight into it: the importer
       replaces its output directory, and this one holds job.json and the
       events file the browser is reading.
    GhostTrace then runs from run_job over the finished survey with the real
    surveys root, so change tracking compares against the other surveys.
    """
    paths = job_paths(job_id)
    title = spec.get("title") or job_id
    out_dir = paths["survey"]
    primary: Path = plan["primary"]
    modules, weights, calib = Path(team["modules"]), Path(team["weights"]), Path(team["calib"])
    work = paths["upload"] / "teammate"
    pipeline_out = work / "pipeline" / primary.stem
    staging = work / "import" / job_id
    for folder in (pipeline_out, staging):
        if folder.exists():
            shutil.rmtree(folder)
    pipeline_out.mkdir(parents=True)

    argv = [sys.executable, str(modules / "sonar_pipeline.py"), str(primary),
            "--weights", str(weights), "--calib", str(calib), "--modules", str(modules)]
    extras = []
    if team["shadow"]:
        argv.append("--shadow")
        extras.append("shadow_check.py")
    if team["anomaly"]:
        argv += ["--anomaly", team["anomaly"]]
        extras.append(f"anomaly.py with {Path(team['anomaly']).name}")
    if plan["kind"] == "image":
        argv += ["--nav", str(plan["nav"])]
    argv += ["--out", str(pipeline_out)]
    for note in team.get("notes") or []:
        events.emit({"type": "log", "level": "warning", "message": note})

    stage("detect", f"team pipeline on {primary.name}: sonar_pipeline.py with {weights.name}"
                    + (f" + {', '.join(extras)}" if extras else " (no shadow check or anomaly "
                                                            "channel delivered)"))
    events.emit({"type": "log", "level": "info",
                 "message": "running " + " ".join(
                     [Path(argv[0]).name, *[a if not a.startswith("/") else Path(a).name
                                            for a in argv[1:]]])})
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    started = time.monotonic()
    code, tail = _run_subprocess(argv, pipeline_out, events, TEAMMATE_TIMEOUT, env)
    if code != 0:
        last = " | ".join(line for line in tail[-6:] if line.strip())
        raise RuntimeError(f"sonar_pipeline.py exited with status {code}: {last or 'no output'}")
    hazards = pipeline_out / "hazards.json"
    if not hazards.is_file():
        raise RuntimeError("sonar_pipeline.py finished but wrote no hazards.json")
    summary = read_json(pipeline_out / "summary.json", {}) or {}
    events.emit({"type": "log", "level": "info",
                 "message": f"sonar_pipeline.py finished in {time.monotonic() - started:.1f} s: "
                            f"{summary.get('detections', '?')} detection(s), "
                            f"{summary.get('unknown_anomalies', 0)} unknown anomal(ies), "
                            f"{summary.get('vetoed_by_shadow_check', 0)} vetoed by the shadow "
                            f"check, navigation {summary.get('navigation', '?')}"})

    if plan["kind"] == "xtf":
        waterfall = pipeline_out / f"{primary.stem}_waterfall.png"
        source = {"xtf": primary}
    elif plan["kind"] == "geotiff":
        # Positions came from the raster transform and are in hazards.json;
        # the importer copies them and registers no navigation of its own.
        waterfall = pipeline_out / f"{primary.stem}_waterfall.png"
        source = {}
    else:
        waterfall = primary
        source = {"nav_csv": plan["nav"]}
    if not waterfall.is_file():
        raise RuntimeError(f"sonar_pipeline.py wrote no waterfall image {waterfall.name}")

    stage("geotag", "geotag.py positions from hazards.json, built into a DeepEcho survey "
                    "(import_geotag: navigation sidecar, tiles, scoring, dedup)")
    from survey_hazard_map import import_geotag
    import_log = logging.getLogger("deepecho.import_geotag")
    handler = _EventLogHandler(events)
    import_log.addHandler(handler)
    import_log.setLevel(logging.INFO)
    try:
        export = import_geotag.import_geotag(
            hazards, staging, waterfall=waterfall, verify="auto", ghosttrace=False,
            overwrite=True, survey_id=job_id, title=title, modules=modules,
            weights=weights.name, **source)
    finally:
        import_log.removeHandler(handler)
    _merge_into(staging, out_dir)
    shutil.rmtree(work / "import", ignore_errors=True)

    verification = (export.get("provenance") or {}).get("verification") or {}
    stage("verify", f"verification path '{verification.get('path')}': "
                    f"{verification.get('choice', 'as recorded')}; "
                    f"{verification.get('suppressed', 0) or 0} suppressed")

    manifest = read_json(out_dir / "manifest.json", {}) or {}
    survey_meta = manifest.get("survey") or {}
    for entry in survey_meta.get("strips") or []:
        strip = entry.get("strip")
        image = paths["strips"] / f"{strip}.png"
        if not strip or not image.is_file():
            continue
        sidecar = read_json(out_dir / entry["sidecar"], {}) if entry.get("sidecar") else {}
        width, height = entry.get("width"), entry.get("height")
        name, scale = strip, 1.0
        if width and height and max(width, height) > PREVIEW_MAX_SIDE:
            # The importer's strip image is full size and other readers use it;
            # the browser gets a scaled copy beside it.
            name = f"{strip}-preview"
            width, height, scale = _write_preview(image, paths["strips"] / f"{name}.png")
        events.emit({"type": "strip", "strip": strip, "width": width, "height": height,
                     "preview_scale": scale, "image_url": f"/survey/{job_id}/strips/{name}.png",
                     "source": primary.name,
                     "synthetic": bool(sidecar.get("synthetic")) or _synthetic_marker(primary),
                     "nadir_col": sidecar.get("nadir_col"),
                     # Slant range: no ground-range scale exists for this strip.
                     "m_per_px_across": None,
                     "m_per_px_along": sidecar.get("m_per_px_along"),
                     "degraded_rows": sidecar.get("degraded_rows"),
                     "track": _track_from_sidecar(out_dir / entry["sidecar"])
                     if entry.get("sidecar") else None})

    detections = export.get("detections", [])
    summary_block = export.get("survey_summary", {})
    stage("export", f"{summary_block.get('total_deduplicated_detections', len(detections))} "
                    f"detection(s), {summary_block.get('total_hotspots', 0)} hotspot(s), "
                    f"{summary_block.get('coordinate_mode')}; export.json written")
    events.emit({"type": "final_detections", "detections": detections})
    for warning in (export.get("metadata") or {}).get("warnings") or []:
        events.emit({"type": "log", "level": "warning", "message": f"import: {warning}"})
    stage("report", "report.csv, report.geojson, actions.csv and map.html written by the importer")
    return export


def run_job(job_id: str) -> int:
    paths = job_paths(job_id)
    spec = read_json(paths["spec"])
    paths["survey"].mkdir(parents=True, exist_ok=True)
    events = EventLog(paths["events"])

    engine_log = logging.getLogger("deepecho.hazard")
    handler = _EventLogHandler(events)
    engine_log.addHandler(handler)
    engine_log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    def terminate(signum, _frame):  # pragma: no cover - signal path
        raise SystemExit(f"worker stopped by signal {signum}")

    signal.signal(signal.SIGTERM, terminate)

    try:
        if not isinstance(spec, dict):
            raise RuntimeError(f"job specification {paths['spec']} is missing or unreadable")

        update_status(job_id, state="running", started_at=utc_now(), pid=os.getpid())
        title = spec.get("title") or job_id
        upload_dir = paths["upload"]
        out_dir = paths["survey"]

        inputs = [upload_dir / name for name in spec.get("inputs", [])]
        missing = [p.name for p in inputs if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"uploaded file(s) missing: {', '.join(missing)}")
        raw_inputs = [p for p in inputs if p.suffix.lower() in SONAR_SUFFIXES]
        image_inputs = [p for p in inputs if p.suffix.lower() in IMAGE_SUFFIXES]
        if not raw_inputs and not image_inputs:
            raise ValueError("no sonar logs or strip images were uploaded")

        # --- which pipeline ---------------------------------------------------
        choice = select_pipeline()
        pipeline, label, reason = choice["pipeline"], choice["label"], choice["reason"]
        not_processed: list[dict[str, str]] = []
        plan: dict[str, Any] = {}
        if choice["error"]:
            update_status(job_id, pipeline=pipeline, pipeline_label=label,
                          pipeline_reason=reason, pipeline_requested=choice["requested"])
            raise RuntimeError(choice["error"])
        if pipeline == "teammate":
            nav_path = upload_dir / spec["nav"] if spec.get("nav") else None
            plan = plan_teammate(raw_inputs, image_inputs, nav_path, spec.get("corners"))
            if plan["primary"] is None:
                reason = f"{reason}; but {plan['fallback_reason']}"
                pipeline, label = "echelon", choice["echelon"]["label"]
            else:
                not_processed = plan["skipped"]
        update_status(job_id, pipeline=pipeline, pipeline_label=label, pipeline_reason=reason,
                      pipeline_requested=choice["requested"], inputs_not_processed=not_processed)

        def stage(name: str, message: str) -> None:
            events.emit({"type": "stage", "stage": name, "message": message,
                         "pipeline": pipeline})

        # --- ingest ---------------------------------------------------------
        reading = [f"{len(group)} {kind}{'' if len(group) == 1 else 's'}"
                   for group, kind in ((raw_inputs, "raw sonar log"),
                                       (image_inputs, "strip image")) if group]
        stage("ingest", f"reading {' and '.join(reading)}")
        events.emit({"type": "pipeline", "pipeline": pipeline, "label": label,
                     "reason": reason, "requested": choice["requested"]})
        events.emit({"type": "log", "level": "info", "message": f"pipeline: {label}; {reason}"})
        for item in not_processed:
            events.emit({"type": "log", "level": "warning",
                         "message": f"{item['file']} {item['reason']}"})

        if pipeline == "teammate":
            export = _run_teammate(job_id, spec, events, stage, plan, choice["teammate"])
        else:
            export = _run_echelon(job_id, spec, events, stage, raw_inputs, image_inputs)

        # --- ghosttrace -----------------------------------------------------
        # Runs on the finished export, so a GhostTrace failure can cost the
        # rescue analysis but never the survey or its reports. Skipped cheaply
        # when the survey holds nothing GhostTrace acts on.
        stage("ghosttrace", "checking detected nets and debris: activity, nearby "
                            "habitat, drift, safety and recovery priority")
        try:
            from ghosttrace.engine import run_ghosttrace

            traced = run_ghosttrace(out_dir, surveys_root=SURVEYS_DIR, on_event=events.emit)
            events.emit({"type": "ghosttrace_done", "summary": traced.get("summary", {})})
        except Exception as exc:
            events.emit({"type": "log", "level": "warning",
                         "message": f"GhostTrace did not run: {type(exc).__name__}: {exc}"})

        summary = dict(export.get("survey_summary", {}))
        detections = export.get("detections", [])
        summary["suppressed_detections"] = sum(1 for d in detections if d.get("suppressed"))
        downloads = _downloads(out_dir)
        events.emit({"type": "done", "survey_id": job_id, "title": title,
                     "summary": summary, "downloads": downloads,
                     "demo": bool(export.get("metadata", {}).get("demo")),
                     "pipeline": pipeline, "pipeline_label": label, "pipeline_reason": reason,
                     "inputs_not_processed": not_processed})
        update_status(job_id, state="done", finished_at=utc_now(), error=None,
                      downloads=downloads)
        return 0

    except BaseException as exc:  # SystemExit from SIGTERM included
        message = str(exc) or type(exc).__name__
        traceback.print_exc()
        try:
            events.emit({"type": "error", "message": message})
        finally:
            update_status(job_id, state="failed", finished_at=utc_now(), error=message)
        return 1
    finally:
        engine_log.removeHandler(handler)
        log.removeHandler(handler)
        events.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one uploaded survey job.")
    parser.add_argument("--job", required=True, help="job id under data/uploads/")
    args = parser.parse_args(argv)

    # Same rule the API applies. The worker is only ever started by the route
    # with an id it created, but a process that takes a path segment on its
    # command line checks it anyway.
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.job):
        print(f"invalid job id {args.job!r}", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    return run_job(args.job)


if __name__ == "__main__":
    sys.exit(main())
