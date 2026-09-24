"""Sonar tile in, detection records out.

One checkpoint is run over the tile:

    models/marine/marine.pt   yolo11s   shipwreck, aircraft, human, pipeline,
                                        fishing_gear, mine_like_object

config.DETECTOR_MODELS is still a dict, so a second model can be added without
changing anything here: boxes from every loaded model are merged by overlap,
the more confident call leads, and a disagreement is kept beside it as a
recorded second opinion rather than settled quietly.

Inference happens in a subprocess. See backend/detector_worker.py for why.

Records that leave here are the shape /chat already accepts. The one addition
is provenance: which model made the call, and what it called the object before
the class map translated it into the corpus's vocabulary.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend import config
from backend.schemas import DetectionRecord, DetectResponse

router = APIRouter()
log = logging.getLogger("deepecho")

_lock = threading.Lock()
_worker: subprocess.Popen | None = None
_status: dict = {"ready": False, "models": [], "classes": {}}


def _spawn() -> subprocess.Popen | None:
    """Start the worker and read its handshake. None if it cannot run."""
    global _status
    try:
        # posix_spawn, not fork: see backend/procs.py for the crash this avoids.
        from backend.procs import spawn_python

        process = spawn_python(
            "survey_hazard_map.detector_worker",
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except Exception:
        return None

    handshake = process.stdout.readline() if process.stdout else ""
    try:
        _status = json.loads(handshake)
    except json.JSONDecodeError:
        process.kill()
        return None

    if not _status.get("ready"):
        process.kill()
        return None
    return process


def load_models() -> dict:
    """Model name to class list. Empty means the stub is in use.

    Named to match what the rest of the application asks for; the models
    themselves live in the worker and are never imported here.
    """
    global _worker
    with _lock:
        if _worker is None or _worker.poll() is not None:
            _worker = _spawn()
        return dict(_status.get("classes", {})) if _worker else {}


def _ask(image_bytes: bytes) -> list[dict] | None:
    """Raw boxes from the worker, or None if it is unavailable."""
    global _worker
    with _lock:
        if _worker is None or _worker.poll() is not None:
            _worker = _spawn()
        if _worker is None:
            return None

        # The image goes via a file rather than the pipe: a few megabytes of
        # base64 through a line-delimited protocol is a needless copy.
        with tempfile.NamedTemporaryFile(suffix=".img", delete=True) as handle:
            handle.write(image_bytes)
            handle.flush()
            try:
                _worker.stdin.write(json.dumps({"image_path": handle.name}) + "\n")
                _worker.stdin.flush()
                reply = json.loads(_worker.stdout.readline())
            except Exception:
                _worker.kill()
                _worker = None
                return None

    if not reply.get("ok"):
        raise HTTPException(status_code=422,
                            detail=f"The detector could not read that tile: {reply.get('error')}")
    return reply["boxes"]


def _iou(a: list[float], b: list[float]) -> float:
    """Overlap of two [x, y, w, h] boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    overlap = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - overlap
    return overlap / union if union > 0 else 0.0


def _merge(boxes: list[dict], filename: str) -> list[dict]:
    """One record per physical contact, with any disagreement kept visible."""
    kept: list[dict] = []
    for box in sorted(boxes, key=lambda b: b["confidence"], reverse=True):
        twin = next((k for k in kept if _iou(k["bbox"], box["bbox"]) >= config.DETECTOR_MERGE_IOU),
                    None)
        if twin is not None:
            # A disagreement between two models is information an operator
            # should see, not a tie for the software to settle quietly.
            if twin["detector_class"] != box["cls"]:
                twin["second_opinion"] = (
                    f"the {box['model']} model called this same box "
                    f"'{box['cls']}' at {box['confidence']:.2f}")
            continue

        floor = config.CLASS_CONFIDENCE_FLOOR.get(box["cls"], 0.0)
        trusted = box["confidence"] >= floor
        record = {
            "object_class": (config.DETECTOR_CLASS_MAP.get(box["cls"], box["cls"])
                             if trusted else config.DOWNGRADE_LABEL),
            "confidence": box["confidence"],
            "bbox": box["bbox"],
            "detector_model": box["model"],
            "detector_class": box["cls"],
            "sensor": config.DETECTOR_SENSOR,
            "platform": config.DETECTOR_PLATFORM,
            "notes": (f"Detected in {filename} by the {box['model']} model as "
                      f"'{box['cls']}'. The tile carries no position, depth or range scale."),
        }
        if not trusted:
            record["downgraded_from"] = config.DOWNGRADE_NOTE.format(
                model=box["model"], cls=box["cls"],
                confidence=box["confidence"], floor=floor)
        kept.append(record)
    return kept


BASIS_VERIFIED = "detector confidence fused with image evidence from this tile"
BASIS_UNVERIFIED = "detector confidence, not verified"
BASIS_STUB = "stub, not verified"


def _json_safe(value):
    """Plain JSON types only: numpy scalars unwrapped, non-finite floats as None.

    The verification block is written straight into a JSON response and a JSON
    column, and both reject NaN and numpy types.
    """
    import math

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _unverified(records: list[dict], basis: str, reason: str) -> list[dict]:
    """Every record gets a 0-100 figure, labelled as the raw detector score."""
    for record in records:
        confidence = float(record.get("confidence") or 0.0)
        record["confidence_pct"] = round(100.0 * confidence, 1)
        record["confidence_pct_basis"] = basis
        record["suppressed"] = False
        record["verification"] = {"status": "not_checked", "reason": reason,
                                  "detector_confidence": round(confidence, 4),
                                  "reasons": [], "hard_reasons": [], "notes": [],
                                  "suppressed": False}
    return records


def _calibration():
    """models/calibration.json resolved against the repository, not the cwd."""
    from survey_hazard_map import hazard_config
    from survey_hazard_map.hazard_verify import load_calibration

    path = Path(hazard_config.CALIBRATION_PATH)
    return load_calibration(path if path.is_absolute() else config.ROOT / path)


def verify_records(records: list[dict], image_bytes: bytes, filename: str) -> list[dict]:
    """The survey pipeline's confidence scoring and noise filtering, on one tile.

    hazard_verify is numpy, scipy and OpenCV only, so it runs here beside the
    retriever rather than in the torch worker. The tile is treated as a strip
    of its own with no sidecar: the nadir is estimated from the pixels, and a
    tile cut away from the water column honestly has none, in which case the
    nadir cue says "unknown" and does not count.

    Adds confidence_pct, confidence_pct_basis, suppressed, verification and,
    when a height could be measured, dimensions. `confidence` is not touched.
    A failure here costs the evidence, never the request: every record falls
    back to its detector confidence, labelled as unverified.
    """
    if not records:
        return records
    try:
        import io

        import numpy as np
        from PIL import Image

        from survey_hazard_map.hazard_verify import strip_context, verify_survey

        with Image.open(io.BytesIO(image_bytes)) as image:
            grey = np.asarray(image.convert("L"), dtype=np.float32)
        strip = filename or "tile"
        ctx = strip_context(None, None, strip=strip, grey=grey)

        probes = []
        for record in records:
            x, y, w, h = (float(v) for v in record.get("bbox") or [0, 0, 0, 0])
            probe = {
                "class": record.get("object_class") or "unknown",
                "confidence": float(record.get("confidence") or 0.0),
                "detector_model": record.get("detector_model"),
                "strip": strip,
                "bbox_global": [x, y, x + w, y + h],
                "global_x": x + w / 2.0,
                "global_y": y + h / 2.0,
            }
            if record.get("downgraded_from") and record.get("detector_class"):
                # The detector's own call is what the image is checked against.
                probe["class_withheld"] = record["detector_class"]
            probes.append(probe)

        calibration = _calibration()
        verify_survey(probes, {strip: ctx}, calibration)
    except Exception as exc:
        log.exception("verification failed for %s; detections returned unverified", filename)
        return _unverified(records, BASIS_UNVERIFIED,
                           f"verification failed: {type(exc).__name__}: {exc}")

    for record, probe in zip(records, probes):
        block = _json_safe(probe["verification"])
        checked = block.get("status") == "checked"
        record["confidence_pct"] = probe["confidence_pct"]
        record["confidence_pct_basis"] = (
            BASIS_VERIFIED + ("" if calibration else " (detector score uncalibrated)")
            if checked else f"{BASIS_UNVERIFIED}: {block.get('reason', 'not checked')}")
        record["suppressed"] = bool(probe["suppressed"])
        record["verification"] = block
        height = (probe.get("dimensions") or {}).get("height_m")
        if height is not None:
            record["dimensions"] = _json_safe(probe["dimensions"])
    return records


def run_detector(image_bytes: bytes, filename: str) -> tuple[list[dict], bool, list[str]]:
    """(detection records, is_stub, models used).

    Real records are verified against the tile before they leave. Stub records
    are not: checking a synthetic box against a real image would dress a
    placeholder up as evidence, so they say "stub, not verified" instead.
    """
    boxes = _ask(image_bytes)
    if boxes is None:
        stub = [dict(record) for record in config.STUB_DETECTIONS]
        return (_unverified(stub, BASIS_STUB,
                            "synthetic stub detection: there is no detector to verify"),
                True, [])
    records = verify_records(_merge(boxes, filename), image_bytes, filename)
    return records, False, sorted(_status.get("models", []))


@router.post("/detect", response_model=DetectResponse)
async def post_detect(tile: UploadFile = File(...)) -> DetectResponse:
    """Accept one sonar tile and return every contact in it.

    A list, because one tile holds several contacts. The client picks one and
    attaches it to a conversation.
    """
    if tile.content_type not in config.ACCEPTED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type {tile.content_type!r}. "
                   f"Accepted: {', '.join(config.ACCEPTED_UPLOAD_TYPES)}")

    image_bytes = await tile.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(image_bytes) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Tile is {len(image_bytes)} bytes; the limit is {config.MAX_UPLOAD_BYTES}.")

    records, is_stub, models = run_detector(image_bytes, tile.filename or "tile")
    return DetectResponse(
        stub=is_stub,
        models=models or ["stub"],
        filename=tile.filename or "tile",
        bytes=len(image_bytes),
        detections=[DetectionRecord(**record) for record in records],
    )
