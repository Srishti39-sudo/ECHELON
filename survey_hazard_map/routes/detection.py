"""Upload a sonar tile, detect what is in it, keep the record.

The detection half runs marine.pt through backend/detect.py, which applies
the per-class confidence floors before anything is written down. The storage half goes through app/store.py, which is
SQLite unless Supabase is configured, so a scan is recorded either way.

Persistence is still best-effort in one specific sense: if the store raises, the
detection is returned anyway with `stored: false` and the reason attached. A
database problem should cost you the history entry, not the analysis you just
waited for.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend import config
from survey_hazard_map import detect
from survey_hazard_map.detection_service import save_detections, verification_fields
from survey_hazard_map.store import get_store

router = APIRouter()
log = logging.getLogger("deepecho")


def _for_storage(record: dict) -> dict:
    """One detection in the shape the detections table expects.

    Two conversions matter. The detector reports [x, y, w, h] and the table
    stores corners, so the box is converted rather than reinterpreted. And
    `anomaly` is true whenever the object is unidentified, which includes a
    class the floors withheld: the contact is real, the name is not.

    A detection verification suppressed as a likely false positive is never
    an anomaly, whatever its label: the flag is what raises alerts and counts,
    and the evidence says there is probably nothing there. It is still stored,
    with its confidence_pct, the suppressed flag and the reasons inside
    `record`, so it can be shown as filtered rather than lost.
    """
    x, y, w, h = record.get("bbox", [0, 0, 0, 0])
    label = record.get("object_class") or ""
    withheld = bool(record.get("downgraded_from"))
    unidentified = withheld or label in config.UNKNOWN_LABELS
    suppressed = bool(record.get("suppressed"))

    from rag_assistant import chat

    return {
        "id": str(uuid.uuid4()),
        "class": label,
        "confidence": record.get("confidence"),
        "bbox": [x, y, x + w, y + h],
        "anomaly": unidentified and not suppressed,
        "severity": chat.severity_for({"label": label}, unidentified),
        # Kept whole so the assistant can be asked about this exact contact
        # later without the detector having to run again.
        "record": record,
    }


@router.post("/detect")
async def detect_route(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(default=None),
    longitude: Optional[float] = Form(default=None),
):
    """Run the detector over one tile and record the result.

    Latitude and longitude are whatever the caller supplies and nothing more.
    A tile carries no position of its own, so absent means null in the row, not
    a guess. They arrive as form fields because the body is already multipart.
    """
    if file.content_type not in config.ACCEPTED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type {file.content_type!r}. "
                   f"Accepted: {', '.join(config.ACCEPTED_UPLOAD_TYPES)}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(image_bytes) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Tile is {len(image_bytes)} bytes; the limit is {config.MAX_UPLOAD_BYTES}.")

    filename = file.filename or "tile"
    records, is_stub, models = detect.run_detector(image_bytes, filename)

    scan_id = str(uuid.uuid4())
    storage_path = f"{scan_id}{os.path.splitext(filename)[1] or '.png'}"
    stored = False
    store_error = None
    saved: list = []

    try:
        store = get_store()
        image_url = store.put_image(storage_path, image_bytes, file.content_type)
        store.create_scan({
            "id": scan_id,
            "image_url": image_url,
            "filename": filename,
            "content_type": file.content_type,
            "bytes": len(image_bytes),
            "latitude": latitude,
            "longitude": longitude,
            "status": "analysed",
            "stub": is_stub,
            "models": models or [],
        })
        saved = [{**row, **verification_fields(row)}
                 for row in save_detections(scan_id, [_for_storage(r) for r in records])]
        stored = True
    except Exception as exc:
        # Not fatal. The operator still gets the analysis; they simply will not
        # find it in history, and the response says which of the two happened.
        log.warning("scan %s was not persisted: %s", scan_id, exc)
        store_error = str(exc)

    return {
        "scan_id": scan_id,
        "image_url": storage_path if stored else None,
        "status": "analysed",
        "stored": stored,
        "store_error": store_error,
        "stub": is_stub,
        "models": models or ["stub"],
        "filename": filename,
        "bytes": len(image_bytes),
        # The full records, with provenance and any withheld class, for a client
        # that wants to reason about them. The stored rows are the flattened
        # version above.
        "detections": records,
        "saved": saved,
        # How many of `detections` verification flagged as likely false
        # positives. They are in the list, marked `suppressed`, not removed.
        "filtered": sum(1 for r in records if r.get("suppressed")),
    }
