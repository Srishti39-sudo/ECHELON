"""Scans that have been run, and what was found in each.

Reads through app/store.py, so this answers from SQLite by default and from
Supabase when it is configured. It used to call require_supabase() directly and
return 503 for every request without credentials.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from survey_hazard_map.detection_service import verification_fields
from survey_hazard_map.store import get_store

router = APIRouter()


def _summarise(scan: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
    """One scan with its counts folded in, for the list view.

    The counts are computed here rather than stored, so they cannot drift away
    from the detection rows they describe.

    Detections verification suppressed as likely false positives are counted
    apart, in `total_filtered`, and left out of every other count: they are
    kept and shown, but they are not contacts an operator is asked to act on.
    Rows stored before verification existed carry no verdict and count as
    reported contacts, exactly as they did before.
    """
    filtered = [d for d in detections if verification_fields(d)["suppressed"]]
    reported = [d for d in detections if not verification_fields(d)["suppressed"]]
    anomalies = sum(1 for d in reported if d.get("anomaly"))
    severities: dict[str, int] = {}
    for d in reported:
        tier = d.get("severity") or "unknown"
        severities[tier] = severities.get(tier, 0) + 1

    return {
        **scan,
        "total_objects": len(reported),
        "total_anomalies": anomalies,
        "severity_counts": severities,
        "total_filtered": len(filtered),
        "total_detected": len(detections),
    }


@router.get("/history")
def get_history(limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, Any]:
    """Every scan, newest first, each with its own counts.

    Detections are fetched once for all scans and grouped in memory rather than
    queried per scan. A hundred scans was a hundred round trips before.
    """
    store = get_store()
    scans = store.list_scans(limit=limit)

    by_scan: dict[str, list[dict[str, Any]]] = {}
    for detection in store.list_detections():
        by_scan.setdefault(detection.get("scan_id"), []).append(detection)

    return {
        "scans": [_summarise(s, by_scan.get(s.get("id"), [])) for s in scans],
        "total": len(scans),
        "storage": store.kind,
    }


@router.get("/history/{scan_id}")
def get_scan_details(scan_id: str) -> dict[str, Any]:
    """One scan, with every detection that came out of it."""
    store = get_store()
    scan = store.get_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"No scan {scan_id!r}.")

    detections = store.list_detections(scan_id)
    return {
        **_summarise(scan, detections),
        "scan_id": scan_id,
        # Every detection, filtered ones included and marked, each with its
        # verification verdict flattened beside the stored columns.
        "detections": [{**d, **verification_fields(d)} for d in detections],
    }


@router.delete("/history/{scan_id}")
def delete_scan(scan_id: str) -> dict[str, Any]:
    """Remove a scan, its detections and its stored tile."""
    if not get_store().delete_scan(scan_id):
        raise HTTPException(status_code=404, detail=f"No scan {scan_id!r}.")
    return {"deleted": scan_id}


def _current_severity(row: dict[str, Any]) -> dict[str, Any]:
    """The severity the CURRENT policy gives this class, not the one stored.

    Severity is a lookup from the class, so it belongs to the policy in force
    when the row is read, not to the day it was written. A wreck stored as
    "medium" before the class floors were raised must read "high" beside a
    wreck stored yesterday, or the table contradicts the survey map for the
    same object. The stored value is kept in `severity_stored` for the record.
    """
    from rag_assistant import chat

    label = row.get("object_class") or ""
    current = chat.severity_for({"label": label}, bool(row.get("unidentified")))
    if current != row.get("severity"):
        row = {**row, "severity_stored": row.get("severity"), "severity": current}
    return row


@router.get("/detections")
def list_detections(
    severity: str | None = Query(default=None),
    anomaly_only: bool = Query(default=False),
    include_filtered: bool = Query(default=True),
    limit: int = Query(default=500, ge=1, le=2000),
) -> dict[str, Any]:
    """Every detection across every scan, for the detections view.

    The filters are applied here rather than in the client so a large history
    does not have to cross the wire to be narrowed down. `include_filtered`
    defaults to true so an older client still sees every row; each row says
    whether it was suppressed, and `filtered` counts them before the limit.
    """
    store = get_store()
    scans = {s.get("id"): s for s in store.list_scans(limit=1000)}
    detections = [_current_severity({**d, **verification_fields(d)}) for d in store.list_detections()]
    filtered = sum(1 for d in detections if d["suppressed"])

    if not include_filtered:
        detections = [d for d in detections if not d["suppressed"]]
    if anomaly_only:
        detections = [d for d in detections if d.get("anomaly")]
    if severity:
        wanted = {s.strip().lower() for s in severity.split(",") if s.strip()}
        detections = [d for d in detections if (d.get("severity") or "unknown").lower() in wanted]

    rows = []
    for d in detections[:limit]:
        scan = scans.get(d.get("scan_id")) or {}
        rows.append({
            **d,
            # Position belongs to the scan, not the box. Carried through so the
            # table can show it without a second request, and null when the
            # upload carried no navigation.
            "latitude": scan.get("latitude"),
            "longitude": scan.get("longitude"),
            "scan_created_at": scan.get("created_at"),
            "filename": scan.get("filename"),
        })

    return {"detections": rows, "total": len(rows), "filtered": filtered,
            "storage": store.kind}


@router.get("/scans/{scan_id}/image")
def get_scan_image(scan_id: str) -> FileResponse:
    """The tile a scan was run on, so a detection can be looked at again.

    Goes through the store rather than the filesystem, so it works the same
    whether the bytes are on disk or in a Supabase bucket.
    """
    store = get_store()
    scan = store.get_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"No scan {scan_id!r}.")

    image_url = scan.get("image_url")
    path = store.open_image(image_url) if image_url else None
    if path is None:
        raise HTTPException(
            status_code=404,
            detail="The scan exists but its image is no longer in storage.")
    return FileResponse(path)
