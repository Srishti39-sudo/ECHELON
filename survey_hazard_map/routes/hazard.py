"""Where the uploaded scans are, and how serious each one is.

Reads through app/store.py so it answers without credentials.

One behaviour changed. The original set a scan's severity to "high" whenever it
contained any anomaly, which contradicts the project's own rule: an unidentified
object gets "unknown", never "high", because asserting a risk level for
something nobody has identified is itself an unsourced claim. Severity is now
the worst severity among the scan's own detections, each of which was already
looked up through the severity tables at detection time. Nothing is inferred
here that was not already decided there.

Note this is not the same map as /survey/{id}/map. That one ranks hotspots
inside one processed survey by class weight times confidence. This one ranks
uploaded scans by position. They answer similar-sounding questions from
different data and the project should eventually keep one.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from survey_hazard_map.store import get_store

router = APIRouter()

# Worst first. "unknown" sits directly below "high" rather than at the bottom,
# because an unidentified contact is handled with more caution than a named
# low-risk one, not less.
SEVERITY_RANK = {"high": 3, "unknown": 2, "medium": 1, "low": 0}


def _worst(severities: list[str]) -> str:
    if not severities:
        return "unknown"
    return max(severities, key=lambda s: SEVERITY_RANK.get(s, 2))


@router.get("/hazard/map")
def get_hazard_map() -> dict[str, Any]:
    store = get_store()
    scans = store.list_scans(limit=1000)

    by_scan: dict[str, list[dict[str, Any]]] = {}
    for detection in store.list_detections():
        by_scan.setdefault(detection.get("scan_id"), []).append(detection)

    hazards = []
    unlocated = 0

    for scan in scans:
        detections = by_scan.get(scan.get("id"), [])
        if not detections:
            continue

        latitude = scan.get("latitude")
        longitude = scan.get("longitude")
        if latitude is None or longitude is None:
            # Counted rather than silently dropped. A scan with contacts and no
            # navigation is a real result the operator should know exists; it
            # simply cannot be drawn on a map.
            unlocated += 1
            continue

        severities = [(d.get("severity") or "unknown").lower() for d in detections]

        hazards.append({
            "scan_id": scan.get("id"),
            "latitude": latitude,
            "longitude": longitude,
            "object_count": len(detections),
            "anomaly_count": sum(1 for d in detections if d.get("anomaly")),
            "severity": _worst(severities),
            "classes": sorted({d.get("object_class") or "unknown" for d in detections}),
            "status": scan.get("status"),
            "created_at": scan.get("created_at"),
        })

    hazards.sort(key=lambda h: SEVERITY_RANK.get(h["severity"], 2), reverse=True)

    return {
        "total_hazards": len(hazards),
        "unlocated_scans": unlocated,
        "hazards": hazards,
        "storage": store.kind,
    }
