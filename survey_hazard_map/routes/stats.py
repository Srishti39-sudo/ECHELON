"""Counts across everything that has been scanned.

Reads through app/store.py so it answers without credentials. The severity
buckets now include `unknown`, which the original silently dropped: an
unidentified object is the case this system treats most carefully, and a
dashboard that counts every tier except that one is the wrong dashboard.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from survey_hazard_map.detection_service import verification_fields
from survey_hazard_map.store import get_store

router = APIRouter()

# Fixed order, and every tier present even at zero, so the interface can render
# four bars without deciding what to do about a missing key.
SEVERITY_TIERS = ("high", "medium", "low", "unknown")


@router.get("/stats")
def get_stats() -> dict[str, Any]:
    store = get_store()
    scans = store.list_scans(limit=1000)
    everything = store.list_detections()
    # Likely false positives, as verification flagged them, are counted apart
    # and left out of every other figure. Rows without a verdict count as before.
    detections = [d for d in everything if not verification_fields(d)["suppressed"]]
    filtered = len(everything) - len(detections)

    severity_counts = {tier: 0 for tier in SEVERITY_TIERS}
    class_counts: dict[str, int] = {}

    for detection in detections:
        tier = (detection.get("severity") or "unknown").lower()
        if tier not in severity_counts:
            tier = "unknown"
        severity_counts[tier] += 1

        label = detection.get("object_class") or "unknown"
        class_counts[label] = class_counts.get(label, 0) + 1

    anomalies = sum(1 for d in detections if d.get("anomaly"))
    located = sum(1 for s in scans
                  if s.get("latitude") is not None and s.get("longitude") is not None)

    return {
        "total_scans": len(scans),
        "total_detections": len(detections),
        "total_anomalies": anomalies,
        "filtered_detections": filtered,
        "severity_counts": severity_counts,
        # Most-seen class first, so the interface does not have to sort it.
        "class_counts": dict(sorted(class_counts.items(),
                                    key=lambda kv: kv[1], reverse=True)),
        "georeferenced_scans": located,
        "storage": store.kind,
    }
