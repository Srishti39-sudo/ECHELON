"""Writing detections down.

Thin on purpose. The row shaping lives in the store so both backends agree on
it; this is the seam the upload route calls so the route never names a table.
"""

from __future__ import annotations

from typing import Any

from backend import config
from survey_hazard_map.store import get_store


def save_detections(scan_id: str, detections: list[dict[str, Any]]) -> list[dict]:
    """Insert every detection for one scan and return the stored rows.

    One statement rather than a round trip per box. The original looped and
    inserted individually, which meant a failure halfway left a scan with some
    of its detections and no way to tell that was what had happened.
    """
    return get_store().insert_detections(scan_id, detections)


# Rows stored before single-tile verification existed carry no verdict. They
# are reported as unverified, never as passed or filtered.
LEGACY_BASIS = "detector confidence, not verified (stored before verification)"


def verification_fields(row: dict[str, Any]) -> dict[str, Any]:
    """The verification verdict of one stored detection, flattened for a client.

    Read from the full detector record kept in the row, because that is where
    both the SQLite column and the upload route put it. Old rows, and Supabase
    rows (whose table has no record column), lack it and come back unverified:
    confidence_pct from the stored detector confidence, not suppressed, no
    reasons.
    """
    record = row.get("record") if isinstance(row.get("record"), dict) else {}
    verification = record.get("verification") if isinstance(record.get("verification"), dict) else {}
    pct = record.get("confidence_pct")
    basis = record.get("confidence_pct_basis")
    if pct is None:
        confidence = row.get("confidence")
        pct = round(100.0 * float(confidence), 1) if isinstance(confidence, (int, float)) else None
        basis = LEGACY_BASIS
    return {
        "confidence_pct": pct,
        "confidence_pct_basis": basis,
        "suppressed": bool(record.get("suppressed")),
        "verified": verification.get("status") == "checked",
        "verification_reasons": list(verification.get("reasons") or []),
        # Unidentified by class, independent of suppression. The stored
        # `anomaly` flag is false for a suppressed row, so a client counting
        # what verification kept from being raised needs this instead.
        "unidentified": bool(row.get("anomaly") or record.get("downgraded_from")
                             or (row.get("object_class") or "").strip().lower()
                             in config.UNKNOWN_LABELS),
    }
