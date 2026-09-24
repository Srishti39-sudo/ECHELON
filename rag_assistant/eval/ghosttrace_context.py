"""Build the assistant's GhostTrace handoff from a ghosttrace.json target.

The rescue queue builds the same object in the browser. This is the Python
mirror, used by the evaluation cases and by tests_assistant.py, so both can be
regenerated from a real ghosttrace.json instead of being typed by hand. Every
value is copied; none is computed.

    python3 eval/ghosttrace_context.py data/surveys/demo-ghosttrace-mannar/ghosttrace.json 0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _stage(block: dict | None) -> dict | None:
    """A stage that could not run is {"available": false}; that is null here."""
    if not isinstance(block, dict) or block.get("available") is False:
        return None
    return block


def _top_impact(impacts: list | None) -> dict | None:
    ranked = sorted((i for i in impacts or [] if isinstance(i, dict)),
                    key=lambda i: -(i.get("probability") or 0))
    if not ranked:
        return None
    top = ranked[0]
    return {k: top.get(k) for k in ("name", "kind", "probability", "first_arrival_hours")}


def context_from_target(doc: dict, target: dict) -> dict:
    priority = target.get("priority") or {}
    activity = _stage(target.get("activity"))
    habitat = _stage(target.get("habitat"))
    drift = _stage(target.get("drift"))
    people = _stage(target.get("people"))
    change = target.get("change") or {}
    refloat = ((target.get("drift_scenarios") or {}).get("if_refloated")
               if isinstance(target.get("drift_scenarios"), dict) else None)
    diver = (people or {}).get("diver_brief") or {}
    evidence = (activity or {}).get("evidence") or {}
    return {
        "kind": "ghosttrace_target",
        "survey_id": doc.get("survey_id"),
        "survey_title": doc.get("title"),
        "synthetic": bool(doc.get("demo") or doc.get("synthetic_inputs")),
        "detection_id": target.get("detection_id"),
        "object_class": target.get("object_class"),
        "latitude": target.get("latitude"),
        "longitude": target.get("longitude"),
        "confidence_pct": target.get("confidence_pct"),
        "priority": {
            "score": priority.get("score"),
            "tier": priority.get("tier"),
            "rank": priority.get("rank"),
            "formula": priority.get("formula"),
            "terms": {name: {k: (term or {}).get(k) for k in ("value", "weight", "contribution")}
                      for name, term in (priority.get("terms") or {}).items()},
        } if priority else None,
        "activity": {
            "level": activity.get("level"),
            "score": activity.get("score"),
            "enrichment_ratio": evidence.get("enrichment_ratio"),
            "echo_clusters_near": evidence.get("echo_clusters_near"),
            "background_clusters_per_window": evidence.get("background_clusters_per_window"),
            "limitations": activity.get("limitations"),
        } if activity else None,
        "habitat_nearest": [
            {"name": h.get("name"), "kind": h.get("kind"), "distance_m": h.get("distance_m"),
             "source": h.get("source")}
            for h in (habitat or {}).get("nearest") or []
        ][:3],
        "drift": {
            "mode": drift.get("mode") or drift.get("requested_mode"),
            "top_impact": _top_impact(drift.get("impacts")),
            "stranding_probability": drift.get("stranding_probability"),
        } if drift else None,
        "refloat_scenario": {"top_impact": _top_impact(_stage(refloat).get("impacts"))}
        if _stage(refloat) else None,
        "people": {
            "propeller_hazard_level": ((people or {}).get("propeller_hazard") or {}).get("level"),
            "diver_recommended_method": diver.get("recommended_method"),
            "seabed_depth_m": diver.get("seabed_depth_m", target.get("seabed_depth_m")),
            "current_mps_at_depth": diver.get("current_mps_at_depth"),
        } if people else None,
        "change": {"status": change.get("status"), "moved_m": change.get("moved_m")},
        "authorities": [{k: a.get(k) for k in ("name", "role", "situation")}
                        for a in (target.get("alert") or {}).get("authorities") or []],
        "caveats": list(doc.get("caveats") or []),
    }


def main() -> int:
    path = Path(sys.argv[1])
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    doc = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(context_from_target(doc, doc["targets"][index]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
