"""The JSON contract, the action list, and the survey summary.

export.json is the boundary between this engine and everything that consumes
it -- the map, the dashboard, the backend, a reviewer six months from now with
no access to this code. Its shape is fixed:

    {
      "metadata":       what produced this file and when
      "survey_summary": the survey in one object
      "detections":     one record per physical object, deduplicated
      "hotspots":       ranked, H001 first
      "configuration":  every threshold and weight the run actually used
      "provenance":     what the numbers rest on
    }

The base contract of {"detections": [], "hotspots": []} is a subset of this, so
a consumer written against the minimal shape keeps working.

Two rules govern what goes in.

EVERY FIELD IS LOAD-BEARING. No key is present because it might be useful one
day. If a value could not be determined it is null and something in provenance
says why, rather than being a plausible default that reads like measurement.

NO SECRETS. The export carries the model's file NAME, never a filesystem path
outside the project, because these files get attached to emails.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map.hazard_severity import policy, severity_tier


def _count_tiers(tiers) -> dict[str, int]:
    """How many records carry each tier WORD. Every tier present, even at zero.

    Counted from the word, not recomputed from the score, because a class
    floor can put a detection's word above its score's tier and the summary
    must agree with the records it summarises.
    """
    counts = {tier: 0 for tier, _ in cfg.SEVERITY_TIERS}
    for tier in tiers:
        counts[tier if tier in counts else severity_tier(0.0)] += 1
    return counts

log = logging.getLogger("deepecho.hazard")


def safe_model_reference(model_path: Any) -> dict[str, Any]:
    """Name the model or models without leaking somebody's home directory."""
    paths = ([Path(model_path)] if isinstance(model_path, (str, Path))
             else [Path(p) for p in model_path])

    def one(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except (ValueError, OSError):
            # Outside the project tree. The name identifies the checkpoint; the
            # absolute path only identifies the machine it ran on.
            return path.name

    return {
        "model_name": " + ".join(p.name for p in paths),
        "model_path": " + ".join(one(p) for p in paths),
        "model_count": len(paths),
    }


def export_detection(detection: dict[str, Any]) -> dict[str, Any]:
    """One deduplicated object, in the contract's vocabulary.

    `object_class` rather than `class`: it is what the rest of the project
    calls this field, and `class` is a reserved word in the languages that will
    read this file.
    """
    return {
        "id": detection["id"],
        "object_class": detection["class"],
        "class_normalized": detection["class_normalized"],
        "confidence": detection["confidence"],
        "class_weight": detection["class_weight"],
        "severity": detection["severity"],
        "severity_tier": detection["severity_tier"],
        # "score" when the tiers table chose the word, "class-floor:<class>"
        # when the class's floor tier did (hazard_config.CLASS_TIER_FLOOR).
        "severity_tier_basis": detection.get("severity_tier_basis", "score"),
        "severity_basis": detection["severity_basis"],
        "recommended_action": detection["recommended_action"],
        "action_basis": detection["action_basis"],

        # Present only when the detector's own class was below the confidence
        # this system requires before asserting it. The original call is kept,
        # never erased, so the operator can see what was withheld and why.
        **({"class_withheld": detection["class_withheld"],
            "class_floor": detection["class_floor"],
            "downgraded_from": detection["downgraded_from"]}
           if "class_withheld" in detection else {}),

        # Relative survey coordinates. Always present, with or without nav.
        "global_x": detection["global_x"],
        "global_y": detection["global_y"],
        "bbox_global": detection["bbox_global"],
        "width_px": detection["width_px"],
        "height_px": detection["height_px"],

        # Geographic position, or an explicit null. Never a guess.
        "latitude": detection.get("latitude"),
        "longitude": detection.get("longitude"),

        # The 0-100 figure an operator reads. Verified, it fuses the calibrated
        # detector probability with the physical evidence in `verification`,
        # every term of which is kept so the number can be recomputed by hand.
        # Unverified, it is the detector's confidence and the basis says so.
        "confidence_pct": detection.get("confidence_pct"),
        **({"confidence_pct_basis": detection["confidence_pct_basis"]}
           if detection.get("confidence_pct_basis") else {}),
        "suppressed": bool(detection.get("suppressed", False)),
        # Physical size where the strip's resolution is known, height where a
        # shadow and an altitude allow one. Null values carry their basis.
        "dimensions": detection.get("dimensions"),
        **({"verification": detection["verification"]}
           if detection.get("verification") else {}),

        "provenance": {
            "strip": detection.get("strip"),
            "representative_tile": detection["representative_tile"],
            "source_tiles": detection["source_tiles"],
            "merged_count": detection["merged_count"],
            "merged_from": detection["merged_from"],
            "merge_spread_px": detection["merge_spread_px"],
            "tile_offset": [detection["tile_x"], detection["tile_y"]],
            "bbox_tile": detection["bbox_tile"],
            **({"detector_model": detection["detector_model"]}
               if detection.get("detector_model") else {}),
            # What another checkpoint called this same box. Kept rather than
            # resolved: a disagreement between two models is something an
            # operator should see.
            **({"second_opinion": detection["second_opinion"],
                "cross_model_views": detection["cross_model_views"]}
               if detection.get("second_opinion") else {}),
        },
    }


def build_summary(*, raw_count: int, detections: list[dict[str, Any]],
                  hotspots: list[dict[str, Any]], strips: list[str], tiles: int,
                  georeferenced: bool, coordinate_mode: str) -> dict[str, Any]:
    """The survey in one object: what was found, how bad, and where to start."""
    severities = [float(d["severity"]) for d in detections]
    classes: dict[str, int] = {}
    for detection in detections:
        classes[str(detection["class"])] = classes.get(str(detection["class"]), 0) + 1

    worst = max(detections, key=lambda d: (float(d["severity"]), float(d["confidence"]), d["id"]),
                default=None)
    top = hotspots[0] if hotspots else None

    return {
        "total_raw_detections": raw_count,
        "total_deduplicated_detections": len(detections),
        "duplicates_removed": raw_count - len(detections),
        # Still inside total_deduplicated_detections and still in the export;
        # counted here because they are excluded from hotspots.
        "suppressed_detections": sum(1 for d in detections if d.get("suppressed")),
        "total_hotspots": len(hotspots),
        "total_severity": round(sum(severities), 4),
        "highest_priority_hotspot": None if top is None else {
            "hotspot_id": top["hotspot_id"],
            "strip": top["strip"],
            "total_severity": top["total_severity"],
            "risk_score": top["risk_score"],
            "dominant_class": top["dominant_class"],
            "detection_count": top["detection_count"],
            "recommended_action": top["recommended_action"],
        },
        "highest_severity_class": None if worst is None else {
            "object_class": worst["class"],
            "severity": worst["severity"],
            "confidence": worst["confidence"],
            "detection_id": worst["id"],
        },
        "class_distribution": dict(sorted(classes.items(), key=lambda kv: (-kv[1], kv[0]))),
        # Tier counts are over detections, which is the unit an operator acts
        # on. Hotspot tiers are on each hotspot record.
        "detections_by_tier": _count_tiers(d.get("severity_tier") for d in detections),
        "hotspots_by_tier": _count_tiers(h.get("severity_tier") for h in hotspots),
        "georeferenced": georeferenced,
        "coordinate_mode": coordinate_mode,
        "strips_processed": len(strips),
        "strips": strips,
        "tiles_processed": tiles,
        "model_confidence_threshold": cfg.CONF_THRESH,
    }


def build_export(*, metadata: dict[str, Any], summary: dict[str, Any],
                 detections: list[dict[str, Any]], hotspots: list[dict[str, Any]],
                 provenance: dict[str, Any], configuration: dict[str, Any]
                 ) -> dict[str, Any]:
    """Assemble the contract. Key order here is the key order on disk."""
    return {
        "metadata": metadata,
        "survey_summary": summary,
        "detections": [export_detection(d) for d in detections],
        "hotspots": hotspots,
        "configuration": configuration,
        "provenance": provenance,
    }


def run_configuration() -> dict[str, Any]:
    """Every knob this run actually used, so the export explains its own numbers."""
    return {
        "tiling": {"TILE": cfg.TILE, "STRIDE": cfg.STRIDE,
                   "overlap_px": cfg.TILE - cfg.STRIDE,
                   "MIN_CONTENT": cfg.MIN_CONTENT, "DENOISE": cfg.DENOISE},
        "detection": {"CONF_THRESH": cfg.CONF_THRESH, "imgsz": cfg.DETECTOR_IMGSZ},
        "deduplication": {"MERGE_DIST": cfg.MERGE_DIST,
                          "cross_model_iou": cfg.DETECTOR_MERGE_IOU,
                          "cross_model_rule": (
                              "same tile, different checkpoints, IoU >= "
                              "cross_model_iou. Merges ACROSS classes, because the "
                              "checkpoints do not share a vocabulary; the losing "
                              "call is kept as second_opinion. A separate stage "
                              "from MERGE_DIST, which never merges classes."),
                          "rule": "same class and same strip, nearest representative "
                                  "within MERGE_DIST survey pixels, highest confidence "
                                  "wins, no chain merging"},
        "hotspots": {"GRID": cfg.GRID, "RISK_DENSITY_WEIGHT": cfg.RISK_DENSITY_WEIGHT,
                     "TOP_N_HOTSPOTS": cfg.TOP_N_HOTSPOTS,
                     "ranked_by": "total_severity descending",
                     "risk_score": "max_severity + RISK_DENSITY_WEIGHT * "
                                   "(total_severity - max_severity)"},
        "severity_policy": policy(),
        "disclaimer": cfg.DISCLAIMER,
    }


def write_export(out_dir: Path, export: dict[str, Any]) -> Path:
    path = Path(out_dir) / cfg.EXPORT_JSON
    path.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_actions(out_dir: Path, hotspots: list[dict[str, Any]],
                  top_n: int | None = None) -> Path:
    """The worklist, in rank order, as a file a survey lead can open anywhere.

    It carries its own `policy_basis` column. The caveat belongs on the row
    because this file gets opened in a spreadsheet, on its own, by someone who
    will never see the README.
    """
    limit = cfg.TOP_N_HOTSPOTS if top_n is None else int(top_n)
    selected = hotspots[:limit] if limit and limit > 0 else hotspots

    columns = ["priority_rank", "hotspot_id", "strip", "severity_tier", "risk_score",
               "total_severity", "max_severity", "detection_count", "hazard_diversity",
               "dominant_class", "dominant_class_severity", "recommended_action",
               "centroid_global_x", "centroid_global_y", "latitude", "longitude",
               "policy_basis", "rationale"]

    path = Path(out_dir) / cfg.ACTIONS_CSV
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for hotspot in selected:
            centroid = hotspot["centroid"]
            writer.writerow({
                "priority_rank": hotspot["priority_rank"],
                "hotspot_id": hotspot["hotspot_id"],
                "strip": hotspot["strip"],
                "severity_tier": hotspot["severity_tier"],
                "risk_score": hotspot["risk_score"],
                "total_severity": hotspot["total_severity"],
                "max_severity": hotspot["max_severity"],
                "detection_count": hotspot["detection_count"],
                "hazard_diversity": hotspot["hazard_diversity"],
                "dominant_class": hotspot["dominant_class"],
                "dominant_class_severity": hotspot["dominant_class_severity"],
                "recommended_action": hotspot["recommended_action"],
                "centroid_global_x": centroid["global_x"],
                "centroid_global_y": centroid["global_y"],
                "latitude": "" if centroid["latitude"] is None else centroid["latitude"],
                "longitude": "" if centroid["longitude"] is None else centroid["longitude"],
                "policy_basis": cfg.ACTION_BASIS,
                "rationale": hotspot["rationale"],
            })
    log.info("wrote %s with %d action rows", path.name, len(selected))
    return path


REPORT_CSV = "report.csv"
REPORT_GEOJSON = "report.geojson"

REPORT_COLUMNS = [
    "detection_id", "object_class", "confidence_pct", "suppressed", "suppression_reasons",
    "latitude", "longitude", "position_basis",
    "length_m", "width_m", "height_m", "dimension_basis", "width_px", "height_px",
    "severity_tier", "recommended_action", "strip", "global_x", "global_y",
    "detector_confidence", "confidence_basis", "policy_basis",
]


def _report_row(detection: dict[str, Any], coordinate_mode: str) -> dict[str, Any]:
    """One detected hazard as the flat record the problem statement asks for:
    where it is, how big it is, what it is, and how sure the system is."""
    dims = detection.get("dimensions") or {}
    verification = detection.get("verification") or {}
    located = detection.get("latitude") is not None
    return {
        "detection_id": detection["id"],
        "object_class": detection["object_class"],
        "confidence_pct": detection.get("confidence_pct"),
        "suppressed": bool(detection.get("suppressed")),
        "suppression_reasons": "; ".join(verification.get("reasons") or [])
                               if detection.get("suppressed") else "",
        "latitude": detection.get("latitude"),
        "longitude": detection.get("longitude"),
        "position_basis": coordinate_mode if located else
                          "not georeferenced: no navigation for this strip",
        "length_m": dims.get("length_m"),
        "width_m": dims.get("width_m"),
        "height_m": dims.get("height_m"),
        "dimension_basis": "; ".join(str(v) for k, v in dims.items()
                                     if k.endswith("basis") and v),
        "width_px": detection.get("width_px"),
        "height_px": detection.get("height_px"),
        "severity_tier": detection.get("severity_tier"),
        "recommended_action": detection.get("recommended_action"),
        "strip": (detection.get("provenance") or {}).get("strip"),
        "global_x": detection.get("global_x"),
        "global_y": detection.get("global_y"),
        "detector_confidence": detection.get("confidence"),
        "confidence_basis": detection.get("confidence_pct_basis")
                            or ("verified: calibrated detector probability fused with "
                                "shadow, clutter, nadir and dropout evidence"
                                if verification else "detector confidence"),
        "policy_basis": cfg.ACTION_BASIS,
    }


def write_reports(out_dir: Path, export: dict[str, Any]) -> tuple[Path, Path]:
    """report.csv and report.geojson: every detected hazard, one record each.

    Suppressed detections are included and flagged rather than left out. A
    report that silently omits what the filter removed cannot be audited, and
    an operator who disagrees with the filter needs the row to disagree with.

    The GeoJSON holds only detections with a real position. A point at (0, 0)
    for an unlocated object would be a fabricated location on a map, so the
    unlocated count is stated in the collection's properties instead.
    """
    out_dir = Path(out_dir)
    mode = export["survey_summary"].get("coordinate_mode", "")
    rows = [_report_row(d, mode) for d in export["detections"]]

    csv_path = out_dir / REPORT_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows([{k: ("" if v is None else v) for k, v in row.items()}
                          for row in rows])

    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
        "properties": {k: v for k, v in row.items() if k not in ("latitude", "longitude")},
    } for row in rows if row["latitude"] is not None and row["longitude"] is not None]

    metadata = export.get("metadata", {})
    geojson = {
        "type": "FeatureCollection",
        "properties": {
            "survey_id": metadata.get("survey_id"),
            "title": metadata.get("title"),
            "processed_at": metadata.get("processed_at"),
            "demo": bool(metadata.get("demo")),
            "coordinate_mode": mode,
            "detections_total": len(rows),
            "detections_located": len(features),
            "detections_unlocated": len(rows) - len(features),
            "disclaimer": cfg.DISCLAIMER,
        },
        "features": features,
    }
    geojson_path = out_dir / REPORT_GEOJSON
    geojson_path.write_text(json.dumps(geojson, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%d rows) and %s (%d located)", csv_path.name, len(rows),
             geojson_path.name, len(features))
    return csv_path, geojson_path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
