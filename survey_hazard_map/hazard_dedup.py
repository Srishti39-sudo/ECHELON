"""One record per physical object, across overlapping tiles.

Tiles overlap by TILE - STRIDE pixels so that nothing lying on a boundary is
cut in half and missed by both neighbours. The cost is that an object in the
overlap is detected two, three or four times. Counting it that way would
inflate every number the system produces: the detection count, the hotspot's
total severity, its priority.

THE RULE
    Two detections are the same object when they have the SAME class and their
    global centres are within MERGE_DIST survey pixels of each other, on the
    same strip.

    Same class, always. A drum lying inside a wreck field is two facts and they
    are merged into one only by a system that has decided proximity outranks
    identity. Different classes are never merged, at any distance.

    Same strip, always. Two strips are two coordinate frames; see hazard_coords
    for why pixel proximity across them means nothing.

NO CHAINS
    Detections are taken highest-confidence first. Each one either joins an
    already-accepted representative or becomes a representative itself, and
    membership is only ever tested against representatives -- never against
    other members. That matters: without it, A merges B, B merges C, C merges
    D, and a line of separate objects each 60 px from the next collapses into
    one cluster hundreds of pixels long. Testing against the representative
    alone caps a merged group at MERGE_DIST from a single fixed point.

WHAT SURVIVES
    The highest-confidence detection is the representative, and its class,
    confidence and box are the ones that go forward unchanged. Everything it
    absorbed is kept as provenance: which tiles the object was seen in, how
    many times, and which tile the surviving call came from.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from survey_hazard_map import hazard_config as cfg
log = logging.getLogger("deepecho.hazard")


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def merge_across_models(detections: list[dict[str, Any]],
                        iou: float | None = None) -> list[dict[str, Any]]:
    """One record per object per tile, when several checkpoints saw it.

    Runs BEFORE the survey-wide deduplication and answers a different question:
    not "did one object appear in several tiles" but "did two models see the
    same box in this one tile". Both stages are needed and neither does the
    other's job.

    This one merges ACROSS classes, which the survey-wide stage never does. The
    reason is that the checkpoints do not share a vocabulary: known.pt says
    "ship" where anomaly.pt says "shipwreck" for the same wreck. Within one
    tile, at high overlap, that is one object described twice. Across tiles,
    proximity between two different classes is two objects, and merging them
    would be the error this system is built to avoid.

    The more confident call survives. The other is kept on it as a structured
    `second_opinion`, never dropped, because two models disagreeing about what
    something is is a fact the operator should see rather than a tie for the
    software to settle quietly.
    """
    threshold = cfg.DETECTOR_MERGE_IOU if iou is None else float(iou)

    ordered = sorted(detections,
                     key=lambda d: (-float(d["confidence"]), d["id"]))
    kept: list[dict[str, Any]] = []
    by_tile: dict[str, list[dict[str, Any]]] = {}
    merged = 0
    disagreements = 0

    for detection in ordered:
        tile = str(detection.get("tile") or "")
        model = detection.get("detector_model")
        candidates = by_tile.setdefault(tile, [])

        twin = None
        for candidate in candidates:
            # Same model twice in one tile is the model's own duplicate box and
            # is left to the survey-wide stage; only cross-model pairs merge here.
            if candidate.get("detector_model") == model:
                continue
            if _iou(candidate["bbox_tile"], detection["bbox_tile"]) >= threshold:
                twin = candidate
                break

        if twin is None:
            candidates.append(detection)
            kept.append(detection)
            continue

        merged += 1
        agreed = str(twin.get("class")) == str(detection.get("class"))
        disagreements += not agreed
        twin.setdefault("second_opinion", []).append({
            "model": model,
            "object_class": detection.get("class"),
            "confidence": detection.get("confidence"),
            "agrees": agreed,
        })
        twin["cross_model_views"] = 1 + len(twin["second_opinion"])

    if merged:
        log.info("cross-model merge: %d box(es) folded into a more confident call "
                 "from another checkpoint, %d of them a disagreement about class "
                 "(IoU >= %.2f, same tile)", merged, disagreements, threshold)
    return kept


def deduplicate(detections: list[dict[str, Any]],
                merge_dist: float | None = None) -> list[dict[str, Any]]:
    """Collapse duplicate views of the same object into one record each."""
    distance = cfg.MERGE_DIST if merge_dist is None else float(merge_dist)

    # Highest confidence first, then by position and id so that two detections
    # of identical confidence always resolve the same way. Determinism here is
    # what makes two runs of the same survey produce the same export.
    ordered = sorted(
        detections,
        key=lambda d: (-float(d["confidence"]), d["global_x"], d["global_y"], d["id"]))

    representatives: list[dict[str, Any]] = []
    # Representatives bucketed by (strip, normalised class), so a detection is
    # only ever compared against candidates it could actually merge with.
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for detection in ordered:
        key = (str(detection.get("strip") or ""), detection.get("class_normalized")
               or str(detection.get("class") or ""))
        candidates = buckets.setdefault(key, [])

        twin = None
        best = distance
        for candidate in candidates:
            gap = math.hypot(detection["global_x"] - candidate["global_x"],
                             detection["global_y"] - candidate["global_y"])
            # Nearest representative within range wins, not merely the first
            # one found, so the outcome does not depend on insertion order.
            if gap <= best:
                twin, best = candidate, gap

        if twin is not None:
            twin["_members"].append(detection)
            continue

        representative = dict(detection)
        representative["_members"] = [detection]
        candidates.append(representative)
        representatives.append(representative)

    merged: list[dict[str, Any]] = []
    for representative in representatives:
        members = representative.pop("_members")
        tiles = sorted({m["tile"] for m in members})
        representative.update({
            "source_tiles": tiles,
            "merged_count": len(members),
            "representative_tile": representative["tile"],
            "merged_from": sorted(m["id"] for m in members if m["id"] != representative["id"]),
            # The spread of the views that were merged. A large value on a
            # merged group is a hint that MERGE_DIST is set too wide.
            "merge_spread_px": round(max(
                (math.hypot(m["global_x"] - representative["global_x"],
                            m["global_y"] - representative["global_y"]) for m in members),
                default=0.0), 2),
        })
        merged.append(representative)

    # Final order is spatial and stable, so detection ids in the export ascend
    # down the survey rather than following confidence.
    merged.sort(key=lambda d: (str(d.get("strip") or ""), d["global_y"], d["global_x"], d["id"]))

    removed = len(detections) - len(merged)
    log.info("deduplication: %d raw detections -> %d objects, %d duplicates removed "
             "(MERGE_DIST=%.0f px, same class and strip only)",
             len(detections), len(merged), removed, distance)
    return merged
