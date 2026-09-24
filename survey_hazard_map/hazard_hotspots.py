"""Detections into hotspots, hotspots into a ranked worklist.

A list of two hundred detections is not a decision. A survey team can dive one
place first, and this module decides which place that is.

AGGREGATION
    The survey is divided into a square grid of GRID pixels per side, per
    strip. Every cell holding at least one detection becomes a hotspot. A cell
    is a fixed frame rather than a grown cluster, which means two runs over the
    same data produce the same hotspots in the same order, and a hotspot's
    extent is a known quantity rather than an emergent one.

    The known limitation, stated because it is real: two detections 10 px apart
    but either side of a cell boundary land in different hotspots. Both are
    still reported, both are still ranked, and neither is lost. Raise GRID if
    your objects cluster tighter than the grid resolves.

RANKING
    By total_severity, descending. Not by count.

    Counting would rank a cell holding six tyres above a cell holding one mine,
    which is the exact failure this system exists to avoid. Total severity is
    the sum of class_weight x confidence over the cell, so six tyres at 0.3
    weight contribute less than one mine at 1.0, and the order comes out right
    without anything being special-cased.

DOMINANT CLASS
    Also severity-weighted, for the same reason. The dominant class of a cell
    is the class with the largest summed severity in it, not the class with the
    most boxes. Three pieces of debris beside one mine leaves the mine as the
    dominant class, and the recommended action is the mine's.

IDENTIFIERS
    H001, H002, and so on, assigned after ranking. H001 is always the highest
    priority hotspot in the export, so the id and the rank never disagree.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map.hazard_geo import Georeference
from survey_hazard_map.hazard_severity import highest_tier, recommended_action, severity_tier

log = logging.getLogger("deepecho.hazard")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dominant(detections: list[dict[str, Any]]) -> tuple[str, float]:
    """(class, its summed severity). Severity-weighted, never a headcount."""
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for detection in detections:
        totals[str(detection["class"])] += float(detection["severity"])
        counts[str(detection["class"])] += 1
    # Tie-broken by count then alphabetically, so an exact severity tie between
    # two classes still resolves identically on every run.
    winner = max(totals, key=lambda c: (totals[c], counts[c], c))
    return winner, round(totals[winner], 4)


def _centroid_geo(detections: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """The mean position, but only when every member actually has one.

    Averaging the located half of a cell would produce a coordinate that sits
    confidently in the wrong place, which is worse than no coordinate at all.
    """
    lats = [d.get("latitude") for d in detections]
    lons = [d.get("longitude") for d in detections]
    if not lats or any(v is None for v in lats) or any(v is None for v in lons):
        return None, None
    return round(_mean(lats), 8), round(_mean(lons), 8)


def _rationale(hotspot: dict[str, Any]) -> str:
    """Why this hotspot has this priority, in a sentence, from its own numbers.

    Written into the export so the question "why is this first?" is answered by
    the file rather than by reading this source.
    """
    top = hotspot["top_detection"]
    others = hotspot["detection_count"] - 1
    tail = (f" and {others} other detection{'s' if others != 1 else ''} adding "
            f"{round(hotspot['total_severity'] - hotspot['max_severity'], 4)} more severity"
            if others > 0 else " and nothing else in the cell")
    return (
        f"Ranked {hotspot['priority_rank']} on a total severity of "
        f"{hotspot['total_severity']} across {hotspot['detection_count']} "
        f"detection{'s' if hotspot['detection_count'] != 1 else ''}. The worst is "
        f"'{top['class']}' at confidence {top['confidence']} and class weight "
        f"{top['class_weight']}, giving severity {top['severity']}{tail}. "
        f"Dominant class '{hotspot['dominant_class']}' is severity-weighted, so it "
        f"reflects consequence rather than headcount.")


def build_hotspots(detections: list[dict[str, Any]],
                   references: dict[str, Georeference] | None = None,
                   grid: int | None = None) -> list[dict[str, Any]]:
    """Group detections into ranked hotspots with their derived metrics."""
    size = cfg.GRID if grid is None else int(grid)
    if size <= 0:
        raise ValueError("GRID must be a positive number of pixels")

    cells: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        strip = str(detection.get("strip") or "")
        cells[(strip,
               int(math.floor(detection["global_x"] / size)),
               int(math.floor(detection["global_y"] / size)))].append(detection)

    cell_area_megapixels = (size * size) / 1_000_000.0
    hotspots: list[dict[str, Any]] = []

    for (strip, cell_x, cell_y), members in cells.items():
        severities = [float(m["severity"]) for m in members]
        confidences = [float(m["confidence"]) for m in members]
        xs = [float(m["global_x"]) for m in members]
        ys = [float(m["global_y"]) for m in members]

        total_severity = round(sum(severities), 4)
        max_severity = round(max(severities), 4)
        dominant_class, dominant_severity = _dominant(members)
        action, action_basis = recommended_action(dominant_class)
        lat, lon = _centroid_geo(members)

        width = round(max(xs) - min(xs), 2)
        height = round(max(ys) - min(ys), 2)
        top = max(members, key=lambda m: (float(m["severity"]), float(m["confidence"]), m["id"]))

        hotspots.append({
            "strip": strip,
            "cell": [cell_x, cell_y],
            "cell_size_px": size,
            "centroid": {
                "global_x": round(_mean(xs), 2),
                "global_y": round(_mean(ys), 2),
                "latitude": lat,
                "longitude": lon,
            },
            "detection_count": len(members),
            "total_severity": total_severity,
            "max_severity": max_severity,
            # The worst member's word, not the word for max_severity: a member
            # can sit above its score's tier through a class floor.
            "severity_tier": highest_tier([str(m.get("severity_tier") or severity_tier(float(m["severity"])))
                                           for m in members]),
            "dominant_class": dominant_class,
            "dominant_class_severity": dominant_severity,
            "recommended_action": action,
            "action_basis": action_basis,

            # --- derived metrics, all of them arithmetic on the numbers above
            # risk_score: the worst object, plus a fraction of everything else.
            "risk_score": round(
                max_severity + cfg.RISK_DENSITY_WEIGHT * (total_severity - max_severity), 4),
            # detections per megapixel of cell area, so cells stay comparable
            # if GRID is ever changed between runs.
            "detection_density": round(len(members) / cell_area_megapixels, 3),
            # distinct classes present. 1 is a single kind of problem; higher
            # means a mixed site, which usually needs a mixed response.
            "hazard_diversity": len({str(m["class"]) for m in members}),
            "confidence_mean": round(_mean(confidences), 4),
            "confidence_max": round(max(confidences), 4),
            "severity_per_detection": round(total_severity / len(members), 4),
            # the bounding box of the detection centres, not of the cell: how
            # spread out the objects actually are inside it.
            "spatial_extent": {
                "width_px": width,
                "height_px": height,
                "diagonal_px": round(math.hypot(width, height), 2),
            },
            "detection_ids": [m["id"] for m in members],
            "top_detection": {
                "id": top["id"],
                "class": top["class"],
                "confidence": top["confidence"],
                "class_weight": top["class_weight"],
                "severity": top["severity"],
                # Deduplication names the surviving view's tile; a detection
                # that has not been through it still knows its own. Reading
                # either keeps this function usable on its own rather than
                # only as the fourth stage of a pipeline.
                "tile": top.get("representative_tile") or top.get("tile"),
            },
        })

    # Severity first, and every subsequent key exists only to make ties
    # resolve the same way twice.
    hotspots.sort(key=lambda h: (-h["total_severity"], -h["max_severity"],
                                 -h["detection_count"], h["strip"],
                                 h["cell"][1], h["cell"][0]))

    ranked: list[dict[str, Any]] = []
    for rank, hotspot in enumerate(hotspots, start=1):
        hotspot["hotspot_id"] = f"H{rank:03d}"
        hotspot["priority_rank"] = rank
        hotspot["rationale"] = _rationale(hotspot)
        # Rebuilt so the identifier and the rank serialise first and a human
        # reading the raw export sees which hotspot it is before anything else.
        ranked.append({"hotspot_id": hotspot["hotspot_id"], "priority_rank": rank,
                       **{k: v for k, v in hotspot.items()
                          if k not in ("hotspot_id", "priority_rank")}})
    hotspots = ranked

    counts = {tier: 0 for tier, _ in cfg.SEVERITY_TIERS}
    for hotspot in hotspots:
        counts[hotspot["severity_tier"]] += 1
    log.info("hotspots: %d cells of %d px (%s), ranked by total severity",
             len(hotspots), size,
             ", ".join(f"{n} {tier}" for tier, n in counts.items()))
    return hotspots
