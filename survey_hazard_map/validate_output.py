#!/usr/bin/env python3
"""Check a survey's output against the contract it claims to follow.

    python3 validate_output.py out/
    python3 validate_output.py out/ --strict     # treat warnings as failures

Exit code is 1 if any check fails, so this can gate a commit or a demo.

This is not a test of the engine. It is a test of one survey's ARTEFACTS, and
it reads them the way a stranger would: opening the files, checking the numbers
against each other, and believing nothing it was not shown. That distinction
matters, because the failure it exists to catch is an export that was generated
correctly months ago and has since been edited, truncated, or half-regenerated.

WHAT IT REFUSES TO ASSUME
    Nothing is recomputed from the engine's own code. Severity is checked as
    class_weight times confidence using the weights recorded IN THE EXPORT, not
    the weights currently in hazard_config. An export whose numbers no longer
    match the policy it was written under is exactly what this should catch, and
    importing the live config would hide it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REQUIRED_FILES = ("manifest.csv", "manifest.json", "export.json", "actions.csv")
OPTIONAL_FILES = ("map.html",)

REQUIRED_BLOCKS = ("metadata", "survey_summary", "detections", "hotspots",
                   "configuration", "provenance")

REQUIRED_DETECTION_FIELDS = ("id", "object_class", "confidence", "class_weight",
                             "severity", "severity_tier", "recommended_action",
                             "global_x", "global_y", "latitude", "longitude",
                             "provenance")

REQUIRED_HOTSPOT_FIELDS = ("hotspot_id", "priority_rank", "centroid",
                           "detection_count", "total_severity", "max_severity",
                           "severity_tier", "dominant_class", "recommended_action",
                           "risk_score", "detection_ids", "rationale")

REQUIRED_MANIFEST_COLUMNS = ("tile", "strip", "x", "y", "lat", "lon", "mean_intensity")

# Floating point: severity is a product of two rounded numbers, so an exact
# equality test would fail on arithmetic rather than on anything real.
TOLERANCE = 1e-6


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.passed = 0

    def check(self, condition: bool, message: str) -> bool:
        if condition:
            self.passed += 1
            return True
        self.failures.append(message)
        return False

    def warn(self, condition: bool, message: str) -> bool:
        if condition:
            self.passed += 1
            return True
        self.warnings.append(message)
        return False


def validate(out_dir: Path, report: Report) -> None:
    # --- files ------------------------------------------------------------
    for name in REQUIRED_FILES:
        report.check((out_dir / name).is_file(), f"missing required file: {name}")
    for name in OPTIONAL_FILES:
        report.warn((out_dir / name).is_file(), f"no {name} (run render_map to produce one)")

    export_path = out_dir / "export.json"
    if not export_path.is_file():
        return

    try:
        export = json.loads(export_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        report.check(False, f"export.json is not valid JSON: {exc}")
        return

    # --- structure --------------------------------------------------------
    for block in REQUIRED_BLOCKS:
        report.check(block in export, f"export.json has no {block!r} block")
    if not all(b in export for b in REQUIRED_BLOCKS):
        return

    detections = export["detections"]
    hotspots = export["hotspots"]
    summary = export["survey_summary"]
    report.check(isinstance(detections, list) and isinstance(hotspots, list),
                 "detections and hotspots must both be lists")

    for index, detection in enumerate(detections):
        missing = [f for f in REQUIRED_DETECTION_FIELDS if f not in detection]
        report.check(not missing,
                     f"detection {index} is missing {', '.join(missing)}")
    for index, hotspot in enumerate(hotspots):
        missing = [f for f in REQUIRED_HOTSPOT_FIELDS if f not in hotspot]
        report.check(not missing, f"hotspot {index} is missing {', '.join(missing)}")

    # --- numeric ranges ---------------------------------------------------
    for detection in detections:
        name = detection.get("id", "?")
        confidence = detection.get("confidence")
        severity = detection.get("severity")
        weight = detection.get("class_weight")

        report.check(isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0,
                     f"{name}: confidence {confidence} is outside 0..1")
        report.check(isinstance(severity, (int, float)) and severity >= 0.0,
                     f"{name}: severity {severity} is negative")
        report.check(isinstance(weight, (int, float)) and weight >= 0.0,
                     f"{name}: class_weight {weight} is negative")

        # The formula, checked against the export's own recorded inputs.
        if all(isinstance(v, (int, float)) for v in (confidence, severity, weight)):
            report.check(abs(severity - round(weight * confidence, 4)) <= TOLERANCE,
                         f"{name}: severity {severity} != class_weight {weight} "
                         f"x confidence {confidence}")

    for hotspot in hotspots:
        name = hotspot.get("hotspot_id", "?")
        report.check(hotspot.get("total_severity", -1) >= 0,
                     f"{name}: total_severity is negative")
        report.check(0.0 <= hotspot.get("max_severity", -1),
                     f"{name}: max_severity is negative")
        report.check(hotspot.get("detection_count", 0) == len(hotspot.get("detection_ids", [])),
                     f"{name}: detection_count disagrees with detection_ids")

    # --- ranking ----------------------------------------------------------
    totals = [h["total_severity"] for h in hotspots]
    report.check(totals == sorted(totals, reverse=True),
                 "hotspots are not in descending total_severity order")
    report.check([h["priority_rank"] for h in hotspots] == list(range(1, len(hotspots) + 1)),
                 "priority_rank is not 1..n in order")
    report.check([h["hotspot_id"] for h in hotspots]
                 == [f"H{i:03d}" for i in range(1, len(hotspots) + 1)],
                 "hotspot ids are not H001..Hnnn in rank order")

    # --- referential integrity -------------------------------------------
    by_id = {d["id"]: d for d in detections}
    report.check(len(by_id) == len(detections), "two detections share an id")
    orphans = [i for h in hotspots for i in h["detection_ids"] if i not in by_id]
    report.check(not orphans,
                 f"hotspots reference {len(orphans)} detection id(s) that do not exist")
    grouped = {i for h in hotspots for i in h["detection_ids"]}
    # A detection verification marked suppressed is kept in the export with its
    # reasons and deliberately left out of every hotspot. Every other one must
    # belong to exactly the hotspots that reference it.
    suppressed = {d["id"] for d in detections if d.get("suppressed")}
    report.check(grouped == set(by_id) - suppressed,
                 f"{len(set(by_id) - suppressed - grouped)} detection(s) belong to no hotspot")
    report.check(not (grouped & suppressed),
                 f"{len(grouped & suppressed)} suppressed detection(s) were counted in a hotspot")

    # --- coordinate honesty ----------------------------------------------
    # The one check this whole file exists for. A survey with no navigation
    # must not carry a single coordinate anywhere.
    georeferenced = bool(summary.get("georeferenced"))
    if not georeferenced:
        stray = [d["id"] for d in detections
                 if d.get("latitude") is not None or d.get("longitude") is not None]
        report.check(not stray,
                     f"survey is not georeferenced yet {len(stray)} detection(s) "
                     f"carry a latitude or longitude: {', '.join(stray[:3])}")
        stray_hotspots = [h["hotspot_id"] for h in hotspots
                          if h["centroid"].get("latitude") is not None]
        report.check(not stray_hotspots,
                     f"survey is not georeferenced yet hotspots {stray_hotspots[:3]} "
                     f"carry a centroid position")
        report.check(summary.get("coordinate_mode") == "Relative Survey Coordinates",
                     f"coordinate_mode should be relative, is "
                     f"{summary.get('coordinate_mode')!r}")
    else:
        for detection in detections:
            lat, lon = detection.get("latitude"), detection.get("longitude")
            if lat is None:
                continue
            report.check(-90.0 <= lat <= 90.0, f"{detection['id']}: latitude {lat} out of range")
            report.check(-180.0 <= lon <= 180.0, f"{detection['id']}: longitude {lon} out of range")

    # Relative coordinates must be present either way.
    report.check(all(isinstance(d.get("global_x"), (int, float))
                     and isinstance(d.get("global_y"), (int, float)) for d in detections),
                 "a detection is missing its relative survey coordinates")

    # --- deduplication ----------------------------------------------------
    # No two surviving detections of the same class on the same strip may sit
    # within MERGE_DIST of each other; if they did, the merge did not run.
    merge_dist = export["configuration"]["deduplication"]["MERGE_DIST"]
    violations = []
    for i, a in enumerate(detections):
        for b in detections[i + 1:]:
            if a["object_class"] != b["object_class"]:
                continue
            if a["provenance"].get("strip") != b["provenance"].get("strip"):
                continue
            gap = math.hypot(a["global_x"] - b["global_x"], a["global_y"] - b["global_y"])
            if gap <= merge_dist:
                violations.append(f"{a['id']} and {b['id']} are {gap:.1f} px apart")
    report.check(not violations,
                 f"deduplication left {len(violations)} same-class pair(s) within "
                 f"MERGE_DIST={merge_dist}: {'; '.join(violations[:3])}")

    # --- summary consistency ---------------------------------------------
    report.check(summary.get("total_deduplicated_detections") == len(detections),
                 "survey_summary detection count disagrees with the detections list")
    report.check(summary.get("total_hotspots") == len(hotspots),
                 "survey_summary hotspot count disagrees with the hotspots list")
    tier_total = sum(summary.get("detections_by_tier", {}).values())
    report.check(tier_total == len(detections),
                 f"tier counts sum to {tier_total}, not {len(detections)}")

    # --- manifest ---------------------------------------------------------
    manifest_csv = out_dir / "manifest.csv"
    if manifest_csv.is_file():
        with manifest_csv.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
        report.check(columns[:len(REQUIRED_MANIFEST_COLUMNS)] == list(REQUIRED_MANIFEST_COLUMNS),
                     f"manifest.csv columns must start {REQUIRED_MANIFEST_COLUMNS}, got "
                     f"{tuple(columns[:7])}")
        if not georeferenced:
            filled = [r["tile"] for r in rows if r.get("lat")]
            report.check(not filled,
                         f"{len(filled)} manifest row(s) carry a latitude on a survey "
                         f"with no navigation")

    # --- actions.csv ------------------------------------------------------
    actions_csv = out_dir / "actions.csv"
    if actions_csv.is_file():
        with actions_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        top_n = export["configuration"]["hotspots"].get("TOP_N_HOTSPOTS", 0)
        expected = len(hotspots) if not top_n else min(top_n, len(hotspots))
        report.check(len(rows) == expected,
                     f"actions.csv has {len(rows)} rows, expected {expected}")
        report.check(all(r.get("recommended_action") for r in rows),
                     "an actions.csv row has no recommended action")
        report.check(all(r.get("policy_basis") for r in rows),
                     "an actions.csv row does not state that it is a heuristic")

    # --- map --------------------------------------------------------------
    map_html = out_dir / "map.html"
    if map_html.is_file():
        html = map_html.read_text(encoding="utf-8", errors="replace")
        report.check(html.lstrip().lower().startswith("<!doctype html>"),
                     "map.html is not a complete HTML document")
        report.warn("data:image" in html, "map.html embeds no imagery")
        if not georeferenced:
            report.check("<th>Latitude</th>" not in html,
                         "map.html shows a latitude row on a survey with no navigation")
        # folium writes popups as $(`...`)[0] and the offline page carries no
        # jQuery, so a map rendered before hazard_mapview replaced that call
        # stops its script at the first popup. A warning: the export is fine,
        # the map needs rendering again.
        report.warn("$(`" not in html,
                    "map.html builds popups with jQuery, which it does not carry; "
                    "re-render it with the current hazard_mapview")
        # A verified export's filtered count is shown on the map. When the map
        # states a count it must be the export's; a map without one predates
        # verification and only needs rendering again.
        verified = "suppressed_detections" in summary or any(
            "suppressed" in d for d in detections)
        if verified:
            suppressed_count = sum(1 for d in detections if d.get("suppressed"))
            report.check(summary.get("suppressed_detections", suppressed_count)
                         == suppressed_count,
                         f"survey_summary.suppressed_detections disagrees with the "
                         f"{suppressed_count} detection(s) marked suppressed")
            stated = re.findall(r"Filtered false positives \((\d+)\)", html)
            if stated:
                report.check(all(int(n) == suppressed_count for n in stated),
                             f"map.html says {sorted(set(stated))} filtered false "
                             f"positive(s); export.json marks {suppressed_count}")
            else:
                report.warn(False, "map.html has no filtered-detections layer although "
                                   "export.json is verified; re-render it")
            unshown = [d["id"] for d in detections
                       if isinstance(d.get("confidence_pct"), (int, float))
                       and f"{float(d['confidence_pct']):.1f}%" not in html]
            report.warn(not unshown, f"map.html does not show confidence_pct for "
                                     f"{len(unshown)} detection(s); re-render it")

    # --- secrets ----------------------------------------------------------
    blob = json.dumps(export)
    for marker in ("API_KEY", "api_key", "SECRET", "password", "Bearer ",
                   "GEMINI", "GROQ", "SUPABASE"):
        report.check(marker not in blob, f"export.json contains {marker!r}")
    report.check("/Users/" not in blob and "/home/" not in blob,
                 "export.json leaks an absolute home directory path")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", help="a survey output directory")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        print(f"not a directory: {out_dir}")
        return 1

    report = Report()
    validate(out_dir, report)

    for warning in report.warnings:
        print(f"  WARN  {warning}")
    for failure in report.failures:
        print(f"  FAIL  {failure}")

    failed = len(report.failures) + (len(report.warnings) if args.strict else 0)
    if failed:
        print(f"\n{out_dir}: {failed} problem(s), {report.passed} checks passed")
        return 1
    print(f"{out_dir}: valid, {report.passed} checks passed"
          + (f", {len(report.warnings)} warning(s)" if report.warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
