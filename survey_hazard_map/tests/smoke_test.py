#!/usr/bin/env python3
"""Prove the engine, with no checkpoint and no real survey.

    python3 smoke_test.py            # quiet, exit 1 on any failure
    python3 smoke_test.py --verbose  # engine logs as well

Everything runs inside a temporary directory that is deleted on the way out, so
this leaves nothing behind and can be run on a clean clone.

Synthetic strips are generated with numpy: textured seabed, a blank margin that
should be filtered out, and bright targets at planned pixel positions. A stand-
in detector then reports boxes derived from those planned positions, which is
what makes the assertions exact rather than approximate -- the test knows where
every object is and what should happen to it.

WHAT IS CHECKED
    1  the four output files exist and parse
    2  the manifest keeps its required columns, in order
    3  MIN_CONTENT drops the blank margin and keeps the textured seabed
    4  tile names are {strip}_{x}_{y}.jpg and the offsets are real pixels
    5  tiles overlap by TILE - STRIDE
    6  severity is exactly class_weight * confidence, every record
    7  an unlisted class falls back to UNKNOWN_CLASS_SEVERITY
    8  an object in the tile overlap is merged into one detection
    9  two different classes at the same place are NOT merged
   10  merging does not chain along a line of separate objects
   11  provenance survives: source_tiles, merged_count, representative_tile
   12  hotspots are ranked by total_severity, not by count
   13  the dominant class is severity-weighted, not a headcount
   14  ids are H001.. and agree with priority_rank
   15  no navigation means every latitude and longitude is null
   16  four-corner navigation puts every position inside the corners
   17  the export contract has all six blocks and the base keys
   18  no absolute filesystem path leaks into the export
   19  two runs over the same input agree on everything but the clock
"""

from __future__ import annotations

import argparse
import csv
import html as html_mod
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map import hazard_theme as theme
from survey_hazard_map.hazard_map import build_hazard_map
from survey_hazard_map.hazard_severity import class_weight, normalize_class
from survey_hazard_map.survey_preparation import prepare_survey

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, message: str) -> bool:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL  {message}")
        return False
    print(f"  ok    {message}")
    return True


# --- synthetic survey ------------------------------------------------------

STRIP_W, STRIP_H = 1400, 1700

# Wider than TILE on purpose. See make_strip().
BLANK_MARGIN = 660

# Objects placed in the strip, as (global_x, global_y, class, confidence).
# Positions are chosen against TILE=640 / STRIDE=512 / MERGE_DIST=60 / GRID=512
# so each group exercises one specific behaviour, named in its comment.
#
# Tile origins for this strip are x = 0, 512, 1024 and y = 0, 512, 1024, 1536.
PLANNED = [
    # In the 1024..1152 overlap of the last two tile columns: seen by both
    # tiles, must collapse to one detection.
    (1080.0, 300.0, "mine", 0.91),
    # Same place, different class. Must NOT merge with the mine, and must lose
    # the dominant-class vote in its cell despite being a real detection.
    (1085.0, 305.0, "debris", 0.77),
    # Four tyres 55 px apart, which is inside MERGE_DIST. Adjacent pairs are
    # therefore the same object by the configured rule and do merge. What must
    # NOT happen is the whole line collapsing into one: that is chain merging,
    # and it would turn four objects 165 px apart end to end into a single
    # contact. Two survivors is the correct answer and one is the bug.
    (700.0, 900.0, "tire", 0.60),
    (755.0, 900.0, "tire", 0.58),
    (810.0, 900.0, "tire", 0.56),
    (865.0, 900.0, "tire", 0.54),
    # Three cans 80 px apart, beyond MERGE_DIST. All three must survive.
    (700.0, 400.0, "can", 0.70),
    (780.0, 400.0, "can", 0.68),
    (860.0, 400.0, "can", 0.66),
    # A class no table lists. Must fall back to UNKNOWN_CLASS_SEVERITY.
    (1300.0, 1100.0, "sea-serpent", 0.80),
    # Three bottles in one cell, well apart. Low weight, high count: this cell
    # must not outrank the cell holding the single mine.
    (700.0, 1600.0, "bottle", 0.90),
    (780.0, 1610.0, "bottle", 0.90),
    (860.0, 1620.0, "bottle", 0.90),
]

# Corner coordinates for the georeferenced run. A small box off Mangaluru.
CORNERS = {
    "top_left": [12.9200, 74.8500],
    "top_right": [12.9200, 74.8620],
    "bottom_left": [12.9080, 74.8500],
    "bottom_right": [12.9080, 74.8620],
}


def make_strip(path: Path) -> None:
    """A sonar-like strip: textured seabed, blank margin, bright targets."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(20260911)
    # Seabed texture. Mid-grey with real variance, so content_score sits well
    # above MIN_CONTENT and the filter has to make a genuine decision.
    strip = rng.normal(110, 26, size=(STRIP_H, STRIP_W)).clip(0, 255)

    # A blank margin down the left edge, wider than one tile, so the whole
    # x = 0 tile column is flat and must be dropped by MIN_CONTENT. A margin
    # narrower than TILE would leave textured seabed in every tile and the
    # filter would correctly keep all of them, testing nothing.
    strip[:, :BLANK_MARGIN] = 12.0

    # Targets: a bright return with a dark acoustic shadow beside it.
    for gx, gy, _, _ in PLANNED:
        x, y = int(gx), int(gy)
        strip[y - 14:y + 14, x - 22:x + 22] = 244
        strip[y - 14:y + 14, x + 22:x + 60] = 6

    Image.fromarray(strip.astype("uint8"), mode="L").save(path)


class PlannedDetector:
    """A detector that reports the planned objects, tile by tile.

    It reads no pixels. It is given the same plan the strip was drawn from and
    reports any object whose centre falls inside the tile, converted into that
    tile's local coordinates. That is exactly what a real detector would emit
    for these tiles, and it makes every downstream assertion exact.
    """

    name = "planned-detector"
    conf = cfg.CONF_THRESH

    def __init__(self) -> None:
        self.classes = sorted({c for _, _, c, _ in PLANNED})
        self.calls = 0

    def __call__(self, image_path: Path) -> list[dict]:
        from survey_hazard_map.hazard_detect import parse_tile_name

        from PIL import Image

        self.calls += 1
        parsed = parse_tile_name(image_path.name)
        if parsed is None:
            return []
        _, tile_x, tile_y = parsed
        # The real tile, not TILE: an edge tile is partial, and a detector
        # cannot report a box outside the pixels it was shown.
        with Image.open(image_path) as tile:
            tile_w, tile_h = tile.size

        boxes = []
        for gx, gy, label, confidence in PLANNED:
            lx, ly = gx - tile_x, gy - tile_y
            if not (0 <= lx < tile_w and 0 <= ly < tile_h):
                continue
            boxes.append({
                "class": label,
                "confidence": confidence,
                "bbox": [lx - 20.0, ly - 12.0, lx + 20.0, ly + 12.0],
            })
        return boxes


# --- the checks ------------------------------------------------------------

def run(workspace: Path) -> None:
    strip_dir = workspace / "survey"
    strip_dir.mkdir(parents=True)
    strip_path = strip_dir / "strip_alpha.png"
    make_strip(strip_path)

    # ---------------------------------------------------------------- part 1
    print("\nMODULE 1  survey preparation, no navigation")
    relative_out = workspace / "relative"
    tiles_dir, manifest_csv = prepare_survey([strip_path], relative_out, nav=None)

    check(tiles_dir.is_dir(), "tiles directory created")
    check(manifest_csv.is_file(), "manifest.csv written")
    check((relative_out / cfg.MANIFEST_JSON).is_file(), "manifest.json written")

    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    check(header[:7] == list(cfg.MANIFEST_REQUIRED_COLUMNS),
          f"manifest required columns first and in order: {cfg.MANIFEST_REQUIRED_COLUMNS}")
    check(all(row["lat"] == "" and row["lon"] == "" for row in rows),
          "no navigation, so every manifest lat/lon is empty")

    payload = json.loads((relative_out / cfg.MANIFEST_JSON).read_text())
    check(payload["survey"]["coordinate_mode"] == cfg.COORD_MODE_RELATIVE,
          f"manifest declares '{cfg.COORD_MODE_RELATIVE}'")
    check(all(t["lat"] is None and t["lon"] is None for t in payload["tiles"]),
          "manifest.json carries explicit nulls, not fabricated positions")

    skipped = payload["survey"]["tiles_skipped_low_content"]
    check(skipped > 0, f"MIN_CONTENT dropped the blank margin ({skipped} tiles skipped)")
    check(len(rows) > 0, f"textured seabed tiles kept ({len(rows)} written)")
    kept_x = {int(r["x"]) for r in rows}
    check(0 not in kept_x, "the flat x=0 margin column was dropped, not kept")

    names_ok = all(t["tile"] == f"{t['strip']}_{t['x']}_{t['y']}.jpg" for t in payload["tiles"])
    check(names_ok, "every tile is named {strip}_{x}_{y}.jpg with real pixel offsets")

    xs = sorted({int(t["x"]) for t in payload["tiles"]})
    check(len(xs) > 1 and xs[1] - xs[0] == cfg.STRIDE,
          f"tile origins step by STRIDE={cfg.STRIDE}, overlapping by {cfg.TILE - cfg.STRIDE} px")
    check(all(t["center_x"] == t["x"] + t["tile_width"] / 2 for t in payload["tiles"]),
          "tile position is taken from the centre, not the top-left corner")

    # ---------------------------------------------------------------- part 2
    print("\nMODULE 2  hazard intelligence, relative coordinates")
    detector = PlannedDetector()
    export = build_hazard_map("models/known.pt", tiles_dir, relative_out,
                              detector=detector)

    for filename in (cfg.EXPORT_JSON, cfg.ACTIONS_CSV):
        check((relative_out / filename).is_file(), f"{filename} written")
    on_disk = json.loads((relative_out / cfg.EXPORT_JSON).read_text())
    check(on_disk == export, "export.json on disk matches the returned object")

    check(list(on_disk) == ["metadata", "survey_summary", "detections",
                            "hotspots", "configuration", "provenance"],
          "export carries all six contract blocks in order")
    check(isinstance(on_disk["detections"], list) and isinstance(on_disk["hotspots"], list),
          "base contract {detections: [], hotspots: []} holds")

    detections = export["detections"]
    summary = export["survey_summary"]

    # -- severity
    bad = [d for d in detections
           if abs(d["severity"] - round(d["class_weight"] * d["confidence"], 4)) > 1e-9]
    check(not bad, "severity == class_weight * confidence for every detection")

    serpent = next((d for d in detections if d["object_class"] == "sea-serpent"), None)
    check(serpent is not None and serpent["class_weight"] == cfg.UNKNOWN_CLASS_SEVERITY,
          f"an unlisted class falls back to UNKNOWN_CLASS_SEVERITY="
          f"{cfg.UNKNOWN_CLASS_SEVERITY}")
    check(serpent is not None and serpent["recommended_action"] == cfg.DEFAULT_ACTION,
          f"an unlisted class gets the default action: '{cfg.DEFAULT_ACTION}'")

    mine = next((d for d in detections if d["object_class"] == "mine"), None)
    check(mine is not None and mine["class_weight"] == cfg.SEVERITY["mine"],
          "a listed class takes its configured weight")

    # -- deduplication
    check(summary["total_raw_detections"] > summary["total_deduplicated_detections"],
          f"overlap produced duplicates and they were merged "
          f"({summary['total_raw_detections']} raw -> "
          f"{summary['total_deduplicated_detections']} objects)")
    check(len([d for d in detections if d["object_class"] == "mine"]) == 1,
          "the mine in the tile overlap is one detection, not two")
    check(mine is not None and mine["provenance"]["merged_count"] > 1,
          f"the merged mine records how many views it came from "
          f"({mine['provenance']['merged_count'] if mine else 0})")
    check(mine is not None and len(mine["provenance"]["source_tiles"]) > 1,
          "the merged mine names every tile it was seen in")
    check(mine is not None and mine["provenance"]["representative_tile"]
          in mine["provenance"]["source_tiles"],
          "the surviving call names the tile it came from")
    check(mine is not None and mine["confidence"] == 0.91,
          "the representative keeps the highest confidence of the merged group")

    check(len([d for d in detections if d["object_class"] == "debris"]) == 1
          and mine is not None,
          "a debris box 5 px from the mine stayed a separate detection: "
          "different classes are never merged on proximity")

    tyres = [d for d in detections if d["object_class"] == "tire"]
    check(len(tyres) == 2,
          f"four tyres 55 px apart did not chain into one contact 165 px long "
          f"({len(tyres)} survived; a chaining implementation gives 1)")
    check(all(t["provenance"]["merge_spread_px"] <= cfg.MERGE_DIST for t in tyres),
          "no merged group is wider than MERGE_DIST from its representative")

    cans = [d for d in detections if d["object_class"] == "can"]
    check(len(cans) == 3,
          f"three cans 80 px apart, beyond MERGE_DIST, all survived (got {len(cans)})")

    # -- hotspots
    hotspots = export["hotspots"]
    totals = [h["total_severity"] for h in hotspots]
    check(totals == sorted(totals, reverse=True),
          "hotspots are ranked by total_severity, descending")
    check([h["hotspot_id"] for h in hotspots] == [f"H{i:03d}" for i in
                                                  range(1, len(hotspots) + 1)],
          "hotspot ids are H001, H002, ... in rank order")
    check(all(h["priority_rank"] == i for i, h in enumerate(hotspots, 1)),
          "priority_rank agrees with position and with the id")

    mine_hotspot = next(h for h in hotspots if mine["id"] in h["detection_ids"])
    bottle_hotspot = next(h for h in hotspots
                          if any(d["object_class"] == "bottle"
                                 and d["id"] in h["detection_ids"] for d in detections))
    check(mine_hotspot["priority_rank"] < bottle_hotspot["priority_rank"],
          f"one mine ({mine_hotspot['hotspot_id']}, {mine_hotspot['detection_count']} "
          f"detections) outranks three bottles ({bottle_hotspot['hotspot_id']}, "
          f"{bottle_hotspot['detection_count']} detections): severity beats count")
    check(mine_hotspot["dominant_class"] == "mine",
          "dominant class is severity-weighted: the mine wins its cell over the debris")
    check(mine_hotspot["recommended_action"] == cfg.ACTIONS["mine"],
          f"the cell's action follows its dominant class: '{cfg.ACTIONS['mine']}'")

    expected_risk = round(mine_hotspot["max_severity"] + cfg.RISK_DENSITY_WEIGHT *
                          (mine_hotspot["total_severity"] - mine_hotspot["max_severity"]), 4)
    check(abs(mine_hotspot["risk_score"] - expected_risk) < 1e-9,
          "risk_score matches its documented formula")
    check(all(k in hotspots[0] for k in
              ("risk_score", "detection_density", "hazard_diversity", "confidence_mean",
               "confidence_max", "severity_per_detection", "spatial_extent",
               "priority_rank", "rationale")),
          "every derived metric is present on a hotspot")
    check(len(mine_hotspot["rationale"]) > 80 and "mine" in mine_hotspot["rationale"],
          "each hotspot explains its own priority in words")

    # -- coordinate honesty
    check(all(d["latitude"] is None and d["longitude"] is None for d in detections),
          "no navigation, so every detection latitude and longitude is null")
    check(all(h["centroid"]["latitude"] is None for h in hotspots),
          "no navigation, so every hotspot centroid is null too")
    check(all(isinstance(d["global_x"], (int, float)) for d in detections),
          "relative survey coordinates are present on every detection regardless")
    check(export["metadata"]["coordinate_mode"] == cfg.COORD_MODE_RELATIVE,
          f"the export declares '{cfg.COORD_MODE_RELATIVE}'")
    check(summary["georeferenced"] is False, "survey_summary says georeferenced: false")

    # -- actions.csv
    with (relative_out / cfg.ACTIONS_CSV).open(newline="", encoding="utf-8") as handle:
        action_rows = list(csv.DictReader(handle))
    check(len(action_rows) == len(hotspots), "actions.csv has one row per hotspot")
    check(action_rows[0]["hotspot_id"] == "H001" and action_rows[0]["priority_rank"] == "1",
          "actions.csv leads with H001")
    check(all(r["policy_basis"] == cfg.ACTION_BASIS for r in action_rows),
          "every action row states it is a heuristic, not an official procedure")
    check("official" not in json.dumps(export).lower().replace(
              cfg.ACTION_BASIS.lower(), "").replace(cfg.DISCLAIMER.lower(), ""),
          "nothing in the export claims official authority")

    # -- provenance and secrets
    provenance = export["provenance"]
    check(all(k in provenance for k in
              ("coordinate_mode", "navigation", "source_strips", "tile_count",
               "model", "confidence_threshold", "severity_formula", "ranking_rule",
               "processing_version")),
          "provenance answers what the numbers rest on")
    check(export["metadata"]["confidence_threshold"] == cfg.CONF_THRESH,
          f"the run records CONF_THRESH={cfg.CONF_THRESH}")
    blob = json.dumps(export)
    check(str(workspace) not in blob and "/Users/" not in blob,
          "no absolute filesystem path leaks into the export")
    check(export["configuration"]["severity_policy"]["class_weights"] == cfg.SEVERITY,
          "the export carries the full severity policy it used")

    # ---------------------------------------------------------------- part 3
    print("\nMODULE 1 + 2  four-corner navigation")
    geo_out = workspace / "geo"
    geo_tiles, _ = prepare_survey([strip_path], geo_out,
                                  nav={"mode": "corners", "strips": {"strip_alpha": CORNERS}})
    geo_export = build_hazard_map("models/known.pt", geo_tiles, geo_out,
                                  detector=PlannedDetector())

    geo_detections = geo_export["detections"]
    located = [d for d in geo_detections if d["latitude"] is not None]
    check(len(located) == len(geo_detections),
          "with corner navigation every detection carries a position")
    lat_lo, lat_hi = CORNERS["bottom_left"][0], CORNERS["top_left"][0]
    lon_lo, lon_hi = CORNERS["top_left"][1], CORNERS["top_right"][1]
    check(all(lat_lo <= d["latitude"] <= lat_hi and lon_lo <= d["longitude"] <= lon_hi
              for d in located),
          "every position falls inside the four corners it was interpolated from")
    check(geo_export["metadata"]["coordinate_mode"] == cfg.COORD_MODE_GEO,
          f"the georeferenced export declares '{cfg.COORD_MODE_GEO}'")
    check(geo_export["survey_summary"]["georeferenced"] is True,
          "survey_summary says georeferenced: true")
    check(all(isinstance(d["global_x"], (int, float)) for d in geo_detections),
          "relative coordinates are preserved alongside latitude and longitude")

    residual = geo_export["provenance"]["navigation"]["references"][0].get(
        "max_residual_degrees")
    check(residual is not None and residual < 1e-4,
          f"the manifest refit reports its own error ({residual} degrees)")

    # a detection north of another must have the larger latitude
    ordered = sorted(located, key=lambda d: d["global_y"])
    check(all(a["latitude"] >= b["latitude"] - 1e-9
              for a, b in zip(ordered, ordered[1:])),
          "latitude decreases down the strip, so the interpolation is not flipped")

    # ---------------------------------------------------------------- part 4
    print("\nDETERMINISM")
    repeat_out = workspace / "repeat"
    repeat_tiles, _ = prepare_survey([strip_path], repeat_out, nav=None)
    repeat = build_hazard_map("models/known.pt", repeat_tiles, repeat_out,
                              detector=PlannedDetector())

    # survey_id and title both default to the output directory's name, so two
    # runs into different directories differ there by design.
    volatile = {"processed_at", "processing_seconds", "survey_id", "title"}
    first = {k: v for k, v in export["metadata"].items() if k not in volatile}
    second = {k: v for k, v in repeat["metadata"].items() if k not in volatile}
    check(first == second, "metadata is identical between runs, clock aside")
    check(export["detections"] == repeat["detections"],
          "two runs produce byte-identical detections, ids included")
    check(export["hotspots"] == repeat["hotspots"],
          "two runs produce identical hotspots in identical rank order")

    tile_names = {p.name for p in tiles_dir.iterdir()}
    repeat_names = {p.name for p in repeat_tiles.iterdir()}
    check(tile_names == repeat_names, "the same tiles are produced with the same names")

    run_extras(workspace, strip_path)
    run_map(workspace, strip_path)




# --- part 5 ----------------------------------------------------------------

def make_second_strip(path: Path) -> None:
    """A shorter RGB strip, to prove colour and multi-strip both work."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(7)
    strip = rng.normal(120, 24, size=(900, 1000, 3)).clip(0, 255)
    # An object at the same pixel position as one in strip_alpha, same class.
    # Two strips are two coordinate frames, so these must stay two detections.
    strip[886:914, 678:722] = 250
    Image.fromarray(strip.astype("uint8"), mode="RGB").save(path)


class TwinDetector:
    """Reports one tyre at (700, 900) of whichever strip it is shown."""

    name = "twin-detector"
    conf = cfg.CONF_THRESH
    classes = ["tire"]

    def __call__(self, image_path: Path) -> list[dict]:
        from survey_hazard_map.hazard_detect import parse_tile_name

        parsed = parse_tile_name(image_path.name)
        if parsed is None:
            return []
        _, tile_x, tile_y = parsed
        lx, ly = 700.0 - tile_x, 900.0 - tile_y
        if not (0 <= lx < cfg.TILE and 0 <= ly < cfg.TILE):
            return []
        return [{"class": "tire", "confidence": 0.62,
                 "bbox": [lx - 20, ly - 12, lx + 20, ly + 12]}]


def run_extras(workspace: Path, strip_path: Path) -> None:
    print("\nMULTIPLE STRIPS, RGB, AND CROSS-STRIP SAFETY")
    second = strip_path.parent / "strip_bravo.png"
    make_second_strip(second)

    multi_out = workspace / "multi"
    multi_tiles, _ = prepare_survey(strip_path.parent, multi_out, nav=None)
    manifest = json.loads((multi_out / cfg.MANIFEST_JSON).read_text())
    strips = {s["strip"] for s in manifest["survey"]["strips"]}
    check(strips == {"strip_alpha", "strip_bravo"},
          "a directory of strips is picked up and both are tiled")
    check(any(t["strip"] == "strip_bravo" for t in manifest["tiles"]),
          "the RGB strip produced tiles alongside the grayscale one")

    multi = build_hazard_map("models/known.pt", multi_tiles, multi_out,
                             detector=TwinDetector())
    tyres = multi["detections"]
    check(len(tyres) == 2 and {d["provenance"]["strip"] for d in tyres}
          == {"strip_alpha", "strip_bravo"},
          "the same class at the same pixel on two strips stayed two detections: "
          "pixel proximity across strips is never a merge")
    check(len({h["strip"] for h in multi["hotspots"]}) == 2,
          "hotspots are scoped per strip, not pooled across frames")
    check(multi["survey_summary"]["strips_processed"] == 2,
          "the summary counts both strips")

    print("\nCONTROL-POINT CSV NAVIGATION")
    nav_csv = workspace / "nav.csv"
    nav_csv.write_text(
        "strip,pixel_x,pixel_y,latitude,longitude\n"
        "strip_alpha,0,0,12.9200,74.8500\n"
        "strip_alpha,1399,0,12.9200,74.8620\n"
        "strip_alpha,0,1699,12.9080,74.8500\n"
        "strip_alpha,1399,1699,12.9080,74.8620\n", encoding="utf-8")

    csv_out = workspace / "csvnav"
    csv_tiles, _ = prepare_survey([strip_path], csv_out, nav=nav_csv)
    csv_manifest = json.loads((csv_out / cfg.MANIFEST_JSON).read_text())
    check(csv_manifest["survey"]["navigation"]["mode"] == "control_points",
          "a CSV path is accepted as navigation")
    check(csv_manifest["survey"]["navigation"]["references"][0]["mode"] == "affine",
          "four non-collinear fixes were fitted as an affine transform")
    located = [t for t in csv_manifest["tiles"] if t["lat"] is not None]
    check(len(located) == len(csv_manifest["tiles"]),
          "every tile of a located strip carries a position")
    check(all(12.9080 <= t["lat"] <= 12.9200 and 74.8500 <= t["lon"] <= 74.8620
              for t in located),
          "every fitted tile position lies inside the control points")

    csv_export = build_hazard_map("models/known.pt", csv_tiles, csv_out,
                                  detector=PlannedDetector())
    check(csv_export["survey_summary"]["georeferenced"] is True,
          "the CSV-navigated survey is marked geo-referenced")

    print("\nALONG-TRACK NAVIGATION, ONE FIX PER PING ROW")
    track_csv = workspace / "track.csv"
    track_csv.write_text(
        "strip,pixel_x,pixel_y,latitude,longitude\n"
        "strip_alpha,700,0,12.9200,74.8560\n"
        "strip_alpha,700,850,12.9140,74.8560\n"
        "strip_alpha,700,1699,12.9080,74.8560\n", encoding="utf-8")
    track_out = workspace / "track"
    prepare_survey([strip_path], track_out, nav=track_csv)
    track_manifest = json.loads((track_out / cfg.MANIFEST_JSON).read_text())
    reference = track_manifest["survey"]["navigation"]["references"][0]
    check(reference["mode"] == "along_track",
          "collinear fixes fall back to along-track interpolation instead of "
          "fitting an underdetermined affine transform")
    check(reference["across_track_resolved"] is False,
          "the export states that across-track position was not resolved")

    print("\nDENOISE")
    denoise_out = workspace / "denoise"
    original = cfg.DENOISE
    cfg.DENOISE = True
    try:
        denoise_tiles, _ = prepare_survey([strip_path], denoise_out, nav=None)
    finally:
        cfg.DENOISE = original
    denoise_manifest = json.loads((denoise_out / cfg.MANIFEST_JSON).read_text())
    check(all(t["denoised"] for t in denoise_manifest["tiles"]),
          "denoised tiles are marked as such in the manifest")
    plain = json.loads((workspace / "relative" / cfg.MANIFEST_JSON).read_text())
    plain_names = {t["tile"] for t in plain["tiles"]}
    denoised_names = {t["tile"] for t in denoise_manifest["tiles"]}
    check(plain_names == denoised_names,
          "denoising changed intensity only: the same tiles at the same offsets")

    print("\nPER-RUN OVERRIDES")
    override_out = workspace / "override"
    override = build_hazard_map(
        "models/known.pt", Path(workspace / "relative" / "tiles"), override_out,
        Path(workspace / "relative" / cfg.MANIFEST_JSON),
        detector=PlannedDetector(),
        nav={"mode": "corners", "strips": {"strip_alpha": CORNERS}}, top_n=2)
    check(all(d["latitude"] is not None for d in override["detections"]),
          "navigation passed straight to build_hazard_map georeferences a "
          "tile set prepared without it")
    check(override["provenance"]["navigation"]["source"].endswith(
              "(supplied to build_hazard_map)"),
          "the export records that navigation came from the caller, not the manifest")
    with (override_out / cfg.ACTIONS_CSV).open(newline="", encoding="utf-8") as handle:
        limited = list(csv.DictReader(handle))
    check(len(limited) == 2, f"top_n limits actions.csv to 2 rows (got {len(limited)})")
    check(override["configuration"]["hotspots"]["TOP_N_HOTSPOTS"] == cfg.TOP_N_HOTSPOTS,
          "an override does not mutate the shared configuration")

    print("\nNO MANIFEST AT ALL")
    orphan = workspace / "orphan"
    orphan_tiles = orphan / "tiles"
    orphan_tiles.mkdir(parents=True)
    for tile in Path(workspace / "relative" / "tiles").iterdir():
        shutil.copy2(tile, orphan_tiles / tile.name)
    orphan_export = build_hazard_map("models/known.pt", orphan_tiles, orphan,
                                     detector=PlannedDetector())
    check(len(orphan_export["detections"]) > 0,
          "tile positions are recovered from {strip}_{x}_{y} filenames alone")
    check(all(d["latitude"] is None for d in orphan_export["detections"]),
          "no manifest means no navigation, so every position is null")
    check(orphan_export["provenance"]["manifest"] is None,
          "the export records that it ran without a manifest")
    relative_ids = {d["id"] for d in json.loads(
        (workspace / "relative" / cfg.EXPORT_JSON).read_text())["detections"]}
    check({d["id"] for d in orphan_export["detections"]} == relative_ids,
          "the same detections are found with and without the manifest")


# --- part 6, the map -------------------------------------------------------

def _tag_balance(html: str, tag: str) -> int:
    """Opens minus closes for a tag that must always be closed explicitly."""
    import re
    opens = len(re.findall(r"<" + tag + r"(?=[\s>])", html, re.I))
    closes = len(re.findall(r"</" + tag + r"\s*>", html, re.I))
    return opens - closes


def _external_resources(html: str) -> list[str]:
    """URLs the browser would actually fetch. An <a href> is not one."""
    import re
    return sorted(set(
        re.findall(r'<link[^>]+href="(https?://[^"]+)"', html)
        + re.findall(r'<script[^>]+src="(https?://[^"]+)"', html)
        + re.findall(r'<img[^>]+src="(https?://[^"]+)"', html)
        + re.findall(r"url\((?:'|\")?(https?://[^)'\"]+)", html)))


def run_map(workspace: Path, strip_path: Path) -> None:
    import re

    from survey_hazard_map.hazard_mapview import render_map

    print("\nMAP, RELATIVE COORDINATES")
    relative = workspace / "relative"
    export = json.loads((relative / cfg.EXPORT_JSON).read_text())
    map_path = render_map(export, relative, relative / "tiles",
                          relative / cfg.MANIFEST_JSON, title="Smoke survey")

    check(map_path.is_file(), "map.html written")
    html = map_path.read_text(encoding="utf-8")
    check(len(html) > 200_000,
          f"map.html carries its imagery inline ({len(html) / 1024:.0f} KB)")

    # -- standalone
    external = _external_resources(html)
    check(not external,
          f"the page fetches nothing from the network "
          f"{'' if not external else '(found: ' + ', '.join(external) + ')'}")
    check("data:image/jpeg;base64," in html,
          "the sonar strip is embedded as a data URI, not linked")
    # Leaflet keeps a @preserve banner through minification, so its presence
    # in the page body means the library itself is there, not a link to it.
    check("a JS library for interactive maps" in html,
          "Leaflet itself is inlined into the page")
    check("simpleheat" in html or "heatLayer" in html,
          "the heat plugin is inlined into the page")
    for unused in ("bootstrap.min.css", "jquery-3", "fontawesome"):
        check(f'src="https://cdn.jsdelivr.net/npm/{unused}' not in html
              and f'href="https://cdn.jsdelivr.net/npm/{unused}' not in html,
              f"the unused {unused.split('.')[0].split('-')[0]} link was removed")

    # -- structure
    for tag in ("div", "script", "style", "table", "button"):
        check(_tag_balance(html, tag) == 0, f"<{tag}> tags balance")

    # An unsubstituted ${token} would silently break the stylesheet rule it
    # sits in, and the page would still load looking almost right.
    check("${" not in html, "no template placeholder survived into the page")

    from html.parser import HTMLParser

    class Strict(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.problems = []

        def error(self, message):
            self.problems.append(message)

    parser = Strict()
    parser.feed(html)
    check(not parser.problems, "the page parses as HTML without errors")
    check(html.lstrip().lower().startswith("<!doctype html>"),
          "the page is a complete HTML document, openable directly from disk")

    # -- coordinate honesty, the whole point of this mode
    check(theme.TEXT["relative_mode"] in html,
          f"the header declares '{theme.TEXT['relative_mode']}'")
    check(theme.TEXT["geo_mode"] not in html,
          "the page never claims to be geo-referenced")
    check("<th>Latitude</th>" not in html and "<th>Longitude</th>" not in html,
          "no popup shows a latitude or longitude row, because there is none")
    check("crs: L.CRS.Simple" in html,
          "the map is drawn on a pixel plane, not a world projection")
    check("tile.openstreetmap.org" not in html,
          "no world basemap is loaded under a survey that has no world position")

    # -- layers
    for key, label in theme.LAYERS.items():
        check(f'>{label}</span>' in html or f">{label}<" in html or label in html,
              f"layer present: {label}")

    # -- popups
    for field in ("Class", "Confidence", "Severity", "Tier", "Survey position",
                  "Source tile", "Views merged", "Action"):
        check(f"<th>{field}</th>" in html, f"detection popup shows {field}")
    for field in ("Priority", "Dominant hazard", "Detections", "Total severity",
                  "Worst single", theme.TEXT["risk_label"], "Centroid"):
        check(f"<th>{field}</th>" in html, f"hotspot popup shows {field}")

    # --- the five faults an operator found in the rendered page ------------
    check("below 0.40" in html and "below 0.75" not in html,
          "legend reads 'low: below 0.40', not the 0.75 it used to claim")
    check('"zoomSnap": 0' in html,
          "fractional zoom is enabled, so fitBounds is not rounded down a whole "
          "power of two and the survey fills the viewport")
    check(f"{theme.MAP_SURROUND} !important" in html,
          "the area outside the survey is painted as canvas, not Leaflet's grey")
    check(theme.TEXT["risk_label"] in html and "index, not a percentage" in html,
          "risk is labelled an index, so a value above 1 is not read as a percentage")
    check(theme.TEXT["total_severity_label"] in html
          and theme.TEXT["max_severity_label"] in html,
          "the panel names total severity and the worst single detection separately")
    check("summed over the cell, can exceed 1" in html,
          "the popup says why a total severity can exceed 1 beside a 0-to-1 tier")

    # A cell on the survey edge must be clipped to the strip, or it drags the
    # fitted bounds out past the data.
    strip_h = max(int(r["y"]) + int(r["tile_height"]) for r in
                  json.loads((relative / cfg.MANIFEST_JSON).read_text())["tiles"])
    fit = re.search(r"fitBounds\(\s*\[\[(-?[\d.]+)", html)
    check(fit is not None and abs(float(fit.group(1))) <= strip_h + 1,
          f"fitted bounds stop at the survey edge ({strip_h} px), not past it")

    # -- the heatmap is weighted by severity, not by count
    heat = re.search(r"L\.heatLayer\(\s*(\[\[.*?\]\])\s*,", html, re.S)
    check(heat is not None, "a heat layer was written")
    if heat:
        weights = sorted(round(float(w), 4) for w in
                         re.findall(r",\s*([0-9.]+)\]", heat.group(1)))
        severities = sorted(round(float(d["severity"]), 4) for d in export["detections"])
        check(weights == severities,
              "every heat point is weighted by its severity, never by a count of 1")

    # -- the priority panel
    hotspots = export["hotspots"]
    check(all(f"data-hotspot='{h['hotspot_id']}'" in html for h in hotspots),
          "every hotspot has a clickable row in the priority panel")
    targets = re.search(r"var targets = (\{.*?\});", html, re.S)
    check(targets is not None, "the panel is wired to map targets")
    if targets:
        parsed = json.loads(targets.group(1))
        check(set(parsed) == {h["hotspot_id"] for h in hotspots},
              "every hotspot row has a map target to pan to")
        check(all(v["marker"] in html for v in parsed.values()),
              "every target names a marker that exists on the page")
    check(theme.TEXT["start_here"] in html, "the top-ranked hotspot is marked to start at")
    check(cfg.DISCLAIMER[:40] in html, "the map states that this is not official procedure")
    check(theme.TEXT["demo_banner"] not in html,
          "a real survey carries no synthetic-data banner")

    print("\nCONFIG CHANGES")
    from survey_hazard_map.hazard_detect import UltralyticsDetector
    from survey_hazard_map.hazard_severity import apply_confidence_floor, class_weight, recommended_action

    fish, _ = class_weight("fish")
    wreck, _ = class_weight("shipwreck")
    check(fish < wreck,
          f"a fish ({fish}) no longer outranks a shipwreck ({wreck})")
    check(recommended_action("aircraft")[0] == "Flag navigation hazard",
          "an aircraft contact gets a navigation-hazard action, not the default")
    check(recommended_action("tire")[0] == "Schedule cleanup",
          "a tyre gets a cleanup action, not the default")
    check(recommended_action("fish")[0] == "Log only, no action",
          "a fish asks for nothing, rather than inheriting the expert-identification "
          "default meant for unidentified objects")
    check(class_weight("fish")[0] * 1.0 < cfg.SEVERITY_TIERS[1][1],
          "a fish cannot leave the low tier at any confidence")
    check(recommended_action("sea-serpent")[0] == cfg.DEFAULT_ACTION,
          "a class nobody has a policy for still falls back to expert identification")

    withheld = apply_confidence_floor({"class": "human", "confidence": 0.463})
    check(withheld["class"] == cfg.DOWNGRADE_LABEL,
          "a human call below its floor is withheld and reported as unidentified")
    check(withheld["class_withheld"] == "human" and "downgraded_from" in withheld,
          "the withheld class and the reason are preserved, never erased")
    kept = apply_confidence_floor({"class": "human", "confidence": 0.80})
    check(kept["class"] == "human" and "class_withheld" not in kept,
          "a human call above its floor is left alone")
    check(apply_confidence_floor({"class": "debris", "confidence": 0.05})["class"]
          == "debris",
          "a class with no floor is never withheld")

    # The documented asymmetry. If this ever silently reverses, the note beside
    # CLASS_CONFIDENCE_FLOOR has become wrong and somebody should notice.
    human_before = class_weight("human")[0] * 0.463
    human_after = class_weight(cfg.DOWNGRADE_LABEL)[0] * 0.463
    air_before = class_weight("aircraft")[0] * 0.55
    air_after = class_weight(cfg.DOWNGRADE_LABEL)[0] * 0.55
    check(human_after < human_before and air_after > air_before,
          "withholding a class lowers severity for human and raises it for "
          "aircraft, exactly as the config documents")

    try:
        UltralyticsDetector(["missing_a.pt", "missing_b.pt"])
        multi = False
    except Exception as exc:
        multi = "missing_a.pt" in str(exc) and "missing_b.pt" in str(exc)
    check(multi, "the detector loader accepts several checkpoints and names every "
                 "one it cannot find")

    # The y axis is the one thing that would put every marker in the wrong
    # place while still looking plausible. Sonar y grows downward, Leaflet
    # latitude grows upward, so a missing negation mirrors the whole survey.
    overlay = re.search(r'imageOverlay\(\s*"data:image[^"]*",\s*\[\[([-\d.]+),\s*'
                        r'([-\d.]+)\],\s*\[([-\d.]+),\s*([-\d.]+)\]\]', html)
    check(overlay is not None, "the sonar overlay is placed with explicit bounds")
    if overlay:
        south, west, north, east = (float(v) for v in overlay.groups())
        check(north == 0.0 and south < 0.0 and west == 0.0 and east > 0.0,
              "the strip hangs from y=0 downward, matching sonar pixel order")
    markers = re.findall(r"L\.circleMarker\(\s*\[([-\d.]+),\s*([-\d.]+)\]", html)
    map_detections = export["detections"]
    check(len(markers) >= len(map_detections),
          f"a marker was drawn for every detection ({len(markers)} for "
          f"{len(map_detections)})")
    placed = {(round(float(a), 2), round(float(b), 2)) for a, b in markers}
    expected = {(round(-d["global_y"], 2), round(d["global_x"], 2))
                for d in map_detections}
    check(expected <= placed,
          "every marker sits at minus its global_y, so the survey is not "
          "mirrored vertically against its own imagery")

    print("\nCROSS-MODEL MERGE")

    class TwoModelDetector:
        """Two checkpoints seeing one object, and one seeing a second object."""

        name = "two-model-stand-in"
        conf = cfg.CONF_THRESH
        classes = ["ship", "shipwreck", "mine"]

        def __call__(self, image_path: Path) -> list[dict]:
            from survey_hazard_map.hazard_detect import parse_tile_name

            parsed = parse_tile_name(image_path.name)
            if parsed is None:
                return []
            _, tile_x, tile_y = parsed
            out = []
            # One wreck at (1080, 300), called differently by each model.
            lx, ly = 1080.0 - tile_x, 300.0 - tile_y
            if 0 <= lx < cfg.TILE and 0 <= ly < cfg.TILE:
                out.append({"class": "ship", "confidence": 0.82, "model": "known",
                            "bbox": [lx - 40, ly - 25, lx + 40, ly + 25]})
                out.append({"class": "shipwreck", "confidence": 0.39, "model": "anomaly",
                            "bbox": [lx - 44, ly - 28, lx + 38, ly + 27]})
            # A mine 300 px away, seen by one model only. Must survive.
            mx, my = 1080.0 - tile_x, 600.0 - tile_y
            if 0 <= mx < cfg.TILE and 0 <= my < cfg.TILE:
                out.append({"class": "mine", "confidence": 0.75, "model": "known",
                            "bbox": [mx - 15, my - 15, mx + 15, my + 15]})
            return out

    cross_out = workspace / "crossmodel"
    cross = build_hazard_map("two.pt", relative / "tiles", cross_out,
                             relative / cfg.MANIFEST_JSON, detector=TwoModelDetector())
    wrecks = [d for d in cross["detections"] if d["object_class"] in ("ship", "shipwreck")]
    check(len(wrecks) == 1,
          f"one wreck seen by two checkpoints is one detection, not two "
          f"(got {len(wrecks)})")
    check(wrecks and wrecks[0]["object_class"] == "ship"
          and wrecks[0]["confidence"] == 0.82,
          "the more confident checkpoint's call is the one that survives")
    opinions = wrecks[0]["provenance"].get("second_opinion", []) if wrecks else []
    check(len(opinions) == 1 and opinions[0]["object_class"] == "shipwreck"
          and opinions[0]["agrees"] is False,
          "the other checkpoint's disagreeing call is kept, not discarded")
    check(any(d["object_class"] == "mine" for d in cross["detections"]),
          "an object only one checkpoint saw is not lost in the cross-model merge")
    check(cross["configuration"]["deduplication"]["cross_model_iou"]
          == cfg.DETECTOR_MERGE_IOU,
          "the export records the overlap threshold the merge used")

    print("\nMAP, GEO-REFERENCED")
    geo = workspace / "geo"
    geo_export = json.loads((geo / cfg.EXPORT_JSON).read_text())
    geo_map = render_map(geo_export, geo, geo / "tiles", geo / cfg.MANIFEST_JSON,
                         title="Smoke survey, navigated")
    geo_html = geo_map.read_text(encoding="utf-8")

    check(theme.TEXT["geo_mode"] in geo_html,
          f"the header declares '{theme.TEXT['geo_mode']}'")
    check(theme.TEXT["relative_mode"] not in geo_html,
          "a navigated survey is not labelled as relative")
    check("<th>Latitude</th>" in geo_html and "<th>Longitude</th>" in geo_html,
          "popups show latitude and longitude when there genuinely is one")
    check("crs: L.CRS.Simple" not in geo_html,
          "a navigated survey is drawn on real coordinates")
    check("<th>Survey position</th>" in geo_html,
          "relative pixel position is still shown beside the geographic one")
    check(not _external_resources(geo_html),
          "the navigated map is self-contained too, with no basemap by default")

    geo_basemap = render_map(geo_export, geo, geo / "tiles", geo / cfg.MANIFEST_JSON,
                             basemap=True, filename="map_basemap.html")
    basemap_html = geo_basemap.read_text(encoding="utf-8")
    check("openstreetmap" in basemap_html.lower(),
          "--basemap adds a street layer to a navigated survey")
    check('"Street basemap": false' in basemap_html.replace("'", '"')
          or "Street basemap" in basemap_html,
          "the street layer exists but is off when the map opens")

    print("\nMAP, DEMO MODE")
    demo_out = workspace / "demo"
    from survey_hazard_map import demo_survey
    demo_survey.main(["--out", str(demo_out), "--quiet"])
    demo_export = json.loads((demo_out / cfg.EXPORT_JSON).read_text())
    demo_html = (demo_out / "map.html").read_text(encoding="utf-8")

    check(demo_export["metadata"]["demo"] is True, "export.json marks the run as demo")
    check(demo_export["metadata"]["data_source"] == theme.TEXT["demo_banner"],
          "export.json names the data source as synthetic")
    check(demo_export["provenance"]["demo"] is True, "provenance marks the run as demo")
    check("SIMULATED" in demo_export["metadata"]["detector"],
          "the export names a simulated detector, not a checkpoint")
    check(theme.TEXT["demo_banner"] in demo_html,
          "the map carries a synthetic-data banner")
    # The banner carries the export's own demo_warning when it has one (an
    # import with simulated navigation words it differently from the demo
    # script), else the theme's generic note. Either way it must say so in words.
    banner_note = demo_export["metadata"].get("demo_warning") or theme.TEXT["demo_note"]
    check(html_mod.escape(banner_note)[:40] in demo_html and "evidence" in banner_note,
          "the banner says in words that nothing here is evidence")
    check(not _external_resources(demo_html), "the demo map is self-contained")

    # -- verification on the map
    demo_suppressed = [d for d in demo_export["detections"] if d.get("suppressed")]
    check(f"Filtered false positives ({len(demo_suppressed)})" in demo_html,
          f"the layer control offers the {len(demo_suppressed)} filtered detection(s)")
    check("$(`" not in demo_html and "deHtml(`" in demo_html,
          "popups are built without jQuery, which the offline page does not carry")
    check(all(f"<b>{float(d['confidence_pct']):.1f}%</b>" in demo_html
              for d in demo_export["detections"]),
          "every detection's popup states its confidence_pct as a percentage")
    if demo_suppressed:
        check("<th>Filtered because</th>" in demo_html,
              "a filtered detection's popup says why it was filtered")

    top = demo_export["hotspots"][0]
    check(top["dominant_class"] == "mine" and top["severity_tier"] == "critical",
          f"the demo leads with the ordnance cluster "
          f"({top['hotspot_id']}, {top['dominant_class']})")
    biggest = max(demo_export["hotspots"], key=lambda h: h["detection_count"])
    check(biggest["priority_rank"] > top["priority_rank"],
          f"the cell with the most contacts ({biggest['detection_count']} in "
          f"{biggest['hotspot_id']}) ranks below the ordnance cluster")

    demo_survey._refuse_to_mix(demo_out)
    real_dir = workspace / "relative"
    try:
        demo_survey._refuse_to_mix(real_dir)
        refused = False
    except SystemExit:
        refused = True
    check(refused, "demo output refuses to overwrite a real survey export")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="show engine logs")
    parser.add_argument("--keep", metavar="DIR",
                        help="write the run into DIR and leave it there")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(message)s")

    workspace = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="hazard-smoke-"))
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        run(workspace)
    finally:
        if not args.keep:
            shutil.rmtree(workspace, ignore_errors=True)

    print()
    if FAILURES:
        print(f"FAILED  {len(FAILURES)} of {CHECKS} checks")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"PASSED  {CHECKS}/{CHECKS} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
