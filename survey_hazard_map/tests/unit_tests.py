#!/usr/bin/env python3
"""Unit tests for the survey hazard engine.

    python3 unit_tests.py
    python3 unit_tests.py --verbose

Exit code is 1 if any case fails, so this can gate a commit.

Plain functions and asserts rather than pytest, because that is what this
repository already does: eval/run.py is a runnable script with an exit code and
no test framework, and adding one for seventeen cases would be a dependency
nobody asked for.

These are UNIT tests. Every one exercises a single function against values
chosen by hand, with no images, no files, no model and no network, so the whole
file runs in well under a second. The end-to-end behaviour is covered by
smoke_test.py and a survey's artefacts by validate_output.py; the three do not
overlap on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from survey_hazard_map import hazard_config as cfg
from survey_hazard_map.hazard_coords import to_global
from survey_hazard_map.hazard_dedup import deduplicate, merge_across_models
from survey_hazard_map.hazard_detect import parse_tile_name
from survey_hazard_map.hazard_export import build_export, build_summary, export_detection, run_configuration
from survey_hazard_map.hazard_geo import Georeference, NavigationError, _interpolate
from survey_hazard_map.hazard_hotspots import build_hotspots
from survey_hazard_map.hazard_severity import (apply_confidence_floor, class_weight, normalize_class,
                             highest_tier, recommended_action, score_detection, severity_tier,
                             tier_counts)
from survey_hazard_map.survey_preparation import _grey, _offsets

CASES = []


def case(name):
    def register(fn):
        CASES.append((name, fn))
        return fn
    return register


def detection(id_, cls, conf, x, y, strip="s", tile="s_0_0.jpg", tx=0, ty=0, model=None):
    """A raw detection as hazard_detect produces one, before any processing."""
    return {
        "id": id_, "class": cls, "confidence": conf,
        "bbox_tile": [x - tx - 10, y - ty - 10, x - tx + 10, y - ty + 10],
        "tile": tile, "strip": strip, "tile_x": tx, "tile_y": ty,
        "detector_model": model,
    }


def prepared(*detections):
    """Raw detections put through the stages dedup and hotspots depend on."""
    items = [dict(d) for d in detections]
    to_global(items)
    for item in items:
        score_detection(item)
    return items


# --- 1. filename parsing ---------------------------------------------------

@case("tile filenames parse back to strip and pixel offsets")
def _():
    assert parse_tile_name("strip_512_1024.jpg") == ("strip", 512, 1024)
    assert parse_tile_name("strip_0_0.jpg") == ("strip", 0, 0)
    # A strip whose own name contains underscores still parses, because the
    # split is from the right. This is why the manifest is the contract and
    # filenames are only a fallback, but the fallback has to work.
    assert parse_tile_name("survey_2026_run_3_640_1280.png") == ("survey_2026_run_3", 640, 1280)


@case("malformed tile filenames are rejected, never guessed")
def _():
    for bad in ("nooffsets.jpg", "strip_abc_def.jpg", "strip_512.jpg",
                "strip_512_notanumber.jpg", "", "_.jpg"):
        assert parse_tile_name(bad) is None, f"{bad!r} should not parse"


# --- 2. global coordinate conversion ---------------------------------------

@case("tile-local boxes convert to survey coordinates")
def _():
    item = {"bbox_tile": [100.0, 50.0, 140.0, 90.0], "tile_x": 512, "tile_y": 1024}
    to_global([item])
    assert item["bbox_center_x"] == 120.0
    assert item["bbox_center_y"] == 70.0
    assert item["global_x"] == 512 + 120.0
    assert item["global_y"] == 1024 + 70.0
    assert item["bbox_global"] == [612.0, 1074.0, 652.0, 1114.0]
    assert item["width_px"] == 40.0 and item["height_px"] == 40.0


@case("a box at a tile's origin lands at the tile's offset")
def _():
    item = {"bbox_tile": [0.0, 0.0, 0.0, 0.0], "tile_x": 2048, "tile_y": 512}
    to_global([item])
    assert (item["global_x"], item["global_y"]) == (2048.0, 512.0)


# --- 3. tile generation ----------------------------------------------------

@case("tile offsets cover the strip and never leave a sliver")
def _():
    assert _offsets(100, 640, 512) == [0]                    # smaller than a tile
    assert _offsets(640, 640, 512) == [0]                    # exactly a tile
    assert _offsets(641, 640, 512) == [0, 512]               # one pixel over
    assert _offsets(1400, 640, 512) == [0, 512, 1024]
    assert _offsets(1700, 640, 512) == [0, 512, 1024, 1536]

    # Every pixel of a strip is covered, and the final partial tile is never
    # narrower than TILE - STRIDE.
    for extent in range(641, 4000, 7):
        offsets = _offsets(extent, 640, 512)
        assert offsets[0] == 0
        assert offsets[-1] + 640 >= extent, f"{extent} is not fully covered"
        last_width = extent - offsets[-1]
        assert last_width > 640 - 512, f"{extent} produced a {last_width}px sliver"


@case("consecutive tiles overlap by exactly TILE minus STRIDE")
def _():
    offsets = _offsets(3000, cfg.TILE, cfg.STRIDE)
    for a, b in zip(offsets, offsets[1:]):
        assert (a + cfg.TILE) - b == cfg.TILE - cfg.STRIDE


# --- 4. content filtering --------------------------------------------------

@case("content score is the documented standard deviation over 255")
def _():
    import numpy as np

    flat = np.full((64, 64), 120, dtype=np.uint8)
    assert float(np.std(_grey(flat))) / 255.0 == 0.0

    textured = np.random.default_rng(0).normal(120, 40, (64, 64)).clip(0, 255).astype(np.uint8)
    score = float(np.std(_grey(textured))) / 255.0
    assert score > cfg.MIN_CONTENT, "textured seabed must survive the filter"

    # An RGB tile is scored through the same luma path as a grey one.
    rgb = np.stack([textured] * 3, axis=-1)
    assert abs(float(np.std(_grey(rgb))) / 255.0 - score) < 1e-3


# --- 5. missing navigation -------------------------------------------------

@case("without navigation nothing carries a position")
def _():
    from survey_hazard_map.hazard_coords import attach_geo

    items = prepared(detection("a", "mine", 0.9, 100, 100))
    located = attach_geo(items, {})
    assert located == 0
    # Explicit nulls, not absent keys: a consumer must be able to tell
    # "not located" from "field missing" without guessing.
    assert items[0]["latitude"] is None and "latitude" in items[0]
    assert items[0]["longitude"] is None
    # Relative coordinates survive regardless.
    assert items[0]["global_x"] == 100.0


@case("too few navigation fixes is refused, not extrapolated")
def _():
    try:
        Georeference.from_control_points("s", [(0.0, 0.0, 12.9, 74.8)])
    except NavigationError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError("one fix should not produce a transform")


# --- 6. navigation interpolation -------------------------------------------

@case("four corners interpolate bilinearly from the tile centre")
def _():
    reference = Georeference.from_corners("s", {
        "top_left": [10.0, 70.0], "top_right": [10.0, 74.0],
        "bottom_left": [6.0, 70.0], "bottom_right": [6.0, 74.0],
    }, width=101, height=101)

    assert reference.locate(0, 0) == (10.0, 70.0)
    assert reference.locate(100, 100) == (6.0, 74.0)
    lat, lon = reference.locate(50, 50)
    assert abs(lat - 8.0) < 1e-6 and abs(lon - 72.0) < 1e-6
    # Latitude falls as the row index rises, which is north-up.
    assert reference.locate(50, 0)[0] > reference.locate(50, 100)[0]


@case("non-collinear fixes fit an affine transform that reproduces them")
def _():
    points = [(0.0, 0.0, 10.0, 70.0), (100.0, 0.0, 10.0, 74.0),
              (0.0, 100.0, 6.0, 70.0), (100.0, 100.0, 6.0, 74.0)]
    reference = Georeference.from_control_points("s", points)
    assert reference.mode == "affine"
    assert reference.detail["max_residual_degrees"] < 1e-9
    for px, py, lat, lon in points:
        got = reference.locate(px, py)
        assert abs(got[0] - lat) < 1e-6 and abs(got[1] - lon) < 1e-6


@case("collinear fixes fall back to along-track and say so")
def _():
    # One fix per ping row, every fix on the same pixel column. An affine fit
    # would be underdetermined across-track, so it must not be attempted.
    points = [(700.0, 0.0, 10.0, 72.0), (700.0, 500.0, 8.0, 72.0),
              (700.0, 1000.0, 6.0, 72.0)]
    reference = Georeference.from_control_points("s", points)
    assert reference.mode == "along_track"
    assert reference.detail["across_track_resolved"] is False
    lat, _ = reference.locate(700.0, 250.0)
    assert abs(lat - 9.0) < 1e-6


@case("interpolation extrapolates from the end segment rather than clamping")
def _():
    knots, values = [0.0, 10.0], [0.0, 100.0]
    assert _interpolate(5.0, knots, values) == 50.0
    # Clamping would return 100.0 here and stack every later tile on one point.
    assert _interpolate(20.0, knots, values) == 200.0
    assert _interpolate(-5.0, knots, values) == -50.0


# --- 7 and 8. deduplication ------------------------------------------------

@case("same-class detections within MERGE_DIST merge into one")
def _():
    items = prepared(detection("a", "mine", 0.90, 100, 100),
                     detection("b", "mine", 0.70, 130, 100))
    merged = deduplicate(items)
    assert len(merged) == 1
    assert merged[0]["confidence"] == 0.90, "the confident call must survive"
    assert merged[0]["merged_count"] == 2
    assert merged[0]["merged_from"] == ["b"]


@case("different classes never merge, however close")
def _():
    items = prepared(detection("a", "mine", 0.90, 100, 100),
                     detection("b", "debris", 0.88, 101, 100))
    assert len(deduplicate(items)) == 2, "a drum inside a wreck field is two facts"


@case("detections beyond MERGE_DIST stay separate")
def _():
    far = cfg.MERGE_DIST + 10
    items = prepared(detection("a", "mine", 0.9, 100, 100),
                     detection("b", "mine", 0.8, 100 + far, 100))
    assert len(deduplicate(items)) == 2


@case("a line of objects does not chain into one contact")
def _():
    # Each 55 px from the next, inside MERGE_DIST, so adjacent pairs merge. The
    # whole line collapsing to one is the bug this guards.
    items = prepared(*[detection(f"t{i}", "tire", 0.6 - i * 0.01, 200 + i * 55, 900)
                       for i in range(4)])
    merged = deduplicate(items)
    assert len(merged) == 2, f"expected 2 representatives, got {len(merged)}"
    assert all(m["merge_spread_px"] <= cfg.MERGE_DIST for m in merged)


@case("the same class on two strips is two objects")
def _():
    items = prepared(detection("a", "mine", 0.9, 100, 100, strip="alpha"),
                     detection("b", "mine", 0.9, 100, 100, strip="bravo"))
    assert len(deduplicate(items)) == 2, "two strips are two coordinate frames"


@case("two checkpoints on one box merge and keep the disagreement")
def _():
    items = prepared(
        detection("a", "ship", 0.82, 100, 100, model="known"),
        detection("b", "shipwreck", 0.39, 102, 101, model="anomaly"))
    merged = merge_across_models(items)
    assert len(merged) == 1
    assert merged[0]["class"] == "ship"
    opinion = merged[0]["second_opinion"][0]
    assert opinion["object_class"] == "shipwreck" and opinion["agrees"] is False


@case("one checkpoint seeing something the other missed keeps it")
def _():
    items = prepared(
        detection("a", "ship", 0.82, 100, 100, model="known"),
        detection("b", "mine", 0.75, 400, 400, model="known"))
    assert len(merge_across_models(items)) == 2


# --- 9 and 10. severity ----------------------------------------------------

@case("severity is exactly class weight times confidence")
def _():
    for cls, conf in (("mine", 0.88), ("debris", 0.55), ("shipwreck", 0.7), ("fish", 0.99)):
        item = score_detection({"class": cls, "confidence": conf})
        assert item["severity"] == round(class_weight(cls)[0] * conf, 4)
        # Both inputs stay beside the result so any score can be recomputed.
        assert item["class_weight"] == round(class_weight(cls)[0], 4)
        assert item["confidence"] == round(conf, 4)


@case("a class the table has never heard of is treated as unidentified")
def _():
    weight, basis = class_weight("sea-serpent")
    assert weight == cfg.UNKNOWN_CLASS_SEVERITY
    assert basis == "default:unknown-class"
    # Never 0.0: a detector trained on classes this policy does not know must
    # not have its findings weighted out of existence.
    assert weight > 0.0


@case("class names are matched after normalising, longest key first")
def _():
    assert normalize_class("Ghost_Gear") == "ghost-gear"
    assert normalize_class("  NAVAL MINE  ") == "naval-mine"
    assert class_weight("moored_mine")[0] == cfg.SEVERITY["mine"]
    assert class_weight("Naval Mine")[0] == cfg.SEVERITY["mine"]
    assert class_weight("shipwreck")[1] == "exact"


@case("a fish cannot outrank a wreck at any confidence")
def _():
    fish, wreck = class_weight("fish")[0], class_weight("shipwreck")[0]
    assert fish < wreck
    assert severity_tier(fish * 1.0) == "low"


@case("a class below its confidence floor is withheld, never dropped")
def _():
    withheld = apply_confidence_floor({"class": "human", "confidence": 0.463})
    assert withheld["class"] == cfg.DOWNGRADE_LABEL
    assert withheld["class_withheld"] == "human"
    assert "downgraded_from" in withheld
    assert apply_confidence_floor({"class": "human", "confidence": 0.9})["class"] == "human"
    assert apply_confidence_floor({"class": "debris", "confidence": 0.01})["class"] == "debris"


@case("withholding a class changes the claim, not always the direction")
def _():
    unknown = class_weight(cfg.DOWNGRADE_LABEL)[0]
    assert class_weight("human")[0] > unknown, "withholding human must lower it"
    assert class_weight("aircraft")[0] < unknown, "withholding aircraft must raise it"


# --- 11. severity tiers ----------------------------------------------------

@case("severity tiers use the configured boundaries")
def _():
    critical, medium = cfg.SEVERITY_TIERS[0][1], cfg.SEVERITY_TIERS[1][1]
    assert severity_tier(critical) == "critical"
    assert severity_tier(critical - 1e-9) == "medium"
    assert severity_tier(medium) == "medium"
    assert severity_tier(medium - 1e-9) == "low"
    assert severity_tier(0.0) == "low"


@case("tier counts list every tier, including the empty ones")
def _():
    counts = tier_counts([0.9, 0.5, 0.1, 0.05])
    assert counts == {"critical": 1, "medium": 1, "low": 2}
    assert set(tier_counts([]).values()) == {0}
    assert set(tier_counts([])) == {t for t, _ in cfg.SEVERITY_TIERS}


# --- 12. action mapping ----------------------------------------------------

@case("actions follow the class, with a safe default")
def _():
    assert recommended_action("mine")[0] == "Deploy EOD team"
    assert recommended_action("shipwreck")[0] == "Flag navigation hazard"
    assert recommended_action("tire")[0] == "Schedule cleanup"
    assert recommended_action("fish")[0] == "Log only, no action"
    assert recommended_action("sea-serpent")[0] == cfg.DEFAULT_ACTION
    assert recommended_action("unknown")[0] == "Send for expert identification"


# --- 13 and 14. hotspots ---------------------------------------------------

@case("detections aggregate into the grid cell they fall in")
def _():
    size = cfg.GRID
    items = prepared(detection("a", "mine", 0.9, 10, 10),
                     detection("b", "debris", 0.5, 20, 20),
                     detection("c", "tire", 0.5, size + 10, 10))
    hotspots = build_hotspots(items)
    assert len(hotspots) == 2
    first = next(h for h in hotspots if h["cell"] == [0, 0])
    assert first["detection_count"] == 2
    assert set(first["detection_ids"]) == {"a", "b"}


@case("hotspots rank by total severity, so one mine beats a few tyres")
def _():
    items = prepared(
        detection("mine", "mine", 0.9, 10, 10),
        *[detection(f"t{i}", "tire", 0.9, cfg.GRID + 10 + i * 80, 10) for i in range(2)])
    hotspots = build_hotspots(items)
    assert hotspots[0]["dominant_class"] == "mine"
    assert hotspots[0]["detection_count"] == 1
    assert hotspots[1]["detection_count"] == 2, "the crowded cell ranks second"
    totals = [h["total_severity"] for h in hotspots]
    assert totals == sorted(totals, reverse=True)


@case("but enough low-severity objects DO outrank one high-severity object")
def _():
    # This is the honest edge of the total-severity rule and it is tested so
    # nobody discovers it on stage. One mine at 0.9 scores 0.900. Five tyres at
    # 0.9 score 0.270 each and sum to 1.350, so the tyre field ranks first.
    #
    # That is the specified rule working, not a defect: the brief ranks by
    # total severity. What separates the two cells is max_severity and
    # risk_score, both of which put the mine ahead, and both of which are in
    # the export and on the map beside the total.
    items = prepared(
        detection("mine", "mine", 0.9, 10, 10),
        *[detection(f"t{i}", "tire", 0.9, cfg.GRID + 10 + i * 80, 10) for i in range(5)])
    hotspots = build_hotspots(items)
    tyres = next(h for h in hotspots if h["dominant_class"] == "tire")
    mine = next(h for h in hotspots if h["dominant_class"] == "mine")

    assert tyres["total_severity"] > mine["total_severity"]
    assert tyres["priority_rank"] < mine["priority_rank"]
    # The two measures that tell them apart, and do not agree with the rank.
    assert mine["max_severity"] > tyres["max_severity"]
    assert mine["risk_score"] > tyres["risk_score"]


@case("the dominant class is severity-weighted, not a headcount")
def _():
    items = prepared(detection("m", "mine", 0.9, 10, 10),
                     detection("d1", "debris", 0.9, 100, 10),
                     detection("d2", "debris", 0.9, 200, 10),
                     detection("d3", "debris", 0.9, 300, 10))
    hotspot = build_hotspots(items)[0]
    assert hotspot["dominant_class"] == "mine"
    assert hotspot["recommended_action"] == cfg.ACTIONS["mine"]


@case("hotspot ids are H001 upward and agree with the rank")
def _():
    items = prepared(*[detection(f"d{i}", "mine", 0.9 - i * 0.1, i * cfg.GRID + 10, 10)
                       for i in range(3)])
    hotspots = build_hotspots(items)
    assert [h["hotspot_id"] for h in hotspots] == ["H001", "H002", "H003"]
    assert [h["priority_rank"] for h in hotspots] == [1, 2, 3]


@case("derived hotspot metrics follow their documented formulas")
def _():
    items = prepared(detection("a", "mine", 0.8, 10, 10),
                     detection("b", "debris", 0.6, 100, 60))
    hotspot = build_hotspots(items)[0]
    expected_risk = round(hotspot["max_severity"] + cfg.RISK_DENSITY_WEIGHT *
                          (hotspot["total_severity"] - hotspot["max_severity"]), 4)
    assert hotspot["risk_score"] == expected_risk
    assert hotspot["hazard_diversity"] == 2
    assert hotspot["severity_per_detection"] == round(hotspot["total_severity"] / 2, 4)
    assert hotspot["spatial_extent"]["width_px"] == 90.0
    assert hotspot["spatial_extent"]["height_px"] == 50.0
    assert abs(hotspot["spatial_extent"]["diagonal_px"] - math.hypot(90, 50)) < 0.01
    # The worst member's word, which a class floor can lift above the score's.
    assert hotspot["severity_tier"] == highest_tier([d["severity_tier"] for d in items])


@case("a hotspot centroid stays null unless every member is located")
def _():
    items = prepared(detection("a", "mine", 0.9, 10, 10),
                     detection("b", "mine", 0.9, 300, 300))
    items[0]["latitude"], items[0]["longitude"] = 12.9, 74.8
    items[1]["latitude"], items[1]["longitude"] = None, None
    hotspot = build_hotspots(items)[0]
    # Averaging the located half would put a confident coordinate in the wrong
    # place, which is worse than having none.
    assert hotspot["centroid"]["latitude"] is None


# --- 15. empty survey ------------------------------------------------------

@case("a survey with no detections produces empty, valid output")
def _():
    assert deduplicate([]) == []
    assert build_hotspots([]) == []
    summary = build_summary(raw_count=0, detections=[], hotspots=[], strips=["s"],
                            tiles=4, georeferenced=False,
                            coordinate_mode=cfg.COORD_MODE_RELATIVE)
    assert summary["total_deduplicated_detections"] == 0
    assert summary["highest_priority_hotspot"] is None
    assert summary["highest_severity_class"] is None
    assert summary["total_severity"] == 0
    assert set(summary["detections_by_tier"].values()) == {0}


# --- 16 and 17. export -----------------------------------------------------

@case("an exported detection carries its whole audit trail")
def _():
    items = deduplicate(prepared(detection("a", "mine", 0.88, 100, 100)))
    record = export_detection(items[0])
    for field in ("id", "object_class", "confidence", "class_weight", "severity",
                  "severity_tier", "recommended_action", "global_x", "global_y",
                  "latitude", "longitude", "provenance"):
        assert field in record, f"export is missing {field}"
    assert record["severity"] == round(record["class_weight"] * record["confidence"], 4)
    assert record["provenance"]["merged_count"] == 1
    assert record["provenance"]["representative_tile"] == "s_0_0.jpg"


@case("the export document is JSON-serialisable and has all six blocks")
def _():
    items = deduplicate(prepared(detection("a", "mine", 0.88, 100, 100)))
    hotspots = build_hotspots(items)
    export = build_export(
        metadata={"engine": cfg.ENGINE_NAME},
        summary=build_summary(raw_count=1, detections=items, hotspots=hotspots,
                              strips=["s"], tiles=1, georeferenced=False,
                              coordinate_mode=cfg.COORD_MODE_RELATIVE),
        detections=items, hotspots=hotspots,
        provenance={"coordinate_mode": cfg.COORD_MODE_RELATIVE},
        configuration=run_configuration())
    assert list(export) == ["metadata", "survey_summary", "detections",
                            "hotspots", "configuration", "provenance"]
    # Round-trips through JSON: no numpy scalars, no sets, no Paths.
    restored = json.loads(json.dumps(export))
    assert restored["detections"][0]["object_class"] == "mine"
    assert restored["configuration"]["severity_policy"]["formula"] == \
        "severity = class_weight * confidence"


@case("the export never claims official authority")
def _():
    configuration = run_configuration()
    text = json.dumps(configuration).lower()
    for word in ("navy", "coast guard", "noaa", "imo"):
        # The disclaimer names them in order to disclaim them, and that is the
        # only place any of them may appear.
        outside = text.replace(cfg.DISCLAIMER.lower(), "")
        assert word not in outside, f"{word!r} appears outside the disclaimer"
    assert "not" in configuration["disclaimer"].lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    failures = []
    for name, fn in CASES:
        try:
            fn()
            if args.verbose:
                print(f"  ok    {name}")
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"FAILED  {len(failures)} of {len(CASES)} unit tests")
        return 1
    print(f"PASSED  {len(CASES)}/{len(CASES)} unit tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
